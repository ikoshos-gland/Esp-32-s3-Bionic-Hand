/*
ESP32-S3 CSV Replay Mode - Offline Model Validation
====================================================

Purpose: Receive pre-recorded EMG data from Python script and run TFLite inference
         for offline model validation without physical sensors.

Architecture:
  Python (csv_replay.py) → Serial (Binary Protocol) → ESP32 (This File)
                                                        ↓
  Window Packet (3007 bytes) → Populate raw_sensor_data → extract_features_from_raw()
                                                        ↓
                                    TFLite Inference → Response Packet (106 bytes)
                                                        ↓
                                    Python (Validation & Logging)

Serial Protocol:
  - Window Packet: Header(2) + GT(1) + Size(2) + Data(3000) + Checksum(2) = 3007 bytes
  - Response Packet: Header(2) + Pred(1) + Conf(4) + Features(96) + GT(1) + Checksum(2) = 106 bytes

Key Features:
  - Bypasses DSP filters (data already clean/filtered)
  - Reuses existing extract_features_from_raw() function
  - Binary protocol with checksums for reliability
  - Comprehensive error handling and logging

Build Command:
  pio run -e csv_replay -t upload && pio device monitor

References:
  - Plan: C:\Users\MERT\.claude\plans\crystalline-petting-knuth.md
  - TFLite patterns: src/main.cpp
  - Serial protocol: src/data_acquisition.cpp
  - Feature extraction: src/functions.cpp
*/

#include <Arduino.h>

// TensorFlow Lite Model
#include "model.h"

// Feature extraction and filters
#include "functions.h"
#include "filters.h"

// TensorFlow Lite ESP32 library (from lib folder)
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// =============================================================================
// BINARY PROTOCOL STRUCTURES
// =============================================================================

// Window Packet: Python → ESP32 (3007 bytes total)
#pragma pack(push, 1)
struct WindowPacket {
  uint8_t header[2];        // Sync bytes: 0xA5, 0x5A
  uint8_t ground_truth;     // Gesture label (0-10, or 255 = unknown)
  uint16_t window_size;     // Should be 250
  uint16_t data[250 * 6];   // 250 samples × 6 sensors = 1500 uint16_t = 3000 bytes
  uint16_t checksum;        // Sum of all data values % 65536
};
#pragma pack(pop)

// Response Packet: ESP32 → Python (106 bytes total)
#pragma pack(push, 1)
struct ResponsePacket {
  uint8_t header[2];           // Sync bytes: 0xB5, 0x6B
  uint8_t predicted_gesture;   // 0-10 or 255 (no prediction)
  float confidence;            // Float32 (0.0-1.0)
  float features[24];          // 24× Float32 TD4 features (96 bytes)
  uint8_t ground_truth;        // Echo back
  uint16_t checksum;           // Validation
};
#pragma pack(pop)

// =============================================================================
// GLOBAL VARIABLES
// =============================================================================

// TensorFlow Lite globals (same pattern as main.cpp)
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena: 30KB for TD4 + Wide & Deep MLP (same as main.cpp)
constexpr int kTensorArenaSize = 30 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // 16-byte aligned for SIMD
}  // namespace

// Gesture names (MUST match Python training order)
const char* gesture_names[] = {
  "Rest", "Fist", "Open", "Point", "Victory",
  "OK", "ThumbUp", "ThumbDn", "Grasp", "Pinch", "WristFlex"
};

// Serial communication buffer
constexpr size_t RX_BUFFER_SIZE = 4096;  // Large enough for WindowPacket

// Local raw sensor data buffer (same format as functions.cpp)
// NOTE: Cannot use extern since functions.cpp declares it as static
// We'll populate this buffer and pass it to extract_features_from_raw() manually
float raw_sensor_data_local[NUM_SENSORS][RAW_WINDOW_SIZE];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Extract features from local raw buffer
 *
 * This is a wrapper that processes our local buffer using the same
 * TD4 feature extraction logic as functions.cpp extract_features_from_raw()
 *
 * @param raw_data 2D array [NUM_SENSORS][RAW_WINDOW_SIZE]
 * @param output_features Output array [NUM_FEATURES]
 */
void extract_features_from_local_buffer(float raw_data[NUM_SENSORS][RAW_WINDOW_SIZE],
                                        float* output_features) {
  int feat_idx = 0;

  // Process each sensor
  for (int s = 0; s < NUM_SENSORS; s++) {
    float* data = raw_data[s];

    // STEP 1: Calculate DC offset (mean value)
    float mean = 0.0f;
    for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
      mean += data[i];
    }
    mean /= RAW_WINDOW_SIZE;

    // STEP 2: Initialize feature accumulators
    float mav = 0.0f;
    float wl = 0.0f;
    int zc = 0;
    int ssc = 0;

    // STEP 3: Process centered signal (remove DC offset)
    float prev_centered = data[0] - mean;

    for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
      float centered = data[i] - mean;  // Center at 0

      // MAV: Mean Absolute Value
      mav += fabs(centered);

      // WL: Waveform Length
      if (i > 0) {
        wl += fabs(centered - prev_centered);
      }

      // ZC: Zero Crossing (with threshold to avoid noise)
      if (i > 0) {
        // Check if sign changed AND difference is significant
        if ((centered > 0 && prev_centered < 0) ||
            (centered < 0 && prev_centered > 0)) {
          float threshold = 15.0f;  // ZC_THRESHOLD_ADC from functions.cpp
          if (fabs(centered - prev_centered) >= threshold) {
            zc++;
          }
        }
      }

      // SSC: Slope Sign Change
      if (i > 0 && i < RAW_WINDOW_SIZE - 1) {
        float next_centered = data[i+1] - mean;
        float left_slope = centered - prev_centered;
        float right_slope = centered - next_centered;
        float product = left_slope * right_slope;

        float threshold = 15.0f;  // SSC_THRESHOLD_ADC from functions.cpp
        if (product >= threshold) {
          ssc++;
        }
      }

      prev_centered = centered;
    }

    // STEP 4: Calculate final feature values
    mav /= RAW_WINDOW_SIZE;

    // STEP 5: Apply GLOBAL normalization (using ADC range)
    mav = mav / ADC_MAX_GLOBAL;  // Normalize to [0,1]
    wl = wl / (ADC_MAX_GLOBAL * RAW_WINDOW_SIZE);  // Match Python

    // ZC and SSC normalized to rate
    float zc_normalized = (float)zc / RAW_WINDOW_SIZE;
    float ssc_normalized = (float)ssc / RAW_WINDOW_SIZE;

    // STEP 6: Store features (order MUST match Python training!)
    output_features[feat_idx++] = mav;
    output_features[feat_idx++] = wl;
    output_features[feat_idx++] = zc_normalized;
    output_features[feat_idx++] = ssc_normalized;
  }
}

/**
 * Calculate checksum for data integrity
 *
 * @param data Pointer to data array
 * @param len Number of uint16_t values
 * @return Checksum (sum % 65536)
 */
uint16_t calculate_checksum(const uint16_t* data, size_t len) {
  uint32_t sum = 0;
  for (size_t i = 0; i < len; i++) {
    sum += data[i];
  }
  return (uint16_t)(sum % 65536);
}

/**
 * Receive window packet from Python via serial
 *
 * @param packet Pointer to WindowPacket structure
 * @return true if packet received and validated successfully
 */
bool receive_window(WindowPacket* packet) {
  // Wait for header sync bytes (0xA5, 0x5A)
  unsigned long timeout_start = millis();
  constexpr unsigned long TIMEOUT_MS = 5000;  // 5 second timeout

  while (millis() - timeout_start < TIMEOUT_MS) {
    if (Serial.available() >= 2) {
      uint8_t byte1 = Serial.read();
      if (byte1 == 0xA5) {
        uint8_t byte2 = Serial.peek();
        if (byte2 == 0x5A) {
          // Header found, read full packet
          packet->header[0] = byte1;
          packet->header[1] = Serial.read();

          // Read remaining packet (3005 bytes)
          size_t remaining = sizeof(WindowPacket) - 2;
          uint8_t* buf = (uint8_t*)packet + 2;

          // Read with timeout
          size_t bytes_read = 0;
          unsigned long read_start = millis();
          while (bytes_read < remaining && (millis() - read_start < TIMEOUT_MS)) {
            if (Serial.available() > 0) {
              buf[bytes_read++] = Serial.read();
            }
          }

          if (bytes_read < remaining) {
            Serial.printf("ERROR: Incomplete packet (got %d/%d bytes)\n",
                         bytes_read + 2, sizeof(WindowPacket));
            return false;
          }

          // Validate checksum
          uint16_t expected_checksum = calculate_checksum(packet->data, 250 * 6);
          if (packet->checksum != expected_checksum) {
            Serial.printf("ERROR: Checksum mismatch (got 0x%04X, expected 0x%04X)\n",
                         packet->checksum, expected_checksum);
            return false;
          }

          // Validate window size
          if (packet->window_size != 250) {
            Serial.printf("ERROR: Invalid window size (got %d, expected 250)\n",
                         packet->window_size);
            return false;
          }

          return true;  // Packet valid
        }
      }
    }
  }

  Serial.println("ERROR: Timeout waiting for window packet");
  return false;
}

/**
 * Send response packet to Python via serial
 *
 * @param response Pointer to ResponsePacket structure
 */
void send_response(ResponsePacket* response) {
  // Calculate checksum (sum of all feature values)
  uint32_t sum = 0;
  for (int i = 0; i < 24; i++) {
    // Convert float to uint32 for checksum (preserve bits)
    uint32_t bits;
    memcpy(&bits, &response->features[i], sizeof(float));
    sum += bits;
  }
  response->checksum = (uint16_t)(sum % 65536);

  // Send entire packet
  Serial.write((uint8_t*)response, sizeof(ResponsePacket));
  Serial.flush();  // Ensure transmission complete
}

/**
 * Populate local raw_sensor_data buffer from window packet
 *
 * This bypasses real-time ADC reading and DSP filtering.
 * The CSV data is already clean/filtered, so we directly populate
 * our local buffer for feature extraction.
 *
 * @param packet Pointer to WindowPacket containing ADC data
 */
void populate_raw_buffer(const WindowPacket* packet) {
  // Window packet data layout: [S1[0], S2[0], ..., S6[0], S1[1], S2[1], ..., S6[1], ...]
  // Reorganize into: raw_sensor_data_local[sensor][sample]

  for (int sample = 0; sample < 250; sample++) {
    for (int sensor = 0; sensor < 6; sensor++) {
      int index = sample * 6 + sensor;
      raw_sensor_data_local[sensor][sample] = (float)packet->data[index];
    }
  }
}

// =============================================================================
// SETUP & MAIN LOOP
// =============================================================================

void setup() {
  // Initialize serial at high baud rate
  Serial.begin(921600);
  Serial.setRxBufferSize(RX_BUFFER_SIZE);  // Increase RX buffer
  delay(1000);

  Serial.println("\n\n=== ESP32-S3 CSV REPLAY MODE ===");
  Serial.println("Build: CSV Replay - Offline Model Validation");
  Serial.println("============================================");

  // Memory diagnostics
  Serial.printf("Free heap: %d bytes (%.1f KB)\n",
                ESP.getFreeHeap(), ESP.getFreeHeap()/1024.0);
  Serial.printf("Free PSRAM: %d bytes (%.1f KB)\n",
                ESP.getFreePsram(), ESP.getFreePsram()/1024.0);
  Serial.printf("Tensor arena: %d bytes (%d KB)\n",
                kTensorArenaSize, kTensorArenaSize/1024);

  // Configure ADC (not used in CSV replay, but maintain consistency)
  analogSetAttenuation(ADC_11db);
  Serial.println("ADC: Configured (11dB attenuation, not used in replay mode)");

  // Initialize filters (will be bypassed via BYPASS_DSP_FILTERS flag)
  filters_init();

  #ifdef BYPASS_DSP_FILTERS
  Serial.println("✓ DSP FILTERS BYPASSED (CSV data pre-filtered)");
  #else
  Serial.println("WARNING: DSP filters enabled (not recommended for CSV replay)");
  #endif

  // Initialize TensorFlow Lite (same pattern as main.cpp)
  Serial.println("\n--- TensorFlow Lite Initialization ---");

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  Serial.println("Loading TFLite model...");
  model = tflite::GetModel(model_tflite);
  Serial.printf("Model schema version: %d (expected: %d)\n",
                model->version(), TFLITE_SCHEMA_VERSION);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    Serial.println("❌ FATAL: Model schema version mismatch");
    Serial.println("Device HALTED - press reset to retry");
    while(1) delay(1000);
  }
  Serial.println("✓ Model loaded successfully");

  // Setup operations resolver
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  Serial.println("Allocating tensors...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  Serial.printf("AllocateTensors() status: %d (0=OK)\n", allocate_status);

  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("❌ FATAL: Tensor allocation failed");
    Serial.printf("Arena size: %d bytes (%d KB)\n",
                  kTensorArenaSize, kTensorArenaSize/1024);
    Serial.println("Device HALTED - press reset to retry");
    while(1) delay(1000);
  }
  Serial.println("✓ Tensors allocated successfully");

  // Get input/output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.printf("Input tensor shape: [%d]\n", input->dims->data[1]);
  Serial.printf("Output tensor shape: [%d]\n", output->dims->data[1]);

  // Verify tensor dimensions
  if (input->dims->data[1] != NUM_FEATURES) {
    Serial.printf("❌ ERROR: Input tensor size mismatch (got %d, expected %d)\n",
                  input->dims->data[1], NUM_FEATURES);
  }
  if (output->dims->data[1] != 11) {
    Serial.printf("❌ ERROR: Output tensor size mismatch (got %d, expected 11)\n",
                  output->dims->data[1]);
  }

  Serial.println("\n=== SETUP COMPLETE ===");
  Serial.println("Ready to receive window packets from Python");
  Serial.println("Protocol: Binary (3007-byte windows → 106-byte responses)");
  Serial.println("Waiting for data...\n");
}

void loop() {
  // Prepare packet structures
  static WindowPacket window_packet;
  static ResponsePacket response_packet;

  // Initialize response header
  response_packet.header[0] = 0xB5;
  response_packet.header[1] = 0x6B;

  // Receive window packet from Python
  if (receive_window(&window_packet)) {
    unsigned long start_time = micros();

    Serial.println("----------------------------------------");
    Serial.printf("Window received: GT=%d (%s), Size=%d\n",
                  window_packet.ground_truth,
                  window_packet.ground_truth < 11 ? gesture_names[window_packet.ground_truth] : "Unknown",
                  window_packet.window_size);

    // Populate raw_sensor_data buffer from packet
    populate_raw_buffer(&window_packet);
    Serial.println("✓ Buffer populated (250 samples × 6 sensors)");

    // Extract features using our local buffer (DC removal + TD4)
    float features[NUM_FEATURES];
    extract_features_from_local_buffer(raw_sensor_data_local, features);
    Serial.println("✓ Features extracted (24 TD4 features)");

    // Display extracted features (for debugging)
    Serial.println("Features by sensor:");
    const char* td4_names[] = {"MAV", "WL", "ZC", "SSC"};
    for (int sensor = 0; sensor < 6; sensor++) {
      Serial.printf("  EMG%d: ", sensor + 1);
      for (int feat = 0; feat < 4; feat++) {
        int idx = sensor * 4 + feat;
        Serial.printf("%s=%.4f ", td4_names[feat], features[idx]);
      }
      Serial.println();
    }

    // Copy features to TFLite input tensor (Float32 - direct copy)
    for (int i = 0; i < NUM_FEATURES; i++) {
      input->data.f[i] = features[i];
    }

    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed");
      Serial.println("❌ ERROR: TFLite inference failed");

      // Send error response
      response_packet.predicted_gesture = 255;  // Invalid
      response_packet.confidence = 0.0f;
      memset(response_packet.features, 0, sizeof(response_packet.features));
      response_packet.ground_truth = window_packet.ground_truth;
      send_response(&response_packet);
      return;
    }
    Serial.println("✓ Inference complete");

    // Display output probabilities
    Serial.print("Probabilities: ");
    for (int i = 0; i < 11; i++) {
      float prob = output->data.f[i];
      Serial.printf("%.3f ", prob);
    }
    Serial.println();

    // Find highest confidence prediction (threshold = 0.8)
    int predicted_gesture = 255;  // Default: no prediction
    float max_confidence = 0.8f;  // Confidence threshold

    for (int i = 0; i < 11; i++) {
      float prob = output->data.f[i];
      if (prob > max_confidence) {
        max_confidence = prob;
        predicted_gesture = i;
      }
    }

    // Populate response packet
    response_packet.predicted_gesture = (uint8_t)predicted_gesture;
    response_packet.confidence = max_confidence;
    memcpy(response_packet.features, features, sizeof(float) * NUM_FEATURES);
    response_packet.ground_truth = window_packet.ground_truth;

    // Display result
    if (predicted_gesture != 255) {
      Serial.printf("PREDICTION: %s (confidence: %.3f)\n",
                    gesture_names[predicted_gesture], max_confidence);

      bool correct = (predicted_gesture == window_packet.ground_truth);
      Serial.printf("Ground Truth: %s → %s\n",
                    gesture_names[window_packet.ground_truth],
                    correct ? "✓ CORRECT" : "✗ INCORRECT");
    } else {
      Serial.println("PREDICTION: None (below threshold)");
    }

    unsigned long elapsed = micros() - start_time;
    Serial.printf("Processing time: %lu μs (%.2f ms)\n", elapsed, elapsed / 1000.0);

    // Send response to Python
    send_response(&response_packet);
    Serial.println("✓ Response sent");
    Serial.println("Ready for next window...\n");

  } else {
    // Packet receive failed - error already logged
    // Send minimal error response to keep Python in sync
    response_packet.predicted_gesture = 255;
    response_packet.confidence = 0.0f;
    memset(response_packet.features, 0, sizeof(response_packet.features));
    response_packet.ground_truth = 255;
    send_response(&response_packet);

    // Flush serial buffer to resync
    while (Serial.available()) {
      Serial.read();
    }
    delay(100);  // Brief pause before retry
  }
}
