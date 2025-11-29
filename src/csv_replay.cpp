/*
ESP32-S3 CSV Replay Mode - Offline Model Validation (8-Sensor)
================================================================

Purpose: Receive pre-recorded 8-channel EMG data from Python script and run TFLite inference
         for offline model validation without physical sensors.

Architecture:
  Python (csv_replay.py) → Serial (Binary Protocol) → ESP32 (This File)
                                                        ↓
  Window Packet (4007 bytes) → Populate raw_sensor_data → extract_features_from_local_buffer()
                                                        ↓
                                    TFLite Inference → Response Packet (138 bytes)
                                                        ↓
                                    Python (Validation & Logging)

Serial Protocol:
  - Window Packet: Header(2) + GT(1) + Size(2) + Data(4000) + Checksum(2) = 4007 bytes
    - 8 sensors × 250 samples × 2 bytes = 4000 bytes
  - Response Packet: Header(2) + Pred(1) + Conf(4) + Features(128) + GT(1) + Checksum(2) = 138 bytes
    - 32 features (8 sensors × 4 TD4 features) × 4 bytes = 128 bytes

Key Features:
  - Supports 8-channel EMG models (independent of 6-sensor hardware config)
  - Bypasses DSP filters (CSV data already clean/filtered)
  - Implements TD4 feature extraction for 8 sensors (32 features total)
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
// BINARY PROTOCOL STRUCTURES - REAL-TIME STREAMING MODE
// =============================================================================

// Sample Packet: Python → ESP32 (21 bytes per sample)
// Simulates real-time ADC sample arrival
#pragma pack(push, 1)
struct SamplePacket {
  uint8_t header[2];        // Sync bytes: 0xAA, 0x55
  uint16_t sensors[8];      // 8 sensors × 2 bytes = 16 bytes
  uint8_t ground_truth;     // Gesture label (0-10)
  uint16_t checksum;        // Sum of sensor bytes % 65536
};
#pragma pack(pop)

// Response Packet: ESP32 → Python (138 bytes total)
// Sent ONLY when a complete window (250 samples) is collected
#pragma pack(push, 1)
struct ResponsePacket {
  uint8_t header[2];           // Sync bytes: 0xB5, 0x6B
  uint8_t predicted_gesture;   // 0-10 or 255 (no prediction)
  float confidence;            // Float32 (0.0-1.0)
  float features[32];          // 32× Float32 TD4 features for 8 sensors (128 bytes)
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
constexpr size_t RX_BUFFER_SIZE = 4096;

// Circular buffer state for real-time streaming
volatile int buffer_write_index = 0;      // Current position in circular buffer
volatile int samples_collected = 0;        // Samples collected in current window
volatile uint8_t window_ground_truth = 0;  // Ground truth for current window
volatile bool window_ready = false;        // Flag: window complete, ready for inference

// Local raw sensor data buffer for 8-sensor CSV replay validation
// NOTE: Using 8 sensors for CSV replay (model trained with 8 channels)
// This is independent of the 6-sensor hardware configuration in functions.h
#define CSV_NUM_SENSORS 8
#define CSV_NUM_FEATURES 32  // 4 TD4 features × 8 sensors

float raw_sensor_data_local[CSV_NUM_SENSORS][RAW_WINDOW_SIZE];

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Extract features from local raw buffer (8-sensor CSV replay version)
 *
 * IMPORTANT: This function processes 8 EMG sensors for CSV replay validation,
 * independent of the 6-sensor hardware configuration.
 *
 * The implementation follows the same logic as functions.cpp::extract_features_from_raw():
 * - Same DC offset removal (mean subtraction)
 * - Same TD4 feature calculations (MAV, WL, ZC, SSC)
 * - Same thresholds (ZC_THRESHOLD_ADC, SSC_THRESHOLD_ADC from functions.h)
 * - Same normalization (ADC_MAX_GLOBAL, RAW_WINDOW_SIZE)
 *
 * @param raw_data 2D array [CSV_NUM_SENSORS][RAW_WINDOW_SIZE]
 * @param output_features Output array [CSV_NUM_FEATURES]
 */
void extract_features_from_local_buffer(float raw_data[CSV_NUM_SENSORS][RAW_WINDOW_SIZE],
                                        float* output_features) {
  int feat_idx = 0;

  // DIAGNOSTIC: Enable detailed logging for first window
  // TEMPORARILY DISABLED - conflicts with binary protocol
  static bool first_window_logged = false;
  bool enable_logging = false;  // Set to true to enable ESP32-side diagnostics

  if (enable_logging) {
    Serial.println("\n=== DIAGNOSTIC: Feature Extraction Pipeline ===");
  }

  // Process each sensor (8 sensors for CSV replay)
  for (int s = 0; s < CSV_NUM_SENSORS; s++) {
    float* data = raw_data[s];

    // DIAGNOSTIC: Log raw ADC values (first 5 samples per sensor)
    if (enable_logging && s < 2) {  // Only log first 2 sensors to save space
      Serial.printf("\nSensor %d - Raw ADC (first 5): ", s);
      for (int i = 0; i < 5; i++) {
        Serial.printf("%.2f ", data[i]);
      }
      Serial.println();
    }

    // STEP 1: Calculate DC offset (mean value)
    float mean = 0.0f;
    for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
      mean += data[i];
    }
    mean /= RAW_WINDOW_SIZE;

    // DIAGNOSTIC: Log DC offset
    if (enable_logging && s < 2) {
      Serial.printf("Sensor %d - DC offset (mean): %.4f\n", s, mean);
    }

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
          if (fabs(centered - prev_centered) >= ZC_THRESHOLD_ADC) {
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

        if (product >= SSC_THRESHOLD_ADC) {
          ssc++;
        }
      }

      prev_centered = centered;
    }

    // STEP 4: Calculate final feature values
    mav /= RAW_WINDOW_SIZE;

    // DIAGNOSTIC: Log raw TD4 features BEFORE normalization
    if (enable_logging && s < 2) {
      Serial.printf("Sensor %d - Raw TD4 (before norm): MAV=%.4f, WL=%.2f, ZC=%d, SSC=%d\n",
                    s, mav, wl, zc, ssc);
    }

    // STEP 5: Apply GLOBAL normalization (using ADC range from functions.h)
    // CRITICAL: Must match functions.cpp::extract_features_from_raw() and Python pipeline
    mav = mav / ADC_MAX_GLOBAL;  // Normalize to [0,1]
    wl = wl / (ADC_MAX_GLOBAL * RAW_WINDOW_SIZE);  // Normalize by 4095 × 250

    // ZC and SSC normalized to rate (counts per sample)
    float zc_normalized = (float)zc / RAW_WINDOW_SIZE;
    float ssc_normalized = (float)ssc / RAW_WINDOW_SIZE;

    // DIAGNOSTIC: Log normalized TD4 features
    if (enable_logging && s < 2) {
      Serial.printf("Sensor %d - Normalized TD4: MAV=%.6f, WL=%.6f, ZC=%.6f, SSC=%.6f\n",
                    s, mav, wl, zc_normalized, ssc_normalized);
    }

    // STEP 6: Store features (order MUST match Python training!)
    output_features[feat_idx++] = mav;
    output_features[feat_idx++] = wl;
    output_features[feat_idx++] = zc_normalized;
    output_features[feat_idx++] = ssc_normalized;
  }

  // DIAGNOSTIC: Mark first window as logged
  if (enable_logging) {
    first_window_logged = true;
    Serial.println("=== END DIAGNOSTIC ===\n");
  }
}

/**
 * Calculate checksum for sample packet
 * 
 * @param sensors Array of 8 sensor values (uint16_t)
 * @return Checksum (sum of bytes % 65536)
 */
uint16_t calculate_sample_checksum(const uint16_t* sensors) {
  uint32_t sum = 0;
  const uint8_t* byteData = (const uint8_t*)sensors;
  size_t byteLen = 8 * 2;  // 8 sensors × 2 bytes
  
  for (size_t i = 0; i < byteLen; i++) {
    sum += byteData[i];
  }
  return (uint16_t)(sum % 65536);
}

/**
 * Receive single sample from Python via serial and add to circular buffer
 * 
 * This function simulates real-time ADC sample arrival. When 250 samples
 * are collected (one complete window), it sets window_ready flag.
 * 
 * @param packet Pointer to SamplePacket structure
 * @return true if sample received and validated successfully
 */
bool receive_sample(SamplePacket* packet) {
  // Wait for header sync bytes (0xAA, 0x55)
  unsigned long timeout_start = millis();
  constexpr unsigned long TIMEOUT_MS = 5000;  // 5 second timeout

  while (millis() - timeout_start < TIMEOUT_MS) {
    if (Serial.available() >= 2) {
      uint8_t byte1 = Serial.read();
      if (byte1 == 0xAA) {
        uint8_t byte2 = Serial.peek();
        if (byte2 == 0x55) {
          // Header found, read full packet
          packet->header[0] = byte1;
          packet->header[1] = Serial.read();

          // Read remaining packet (19 bytes)
          size_t remaining = sizeof(SamplePacket) - 2;
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
            // Incomplete packet - silently return false
            return false;
          }

          // Validate checksum
          uint16_t expected_checksum = calculate_sample_checksum(packet->sensors);
          if (packet->checksum != expected_checksum) {
            // Checksum mismatch - silently return false
            return false;
          }

          // Sample validated - add to circular buffer
          for (int sensor = 0; sensor < CSV_NUM_SENSORS; sensor++) {
            raw_sensor_data_local[sensor][buffer_write_index] = (float)packet->sensors[sensor];
          }

          // Store ground truth (first sample in window sets it)
          if (samples_collected == 0) {
            window_ground_truth = packet->ground_truth;
          }

          // Increment counters
          buffer_write_index++;
          samples_collected++;

          // Check if window is complete
          if (samples_collected >= RAW_WINDOW_SIZE) {
            window_ready = true;
            buffer_write_index = 0;  // Reset for next window
            samples_collected = 0;
          }

          return true;  // Sample received and added to buffer
        }
      }
    }
  }

  // Timeout - no sample received
  return false;
}

/**
 * Send response packet to Python via serial
 *
 * CRITICAL FIX: Checksum calculation matches Python's byte-based approach
 * Python expects: sum of ALL bytes before checksum field (134 bytes)
 * - predicted_gesture (1 byte)
 * - confidence (4 bytes)
 * - features (128 bytes = 32 × float32)
 * - ground_truth (1 byte)
 *
 * @param response Pointer to ResponsePacket structure
 */
void send_response(ResponsePacket* response) {
  // Calculate checksum: sum of all data bytes before checksum field
  const uint8_t* data = (const uint8_t*)response;
  size_t checksum_offset = offsetof(ResponsePacket, checksum);

  uint32_t sum = 0;
  for (size_t i = 2; i < checksum_offset; i++) {  // Skip header
    sum += data[i];
  }
  response->checksum = (uint16_t)(sum % 65536);

  // Send entire packet
  Serial.write((uint8_t*)response, sizeof(ResponsePacket));
  Serial.flush();
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

  // Verify tensor dimensions (8 sensors = 32 features)
  if (input->dims->data[1] != CSV_NUM_FEATURES) {
    Serial.printf("❌ ERROR: Input tensor size mismatch (got %d, expected %d)\n",
                  input->dims->data[1], CSV_NUM_FEATURES);
  }
  if (output->dims->data[1] != 7) {
    Serial.printf("❌ ERROR: Output tensor size mismatch (got %d, expected 7)\n",
                  output->dims->data[1]);
  }

  Serial.println("\n=== SETUP COMPLETE ===");
  Serial.println("Ready to receive window packets from Python");
  Serial.println("Protocol: Binary (3007-byte windows → 106-byte responses)");
  Serial.println("Waiting for data...\n");
}

void loop() {
  // Prepare packet structures
  static SamplePacket sample_packet;
  static ResponsePacket response_packet;

  // Initialize response header
  response_packet.header[0] = 0xB5;
  response_packet.header[1] = 0x6B;

  // Receive samples one-by-one (simulates real-time ADC)
  if (receive_sample(&sample_packet)) {
    // Sample successfully added to circular buffer
    // Check if we have a complete window
    
    if (window_ready) {
      // --- WINDOW COMPLETE - START INFERENCE ---
      window_ready = false;  // Reset flag

      // DIAGNOSTIC: Track window count
      // TEMPORARILY DISABLED - conflicts with binary protocol
      static uint32_t window_count = 0;
      window_count++;
      bool enable_model_logging = false;  // Set to true to enable ESP32-side model diagnostics

      // 1. Extract features from the complete window
      float features[CSV_NUM_FEATURES];
      extract_features_from_local_buffer(raw_sensor_data_local, features);

      // DIAGNOSTIC: Log input tensor information (first window only)
      if (enable_model_logging) {
        Serial.println("\n=== DIAGNOSTIC: Model Input/Output ===");
        Serial.printf("Input tensor type: %s\n",
                      input->type == kTfLiteFloat32 ? "Float32" :
                      input->type == kTfLiteInt8 ? "Int8" : "Unknown");

        if (input->type == kTfLiteInt8) {
          Serial.printf("Input quantization: scale=%.8f, zero_point=%d\n",
                        input->params.scale, input->params.zero_point);
        }

        Serial.println("\nAll 32 feature values (to be fed to model):");
        for (int i = 0; i < CSV_NUM_FEATURES; i++) {
          Serial.printf("  F[%2d]: %.8f", i, features[i]);
          if (i % 4 == 3) Serial.println();  // New line every 4 features
        }
        Serial.println();
      }

      // 2. Copy features to TFLite input tensor
      for (int i = 0; i < CSV_NUM_FEATURES; i++) {
        input->data.f[i] = features[i];
      }

      // 3. Run inference
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        // Inference failed - send error response
        response_packet.predicted_gesture = 255;
        response_packet.confidence = 0.0f;
        memset(response_packet.features, 0, sizeof(response_packet.features));
        response_packet.ground_truth = window_ground_truth;
        send_response(&response_packet);
        return;
      }

      // 4. Find highest confidence prediction
      int predicted_gesture = 0;
      float max_confidence = output->data.f[0];

      // DIAGNOSTIC: Log output tensor information
      if (enable_model_logging) {
        Serial.printf("\nOutput tensor type: %s\n",
                      output->type == kTfLiteFloat32 ? "Float32" :
                      output->type == kTfLiteInt8 ? "Int8" : "Unknown");

        if (output->type == kTfLiteInt8) {
          Serial.printf("Output quantization: scale=%.8f, zero_point=%d\n",
                        output->params.scale, output->params.zero_point);
        }

        Serial.println("\nModel output probabilities (all 7 gestures):");
      }

      for (int i = 1; i < 7; i++) {
        float prob = output->data.f[i];

        // DIAGNOSTIC: Log each probability
        if (enable_model_logging) {
          Serial.printf("  Gesture[%2d]: %.6f", i, prob);
          if (prob == max_confidence) Serial.print(" <- MAX");
          Serial.println();
        }

        if (prob > max_confidence) {
          max_confidence = prob;
          predicted_gesture = i;
        }
      }

      // DIAGNOSTIC: Log first gesture probability (already set as max_confidence initially)
      if (enable_model_logging) {
        Serial.printf("  Gesture[%2d]: %.6f", 0, output->data.f[0]);
        if (output->data.f[0] == max_confidence) Serial.print(" <- MAX");
        Serial.println();
      }

      // Apply confidence threshold
      if (max_confidence < 0.3f) {
        predicted_gesture = 255;  // No prediction
      }

      // DIAGNOSTIC: Log final prediction
      if (enable_model_logging) {
        Serial.printf("\nFinal prediction: Gesture %d (confidence=%.6f, threshold=0.3)\n",
                      predicted_gesture, max_confidence);
        Serial.printf("Ground truth (from packet): %d\n", window_ground_truth);
        Serial.println("=== END DIAGNOSTIC ===\n");
      }

      // 5. Populate response packet
      response_packet.predicted_gesture = (uint8_t)predicted_gesture;
      response_packet.confidence = max_confidence;
      memcpy(response_packet.features, features, sizeof(float) * CSV_NUM_FEATURES);
      response_packet.ground_truth = window_ground_truth;

      // 6. Send binary response to Python
      send_response(&response_packet);
    }
    // If window not ready yet, continue collecting samples
  } else {
    // Failed to receive sample - could be timeout or checksum error
    // Flush serial buffer to resync
    while (Serial.available()) {
      Serial.read();
    }
    delay(10);  // Brief pause
  }
}
