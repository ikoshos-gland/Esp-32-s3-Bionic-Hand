/*Libraries*/
#include <Arduino.h>
//Tensorflow model converted to C++
#include "model.h"
//functions
#include "functions.h"
//DSP filters for EMG signal conditioning
#include "filters.h"
//Servo controller for robotic hand
#include "servo_controller.h"

//Tensorflow custom library for ESP32 (from lib folder)
#include <TensorFlowLite_ESP32.h>
// Use the experimental micro API (which is in lib folder)
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Create an area of memory for input, output, and intermediate arrays
// TD4 features: 24 features (4 TD4 × 6 sensors) with Wide & Deep MLP
// Float32 model (no quantization for TFLite v2.1.1 compatibility)
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB for TD4 + MLP
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // 16-byte aligned
}  // namespace

int this_predict = -1;
int last_predict = -1;

// Debouncing: Require stable predictions across multiple frames
#define DEBOUNCE_FRAMES 2              // 2 consecutive predictions (2×270ms = ~540ms)
int debounce_counter = 0;              // How many times current gesture seen consecutively
int debounce_gesture = -1;             // Which gesture we're currently debouncing
int confirmed_gesture = -1;            // Last confirmed stable gesture

// Motion Locking: Prevent rapid re-detection after gesture confirmation
#define MOTION_LOCK_MS 1000            // 1000ms cooldown after gesture detection
unsigned long last_gesture_time = 0;   // Timestamp of last detected gesture (millis)

// Servo Controller: Controls 6 servos for robotic hand gestures
ServoController servoController;

void setup() {
  Serial.begin(921600);
  Serial.println("BOOT: Serial started at 921600 baud");
  delay(1000);
  Serial.println("BOOT: Delay complete, initializing...");

  Serial.println("\n\n=== STARTING SETUP ===");

  // ADDED: Memory diagnostics to detect memory issues
  Serial.printf("BOOT: Free heap: %d bytes (%.1f KB)\n", ESP.getFreeHeap(), ESP.getFreeHeap()/1024.0);
  Serial.printf("BOOT: Free PSRAM: %d bytes (%.1f KB)\n", ESP.getFreePsram(), ESP.getFreePsram()/1024.0);
  Serial.printf("BOOT: Tensor arena size: %d bytes (%d KB)\n", kTensorArenaSize, kTensorArenaSize/1024);

  // Configure ADC attenuation for 0-3.3V range (12-bit: 0-4095)
  // ESP32-S3 ADC defaults to 11dB attenuation (0-2500mV), we need full range
  analogSetAttenuation(ADC_11db);  // 0-3.3V mapping to 0-4095
  Serial.println("✅ ADC configured: 11dB attenuation (0-3.3V → 0-4095)");

  // Initialize DSP filters (HPF + LPF + Notch)
  filters_init();

  // Set up TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model
  Serial.println("BOOT: Loading TFLite model...");
  model = tflite::GetModel(model_tflite);
  Serial.printf("BOOT: Model schema version: %d (expected: %d)\n",
                model->version(), TFLITE_SCHEMA_VERSION);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    Serial.println("❌ FATAL: Model schema version mismatch");
    Serial.println("   Device is HALTED - Press reset button to retry");
    while(1) {
      delay(1000);
      Serial.print(".");  // Heartbeat shows device is alive but halted
    }
  }
  Serial.println("BOOT: Model loaded successfully");

  // Pull in all operations
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  Serial.println("BOOT: Allocating TensorFlow tensors...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  Serial.printf("BOOT: AllocateTensors() status: %d (0=OK, other=FAIL)\n", allocate_status);

  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("❌ FATAL: TensorFlow Lite memory allocation failed");
    Serial.printf("   Tensor arena size: %d bytes (%d KB)\n", kTensorArenaSize, kTensorArenaSize/1024);
    Serial.println("   Possible causes:");
    Serial.println("   - Model too large for tensor arena");
    Serial.println("   - Insufficient memory available");
    Serial.println("   - Increase kTensorArenaSize in main.cpp");
    Serial.println("\n🛑 Device HALTED - Press reset button to retry");
    while(1) {
      delay(1000);
      Serial.print(".");  // Heartbeat shows device is alive but halted
    }
  }
  Serial.println("BOOT: Tensors allocated successfully");

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Initialize servo controller (after TFLite to ensure memory allocation succeeds)
  Serial.println("BOOT: Initializing servo controller...");
  servoController.begin();
  servoController.setHome();
  Serial.println("✅ Servo controller initialized and moved to home position");

  Serial.println("Setup complete! Starting gesture classification...");
  Serial.println("==================================================");
  Serial.println();
  Serial.println("✅ TD4 Feature Extraction Active (Literature-Based)");
  Serial.printf("   Features: %d (4 TD4 × 6 EMG sensors)\n", NUM_FEATURES);
  Serial.println("   Model: Wide & Deep MLP (Float32 - No Quantization)");
  Serial.println("   TD4: MAV, WL, ZC, SSC (Hudgins et al.)");
  Serial.println("   TFLite: v2.1.1 compatible (SOFTMAX v1)");
}



void loop() {
  // Get sensor features (24 TD4 features from 6 EMG sensors)
  float* features = prelim_collection();

  // Copy features into TFLite input tensor buffer
  // Float32 model - direct copy, no quantization needed
  for (int i = 0; i < NUM_FEATURES; i++) {
    input->data.f[i] = features[i];
  }

  // DEBUG: Display TD4 feature values
  Serial.println("----------------------------------");
  Serial.println("TD4 Features (24 total):");
  Serial.printf("  Input tensor size: %d\n", input->dims->data[1]);
  Serial.printf("  Features extracted: %d\n", NUM_FEATURES);

  // Display features by sensor (4 features per sensor)
  const char* td4_names[] = {"MAV", "WL", "ZC", "SSC"};
  for (int sensor = 0; sensor < 6; sensor++) {
    Serial.printf("  EMG%d: ", sensor + 1);
    for (int feat = 0; feat < 4; feat++) {
      int idx = sensor * 4 + feat;
      Serial.printf("%s=%.3f ", td4_names[feat], features[idx]);
    }
    Serial.println();
  }

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // Display output probabilities (Float32 - no dequantization needed)
  Serial.print("Probabilities: ");
  for(int i = 0; i < 11; i++) {
    float prob = output->data.f[i];  // Direct Float32 access
    Serial.printf("%.3f ", prob);
  }
  Serial.println();

  // Find highest confidence gesture (above threshold)
  this_predict = -1;
  float max_confidence = 0.8;  // Threshold (TEMPORARILY LOWERED: 0.5 = 50%, normally 0.8 = 80%)
  for (int i = 0; i < 11; i++) {
    float prob = output->data.f[i];  // Direct Float32 access
    if (prob > max_confidence) {
      max_confidence = prob;
      this_predict = i;
    }
  }

  // Debouncing: Require DEBOUNCE_FRAMES consecutive identical predictions
  if (this_predict != -1) {
    if (this_predict == debounce_gesture) {
      // Same gesture as last frame, increment counter
      debounce_counter++;
      if (debounce_counter >= DEBOUNCE_FRAMES) {
        // Gesture is stable across required frames, confirm it
        confirmed_gesture = this_predict;
      }
    } else {
      // Different gesture detected, reset debounce counter
      debounce_gesture = this_predict;
      debounce_counter = 1;
      confirmed_gesture = -1;  // Not yet confirmed
    }
    // Only use confirmed stable gestures
    this_predict = confirmed_gesture;
  }

  // Gesture names
  const char* gesture_names[] = {"Rest", "Fist", "Open", "Point", "Victory",
                                  "OK", "ThumbUp", "ThumbDn", "Grasp", "Pinch", "WristFlex"};

  // Motion Locking: Prevent rapid re-detection with cooldown period
  unsigned long current_time = millis();
  bool in_lock_period = (current_time - last_gesture_time) < MOTION_LOCK_MS;
  bool is_rest = (this_predict == 0);  // Rest (class 0) always overrides lock

  if (this_predict != -1) {
    // Check if we're in lock period (Rest gesture always bypasses lock)
    if (!in_lock_period || is_rest) {
      // Only output if gesture changed from last prediction
      if (this_predict != last_predict) {
        Serial.print("GESTURE DETECTED: ");
        Serial.println(gesture_names[this_predict]);

        // Debug: Show debouncing info
        Serial.print("  [Debounced: ");
        Serial.print(debounce_counter);
        Serial.print("/");
        Serial.print(DEBOUNCE_FRAMES);
        Serial.println(" frames]");

        // Trigger servo movement for detected gesture
        servoController.moveToGesture(this_predict);

        last_predict = this_predict;

        // Start lock period (except for Rest gesture)
        if (!is_rest) {
          last_gesture_time = current_time;
        }
      }
    } else {
      // In lock period, ignore new predictions
      Serial.print("  [LOCKED: Ignoring prediction for ");
      Serial.print(MOTION_LOCK_MS - (current_time - last_gesture_time));
      Serial.println("ms]");
    }
  }
}
