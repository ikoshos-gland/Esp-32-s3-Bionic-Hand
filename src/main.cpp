/*Libraries*/
#include <Arduino.h>
//Tensorflow model converted to C++
#include "model.h"
//functions
#include "functions.h"

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
// FIXED: Added 16-byte alignment as required by TensorFlow Lite
// INCREASED: 30KB (was 20KB) for better safety margin
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB for TD4 + MLP
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // ← 16-byte aligned

// Helper: Safe Int8 clamping for quantization (prevents overflow)
inline int8_t clamp_int8(float value) {
  if (value > 127.0f) return 127;
  if (value < -128.0f) return -128;
  return (int8_t)value;
}

// Helper: Clamp probability to valid range [0.0, 1.0]
inline float clamp_prob(float value) {
  if (value > 1.0f) return 1.0f;
  if (value < 0.0f) return 0.0f;
  return value;
}
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

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n\n=== STARTING SETUP ===");

  // Configure ADC attenuation for 0-3.3V range (12-bit: 0-4095)
  // ESP32-S3 ADC defaults to 11dB attenuation (0-2500mV), we need full range
  analogSetAttenuation(ADC_11db);  // 0-3.3V mapping to 0-4095
  Serial.println("✅ ADC configured: 11dB attenuation (0-3.3V → 0-4095)");

  // Set up TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    Serial.println("❌ FATAL: Model schema version mismatch - HALTING");
    while(1) { delay(1000); }  // Halt execution
  }

  // Pull in all operations
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    Serial.println("❌ FATAL: TensorFlow Lite memory allocation failed");
    Serial.printf("   Tensor arena size: %d bytes (%d KB)\n", kTensorArenaSize, kTensorArenaSize/1024);
    Serial.println("   Possible causes:");
    Serial.println("   - Model too large for tensor arena");
    Serial.println("   - Insufficient memory available");
    Serial.println("   - Increase kTensorArenaSize in main.cpp");
    Serial.println("\n🛑 HALTING - Cannot continue without TFLite");
    while(1) { delay(1000); }  // Halt execution - do not continue to loop()
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Setup complete! Starting gesture classification...");
  Serial.println("==================================================");
  Serial.println();
  Serial.println("✅ TD4 Feature Extraction Active (Literature-Based)");
  Serial.printf("   Features: %d (4 TD4 × 6 EMG sensors)\n", NUM_FEATURES);
  Serial.println("   Model: Wide & Deep MLP with Int8 Quantization");
  Serial.println("   TD4: MAV, WL, ZC, SSC (Hudgins et al.)");
  Serial.println();
}



void loop() {
  // Get sensor features (Phase 1: 57 time-domain features)
  float* features = prelim_collection();

  // Copy features into TFLite input tensor buffer
  // FIXED: Proper Int8 quantization using TFLite parameters
  // SAFETY: Clamp values to prevent overflow from sensor spikes/noise
  // Model expects Int8 input, features are in [0.0, 1.0] range
  for (int i = 0; i < NUM_FEATURES; i++) {
    // Quantize: float → int8 using TFLite's quantization params
    // Formula: quantized = (float_value / scale) + zero_point
    float scaled = (features[i] / input->params.scale) + input->params.zero_point;
    input->data.int8[i] = clamp_int8(scaled);  // SAFE: Prevents overflow [-128, 127]
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

  // Display output (optimized compact format)
  // FIXED: Proper Int8 dequantization using TFLite parameters
  // SAFETY: Clamp probabilities to valid range [0.0, 1.0]
  Serial.print("Probabilities: ");
  for(int i = 0; i < 11; i++) {
    // Dequantize: int8 → float using TFLite's quantization params
    // Formula: float_value = (quantized - zero_point) * scale
    float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    prob = clamp_prob(prob);  // SAFE: Ensure [0.0, 1.0] range
    Serial.printf("%.3f ", prob);
  }
  Serial.println();

  // Find highest confidence gesture (above threshold)
  this_predict = -1;
  float max_confidence = 0.8;  // Threshold
  for (int i = 0; i < 11; i++) {
    // FIXED: Dequantize int8 output before comparison
    // SAFETY: Clamp to prevent invalid probability values
    float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    prob = clamp_prob(prob);  // SAFE: Ensure [0.0, 1.0] range
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
