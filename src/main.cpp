/*
 * ESP32-S3 Bionic Hand: real-time EMG gesture classification
 * ============================================================
 *
 * Pipeline: continuous 1 kHz sampling (esp_timer) -> IIR filters -> 250 ms
 * window -> TD4 features -> TFLite Micro MLP -> decision layer -> servos.
 *
 * Decision layer:
 *   - softmax confidence threshold (CONFIDENCE_THRESHOLD)
 *   - debounce: DEBOUNCE_FRAMES identical confident predictions in a row;
 *     any low-confidence frame resets the counter
 *   - rest fallback: REST_FALLBACK_FRAMES low-confidence frames in a row
 *     emit REST (hand opens instead of staying frozen)
 *   - motion lock: MOTION_LOCK_MS after a movement; REST bypasses the lock and
 *     never starts one
 *
 * Class names and REST index come from model_meta.h (generated at training
 * time), so a retrained model cannot silently change the label order.
 */

#include <Arduino.h>
#include "model.h"        // model_tflite[] byte array (include ONLY here)
#include "model_meta.h"   // GESTURE_NAMES[], MODEL_NUM_CLASSES, REST_CLASS_INDEX
#include "functions.h"
#include "filters.h"
#include "servo_controller.h"

#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// ============================================================================
// DECISION LAYER CONFIGURATION
// ============================================================================
#define CONFIDENCE_THRESHOLD 0.8f
#define DEBOUNCE_FRAMES      2      // consecutive confident frames to confirm
#define REST_FALLBACK_FRAMES 4      // low-confidence frames before emitting REST (~1 s)
#define MOTION_LOCK_MS       1000   // cooldown after a movement gesture

// ============================================================================
// TFLITE GLOBALS
// ============================================================================
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Weights stay in flash; the arena holds activations and tensor metadata only.
constexpr int kTensorArenaSize = 30 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// ============================================================================
// DECISION STATE
// ============================================================================
static int debounce_gesture = -1;
static int debounce_counter = 0;
static int no_confidence_frames = 0;
static int last_emitted = -1;                 // last gesture sent to the hand
static unsigned long last_movement_time = 0;  // millis() of last non-REST emission

ServoController servoController;

// Called by prelim_collection() while waiting for the next window.
void sampling_idle_hook() { servoController.update(); }

static void halt_forever(const char* why) {
  Serial.print("FATAL: "); Serial.println(why);
  Serial.println("Device HALTED - press reset");
  while (true) { delay(1000); Serial.print("."); }
}

static void emit_gesture(int idx, const char* reason) {
  Serial.printf("GESTURE DETECTED: %s (%s)\n", GESTURE_NAMES[idx], reason);
  servoController.moveToGesture(idx);
  last_emitted = idx;
  if (idx != REST_CLASS_INDEX) last_movement_time = millis();
}

// ============================================================================
// SETUP
// ============================================================================
void setup() {
  Serial.begin(921600);
  delay(1000);
  Serial.println("\n=== ESP32-S3 Bionic Hand: real-time inference ===");
  Serial.printf("Free heap: %u bytes, free PSRAM: %u bytes\n", ESP.getFreeHeap(), ESP.getFreePsram());
  Serial.printf("Model: %s\n", MODEL_DESCRIPTION);
  Serial.printf("Trained on: %s\n", MODEL_TRAINING_DATA);
  Serial.printf("Classes (%d):", MODEL_NUM_CLASSES);
  for (int i = 0; i < MODEL_NUM_CLASSES; i++) Serial.printf(" [%d]%s", i, GESTURE_NAMES[i]);
  Serial.printf("\nREST index: %d\n", REST_CLASS_INDEX);

  // TFLite Micro
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) halt_forever("model schema version mismatch");

  static tflite::ops::micro::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena,
                                                     kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) halt_forever("AllocateTensors failed (increase kTensorArenaSize)");

  input = interpreter->input(0);
  output = interpreter->output(0);
  if (input->dims->data[input->dims->size - 1] != NUM_FEATURES) halt_forever("model input size != NUM_FEATURES");
  if (output->dims->data[output->dims->size - 1] != MODEL_NUM_CLASSES) halt_forever("model output size != MODEL_NUM_CLASSES");
  if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) halt_forever("model is not float32");
  Serial.println("TFLite ready");

  // Servos, then sampling (sampling last so the first window is clean)
  servoController.begin();
  servoController.setHome();
  last_emitted = REST_CLASS_INDEX;

  sampling_start();
  Serial.println("Setup complete. Classifying...\n");
}

// ============================================================================
// LOOP
// ============================================================================
void loop() {
  servoController.update();

  float* feats = prelim_collection();
  for (int i = 0; i < NUM_FEATURES; i++) input->data.f[i] = feats[i];

  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Invoke failed");
    return;
  }

  // argmax with confidence threshold
  int pred = -1;
  float best = CONFIDENCE_THRESHOLD;
  for (int i = 0; i < MODEL_NUM_CLASSES; i++) {
    float p = output->data.f[i];
    if (p > best) { best = p; pred = i; }
  }

  // Debug line: features summary + probabilities
  Serial.print("MAV:");
  for (int s = 0; s < NUM_SENSORS; s++) Serial.printf(" %.4f", feats[4 * s]);
  Serial.print(" | P:");
  for (int i = 0; i < MODEL_NUM_CLASSES; i++) Serial.printf(" %.2f", output->data.f[i]);
  Serial.println();

  // ---- debounce + rest fallback ----
  int confirmed = -1;
  if (pred == -1) {
    debounce_gesture = -1;
    debounce_counter = 0;
    no_confidence_frames++;
    if (no_confidence_frames >= REST_FALLBACK_FRAMES && last_emitted != REST_CLASS_INDEX) {
      emit_gesture(REST_CLASS_INDEX, "fallback: no confident class");
    }
  } else {
    no_confidence_frames = 0;
    if (pred == debounce_gesture) {
      debounce_counter++;
    } else {
      debounce_gesture = pred;
      debounce_counter = 1;
    }
    if (debounce_counter >= DEBOUNCE_FRAMES) confirmed = pred;
  }
  if (confirmed == -1) return;

  // ---- motion lock ----
  unsigned long now = millis();
  bool in_lock = (now - last_movement_time) < MOTION_LOCK_MS;

  if (confirmed == REST_CLASS_INDEX) {
    if (last_emitted != REST_CLASS_INDEX) emit_gesture(REST_CLASS_INDEX, "confirmed");
    return;                                    // REST bypasses and never starts a lock
  }
  if (in_lock) {
    Serial.printf("  [LOCKED %lu ms] %s ignored\n",
                  MOTION_LOCK_MS - (now - last_movement_time), GESTURE_NAMES[confirmed]);
    return;
  }
  if (confirmed != last_emitted) emit_gesture(confirmed, "confirmed");

  static uint32_t last_stats_ms = 0;
  if (now - last_stats_ms > 10000) {
    SamplingStats st = sampling_stats();
    Serial.printf("  [sampler] samples=%lu overruns=%lu max_cb=%lu us\n",
                  (unsigned long)st.samples_total, (unsigned long)st.overruns, (unsigned long)st.max_callback_us);
    last_stats_ms = now;
  }
}
