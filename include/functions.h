#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_

#include <Arduino.h>
#include <stdint.h>
#include <cmath>

// Class names, REST index and feature count come from the auto-generated
// model header so that firmware and trained model can never disagree.
#include "model_meta.h"

// ============================================================================
// FEATURE CONFIGURATION (Hudgins TD4: MAV, WL, ZC, SSC per sensor)
// ============================================================================
#define NUM_SENSORS      6
#define NUM_FEATURES     (4 * NUM_SENSORS)

static_assert(NUM_FEATURES == MODEL_NUM_FEATURES,
              "model_meta.h: deployed model input size does not match NUM_FEATURES");
static_assert(MODEL_NUM_CLASSES >= 2, "model_meta.h: model must have at least 2 classes");
static_assert(REST_CLASS_INDEX >= 0 && REST_CLASS_INDEX < MODEL_NUM_CLASSES,
              "model_meta.h: REST_CLASS_INDEX out of range");

// ============================================================================
// SAMPLING CONFIGURATION (unified 1000 Hz, must match Python training)
// ============================================================================
#define SAMPLING_FREQ    1000                  // Hz
#define SAMPLE_PERIOD_US (1000000 / SAMPLING_FREQ)
#define RAW_WINDOW_SIZE  250                   // 250 ms window at 1 kHz
#define INFERENCE_HOP    250                   // new samples between two inferences
#define RING_SIZE        1024                  // per-sensor ring buffer (power of two)

static_assert((RING_SIZE & (RING_SIZE - 1)) == 0, "RING_SIZE must be a power of two");
static_assert(RING_SIZE >= 2 * RAW_WINDOW_SIZE, "RING_SIZE must hold at least two windows");
static_assert(INFERENCE_HOP >= 1 && INFERENCE_HOP <= RAW_WINDOW_SIZE, "invalid INFERENCE_HOP");

// Global ADC normalization bounds (12-bit ADC on ESP32-S3)
#define ADC_MIN_GLOBAL   0.0f
#define ADC_MAX_GLOBAL   4095.0f

// TD4 thresholds in ADC units (must match Python)
#define ZC_THRESHOLD_ADC  15.0f
#define SSC_THRESHOLD_ADC 15.0f

// ============================================================================
// GPIO (ADC1 pins 4-7, ADC2 pins 15-16; GPIO1/2 are UART0 and must be avoided)
// ============================================================================
#define pin_MW1 4   // ADC1_CH3
#define pin_MW2 5   // ADC1_CH4
#define pin_MW3 6   // ADC1_CH5
#define pin_MW4 7   // ADC1_CH6
#define pin_MW5 15  // ADC2_CH4
#define pin_MW6 16  // ADC2_CH5

// ============================================================================
// BIPOLAR SIGNAL ENCODING (data acquisition serial protocol)
// ============================================================================
// The HPF produces bipolar values; the 16-bit serial protocol carries [0, 4095].
// Encoding adds an offset; Python decodes with  value - 2047.5
#define BIPOLAR_OFFSET 2047.5f

inline uint16_t encode_bipolar(float filtered) {
  float v = filtered + BIPOLAR_OFFSET;
  if (v < 0.0f) v = 0.0f;
  if (v > 4095.0f) v = 4095.0f;
  return (uint16_t)v;
}

inline float decode_bipolar(uint16_t encoded) {
  return (float)encoded - BIPOLAR_OFFSET;
}

// ============================================================================
// CONTINUOUS SAMPLING (esp_timer, 1 kHz, filters run per sample, never paused)
// ============================================================================

// Configure ADC, initialise filters and start the periodic sampler.
void sampling_start();

// Statistics for diagnostics.
struct SamplingStats {
  uint32_t samples_total;     // samples written since start
  uint32_t overruns;          // windows where the consumer fell behind by > RING_SIZE
  uint32_t max_callback_us;   // worst-case sampler callback duration
};
SamplingStats sampling_stats();

// Called repeatedly while prelim_collection() waits for the next window.
// Defined weak so that builds without a servo controller need not provide it.
void sampling_idle_hook() __attribute__((weak));

// Wait until INFERENCE_HOP new samples exist, snapshot the last RAW_WINDOW_SIZE
// samples of every sensor and return the 24 TD4 features.
float *prelim_collection();

// TD4 extraction on the current snapshot (exposed for tests).
float *extract_features_from_raw();

// TD4 primitives (single implementation, used by extract_features_from_raw)
float compute_mav(const float data[], int len);
float compute_wl(const float data[], int len);
// zc: sign test on centered[], threshold on |raw[i]-raw[i+1]|; ssc: slopes from raw[]
int   compute_zc(const float centered[], const float raw[], int len, float threshold);
int   compute_ssc(const float raw[], int len, float threshold);

#endif  // FUNCTIONS_H_
