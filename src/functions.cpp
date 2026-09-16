/*
 * TD4 Feature Extraction + continuous sampling for ESP32-S3 Bionic Hand
 * =====================================================================
 *
 * Sampling: an esp_timer fires every 1000 us (ESP_TIMER_TASK dispatch), reads
 * the six ADC channels, runs the per-sensor IIR filter cascade and writes the
 * filtered bipolar sample into a ring buffer. Sampling never stops, so the
 * filter states stay continuous across inference, serial printing and servo
 * motion. prelim_collection() only waits for INFERENCE_HOP new samples and
 * snapshots the newest RAW_WINDOW_SIZE samples.
 *
 * Features (per sensor, on the DC-removed window): MAV, WL, ZC, SSC.
 * Normalisation must match scripts_ai/critical/feature_extraction.py exactly:
 *   MAV / 4095,  WL / (4095 * N),  ZC / N,  SSC / N
 *
 * References: Hudgins et al. 1993; Phinyomark et al. 2012.
 */

#include "functions.h"
#include "filters.h"
#include <Arduino.h>
#include <cmath>
#include "esp_timer.h"

// ============================================================================
// RING BUFFER (written by the sampler task, read by the main loop)
// ============================================================================
static float    ring[NUM_SENSORS][RING_SIZE];
static volatile uint32_t write_count = 0;     // total samples written
static uint32_t last_window_end = 0;          // write_count at the last snapshot
static SamplingStats stats = {0, 0, 0};
static esp_timer_handle_t sampler_handle = nullptr;

static const int sensor_pins[NUM_SENSORS] = {pin_MW1, pin_MW2, pin_MW3, pin_MW4, pin_MW5, pin_MW6};

// Snapshot of the newest window and the feature output
static float raw_sensor_data[NUM_SENSORS][RAW_WINDOW_SIZE];
static float features[NUM_FEATURES];

// ============================================================================
// SAMPLER
// ============================================================================
static void sampler_callback(void*) {
  uint32_t t0 = micros();
  uint32_t idx = write_count & (RING_SIZE - 1);
  for (int s = 0; s < NUM_SENSORS; s++) {
    float raw = (float)analogRead(sensor_pins[s]);
    ring[s][idx] = filter_sample(s, raw);
  }
  write_count = write_count + 1;   // single writer; readers only compare counts
  uint32_t dt = micros() - t0;
  if (dt > stats.max_callback_us) stats.max_callback_us = dt;
  stats.samples_total = write_count;
}

void sampling_start() {
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);   // ~0-3.1 V full scale on ESP32-S3
  for (int s = 0; s < NUM_SENSORS; s++) {
    pinMode(sensor_pins[s], INPUT);
    (void)analogRead(sensor_pins[s]);   // one-time ADC init outside the timer task
  }
  filters_init();

  esp_timer_create_args_t args = {};
  args.callback = &sampler_callback;
  args.arg = nullptr;
  args.dispatch_method = ESP_TIMER_TASK;
  args.name = "emg_sampler";
  if (esp_timer_create(&args, &sampler_handle) != ESP_OK) {
    Serial.println("FATAL: esp_timer_create failed");
    while (true) { delay(1000); Serial.print("."); }
  }
  write_count = 0;
  last_window_end = 0;
  esp_timer_start_periodic(sampler_handle, SAMPLE_PERIOD_US);
  Serial.printf("Sampler started: %d Hz, window %d, hop %d, ring %d\n",
                SAMPLING_FREQ, RAW_WINDOW_SIZE, INFERENCE_HOP, RING_SIZE);
}

SamplingStats sampling_stats() { return stats; }

// ============================================================================
// TD4 PRIMITIVES (single implementation)
// ============================================================================
float compute_mav(const float data[], int len) {
  float sum = 0.0f;
  for (int i = 0; i < len; i++) sum += fabsf(data[i]);
  return sum / len;
}

float compute_wl(const float data[], int len) {
  float wl = 0.0f;
  for (int i = 1; i < len; i++) wl += fabsf(data[i] - data[i - 1]);
  return wl;
}

// Sign test on the centered signal, difference on the RAW samples:
// (x_i - m) - (x_j - m) == x_i - x_j exactly, and integer differences are exact
// in float, so threshold ties are decided like in Python.
// Python: zc = sum((c[:-1]*c[1:] < 0) & (|x[:-1]-x[1:]| >= threshold))
int compute_zc(const float centered[], const float raw[], int len, float threshold) {
  int count = 0;
  for (int i = 0; i < len - 1; i++) {
    float product = centered[i] * centered[i + 1];
    float diff = fabsf(raw[i] - raw[i + 1]);
    if (product < 0.0f && diff >= threshold) count++;
  }
  return count;
}

// Python: ssc = sum(((x[i]-x[i-1]) * (x[i]-x[i+1])) >= threshold)  (raw samples)
int compute_ssc(const float raw[], int len, float threshold) {
  if (len < 3) return 0;
  int count = 0;
  for (int i = 1; i < len - 1; i++) {
    float left  = raw[i] - raw[i - 1];
    float right = raw[i] - raw[i + 1];
    if (left * right >= threshold) count++;
  }
  return count;
}

// ============================================================================
// WINDOW ACQUISITION
// ============================================================================
float* prelim_collection() {
  // Wait for INFERENCE_HOP new samples (first call waits for a full window)
  uint32_t need = (last_window_end == 0) ? RAW_WINDOW_SIZE : INFERENCE_HOP;
  while ((uint32_t)(write_count - last_window_end) < need) {
    if (sampling_idle_hook) sampling_idle_hook();
    delayMicroseconds(200);
  }

  uint32_t end = write_count;                 // newest sample index (exclusive)
  if ((uint32_t)(end - last_window_end) > (uint32_t)(RING_SIZE - RAW_WINDOW_SIZE)) {
    stats.overruns++;                          // consumer fell behind, resync to newest
  }
  last_window_end = end;

  uint32_t start = end - RAW_WINDOW_SIZE;
  for (int s = 0; s < NUM_SENSORS; s++) {
    for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
      raw_sensor_data[s][i] = ring[s][(start + i) & (RING_SIZE - 1)];
    }
  }
  return extract_features_from_raw();
}

// ============================================================================
// FEATURE EXTRACTION (DC removal per window, then TD4, then global scaling)
// ============================================================================
float* extract_features_from_raw() {
  static float centered[RAW_WINDOW_SIZE];
  int feat_idx = 0;

  for (int s = 0; s < NUM_SENSORS; s++) {
    const float* data = raw_sensor_data[s];

    float mean = 0.0f;
    for (int i = 0; i < RAW_WINDOW_SIZE; i++) mean += data[i];
    mean /= RAW_WINDOW_SIZE;
    for (int i = 0; i < RAW_WINDOW_SIZE; i++) centered[i] = data[i] - mean;

    float mav = compute_mav(centered, RAW_WINDOW_SIZE);
    float wl  = compute_wl(data, RAW_WINDOW_SIZE);              // diffs identical to centered
    int   zc  = compute_zc(centered, data, RAW_WINDOW_SIZE, ZC_THRESHOLD_ADC);
    int   ssc = compute_ssc(data, RAW_WINDOW_SIZE, SSC_THRESHOLD_ADC);

    features[feat_idx++] = mav / ADC_MAX_GLOBAL;
    features[feat_idx++] = wl / (ADC_MAX_GLOBAL * RAW_WINDOW_SIZE);
    features[feat_idx++] = (float)zc / RAW_WINDOW_SIZE;
    features[feat_idx++] = (float)ssc / RAW_WINDOW_SIZE;
  }

  if (feat_idx != NUM_FEATURES) {
    Serial.printf("ERROR: feature count mismatch, expected %d got %d\n", NUM_FEATURES, feat_idx);
  }
  return features;
}
