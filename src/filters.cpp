/*
 * DSP Filter Implementation for ESP32-S3 EMG System
 * ==================================================
 *
 * IIR filter cascade for EMG signal conditioning.
 * Uses Direct Form II Transposed biquad structure.
 *
 * Author: ESP32 Bionic Hand Project
 * Date: 2025
 */

#include "filters.h"
#include "filter_coefficients_50hz.h"  // Auto-generated, 1000 Hz unless FILTER_SAMPLING_2000HZ

// ============================================================================
// GLOBAL FILTER STATES (6 sensors)
// ============================================================================

SensorFilterState filter_states[NUM_SENSORS];

// ============================================================================
// BIQUAD PROCESSING (Direct Form II Transposed)
// ============================================================================

/**
 * Process one sample through a biquad filter
 *
 * Direct Form II Transposed:
 *   y[n] = b0*x[n] + z1
 *   z1   = b1*x[n] - a1*y[n] + z2
 *   z2   = b2*x[n] - a2*y[n]
 *
 * This structure is numerically stable and minimizes coefficient
 * quantization error compared to Direct Form I.
 */
float biquad_process(BiquadFilter* bq, float input) {
  // Output = b0*input + delayed feedback/feedforward
  float output = bq->b0 * input + bq->z1;

  // Update delay line
  bq->z1 = bq->b1 * input - bq->a1 * output + bq->z2;
  bq->z2 = bq->b2 * input - bq->a2 * output;

  return output;
}

// ============================================================================
// MOVING AVERAGE (Optional)
// ============================================================================

#if ENABLE_MOVING_AVG
/**
 * Apply moving average filter (circular buffer)
 *
 * Simple FIR smoothing filter. Adds latency but reduces noise.
 * Disabled by default (use ENABLE_MOVING_AVG in filters.h to enable).
 */
float moving_average(SensorFilterState* state, float input) {
  // Subtract oldest value from sum
  state->ma_sum -= state->ma_buffer[state->ma_index];

  // Add new value to buffer and sum
  state->ma_buffer[state->ma_index] = input;
  state->ma_sum += input;

  // Advance circular buffer index
  state->ma_index = (state->ma_index + 1) % MA_WINDOW_SIZE;

  // Return average
  return state->ma_sum / MA_WINDOW_SIZE;
}
#endif

// ============================================================================
// FILTER INITIALIZATION
// ============================================================================

/**
 * Initialize filter coefficients for one biquad
 */
void init_biquad(BiquadFilter* bq, float b0, float b1, float b2, float a1, float a2) {
  bq->b0 = b0;
  bq->b1 = b1;
  bq->b2 = b2;
  bq->a1 = a1;
  bq->a2 = a2;
  bq->z1 = 0.0f;
  bq->z2 = 0.0f;
}

/**
 * Initialize all filters for all sensors
 *
 * Call once during setup() before starting data collection.
 */
void filters_init() {
  Serial.println("\n=== Initializing DSP Filters ===");

  for (int i = 0; i < NUM_SENSORS; i++) {
    SensorFilterState* state = &filter_states[i];

    // Initialize High-Pass Filter (20 Hz, 4th-order)
    #if ENABLE_HPF
      init_biquad(&state->hpf_stage1,
                  HPF_STAGE1_B0, HPF_STAGE1_B1, HPF_STAGE1_B2,
                  HPF_STAGE1_A1, HPF_STAGE1_A2);
      init_biquad(&state->hpf_stage2,
                  HPF_STAGE2_B0, HPF_STAGE2_B1, HPF_STAGE2_B2,
                  HPF_STAGE2_A1, HPF_STAGE2_A2);
    #endif

    // Initialize Low-Pass Filter (450 Hz, 4th-order)
    #if ENABLE_LPF
      init_biquad(&state->lpf_stage1,
                  LPF_STAGE1_B0, LPF_STAGE1_B1, LPF_STAGE1_B2,
                  LPF_STAGE1_A1, LPF_STAGE1_A2);
      init_biquad(&state->lpf_stage2,
                  LPF_STAGE2_B0, LPF_STAGE2_B1, LPF_STAGE2_B2,
                  LPF_STAGE2_A1, LPF_STAGE2_A2);
    #endif

    // Initialize Notch Filter (50/60 Hz, 2nd-order)
    #if ENABLE_NOTCH
      init_biquad(&state->notch,
                  NOTCH_B0, NOTCH_B1, NOTCH_B2,
                  NOTCH_A1, NOTCH_A2);
    #endif

    // Initialize Moving Average
    #if ENABLE_MOVING_AVG
      for (int j = 0; j < MA_WINDOW_SIZE; j++) {
        state->ma_buffer[j] = 0.0f;
      }
      state->ma_index = 0;
      state->ma_sum = 0.0f;
    #endif
  }

  Serial.println("Filter Configuration:");
  Serial.printf("  Sampling Rate: %d Hz (coefficients designed for this rate)\n", FILTER_SAMPLING_RATE_HZ);

  Serial.print("  High-Pass Filter (20 Hz): ");
  Serial.println(ENABLE_HPF ? "ENABLED" : "DISABLED");

  Serial.print("  Low-Pass Filter (450 Hz): ");
  Serial.println(ENABLE_LPF ? "ENABLED" : "DISABLED");

  Serial.print("  Notch Filter (");
  Serial.print(POWERLINE_FREQ_HZ);
  Serial.print(" Hz): ");
  Serial.println(ENABLE_NOTCH ? "ENABLED" : "DISABLED");

  Serial.print("  Moving Average (N=");
  Serial.print(MA_WINDOW_SIZE);
  Serial.print("): ");
  Serial.println(ENABLE_MOVING_AVG ? "ENABLED" : "DISABLED");

  Serial.println("=== Filters Initialized Successfully ===\n");
}

/**
 * Reset all filter states to zero
 *
 * Call when switching gestures or after long idle periods.
 * Clears transient response from previous signals.
 */
void filters_reset() {
  for (int i = 0; i < NUM_SENSORS; i++) {
    SensorFilterState* state = &filter_states[i];

    // Reset biquad states
    #if ENABLE_HPF
      state->hpf_stage1.z1 = 0.0f;
      state->hpf_stage1.z2 = 0.0f;
      state->hpf_stage2.z1 = 0.0f;
      state->hpf_stage2.z2 = 0.0f;
    #endif

    #if ENABLE_LPF
      state->lpf_stage1.z1 = 0.0f;
      state->lpf_stage1.z2 = 0.0f;
      state->lpf_stage2.z1 = 0.0f;
      state->lpf_stage2.z2 = 0.0f;
    #endif

    #if ENABLE_NOTCH
      state->notch.z1 = 0.0f;
      state->notch.z2 = 0.0f;
    #endif

    // Reset moving average
    #if ENABLE_MOVING_AVG
      for (int j = 0; j < MA_WINDOW_SIZE; j++) {
        state->ma_buffer[j] = 0.0f;
      }
      state->ma_index = 0;
      state->ma_sum = 0.0f;
    #endif
  }
}

// ============================================================================
// FILTER CASCADE
// ============================================================================

/**
 * Process a single ADC sample through the complete filter cascade
 *
 * Filter order: Raw → HPF → LPF → Notch → (MA) → Filtered
 *
 * Args:
 *   sensor_index: Sensor number (0-5)
 *   raw_sample: Raw ADC value (0-4095)
 *
 * Returns:
 *   Filtered ADC value
 */
float filter_sample(int sensor_index, float raw_sample) {
  // Validate sensor index
  if (sensor_index < 0 || sensor_index >= NUM_SENSORS) {
    return raw_sample;  // Return unfiltered if invalid index
  }

  SensorFilterState* state = &filter_states[sensor_index];
  float signal = raw_sample;

  // High-Pass Filter (20 Hz, 4th-order = 2 biquad stages)
  // Removes DC drift and motion artifacts (0-20 Hz)
  #if ENABLE_HPF
    signal = biquad_process(&state->hpf_stage1, signal);
    signal = biquad_process(&state->hpf_stage2, signal);
  #endif

  // Low-Pass Filter (450 Hz, 4th-order = 2 biquad stages)
  // Removes electronic noise (>500 Hz)
  #if ENABLE_LPF
    signal = biquad_process(&state->lpf_stage1, signal);
    signal = biquad_process(&state->lpf_stage2, signal);
  #endif

  // Notch Filter (50/60 Hz, 2nd-order = 1 biquad)
  // Removes powerline interference
  #if ENABLE_NOTCH
    signal = biquad_process(&state->notch, signal);
  #endif

  // Moving Average (optional smoothing)
  #if ENABLE_MOVING_AVG
    signal = moving_average(state, signal);
  #endif

  return signal;
}
