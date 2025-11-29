/*
 * DSP Filter Library for ESP32-S3 EMG System
 * ===========================================
 *
 * IIR (Infinite Impulse Response) filters for EMG signal conditioning.
 * Uses Direct Form II Transposed biquad structure for numerical stability.
 *
 * Filter Chain:
 * Raw ADC → HPF (20Hz) → LPF (450Hz) → Notch (50/60Hz) → DC Removal → TD4
 *
 * Features:
 * - High-Pass Filter: Removes DC drift and motion artifacts (0-20 Hz)
 * - Low-Pass Filter: Removes electronic noise (>500 Hz)
 * - Notch Filter: Removes powerline interference (50/60 Hz)
 * - Per-sensor state management (6 independent filter chains)
 *
 * Performance:
 * - Memory: ~480 bytes (360 bytes state + 120 bytes coefficients)
 * - Latency: ~12 ms total
 * - CPU: <0.1% @ 240 MHz
 *
 * Author: ESP32 Bionic Hand Project
 * Date: 2025
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <Arduino.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

// Powerline frequency (50 Hz for Europe/Asia, 60 Hz for Americas)
#define POWERLINE_FREQ_HZ 50

// Enable/disable individual filters (1 = enabled, 0 = disabled)
#ifdef BYPASS_DSP_FILTERS
  #define ENABLE_HPF         0  // DISABLED in CSV replay
  #define ENABLE_LPF         0  // DISABLED in CSV replay
  #define ENABLE_NOTCH       0  // DISABLED in CSV replay
#else
  #define ENABLE_HPF         1  // Enabled in production
  #define ENABLE_LPF         1
  #define ENABLE_NOTCH       1
#endif
#define ENABLE_MOVING_AVG  0  // Moving average (adds latency, disabled by default)

// Number of EMG sensors
#define NUM_SENSORS        6

// Moving average window size (if enabled)
#define MA_WINDOW_SIZE     5

// ============================================================================
// BIQUAD FILTER STRUCTURE (Direct Form II Transposed)
// ============================================================================

/**
 * Biquad Filter Structure
 *
 * Transfer Function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
 *
 * Direct Form II Transposed Update Equations:
 *   y[n] = b0*x[n] + z1
 *   z1   = b1*x[n] - a1*y[n] + z2
 *   z2   = b2*x[n] - a2*y[n]
 *
 * Why Direct Form II Transposed?
 * - Most numerically stable IIR structure
 * - Minimizes coefficient quantization error
 * - Only 2 state variables (minimal memory)
 * - Industry standard (used by ARM CMSIS-DSP)
 */
typedef struct {
  float b0, b1, b2;  // Numerator coefficients
  float a1, a2;      // Denominator coefficients (a0 = 1.0, normalized)
  float z1, z2;      // State variables (delay line)
} BiquadFilter;

// ============================================================================
// PER-SENSOR FILTER STATE
// ============================================================================

/**
 * Complete filter state for one EMG sensor
 *
 * Contains all biquad filters and moving average buffer.
 * Each sensor has independent state to prevent cross-contamination.
 */
typedef struct {
  // High-Pass Filter (4th-order = 2 biquad stages)
  BiquadFilter hpf_stage1;
  BiquadFilter hpf_stage2;

  // Low-Pass Filter (4th-order = 2 biquad stages)
  BiquadFilter lpf_stage1;
  BiquadFilter lpf_stage2;

  // Notch Filter (2nd-order = 1 biquad)
  BiquadFilter notch;

  // Moving Average (optional)
  #if ENABLE_MOVING_AVG
    float ma_buffer[MA_WINDOW_SIZE];
    int ma_index;
    float ma_sum;
  #endif
} SensorFilterState;

// ============================================================================
// GLOBAL FILTER STATES (6 sensors)
// ============================================================================

extern SensorFilterState filter_states[NUM_SENSORS];

// ============================================================================
// FUNCTION PROTOTYPES
// ============================================================================

/**
 * Initialize all filters for all sensors
 *
 * Call once during setup() before starting data collection.
 * Loads filter coefficients and zeros all state variables.
 */
void filters_init();

/**
 * Reset filter states to zero
 *
 * Call when switching gestures or after long idle periods.
 * Clears transient response from previous signals.
 */
void filters_reset();

/**
 * Process a single ADC sample through the filter cascade
 *
 * Args:
 *   sensor_index: Sensor number (0-5)
 *   raw_sample: Raw ADC value (0-4095)
 *
 * Returns:
 *   Filtered ADC value
 *
 * Processing order:
 *   raw → HPF → LPF → Notch → (optional MA) → filtered
 */
float filter_sample(int sensor_index, float raw_sample);

/**
 * Process a single sample through one biquad filter
 *
 * Args:
 *   bq: Pointer to biquad filter structure
 *   input: Input sample
 *
 * Returns:
 *   Filtered output sample
 *
 * Internal use only (called by filter_sample)
 */
float biquad_process(BiquadFilter* bq, float input);

/**
 * Apply moving average filter (if enabled)
 *
 * Args:
 *   state: Pointer to sensor filter state
 *   input: Input sample
 *
 * Returns:
 *   Smoothed output
 *
 * Internal use only (called by filter_sample)
 */
#if ENABLE_MOVING_AVG
float moving_average(SensorFilterState* state, float input);
#endif

#endif  // FILTERS_H
