#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_ //header guard to prevent adding it multiple times

#include <Arduino.h>  // For constrain() macro (needed by encode_bipolar)
#include <stdint.h>
//C++ mathematical functions
#include <cmath>

// Feature extraction configuration - TD4 Features (Literature-Based)
// Hudgins' TD4: MAV, WL, ZC, SSC (4 features × 6 EMG sensors = 24 total)
#define NUM_FEATURES 24
#define NUM_SENSORS  6

// Raw data collection configuration
// CHANGED: Using raw signal processing instead of RMS windows
#define RAW_WINDOW_SIZE  250   // 250ms window (250 samples at 1kHz) - Unified 1000 Hz sampling
#define SAMPLING_FREQ    1000  // 1000 Hz sampling rate - Unified for all modes

// Global ADC normalization bounds (12-bit ADC on ESP32)
// CRITICAL: Must match Python training pipeline normalization
#define ADC_MIN_GLOBAL   0.0f
#define ADC_MAX_GLOBAL   4095.0f

// Safe ESP32-S3 GPIO pins (avoiding UART0/Serial conflicts)
// CHANGED: GPIO 1-2 conflict with UART0, now using ADC1 pins 4-7 and ADC2 pins 15-16
// WARNING: You must physically reconnect sensors to these new pins!
#define pin_MW1 4   // GPIO 4  (ADC1_CH3)
#define pin_MW2 5   // GPIO 5  (ADC1_CH4)
#define pin_MW3 6   // GPIO 6  (ADC1_CH5)
#define pin_MW4 7   // GPIO 7  (ADC1_CH6)
#define pin_MW5 15  // GPIO 15 (ADC2_CH4)
#define pin_MW6 16  // GPIO 16 (ADC2_CH5)

// ============================================================================
// BIPOLAR SIGNAL ENCODING/DECODING
// ============================================================================
// HPF (High-Pass Filter) creates bipolar signals (negative values after DC removal),
// but uint16_t serial protocol requires [0, 4095] range.
// Solution: Use offset encoding to preserve negative values during transmission.

#define BIPOLAR_OFFSET 2047.5f  // ADC midpoint (4095 / 2) for symmetric bipolar range

/**
 * Encode bipolar float to uint16_t for serial transmission
 *
 * Maps: float [-2047.5, +2047.5] → uint16_t [0, 4095]
 *
 * Example transformations:
 *   -2000.0 →   48  (negative preserved as low values)
 *       0.0 → 2048  (DC center maps to ADC midpoint)
 *   +2000.0 → 4048  (positive maps to high values)
 *
 * @param filtered Bipolar filtered signal (can be negative)
 * @return Encoded uint16_t in range [0, 4095]
 */
inline uint16_t encode_bipolar(float filtered) {
    return (uint16_t)constrain(filtered + BIPOLAR_OFFSET, 0.0f, 4095.0f);
}

/**
 * Decode uint16_t to bipolar float (for documentation - Python uses own decoding)
 *
 * Maps: uint16_t [0, 4095] → float [-2047.5, +2047.5]
 *
 * Note: This function is for C++ reference only.
 * Python training pipeline implements its own decoding:
 *   df[col] = df[col] - 2047.5
 *
 * @param encoded Encoded uint16_t from serial transmission
 * @return Decoded bipolar float
 */
inline float decode_bipolar(uint16_t encoded) {
    return (float)encoded - BIPOLAR_OFFSET;
}

// Raw signal buffers (6 sensors × RAW_WINDOW_SIZE samples)
// CHANGED: Storing raw ADC values instead of RMS windows
// NOTE: Buffer is private to functions.cpp (static) for thread safety

// Main pipeline functions
// CHANGED: Simplified pipeline - removed RMS preprocessing
float *prelim_collection();
float *extract_features_from_raw();

// TD4 Feature Extraction Functions (Hudgins et al.)
// Literature-validated time-domain features for EMG pattern recognition

// MAV: Mean Absolute Value - average signal amplitude
float compute_mav(float data[], int len);

// WL: Waveform Length - signal complexity measure
float compute_wl(float data[], int len);

// ZC: Zero Crossings - frequency estimate (with threshold for noise reduction)
int compute_zc(float data[], int len, float threshold);

// SSC: Slope Sign Changes - frequency content indicator (with threshold)
int compute_ssc(float data[], int len, float threshold);

#endif  // FUNCTIONS_H_
