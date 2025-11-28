/*
 * TD4 Feature Extraction for ESP32-S3 Bionic Hand
 * =================================================
 *
 * REFACTORED: Now processes RAW signals with DC offset removal
 * (Previously used RMS preprocessing which broke ZC/SSC features)
 *
 * Implements Hudgins' TD4 (Time-Domain 4) features - Literature-validated
 * "Gold Standard" for EMG pattern recognition.
 *
 * Features (per EMG sensor):
 * - MAV (Mean Absolute Value): Average signal amplitude
 * - WL (Waveform Length): Signal complexity measure
 * - ZC (Zero Crossings): Frequency estimate
 * - SSC (Slope Sign Changes): Frequency content indicator
 *
 * Total: 4 features × 6 EMG sensors = 24 features
 *
 * CRITICAL CHANGES:
 * 1. RAW ADC data collection (not RMS windows)
 * 2. DC offset removal (centering signal at 0)
 * 3. ZC/SSC calculated on centered bipolar signals
 * 4. Thresholds in ADC units (not normalized)
 * 5. Global normalization (not per-window)
 *
 * WARNING: Model must be RETRAINED with updated Python pipeline!
 *
 * References:
 * - Hudgins et al. (1993): "A New Strategy for Multifunction Myoelectric Control"
 * - Phinyomark et al. (2012): "Feature Reduction and Selection for EMG Signal Classification"
 */

#include "functions.h"
#include "filters.h"
#include <Arduino.h>
#include <cmath>

// Forward declaration of servo controller (only for real-time inference mode)
#ifdef REAL_TIME_INFERENCE_MODE
#include "servo_controller.h"
extern ServoController servoController;
#endif

// TD4 thresholds in ADC units (NOT normalized values)
// CHANGED: Using ADC units for raw signal processing (was 0.01 for normalized RMS)
#define ZC_THRESHOLD_ADC  15.0f   // ADC units - reduces noise-induced crossings
#define SSC_THRESHOLD_ADC 15.0f   // ADC units - reduces noise sensitivity

// Raw signal buffers for 6 EMG sensors (RAW_WINDOW_SIZE samples each)
// CHANGED: Storing raw ADC values instead of RMS windows
// THREAD SAFETY: Static ensures single instance, prevents multi-core race conditions
static float raw_sensor_data[NUM_SENSORS][RAW_WINDOW_SIZE];

// Feature output array (24 features total)
float features[NUM_FEATURES];


// =============================================================================
// TD4 FEATURE EXTRACTION FUNCTIONS
// =============================================================================

/**
 * MAV: Mean Absolute Value
 *
 * MAV = (1/N) × Σ|x[i]|
 *
 * Represents the average amplitude of the signal.
 * Most commonly used EMG feature in literature.
 *
 * @param data Input signal array
 * @param len Array length
 * @return MAV feature value
 */
float compute_mav(float data[], int len) {
  float sum = 0.0;
  for (int i = 0; i < len; i++) {
    sum += fabs(data[i]);
  }
  return sum / len;
}

/**
 * WL: Waveform Length
 *
 * WL = Σ|x[i+1] - x[i]|
 *
 * Measures the complexity of the EMG signal.
 * Related to signal frequency content and amplitude.
 *
 * @param data Input signal array
 * @param len Array length
 * @return WL feature value
 */
float compute_wl(float data[], int len) {
  float wl = 0.0;
  for (int i = 1; i < len; i++) {
    wl += fabs(data[i] - data[i-1]);
  }
  return wl;
}

/**
 * ZC: Zero Crossings
 *
 * ZC = Σ sgn(x[i] × x[i+1] < -threshold)
 *
 * Counts the number of times the signal crosses zero.
 * Provides approximate measure of frequency content.
 * Threshold reduces noise-induced crossings.
 *
 * IMPORTANT: Implementation matches Python exactly:
 * - Count crossings where product is negative
 * - AND absolute difference exceeds threshold
 *
 * @param data Input signal array (normalized)
 * @param len Array length
 * @param threshold Minimum difference for valid crossing
 * @return ZC count
 */
int compute_zc(float data[], int len, float threshold) {
  int zc_count = 0;

  for (int i = 0; i < len - 1; i++) {
    float product = data[i] * data[i+1];
    float diff = fabs(data[i] - data[i+1]);

    // Count crossing if product is negative AND difference exceeds threshold
    if (product < 0 && diff >= threshold) {
      zc_count++;
    }
  }

  return zc_count;
}

/**
 * SSC: Slope Sign Changes
 *
 * SSC = Σ sgn((x[i] - x[i-1]) × (x[i] - x[i+1]) > threshold)
 *
 * Counts the number of times the signal slope changes sign.
 * Related to signal frequency content.
 * Threshold reduces noise sensitivity.
 *
 * IMPORTANT: Implementation matches Python exactly:
 * - Calculate left_slope = x[i] - x[i-1]
 * - Calculate right_slope = x[i] - x[i+1]
 * - Count where product >= threshold
 *
 * @param data Input signal array (normalized)
 * @param len Array length
 * @param threshold Minimum product for valid sign change
 * @return SSC count
 */
int compute_ssc(float data[], int len, float threshold) {
  if (len < 3) return 0;

  int ssc_count = 0;

  for (int i = 1; i < len - 1; i++) {
    float left_slope = data[i] - data[i-1];
    float right_slope = data[i] - data[i+1];
    float product = left_slope * right_slope;

    // Count sign change if product >= threshold
    if (product >= threshold) {
      ssc_count++;
    }
  }

  return ssc_count;
}


// =============================================================================
// DATA COLLECTION PIPELINE
// =============================================================================

/**
 * Collect raw EMG data from 6 sensors
 *
 * REFACTORED Process:
 * 1. Collect RAW_WINDOW_SIZE samples (250ms at 1000Hz = 250 samples per sensor)
 * 2. Store in raw_sensor_data buffers (no RMS preprocessing)
 * 3. Pass to feature extraction
 *
 * CHANGED: No longer uses RMS windows - collects raw ADC values directly
 * FIXED: Reduced from 500 samples @ 2kHz to 250 samples @ 1kHz (matches documentation)
 *
 * @return Pointer to feature array (after full pipeline)
 */
float* prelim_collection() {
  unsigned long next_sample_time = micros();
  
  // Collect RAW_WINDOW_SIZE raw samples from all 6 sensors
  for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
    // Read raw ADC values (0-4095 on ESP32 12-bit ADC) and apply DSP filters
    // CRITICAL FIX: DSP filters (HPF) can output NEGATIVE values (DC offset removal)
    // Casting negative float to uint16_t causes OVERFLOW → Clamp to [0, 4095]
    
    float filtered1 = filter_sample(0, (float)analogRead(pin_MW1));
    float filtered2 = filter_sample(1, (float)analogRead(pin_MW2));
    float filtered3 = filter_sample(2, (float)analogRead(pin_MW3));
    float filtered4 = filter_sample(3, (float)analogRead(pin_MW4));
    float filtered5 = filter_sample(4, (float)analogRead(pin_MW5));
    float filtered6 = filter_sample(5, (float)analogRead(pin_MW6));
    
    // Clamp to 12-bit ADC range [0, 4095] to prevent overflow in feature extraction
    raw_sensor_data[0][i] = constrain(filtered1, 0.0f, 4095.0f);
    raw_sensor_data[1][i] = constrain(filtered2, 0.0f, 4095.0f);
    raw_sensor_data[2][i] = constrain(filtered3, 0.0f, 4095.0f);
    raw_sensor_data[3][i] = constrain(filtered4, 0.0f, 4095.0f);
    raw_sensor_data[4][i] = constrain(filtered5, 0.0f, 4095.0f);
    raw_sensor_data[5][i] = constrain(filtered6, 0.0f, 4095.0f);

    // Update servos every 10 samples (~10ms interval) for smooth motion
    #ifdef REAL_TIME_INFERENCE_MODE
    if (i % 10 == 0) {
      servoController.update();
    }
    #endif

    // REMOVED: Watchdog reset (watchdog disabled in main.cpp setup)

    // Maintain 1000Hz sampling rate (1000 microseconds = 1ms per sample) - FIXED from 2000Hz
    next_sample_time += 1000;
    while (micros() < next_sample_time) {
      delayMicroseconds(10);
    }
  }
  
  // Extract features from raw signals
  return extract_features_from_raw();
}


/**
 * Extract TD4 features from raw ADC signals
 *
 * REFACTORED to process RAW signals correctly:
 * 1. Calculate DC offset (mean) for each sensor
 * 2. Center signal by subtracting mean (creates bipolar signal)
 * 3. Calculate TD4 features on centered signal
 * 4. Apply GLOBAL normalization (not per-window)
 *
 * This fixes the Zero Crossing and Slope Sign Change features which
 * were previously always zero due to RMS preprocessing.
 *
 * @return Pointer to feature array (24 features)
 */
float* extract_features_from_raw() {
  int feat_idx = 0;
  
  // Process each sensor
  for (int s = 0; s < NUM_SENSORS; s++) {
    float* data = raw_sensor_data[s];
    
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
    
    // STEP 5: Apply GLOBAL normalization (using ADC range)
    // This preserves amplitude differences between gestures
    // (Unlike per-window normalization which destroyed this information)
    mav = mav / ADC_MAX_GLOBAL;  // Normalize to [0,1]
    wl = wl / (ADC_MAX_GLOBAL * RAW_WINDOW_SIZE);  // CRITICAL FIX: Must match Python (divides by 4095 × 250)
    
    // ZC and SSC are counts, optionally normalize if needed
    // For now, keep as raw counts (model can learn appropriate scaling)
    float zc_normalized = (float)zc / RAW_WINDOW_SIZE;  // Normalize to rate
    float ssc_normalized = (float)ssc / RAW_WINDOW_SIZE;  // Normalize to rate
    
    // STEP 6: Store features (order MUST match Python training!)
    features[feat_idx++] = mav;
    features[feat_idx++] = wl;
    features[feat_idx++] = zc_normalized;
    features[feat_idx++] = ssc_normalized;
  }
  
  // Verify feature count
  if (feat_idx != NUM_FEATURES) {
    Serial.println("ERROR: Feature count mismatch!");
    Serial.printf("Expected %d features, extracted %d\n", NUM_FEATURES, feat_idx);
  }
  
  return features;
}
