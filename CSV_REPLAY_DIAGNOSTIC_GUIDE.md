# CSV Replay Diagnostic Logging Guide

## Overview

Comprehensive diagnostic logging has been added to the CSV replay system to help identify preprocessing mismatches between training and inference that may be causing 0% accuracy.

**Date**: 2025-11-29
**Purpose**: Phase 1 of the fix plan - diagnostic logging to pinpoint exact preprocessing differences

---

## Changes Made

### 1. ESP32 Side (src/csv_replay.cpp)

Added detailed logging for the **first window only** to avoid overwhelming the serial output:

#### Feature Extraction Pipeline Logging (lines 148-262):
- **Raw ADC values** after scaling from normalized floats (first 5 samples, first 2 sensors)
- **DC offset (mean)** calculated for each sensor
- **Raw TD4 features BEFORE normalization** (MAV, WL, ZC count, SSC count)
- **Normalized TD4 features AFTER normalization** (all 4 features per sensor)

#### Model Input/Output Logging (lines 509-607):
- **Input tensor type** (Float32 vs Int8)
- **Input quantization parameters** (if Int8): scale and zero_point
- **All 32 feature values** being fed to the model
- **Output tensor type** (Float32 vs Int8)
- **Output quantization parameters** (if Int8)
- **All 11 gesture probabilities** from model output
- **Final prediction** and ground truth comparison

### 2. Python Side (scripts_ai/validation/csv_replay.py)

Added logging for the **first window only**:

#### Before Sending (lines 437-446):
- **Normalized EMG values** (first 5 samples, all 8 sensors) from CSV file
- **Scaled ADC values** (first 5 samples, all 8 sensors) after scale_to_adc()

#### After Receiving (lines 475-486):
- **Ground truth** label
- **Predicted** gesture from ESP32
- **Confidence** score
- **All 32 features** received from ESP32 (organized by sensor and feature type)

---

## How to Use the Diagnostic Logging

### Step 1: Run CSV Replay with Logging Enabled

1. **Build and upload** the csv_replay firmware:
   ```bash
   pio run -e csv_replay -t upload
   ```

2. **Run the Python validation script**:
   ```bash
   cd scripts_ai/validation
   python csv_replay.py path/to/your/csv_file.csv --port COM11 --baud 921600
   ```

3. **Capture the output**:
   - The first window will produce extensive diagnostic output
   - Save this output for analysis

### Step 2: Analyze the Diagnostic Output

The logging will help you identify where the preprocessing diverges. Look for:

#### A. Raw ADC Scaling Issues
```
=== PYTHON DIAGNOSTIC: First Window ===
Normalized EMG (first 5 samples, all sensors):
  Sample 0: [ 0.053864 -0.09659  -0.026093  0.027924  0.023346  0.0010681  0.039521  0.087434]
Scaled ADC (first 5 samples, all sensors):
  Sample 0: [2157.  1849.  1993.  2104.  2095.  2049.  2128.  2226.]
```

Then on ESP32:
```
=== DIAGNOSTIC: Feature Extraction Pipeline ===
Sensor 0 - Raw ADC (first 5): 2157.00 ...
Sensor 0 - DC offset (mean): 2048.1234
```

**Check**: Do the ADC values match between Python output and ESP32 input?

#### B. DC Offset Removal
```
Sensor 0 - DC offset (mean): 2048.1234
```

**Check**: Is the mean reasonable? (Should be around 2048 for centered signals)

#### C. Raw TD4 Features (Before Normalization)
```
Sensor 0 - Raw TD4 (before norm): MAV=123.45, WL=5678.90, ZC=15, SSC=28
Sensor 1 - Raw TD4 (before norm): MAV=98.76, WL=4321.10, ZC=12, SSC=22
```

**Check**:
- MAV should be in range ~50-500 ADC units
- WL should be in range ~1000-50000
- ZC/SSC should be reasonable counts (5-80)
- Compare these with your training data features BEFORE normalization

#### D. Normalized TD4 Features (After Normalization)
```
Sensor 0 - Normalized TD4: MAV=0.030147, WL=0.005548, ZC=0.060000, SSC=0.112000
Sensor 1 - Normalized TD4: MAV=0.024116, WL=0.004221, ZC=0.048000, SSC=0.088000
```

**Check**:
- MAV should be ~0.01-0.12 (reasonable amplitude)
- WL should be ~0.001-0.05 (complexity measure)
- ZC/SSC should be ~0.02-0.32 (frequency content)
- **These MUST match your training data feature ranges!**

#### E. Model Input Tensor
```
All 32 feature values (to be fed to model):
  F[ 0]: 0.03014700  F[ 1]: 0.00554800  F[ 2]: 0.06000000  F[ 3]: 0.11200000
  F[ 4]: 0.02411600  F[ 5]: 0.00422100  F[ 6]: 0.04800000  F[ 7]: 0.08800000
  ...
```

**Check**: Compare these with features from your training pipeline on the SAME CSV data

#### F. Model Output Probabilities
```
Model output probabilities (all 11 gestures):
  Gesture[ 0]: 0.123456
  Gesture[ 1]: 0.234567
  Gesture[ 2]: 0.098765 <- MAX
  ...
```

**Check**:
- Are probabilities uniform (~0.09 each)? → Model is confused (preprocessing mismatch!)
- Is one probability dominant? → Model is working but might predict wrong class
- Are probabilities NaN or extremely small? → Quantization or normalization issue

---

## Common Issues to Look For

### Issue 1: Scale Mismatch
**Symptom**: Normalized features are 100x or 1000x different from training
**Cause**: Different ADC_MAX values or normalization formulas
**Fix**: Adjust normalization constants to match training

### Issue 2: DC Offset Timing
**Symptom**: Features are offset from training data
**Cause**: DC offset removed at different pipeline stage
**Fix**: Move DC offset removal to match training

### Issue 3: Threshold Mismatch
**Symptom**: ZC/SSC counts drastically different from training
**Cause**: Different threshold values (15.0 ADC vs normalized units)
**Fix**: Adjust ZC_THRESHOLD_ADC and SSC_THRESHOLD_ADC

### Issue 4: Quantization Mismatch (Int8 models only)
**Symptom**: Model outputs are random or uniform
**Cause**: Features not being quantized, or wrong scale/zero_point
**Fix**: Verify quantization parameters match training

---

## Next Steps After Diagnosis

Once you've captured the diagnostic output:

1. **Compare with your training code**:
   - Run the same CSV data through your training feature extraction
   - Compare the feature values at each stage

2. **Identify the exact mismatch point**:
   - Is it in scaling? DC offset? Feature calculation? Normalization?

3. **Apply targeted fixes**:
   - Modify csv_replay.cpp to match your training pipeline exactly
   - Re-run validation to verify improvement

4. **If you need help**:
   - Share the diagnostic output (first window only)
   - Share your training feature extraction code
   - We can identify the exact fix needed

---

## Disabling Diagnostic Logging

The logging only activates for the **first window** and is automatically disabled afterward to avoid performance impact. No changes needed for production use.

If you want to log additional windows, modify these lines:
- `csv_replay.cpp:149`: Change `first_window_logged` logic
- `csv_replay.cpp:512`: Change `window_count == 1` condition
- `csv_replay.py:438`: Change `window_idx == 0` condition

---

## Files Modified

1. **src/csv_replay.cpp**
   - Lines 148-262: Feature extraction logging
   - Lines 509-607: Model input/output logging

2. **scripts_ai/validation/csv_replay.py**
   - Lines 437-446: Python-side input logging
   - Lines 475-486: Python-side response logging

---

## Example Complete Diagnostic Output

See `CSV_REPLAY_DIAGNOSTIC_EXAMPLE.txt` (to be generated after first run) for a complete example of the diagnostic output and how to interpret it.

---

## Support

If you encounter issues or need help interpreting the diagnostic output, refer back to the implementation plan in:
- `C:\Users\MERT\.claude\plans\goofy-herding-candy.md`
