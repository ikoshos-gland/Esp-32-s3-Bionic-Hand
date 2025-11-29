# CSV Replay Diagnostic Implementation - Complete

## Summary

**Phase 1 (Diagnostic Logging)** has been successfully implemented to identify preprocessing mismatches causing 0% accuracy in the CSV replay validation system.

**Date**: 2025-11-29
**Status**: ✅ Ready to Test

---

## What Was Done

### 1. Analysis of Original Suggestions

After thorough codebase analysis, I determined:

| Suggestion | Status | Reason |
|------------|--------|---------|
| **#1: Magnitude Scaling Fix** | ❌ Not Applicable | Your setup correctly scales normalized floats to ADC. No 2000x magnitude difference exists. |
| **#2: Timing Fix** | ✅ Already Implemented | Python already waits for ESP32 response after sending all samples. No UART overflow. |
| **#3: Filter Bypass Fix** | ✅ Correct As-Is | Your data is pre-filtered, so bypassing ESP32 filters is the right approach. |

**Conclusion**: The three original suggestions don't apply to your 8-sensor, normalized float, pre-filtered configuration.

---

### 2. Root Cause Hypothesis

The 0% accuracy is likely caused by **subtle preprocessing differences** between:
- How your model was trained (external pipeline - unknown)
- How csv_replay extracts features (current implementation)

Potential mismatches:
- DC offset removal timing
- Normalization scheme differences
- ZC/SSC threshold values
- Int8 quantization parameters (if applicable)

---

### 3. Diagnostic Logging Added

#### ESP32 Side ([src/csv_replay.cpp](src/csv_replay.cpp))

**Feature Extraction Pipeline** (lines 148-262):
```cpp
// For FIRST WINDOW only:
✓ Raw ADC values (first 5 samples, first 2 sensors)
✓ DC offset (mean) per sensor
✓ Raw TD4 features BEFORE normalization
✓ Normalized TD4 features AFTER normalization
```

**Model Input/Output** (lines 509-607):
```cpp
// For FIRST WINDOW only:
✓ Input tensor type (Float32/Int8)
✓ Input quantization params (if Int8)
✓ All 32 feature values fed to model
✓ Output tensor type
✓ Output quantization params (if Int8)
✓ All 11 gesture probabilities
✓ Final prediction vs ground truth
```

#### Python Side ([scripts_ai/validation/csv_replay.py](scripts_ai/validation/csv_replay.py))

**Before Sending** (lines 437-446):
```python
# For FIRST WINDOW only:
✓ Normalized EMG from CSV (first 5 samples, all 8 sensors)
✓ Scaled ADC after scale_to_adc() (first 5 samples, all 8 sensors)
```

**After Receiving** (lines 475-486):
```python
# For FIRST WINDOW only:
✓ Ground truth label
✓ Predicted gesture from ESP32
✓ Confidence score
✓ All 32 features received from ESP32
```

---

## Next Steps - Action Required

### Step 1: Build and Upload Firmware

```bash
# Navigate to project root
cd C:\Users\MERT\Documents\PlatformIO\Projects\real_time_esp322

# Build and upload csv_replay environment
pio run -e csv_replay -t upload

# Optional: Monitor serial output during upload
pio run -e csv_replay -t upload && pio device monitor
```

**Expected Output**:
```
=== CSV Replay Mode - 8-Channel Validation ===
✓ Model loaded
✓ Interpreter created
✓ Tensors allocated successfully
Input tensor shape: [32]
Output tensor shape: [11]

=== SETUP COMPLETE ===
Ready to receive window packets from Python
Waiting for data...
```

---

### Step 2: Run CSV Replay Validation

```bash
# Navigate to validation scripts
cd scripts_ai\validation

# Run replay on your CSV file
python csv_replay.py ..\..\data\raw\S1_20P_C1_R1.csv --port COM11 --baud 921600
```

**What You'll See**:

The script will output diagnostic information for the **first window only**:

```
=== PYTHON DIAGNOSTIC: First Window ===
Normalized EMG (first 5 samples, all sensors):
  Sample 0: [ 0.053864 -0.09659  -0.026093 ...]
Scaled ADC (first 5 samples, all sensors):
  Sample 0: [2157.  1849.  1993. ...]
=== END PYTHON DIAGNOSTIC ===

Window   0/X: GT=GestureName   PRED=GestureName  (0.XX) STATUS

=== DIAGNOSTIC: Feature Extraction Pipeline ===
Sensor 0 - Raw ADC (first 5): 2157.00 1849.00 ...
Sensor 0 - DC offset (mean): 2048.1234
Sensor 0 - Raw TD4 (before norm): MAV=123.45, WL=5678.90, ZC=15, SSC=28
Sensor 0 - Normalized TD4: MAV=0.030147, WL=0.005548, ZC=0.060000, SSC=0.112000
...
=== END DIAGNOSTIC ===

=== DIAGNOSTIC: Model Input/Output ===
Input tensor type: Float32
All 32 feature values (to be fed to model):
  F[ 0]: 0.03014700  F[ 1]: 0.00554800  F[ 2]: 0.06000000  F[ 3]: 0.11200000
  ...

Output tensor type: Float32
Model output probabilities (all 11 gestures):
  Gesture[ 0]: 0.123456
  Gesture[ 1]: 0.234567
  ...

Final prediction: Gesture X (confidence=0.XX, threshold=0.3)
Ground truth (from packet): Y
=== END DIAGNOSTIC ===

=== PYTHON DIAGNOSTIC: ESP32 Response ===
Ground truth: Y
Predicted: X
Confidence: 0.XX
Features received from ESP32 (all 32):
  Sensor1_MAV: 0.03014700
  Sensor1_WL: 0.00554800
  ...
=== END PYTHON DIAGNOSTIC ===
```

Then validation continues for remaining windows (without diagnostic output).

---

### Step 3: Analyze Diagnostic Output

#### Save the Output

```bash
# Run with output redirection to save diagnostics
python csv_replay.py ..\..\data\raw\S1_20P_C1_R1.csv --port COM11 > diagnostic_output.txt 2>&1
```

#### Key Things to Check

1. **Raw ADC Scaling**
   - Python sends: `[2157. 1849. 1993. ...]`
   - ESP32 receives: `2157.00 1849.00 1993.00 ...`
   - **Should match exactly!**

2. **DC Offset (Mean)**
   - Should be around 2048 for centered signals
   - If very different, indicates scaling issue

3. **Raw TD4 Features (Before Normalization)**
   - MAV: ~50-500 ADC units
   - WL: ~1000-50000
   - ZC/SSC: ~5-80 counts
   - **Compare with your training pipeline output!**

4. **Normalized TD4 Features**
   - MAV: ~0.01-0.12
   - WL: ~0.001-0.05
   - ZC/SSC: ~0.02-0.32
   - **Must match training data ranges!**

5. **Model Output Probabilities**
   - **Uniform distribution (~0.09 each)** = Model is confused (preprocessing mismatch!)
   - **One dominant probability** = Model works but might predict wrong class
   - **NaN or extreme values** = Quantization/normalization issue

---

### Step 4: Compare with Your Training Pipeline

**Critical**: You need to run the **same CSV data** through your training feature extraction code and compare:

```python
# Your training code (example)
import your_training_module as train

# Load same CSV file
data = train.load_csv('data/raw/S1_20P_C1_R1.csv')

# Extract features for first window
first_window = data[0:250, :]  # First 250 samples, all 8 sensors
features = train.extract_features(first_window)

# Print features
print("Training pipeline features:")
for i, feat in enumerate(features):
    sensor = i // 4 + 1
    feat_type = ['MAV', 'WL', 'ZC', 'SSC'][i % 4]
    print(f"Sensor{sensor}_{feat_type}: {feat:.8f}")
```

**Compare** these features with the ones logged by ESP32!

---

## Expected Outcomes

### Scenario A: Features Match
- ESP32 features ≈ Training features (within 5% tolerance)
- **Action**: Issue is not preprocessing - check model file, quantization, or other factors

### Scenario B: Features Don't Match
- ESP32 features significantly different from training
- **Action**: Identify which stage differs (DC offset, normalization, thresholds)
- **Next Step**: Apply targeted fixes in Phase 2

### Scenario C: Model Outputs Uniform
- All probabilities ≈ 0.09 (1/11)
- **Cause**: Model is completely confused
- **Most Likely**: Preprocessing mismatch or quantization issue

---

## Phase 2: Apply Fixes (After Diagnosis)

Once you identify the exact mismatch from diagnostic output, we'll apply targeted fixes:

### Fix Option A: Adjust Normalization
Modify `csv_replay.cpp` lines 238-239 to match training normalization

### Fix Option B: Remove scale_to_adc()
If training used normalized floats directly, modify `csv_replay.py` and `csv_replay.cpp` to skip ADC scaling

### Fix Option C: Adjust ZC/SSC Thresholds
Modify `functions.h` constants to match training thresholds

### Fix Option D: Fix Int8 Quantization
Add proper quantization in `csv_replay.cpp` if model uses Int8

---

## Documentation

- **Implementation Guide**: [CSV_REPLAY_DIAGNOSTIC_GUIDE.md](CSV_REPLAY_DIAGNOSTIC_GUIDE.md)
- **Implementation Plan**: [C:\Users\MERT\.claude\plans\goofy-herding-candy.md](.claude/plans/goofy-herding-candy.md)

---

## Files Modified

| File | Lines | Changes |
|------|-------|---------|
| `src/csv_replay.cpp` | 148-262 | Feature extraction diagnostic logging |
| `src/csv_replay.cpp` | 509-607 | Model input/output diagnostic logging |
| `scripts_ai/validation/csv_replay.py` | 437-446 | Python input diagnostic logging |
| `scripts_ai/validation/csv_replay.py` | 475-486 | Python response diagnostic logging |

---

## Support

If you encounter issues or need help:

1. **Share diagnostic output** (first window only) from `diagnostic_output.txt`
2. **Share training code** feature extraction implementation
3. I can identify the exact fix needed

---

## Quick Reference Commands

```bash
# 1. Build & upload firmware
pio run -e csv_replay -t upload

# 2. Run validation with logging
cd scripts_ai\validation
python csv_replay.py ..\..\data\raw\S1_20P_C1_R1.csv --port COM11 > diagnostic_output.txt 2>&1

# 3. View diagnostic output
type diagnostic_output.txt

# 4. Check accuracy
# Look for "REPLAY COMPLETE" section at end of output
```

---

**Status**: ✅ Ready for Testing
**Next Action**: Run Step 1 (Build & Upload) and Step 2 (Run Validation)
