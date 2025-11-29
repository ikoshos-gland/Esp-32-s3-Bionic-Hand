# Diagnostic Output Interpretation Reference

Quick reference guide for interpreting CSV replay diagnostic output.

---

## Normal vs Problematic Values

### 1. Raw ADC Values (After scale_to_adc)

**Expected Range**: 0 - 4095 (12-bit ADC)
**Typical Centered Signal**: 1500 - 2500 (centered around 2048)

```
✅ GOOD:  2157.00 1849.00 1993.00 2104.00
❌ BAD:   4095.00 4095.00 4095.00 4095.00  (all maxed out → scaling issue!)
❌ BAD:   0.05 -0.09 -0.02 0.02              (still normalized → scale_to_adc failed!)
```

---

### 2. DC Offset (Mean)

**Expected Range**: 1800 - 2300 (around 2048 for centered signals)

```
✅ GOOD:  DC offset (mean): 2048.1234
✅ GOOD:  DC offset (mean): 2156.7890
❌ BAD:   DC offset (mean): 4000.0000  (signal too high → clipping)
❌ BAD:   DC offset (mean): 100.0000   (signal too low → scaling wrong)
⚠️  WARN: DC offset (mean): 0.0534     (not scaled to ADC → major issue!)
```

---

### 3. Raw TD4 Features (Before Normalization)

#### MAV (Mean Absolute Value)
**Expected Range**: 50 - 500 ADC units
**Typical**: 100 - 300

```
✅ GOOD:  MAV=123.45
✅ GOOD:  MAV=287.90
❌ BAD:   MAV=2000.00    (too high → signal not centered)
❌ BAD:   MAV=0.05       (too low → not scaled to ADC)
```

#### WL (Waveform Length)
**Expected Range**: 1000 - 50000
**Typical**: 3000 - 15000

```
✅ GOOD:  WL=5678.90
✅ GOOD:  WL=12345.67
❌ BAD:   WL=100000.00   (extremely high → noise or scaling issue)
❌ BAD:   WL=0.50        (too low → not scaled to ADC)
```

#### ZC (Zero Crossings)
**Expected Range**: 5 - 80 counts
**Typical**: 10 - 40

```
✅ GOOD:  ZC=15
✅ GOOD:  ZC=32
❌ BAD:   ZC=0      (no crossings → DC not removed or threshold too high)
❌ BAD:   ZC=200    (too many → noise or threshold too low)
```

#### SSC (Slope Sign Changes)
**Expected Range**: 10 - 150 counts
**Typical**: 20 - 80

```
✅ GOOD:  SSC=28
✅ GOOD:  SSC=65
❌ BAD:   SSC=0      (no changes → smooth signal or threshold too high)
❌ BAD:   SSC=250    (too many → noisy signal)
```

---

### 4. Normalized TD4 Features (After Normalization)

#### MAV (Normalized)
**Expected Range**: 0.01 - 0.12
**Typical**: 0.02 - 0.08

```
✅ GOOD:  MAV=0.030147
✅ GOOD:  MAV=0.067890
❌ BAD:   MAV=0.50000   (too high → normalization wrong)
❌ BAD:   MAV=0.00001   (too low → very weak signal)
```

#### WL (Normalized)
**Expected Range**: 0.001 - 0.05
**Typical**: 0.003 - 0.015

```
✅ GOOD:  WL=0.005548
✅ GOOD:  WL=0.012345
❌ BAD:   WL=0.10000    (too high → normalization wrong)
❌ BAD:   WL=0.000001   (too low → flat signal)
```

#### ZC (Normalized - Rate)
**Expected Range**: 0.02 - 0.32
**Typical**: 0.04 - 0.16

```
✅ GOOD:  ZC=0.060000   (15 crossings / 250 samples)
✅ GOOD:  ZC=0.128000   (32 crossings / 250 samples)
❌ BAD:   ZC=0.000000   (no crossings detected)
❌ BAD:   ZC=0.800000   (way too many crossings)
```

#### SSC (Normalized - Rate)
**Expected Range**: 0.04 - 0.60
**Typical**: 0.08 - 0.30

```
✅ GOOD:  SSC=0.112000  (28 changes / 250 samples)
✅ GOOD:  SSC=0.260000  (65 changes / 250 samples)
❌ BAD:   SSC=0.000000  (no slope changes)
❌ BAD:   SSC=1.000000  (every sample changes → noise)
```

---

### 5. Model Output Probabilities

#### Healthy Model Output
**Pattern**: One dominant probability, others low

```
✅ GOOD:
  Gesture[ 0]: 0.012345
  Gesture[ 1]: 0.876543 <- MAX  (clear winner!)
  Gesture[ 2]: 0.034567
  ...

✅ ACCEPTABLE:
  Gesture[ 0]: 0.234567
  Gesture[ 1]: 0.456789 <- MAX  (moderate confidence)
  Gesture[ 2]: 0.123456
  ...
```

#### Problematic Model Output
**Pattern**: Uniform distribution (model confused)

```
❌ BAD (Preprocessing Mismatch):
  Gesture[ 0]: 0.090909
  Gesture[ 1]: 0.090909
  Gesture[ 2]: 0.090909
  ...
  → All ~0.09 (1/11) = Model has no idea!
```

```
❌ BAD (Extreme Softmax):
  Gesture[ 0]: 0.999999 <- MAX
  Gesture[ 1]: 0.000000
  Gesture[ 2]: 0.000000
  ...
  → One probability dominates completely = Overfitting or bad features
```

```
❌ BAD (NaN or Inf):
  Gesture[ 0]: nan
  Gesture[ 1]: inf
  ...
  → Numerical instability or quantization issue
```

---

## Common Problem Patterns

### Problem A: "All ADC Values at 4095"
**Symptom**:
```
Raw ADC (first 5): 4095.00 4095.00 4095.00 4095.00 4095.00
```

**Cause**: scale_to_adc() applied to already-scaled ADC values
**Fix**: Remove scale_to_adc() or fix input data format

---

### Problem B: "ADC Values Still Normalized"
**Symptom**:
```
Raw ADC (first 5): 0.05 -0.09 -0.02 0.03 0.02
```

**Cause**: scale_to_adc() not being called or failing silently
**Fix**: Debug scale_to_adc() function in Python

---

### Problem C: "Zero Crossings Always 0"
**Symptom**:
```
Raw TD4 (before norm): MAV=123.45, WL=5678.90, ZC=0, SSC=0
```

**Cause**:
- DC offset not removed (signal not centered at 0)
- Threshold too high (15 ADC units inappropriate)

**Fix**:
- Verify DC offset removal
- Adjust ZC_THRESHOLD_ADC in functions.h

---

### Problem D: "Normalized Features 100x Wrong"
**Symptom**:
```
# ESP32:
MAV=0.500000 (should be ~0.03)

# vs Training:
MAV=0.005000
```

**Cause**: Different normalization constants (4095 vs something else)
**Fix**: Match ADC_MAX_GLOBAL to training value

---

### Problem E: "Uniform Model Outputs"
**Symptom**:
```
All probabilities ≈ 0.09 (1/11)
Final prediction: Gesture X (confidence=0.09, threshold=0.3)
```

**Cause**: Model receiving features completely different from training
**Root Causes**:
- Preprocessing pipeline mismatch
- Features in wrong range
- Wrong feature order

**Fix**: Compare ESP32 features with training features line-by-line

---

## Diagnostic Checklist

Use this checklist when analyzing diagnostic output:

- [ ] **Raw ADC values** are in range 0-4095 (not normalized floats)
- [ ] **DC offset** is around 2048 ± 500
- [ ] **Raw MAV** is in range 50-500
- [ ] **Raw WL** is in range 1000-50000
- [ ] **Raw ZC** is in range 5-80
- [ ] **Raw SSC** is in range 10-150
- [ ] **Normalized MAV** is in range 0.01-0.12
- [ ] **Normalized WL** is in range 0.001-0.05
- [ ] **Normalized ZC** is in range 0.02-0.32
- [ ] **Normalized SSC** is in range 0.04-0.60
- [ ] **Model outputs** show non-uniform distribution
- [ ] **ESP32 features** match training features (±5%)

---

## Quick Decision Tree

```
START: Run diagnostic

↓

Are ADC values 0-4095?
├─ NO (still -1 to +1) → Fix: Check scale_to_adc()
├─ NO (all 4095)      → Fix: Remove double-scaling
└─ YES ↓

Is DC offset around 2048?
├─ NO (very high/low) → Fix: Check input data format
└─ YES ↓

Are raw TD4 values in expected ranges?
├─ NO (too high)   → Fix: Check centering/normalization
├─ NO (too low)    → Fix: Check scaling applied
└─ YES ↓

Are normalized TD4 values in expected ranges?
├─ NO → Fix: Adjust ADC_MAX_GLOBAL constant
└─ YES ↓

Are model outputs uniform (~0.09 each)?
├─ YES → Compare ESP32 vs training features
│        Find first mismatched feature
│        Fix that specific preprocessing step
└─ NO → Check if predictions are correct
        ├─ Correct → Success!
        └─ Wrong   → Model issue (not preprocessing)
```

---

## Next Steps Based on Findings

### If All Checks Pass but Accuracy Still 0%
**Possible Causes**:
1. Feature **order** mismatch (sensors swapped)
2. Model file mismatch (wrong .tflite loaded)
3. Label mismatch (gesture 0 vs 1 indexing)

**Actions**:
- Verify feature order matches training exactly
- Check model.h contains correct model
- Verify gesture labels are 0-indexed

### If Any Check Fails
**Action**: Apply targeted fix for that specific stage
**Then**: Re-run diagnostic to verify fix

---

**Remember**: The diagnostic logging only runs for the **first window**, so all analysis should be done on that first window's output!
