# CSV Replay System - Critical Fixes Summary

**Date:** 2025-11-29
**Status:** ✅ All fixes implemented and ready for testing

---

## Overview

Fixed critical issues in the CSV replay validation system that would have caused **100% packet rejection** and validation failures. The system now properly validates ESP32 model inference against pre-recorded CSV data.

---

## Fixes Applied

### 1. ✅ CRITICAL: Window Packet Checksum Mismatch

**Problem:**
- **Python** (`csv_replay.py`): Calculated checksum as sum of individual **bytes**
  ```python
  checksum = sum(data_bytes) % 65536  # data_bytes is bytearray
  ```

- **ESP32** (`csv_replay.cpp`): Calculated checksum as sum of **16-bit integers**
  ```cpp
  sum += data[i];  // data[i] is uint16_t (0-4095)
  ```

**Impact:** 100% packet rejection - no window would be accepted by ESP32

**Fix:** Updated `csv_replay.cpp::calculate_checksum()` to match Python's byte-based approach
```cpp
// Now treats data as byte array
const uint8_t* byteData = (const uint8_t*)data;
size_t byteLen = len * 2;  // uint16 count → byte count

for (size_t i = 0; i < byteLen; i++) {
  sum += byteData[i];  // Sum individual bytes
}
```

**Files Modified:**
- [src/csv_replay.cpp](src/csv_replay.cpp):218-227

---

### 2. ✅ MAJOR: Code Duplication & Hardcoded Thresholds

**Problem:**
- `csv_replay.cpp::extract_features_from_local_buffer()` completely **reimplemented** TD4 feature extraction logic
- Hardcoded thresholds (`15.0f`) instead of using shared constants from `functions.h`
- If you changed `ZC_THRESHOLD_ADC` in `functions.cpp`, CSV replay validation would use **old values**
- Risk of **false positive/negative validation results**

**Fix:**
1. **Exported shared constants** to `functions.h`:
   ```cpp
   #define ZC_THRESHOLD_ADC   15.0f   // Zero crossing threshold
   #define SSC_THRESHOLD_ADC  15.0f   // Slope sign change threshold
   ```

2. **Updated csv_replay.cpp** to use shared constants:
   ```cpp
   if (fabs(centered - prev_centered) >= ZC_THRESHOLD_ADC) {  // Was: 15.0f
   ```

3. **Added comprehensive documentation** explaining:
   - Why code is duplicated (static vs. local buffer)
   - Synchronization requirements with `functions.cpp`
   - TODO for future refactoring

**Files Modified:**
- [src/functions.h](src/functions.h):23-27 (added threshold exports)
- [src/functions.cpp](src/functions.cpp):44 (removed duplicate definitions)
- [src/csv_replay.cpp](src/csv_replay.cpp):116-133 (added sync documentation)
- [src/csv_replay.cpp](src/csv_replay.cpp):174,187 (use shared constants)

---

### 3. ✅ CRITICAL: Response Packet Checksum Mismatch

**Problem:**
- **Python** (`csv_replay.py`): Expected checksum of **all response bytes** (102 bytes):
  ```python
  # pred(1) + conf(4) + features(96) + gt_echo(1) = 102 bytes
  expected_checksum = sum(data_for_checksum) % 65536
  ```

- **ESP32** (`csv_replay.cpp`): Only checksummed **feature values** using float bit representation
  ```cpp
  // WRONG: Only checksums features, converts floats to uint32
  for (int i = 0; i < 24; i++) {
    memcpy(&bits, &response->features[i], sizeof(float));
    sum += bits;
  }
  ```

**Impact:** Response validation would always fail, accuracy metrics would be unreliable

**Fix:** Updated `send_response()` to checksum all response bytes before checksum field:
```cpp
// Now matches Python: sum of ALL bytes before checksum
const uint8_t* data = (const uint8_t*)response;
size_t checksum_offset = offsetof(ResponsePacket, checksum);  // 104

for (size_t i = 2; i < checksum_offset; i++) {  // Skip header (2 bytes)
  sum += data[i];  // Sum bytes 2-103 (102 bytes total)
}
```

**Files Modified:**
- [src/csv_replay.cpp](src/csv_replay.cpp):302-329

---

### 4. ✅ Documentation: Data Flow & DC Offset Handling

**Problem:**
- Unclear how CSV data (already filtered, centered at 0) flows through the system
- Why Python **adds** DC offset, then ESP32 **removes** it again
- What DSP filters are bypassed and why

**Fix:** Added comprehensive documentation to `populate_raw_buffer()`:
- **CSV Source Data**: Pre-filtered, normalized to [-1, +1], centered at 0
- **Python Processing**: Scales to [0-4095], adds DC offset (~2048) to simulate ADC
- **ESP32 Processing**: Removes DC offset via mean subtraction
- **DSP Filter Bypass**: CSV data pre-filtered, IIR filters disabled
- **Validation Scope**: Tests TD4 + model inference, NOT DSP filters

**Files Modified:**
- [src/csv_replay.cpp](src/csv_replay.cpp):331-365

---

## Testing Instructions

### 1. Build and Upload CSV Replay Firmware
```bash
# Clean previous build
pio run -e csv_replay --target clean

# Build and upload
pio run -e csv_replay -t upload

# Monitor serial output
pio device monitor
```

**Expected Output:**
```
=== ESP32-S3 CSV REPLAY MODE ===
Build: CSV Replay - Offline Model Validation
✓ DSP FILTERS BYPASSED (CSV data pre-filtered)
✓ Model loaded successfully
✓ Tensors allocated successfully
Ready to receive window packets from Python
```

### 2. Run Python CSV Replay Script
```bash
cd scripts_ai/validation
python csv_replay.py ../../data/your_data.csv --port COM11 --baud 921600
```

**Expected Behavior:**
- ✅ No checksum mismatch errors
- ✅ Window packets accepted (not "ERROR: Checksum mismatch")
- ✅ Response packets validated (checksum_valid = True)
- ✅ Accuracy metrics calculated correctly
- ✅ Confusion matrix generated

**Previous Behavior (BEFORE fixes):**
- ❌ ERROR: Checksum mismatch (got 0x1234, expected 0x5678)
- ❌ 0% packet acceptance rate
- ❌ No validation results

### 3. Verify Synchronization

Run this test to ensure C++ and Python preprocessing match:
```bash
cd scripts_ai/validation
python validate_preprocessing.py
```

**Expected Output:**
```
✓ Feature extraction matches Python pipeline
✓ ZC thresholds: C++ (15.0) == Python (15.0)
✓ SSC thresholds: C++ (15.0) == Python (15.0)
✓ Normalization: C++ (4095) == Python (4095)
```

---

## Impact Summary

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| Window checksum mismatch | **CRITICAL** | 100% packet rejection | ✅ FIXED |
| Response checksum mismatch | **CRITICAL** | Invalid validation results | ✅ FIXED |
| Hardcoded thresholds | **MAJOR** | False validation results | ✅ FIXED |
| Code duplication | **MAJOR** | Maintenance drift risk | ✅ DOCUMENTED |
| Missing data flow docs | **MINOR** | Developer confusion | ✅ FIXED |

---

## Future Improvements (TODO)

### 1. Eliminate Code Duplication

**Current State:**
- `extract_features_from_local_buffer()` duplicates `extract_features_from_raw()` logic
- Must manually keep both implementations synchronized

**Proposed Fix:**
```cpp
// In functions.cpp:
float* extract_features_from_buffer(float buffer[NUM_SENSORS][RAW_WINDOW_SIZE],
                                     float* output_features);

// In functions.cpp (refactored):
float* extract_features_from_raw() {
  return extract_features_from_buffer(raw_sensor_data, features);
}

// In csv_replay.cpp (refactored):
void extract_features_from_local_buffer(...) {
  extract_features_from_buffer(raw_data_local, output_features);
}
```

### 2. Add Unit Tests

Create automated tests for:
- Checksum calculation (window and response)
- Feature extraction parity (C++ vs Python)
- Packet serialization/deserialization

---

## Validation Checklist

Before deploying:
- [ ] Build succeeds with no warnings (`pio run -e csv_replay`)
- [ ] ESP32 boots and shows "Ready to receive window packets"
- [ ] Python script connects without errors
- [ ] Window packets accepted (no checksum errors)
- [ ] Response packets validated (checksum_valid = True)
- [ ] Confusion matrix generated
- [ ] Accuracy > 70% (if model is trained)
- [ ] No memory errors or crashes during replay

---

## Files Changed

### Modified Files
1. [src/functions.h](src/functions.h) - Added threshold exports
2. [src/functions.cpp](src/functions.cpp) - Removed duplicate thresholds
3. [src/csv_replay.cpp](src/csv_replay.cpp) - All fixes implemented

### No Changes Required
- [scripts_ai/validation/csv_replay.py](scripts_ai/validation/csv_replay.py) - Python code was correct
- [src/model.h](src/model.h) - Model unchanged
- [platformio.ini](platformio.ini) - Build config unchanged

---

## Additional Notes

### Why Checksums Matter

**Without proper checksums:**
- Corrupted serial data goes undetected
- Features might be garbled (garbage predictions)
- Validation metrics are meaningless
- False confidence in model performance

**With proper checksums:**
- Data integrity guaranteed
- Errors detected immediately
- Reliable validation results
- Trustworthy accuracy metrics

### Why Threshold Synchronization Matters

**If thresholds drift:**
- Python training uses different ZC/SSC thresholds than ESP32
- Model learns to recognize features that ESP32 never produces
- Inference accuracy collapses in production
- Debugging is nearly impossible (features look correct but aren't)

**With synchronized thresholds:**
- Training and inference use identical preprocessing
- Features match exactly between Python and C++
- Model performance is reproducible
- Debugging is straightforward

---

## Contact & Support

If you encounter issues after applying these fixes:

1. **Check serial monitor** for detailed error messages
2. **Verify build environment**: `pio run -e csv_replay --verbose`
3. **Test with small CSV**: Start with 10-20 windows
4. **Compare checksums**: Print intermediate values in both Python and C++
5. **Validate packet sizes**: 3007 bytes (window), 106 bytes (response)

---

**END OF SUMMARY**
