# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time gesture classification system for **ESP32-S3 DevKitC-1** using TensorFlow Lite Micro. The system uses **6 MyoWare EMG sensors** with **literature-validated DSP filtering and TD4 features** (Hudgins et al. 1993) to classify 11 hand gestures for prosthetic control.

**Gestures recognized:** Rest, Fist, Open, Point, Victory, OK, ThumbUp, ThumbDn, Grasp, Pinch, WristFlex

**Project README:** See [README.md](README.md) for comprehensive project documentation, installation, and usage guide.

## Hardware Configuration

- **Board:** ESP32-S3 DevKitC-1 (N16R8 variant)
  - 16MB Flash, 8MB PSRAM (Octal OPI mode)
  - Dual-core Xtensa LX7 @ 240MHz
- **Serial Port:** COM11 (921600 baud) - Update in [platformio.ini](platformio.ini) if different
- **MPU6050:** NOT USED (IMU removed from system - EMG-only)
- **ADC Pins (6 EMG Sensors):**
  - GPIO4  → MW1 (EMG Sensor 1) - ADC1_CH3
  - GPIO5  → MW2 (EMG Sensor 2) - ADC1_CH4
  - GPIO6  → MW3 (EMG Sensor 3) - ADC1_CH5
  - GPIO7  → MW4 (EMG Sensor 4) - ADC1_CH6
  - GPIO15 → MW5 (EMG Sensor 5) - ADC1_CH6
  - GPIO16 → MW6 (EMG Sensor 6) - ADC2_CH5

**⚠️ Pin Change Warning:**
- **GPIO pins changed from 0-5 to 4-7, 15-16** to avoid UART0 conflicts
- GPIO1-2 are used by USB Serial (UART0 TX/RX) and must be avoided
- **You must physically reconnect sensors to the new pins!**
- ADC2 (GPIO15-16) is safe on ESP32-S3 (no Wi-Fi conflict like ESP32-C3)

## Build & Development Commands

### Add to Path
$env:Path += ";C:\Users\MERT\.platformio\penv\Scripts"

### Environment Overview

The project has **2 separate environments** for different purposes:

1. **`data_acquisition`** - For collecting training data from 6 EMG sensors
   - Includes DSP filters (same as real-time inference for consistency)
   - No TFLite model (faster compilation, smaller binary)
   - Streams filtered ADC values over serial at 1000 Hz
   - Used with `scripts_ai/training_data_collection.py`

2. **`real_time_inference`** - For real-time gesture classification
   - Includes DSP filters + TFLite Micro model
   - Classifies gestures using filtered TD4 features at 1000 Hz
   - Outputs gesture predictions over serial

### Data Acquisition (Training Data Collection)
```bash
# Build data acquisition firmware
pio run -e data_acquisition

# Upload to device
pio run -e data_acquisition -t upload

# Upload and monitor
pio run -e data_acquisition -t upload && pio device monitor

# Then run Python script to collect data:
cd scripts_ai
python training_data_collection.py
```

### Real-Time Inference (Gesture Classification)
```bash
# Build real-time inference firmware
pio run -e real_time_inference

# Upload to device
pio run -e real_time_inference -t upload

# Upload and monitor
pio run -e real_time_inference -t upload && pio device monitor
```

### General Commands
```bash
# Monitor serial output (works for both environments)
pio device monitor

# Build default environment (legacy, same as real_time_inference)
pio run

# Upload default environment
pio run -t upload
```

### Cleaning
```bash
# Clean build files
pio run --target clean

# Full clean (including dependencies)
pio run --target fullclean
```

### Device Management
```bash
# List connected devices
pio device list

# Update platforms and libraries
pio pkg update
```

## Code Architecture

### Signal Processing Pipeline (Sigma Project)

The system processes sensor data through a **DUAL-STAGE** literature-based pipeline:

**STAGE 1: DSP Filtering (Real-Time IIR Filters)**
- Applied **per-sample** during data collection in `prelim_collection()`
- Runs at 1000 Hz (unified for both data acquisition and real-time inference)
- Filter cascade: **Raw ADC → HPF (20Hz) → LPF (450Hz) → Notch (50/60Hz)**
- Uses **Direct Form II Transposed biquads** for numerical stability
- **6 independent filter chains** (one per sensor) to prevent cross-contamination
- Implemented in [filters.cpp](src/filters.cpp) / [filters.h](src/filters.h)

**STAGE 2: TD4 Feature Extraction**
- Applied **per-window** on filtered data (250ms window)
- Extracts Hudgins' TD4 features: MAV, WL, ZC, SSC
- Applied DC offset removal before feature calculation
- Global normalization using ADC_MAX = 4095
- Implemented in [functions.cpp](src/functions.cpp)

### Detailed Pipeline Steps

1. **Real-Time DSP Filtering** (`prelim_collection()` in [functions.cpp](src/functions.cpp):194-209)
   - Collects **250ms window** of **filtered ADC data** at 1000 Hz
   - **250 samples per sensor** (1000Hz × 0.25s)
   - Each raw ADC sample passes through `filter_sample()` from [filters.cpp](src/filters.cpp):301-336
   - **Filter chain per sample:**
     ```
     Raw ADC (0-4095) → HPF Stage1 → HPF Stage2 →
     LPF Stage1 → LPF Stage2 → Notch → Filtered ADC
     ```
   - **HPF (20 Hz, 4th-order)**: Removes DC drift and motion artifacts (0-20 Hz)
   - **LPF (450 Hz, 4th-order)**: Removes electronic noise (>500 Hz)
   - **Notch (50/60 Hz, 2nd-order)**: Removes powerline interference
   - Stores filtered data in `raw_sensor_data[6][250]` buffers
   - **Response time: 250ms** (fast gesture detection)

2. **DC Offset Removal & Feature Extraction** (`extract_features_from_raw()` in [functions.cpp](src/functions.cpp):228-340)
   - **STEP 1: Calculate mean** (remaining DC offset) for each sensor on filtered data
   - **STEP 2: Center signal** by subtracting mean (creates bipolar signal centered at 0)
   - **STEP 3: Extract TD4 features** on centered signal:
     - **MAV** (Mean Absolute Value): Average signal amplitude
     - **WL** (Waveform Length): Signal complexity measure
     - **ZC** (Zero Crossings): Frequency estimate (threshold: 15.0 ADC units)
     - **SSC** (Slope Sign Changes): Frequency content (threshold: 15.0 ADC units)
   - **STEP 4: Global normalization** using ADC_MAX = 4095 (preserves amplitude info)
   - **Output:** float array[24] for TFLite model input
   - **CRITICAL:** ZC/SSC work correctly on bipolar centered signal

3. **TD4 Feature Set** (Hudgins et al. 1993 - Literature-Validated)
   - **24 total features:** 4 TD4 features × 6 EMG sensors
   - **Feature order** (MUST match Python training!):
     ```
     [MAV₁, WL₁, ZC₁, SSC₁,  // Sensor 1
      MAV₂, WL₂, ZC₂, SSC₂,  // Sensor 2
      MAV₃, WL₃, ZC₃, SSC₃,  // Sensor 3
      MAV₄, WL₄, ZC₄, SSC₄,  // Sensor 4
      MAV₅, WL₅, ZC₅, SSC₅,  // Sensor 5
      MAV₆, WL₆, ZC₆, SSC₆]  // Sensor 6
     ```

4. **TFLite Inference** ([main.cpp](src/main.cpp):83-159)
   - **Int8 Quantization**: Features quantized using `input->data.int8[i]` with proper scale/zero_point
   - **Dequantization**: Output probabilities dequantized from Int8 to Float32
   - Loads model from `model_tflite` array (defined in [model.h](src/model.h))
   - **30KB tensor arena** (increased from 20KB) with 16-byte alignment for SIMD
   - Model: Wide & Deep MLP with Int8 quantization
   - Architecture: 24 → 256 → 128 → 64 → 11 (with dropout)
   - Outputs 11 class probabilities
   - Threshold: 0.8 confidence for gesture detection ([main.cpp](src/main.cpp):132)

### DSP Filter Implementation

**Biquad Structure:** Direct Form II Transposed
```
Transfer Function: H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)

Update Equations:
  y[n] = b0*x[n] + z1
  z1   = b1*x[n] - a1*y[n] + z2
  z2   = b2*x[n] - a2*y[n]
```

**Why Direct Form II Transposed?**
- Most numerically stable IIR structure
- Minimizes coefficient quantization error
- Only 2 state variables (minimal memory: ~480 bytes total)
- Industry standard (used by ARM CMSIS-DSP)
- Total latency: ~12 ms

**Filter Coefficients:** Auto-generated by [generate_filter_coefficients.py](scripts_ai/generate_filter_coefficients.py)
- Unified 1000 Hz coefficients for both data acquisition and real-time inference
- Stored in [filters.cpp](src/filters.cpp) lines 20-48

### File Structure

**Source Files:**
- [main.cpp](src/main.cpp) - **Real-time inference**: Arduino setup/loop, TFLite integration, gesture detection
- [data_acquisition.cpp](src/data_acquisition.cpp) - **Data collection**: Streams filtered EMG data for training
- [functions.cpp](src/functions.cpp) - **TD4 feature extraction**, DC offset removal, prelim_collection
- [functions.h](src/functions.h) - Function declarations, pin definitions, constants (`NUM_FEATURES = 24`, `ADC_MAX = 4095`)
- [filters.cpp](src/filters.cpp) - **DSP filter implementation**: IIR biquad cascade (HPF, LPF, Notch)
- [filters.h](src/filters.h) - Filter structures, configuration, function prototypes
- [filter_coefficients_50hz.h](src/filter_coefficients_50hz.h) - Auto-generated filter coefficients (50Hz powerline)
- [model.h](src/model.h) - Auto-generated TFLite model as C++ byte array (Wide & Deep MLP, Int8)
- ~~modeldata.h~~ - **DELETED** (obsolete, replaced by model.h)
- ~~modeldata.cpp~~ - **DELETED** (obsolete, replaced by model.h auto-generation)

**Build Configuration:**
- [platformio.ini](platformio.ini) - Defines 2 environments: `data_acquisition` and `real_time_inference`

**Documentation:**
- [ADVANCED_FEATURES_PROPOSAL.md](ADVANCED_FEATURES_PROPOSAL.md) - Proposal for 57-114 feature expansion (not currently implemented)

**Libraries:**
- [lib/TensorFlowLite_ESP32/](lib/TensorFlowLite_ESP32/) - Custom TFLite Micro port for ESP32
  - Based on TensorFlow v2.1.1
  - Modified from upstream: .cc → .cpp, include paths adjusted
  - Uses experimental micro API

**AI Training Pipeline:**
- [scripts_ai/](scripts_ai/) - **Reorganized Python ML pipeline** (2025-11-26 update)
  - [📄 README.md](scripts_ai/README.md) - Pipeline overview, folder structure, workflow
  - [📄 SCRIPT_EXPLANATIONS.md](scripts_ai/SCRIPT_EXPLANATIONS.md) - Detailed Turkish explanations of all scripts
  - [📄 QUICK_START.md](scripts_ai/QUICK_START.md) - 3-minute quick start guide
  - [🎯 critical/](scripts_ai/critical/) - **END-TO-END PIPELINE (CORE)**
    - [feature_extraction.py](scripts_ai/critical/feature_extraction.py) - CSV → TD4 Features (NPZ)
    - [train_test_model.py](scripts_ai/critical/train_test_model.py) - NPZ → TFLite Model + C++ Header
  - [🔧 filters/](scripts_ai/filters/) - **DSP Filter Tools**
    - [dsp_filters.py](scripts_ai/filters/dsp_filters.py) - Python DSP filter bank (matches C++)
    - [generate_filter_coefficients.py](scripts_ai/filters/generate_filter_coefficients.py) - C++ coefficient generator
  - [✅ validation/](scripts_ai/validation/) - **Quality Control & Testing**
    - [validate_noise_data.py](scripts_ai/validation/validate_noise_data.py) - ADC overflow, sensor quality check
    - [validate_filters.py](scripts_ai/validation/validate_filters.py) - Python-C++ filter consistency test
    - [validate_preprocessing.py](scripts_ai/validation/validate_preprocessing.py) - Feature extraction verification
    - [validate_pipeline.py](scripts_ai/validation/validate_pipeline.py) - End-to-end pipeline test
  - [📊 visualization/](scripts_ai/visualization/) - **Signal Quality Analysis**
    - [visualize_signal_quality.py](scripts_ai/visualization/visualize_signal_quality.py) - EMG signal quality plots
  - 📁 data/ - Auto-generated outputs
    - `features/` - TD4 features NPZ files
    - `models/` - TFLite models (Int8) and C headers
    - `plots/` - Training curves and confusion matrices

## Important Constants

From [functions.h](src/functions.h):
```cpp
// Feature configuration
#define NUM_FEATURES      24     // TD4 features: 4 features × 6 EMG sensors
#define NUM_SENSORS       6      // Number of EMG sensors

// Raw data collection
#define RAW_WINDOW_SIZE   250    // 250ms window (250 samples at 1kHz)
#define SAMPLING_FREQ     1000   // 1000 Hz sampling rate

// Global ADC normalization (12-bit ADC on ESP32)
#define ADC_MIN_GLOBAL    0.0f   // Minimum ADC value
#define ADC_MAX_GLOBAL    4095.0f  // Maximum ADC value (12-bit = 4095)

// GPIO Pin Assignments (ESP32-S3)
#define pin_MW1  4    // GPIO 4  (ADC1_CH3)
#define pin_MW2  5    // GPIO 5  (ADC1_CH4)
#define pin_MW3  6    // GPIO 6  (ADC1_CH5)
#define pin_MW4  7    // GPIO 7  (ADC1_CH6)
#define pin_MW5  15   // GPIO 15 (ADC2_CH4)
#define pin_MW6  16   // GPIO 16 (ADC2_CH5)
```

From [filters.h](src/filters.h):
```cpp
// Filter configuration
#define POWERLINE_FREQ_HZ 50    // 50Hz for Europe/Asia, 60Hz for Americas
#define ENABLE_HPF        1     // High-pass filter (20 Hz)
#define ENABLE_LPF        1     // Low-pass filter (450 Hz)
#define ENABLE_NOTCH      1     // Notch filter (50/60 Hz)
#define ENABLE_MOVING_AVG 0     // Moving average (disabled, adds latency)
#define NUM_SENSORS       6     // 6 EMG sensors
#define MA_WINDOW_SIZE    5     // Moving average window (if enabled)
```

From [functions.cpp](src/functions.cpp):
```cpp
// TD4 thresholds in ADC units (NOT normalized values)
#define ZC_THRESHOLD_ADC   15.0f  // Zero crossing threshold (ADC units)
#define SSC_THRESHOLD_ADC  15.0f  // Slope sign change threshold (ADC units)
```

From [main.cpp](src/main.cpp):
```cpp
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB for TFLite (increased from 20KB)
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // 16-byte aligned for SIMD
float threshold = 0.8;  // Confidence threshold for gesture detection
```

## 🚨 CRITICAL: Gesture Name Order Consistency

**EXTREMELY IMPORTANT:** The gesture order MUST be identical across ALL files to prevent label mismatch between training and inference!

**Fixed Gesture Order (DO NOT CHANGE):**
```
Index 0:  Rest
Index 1:  Fist
Index 2:  Open
Index 3:  Point
Index 4:  Victory
Index 5:  OK
Index 6:  ThumbUp
Index 7:  ThumbDn
Index 8:  Grasp
Index 9:  Pinch
Index 10: WristFlex
```

**Files that MUST use this exact order:**
1. [data_acquisition/scripts/training_collection new.py](data_acquisition/scripts/training_collection new.py) - `GESTURE_NAMES` array
2. [scripts_ai/train_test_model.py](scripts_ai/train_test_model.py) - `GESTURE_NAMES_FIXED` array
3. [scripts_ai/generate_dummy_data.py](scripts_ai/generate_dummy_data.py) - `GESTURE_NAMES` array
4. [src/main.cpp](src/main.cpp) - `gesture_names[]` array
5. [src/servo_controller.cpp](src/servo_controller.cpp) - `gesture_names[]` array

**Why this matters:**
- The model is trained with gestures mapped to integer indices (0-10)
- ESP32 C++ code uses the same indices to look up gesture names
- If the order differs, predictions will be completely wrong
  - Example: Model predicts "Fist" (index 1) → ESP32 displays wrong gesture
- Previous bug: `train_test_model.py` used alphabetical sorting → caused total mismatch

**Fixed (2025-11-26):** `train_test_model.py` now uses `GESTURE_NAMES_FIXED` to match data collection and C++ code.

## Training & Deploying Models

### DSP-Enhanced TD4 Feature Extraction & MLP Training Pipeline

The project uses a **dual-stage** approach: DSP filtering + TD4 features + Wide & Deep MLP.

**Training workflow:**
```bash
cd scripts_ai

# 1. Generate filter coefficients (if not already done)
python filters/generate_filter_coefficients.py 50  # 50Hz for Europe/Asia, 60Hz for Americas
# Output: ../src/filter_coefficients_50hz.h

# 2. Validate C++ vs Python filter consistency
python validation/validate_filters.py
# Ensures DSP filters match between training and inference

# 3. Collect training data from ESP32 (6 EMG sensors)
cd ../data_acquisition/scripts
python training_data_collection.py
# Output: ../../data/training_data_TIMESTAMP.csv

# 4. Validate data quality
cd ../../scripts_ai/validation
python validate_noise_data.py ../../data/training_data_TIMESTAMP.csv
# Checks: ADC overflow, sensor variance, gesture separability

# 5. Extract TD4 features (applies DSP filters first, then MAV/WL/ZC/SSC)
cd ../critical
python feature_extraction.py ../../data/training_data_TIMESTAMP.csv
# Output: ../data/features/td4_features_TIMESTAMP.npz

# 6. Train Wide & Deep MLP with Int8 quantization
python train_test_model.py
# Output: 
#   ../data/models/emg_model_TIMESTAMP.tflite
#   ../data/models/emg_model_TIMESTAMP.h

# 7. Copy generated model.h to src/
cp ../data/models/emg_model_TIMESTAMP.h ../../src/model.h

# 8. Build and upload to ESP32
cd ../..
pio run -e real_time_inference -t upload
```

### Model Architecture

**Input:** 24 TD4 features (4 features × 6 EMG sensors)
- Features extracted from **DSP-filtered signals**
- MAV, WL, ZC, SSC per sensor

**Wide & Deep MLP:**
```
Input(24) → Dense(256, relu) → Dropout(0.3) →
Dense(128, relu) → Dropout(0.2) →
Dense(64, relu) → Dropout(0.1) →
Output(11, softmax)
```

**Quantization:** Post-Training Int8 (75% size reduction, 2-3× faster inference)

**Output:** 11 gestures (Rest, Fist, Open, Point, Victory, OK, ThumbUp, ThumbDn, Grasp, Pinch, WristFlex)

### Manual Model Replacement

To replace the model:

1. Train a new model with **24 TD4 input features** → 11 output classes
2. Ensure TD4 feature extraction matches [feature_extraction.py](scripts_ai/feature_extraction.py)
3. Ensure DSP filters are applied consistently (validate with [validate_filters.py](scripts_ai/validate_filters.py))
4. Replace [model.h](src/model.h) with new auto-generated .h file
5. Verify `NUM_FEATURES = 24` in [functions.h](src/functions.h)
6. Adjust `kTensorArenaSize` in [main.cpp](src/main.cpp) if needed

## Preprocessing Alignment (2025-11-25 CRITICAL UPDATE - Sigma Project)

**🚨 MAJOR UPDATE**: The preprocessing pipeline was **ENHANCED** with real-time DSP filtering to improve signal quality and feature extraction reliability.

### The Enhancement

**Previous Approach (2D-CNN Branch):**
```
Raw ADC → DC offset removal → TD4 features → Global normalization
```

**Current Approach (Sigma Project):**
```
Raw ADC → DSP Filters (HPF→LPF→Notch) → DC offset removal → TD4 features → Global normalization
```

| Component | Old (2D-CNN) | **New (Sigma Project)** |
|-----------|--------------|-------------------------|
| **Pre-filtering** | None | **IIR cascade (HPF+LPF+Notch)** |
| **DC Removal** | Mean subtraction | **Still used (removes residual offset)** |
| **Signal Quality** | Raw noisy ADC | **Clean filtered EMG band** |
| **Filter Order** | N/A | **HPF(20Hz) → LPF(450Hz) → Notch(50Hz)** |
| **Memory Usage** | ~1.5 KB | **~2 KB (~480B filters + buffers)** |
| **Latency** | 250ms | **~282ms (250ms + 12ms filters)** |
| **CPU Usage** | ~5% | **~5.5% (~0.5% for filters)** |

### Why DSP Filtering Matters

**Benefits of DSP Pre-filtering:**
1. **Removes DC drift**: HPF eliminates slow baseline wander (motion artifacts)
2. **Isolates EMG band**: LPF removes high-frequency electronic noise (>500 Hz)
3. **Eliminates powerline**: Notch removes 50/60 Hz interference
4. **Better TD4 features**: ZC/SSC more reliable on band-limited signals
5. **Consistent preprocessing**: Same filters used in training and inference

**Before (2D-CNN):**
```python
# Training
raw_adc = [2048, 2100, 2050, ...]  # Noisy, DC offset, powerline interference
mean = np.mean(raw_adc)  # 2048
centered = raw_adc - mean  # [-48, +52, +2, ...]
zc = count_zero_crossings(centered)  # Noisy, unreliable
```

**After (Sigma Project):**
```python
# Training
raw_adc = [2048, 2100, 2050, ...]  # Noisy, DC offset, powerline
filtered = apply_dsp_filters(raw_adc)  # [1950, 2020, 1980, ...] (clean EMG band)
mean = np.mean(filtered)  # Residual DC
centered = filtered - mean  # [-70, +50, -20, ...] (clean bipolar signal)
zc = count_zero_crossings(centered)  # Reliable frequency estimate
```

### Changes Made

**C++ Side** ([filters.cpp](src/filters.cpp), [filters.h](src/filters.h)):
- ✅ **Added IIR filter cascade**: HPF (20Hz, 4th-order) + LPF (450Hz, 4th-order) + Notch (50/60Hz, 2nd-order)
- ✅ **Direct Form II Transposed**: Most stable biquad structure
- ✅ **Per-sensor state**: 6 independent filter chains (no cross-contamination)
- ✅ **Minimal overhead**: <0.5% CPU, ~480 bytes memory, ~12ms latency
- ✅ **Integrated in prelim_collection()**: Filters applied per-sample during data collection

**Python Side** ([dsp_filters.py](scripts_ai/dsp_filters.py)):
- ✅ **Matching IIR filters**: Uses SciPy with identical specifications
- ✅ **EMGFilterBank class**: Easy-to-use filter bank matching C++ implementation
- ✅ **Validation script**: [validate_filters.py](scripts_ai/validate_filters.py) ensures bit-exact match
- ✅ **Integrated in feature_extraction.py**: Applies filters before TD4 extraction

**Filter Coefficient Generation** ([generate_filter_coefficients.py](scripts_ai/generate_filter_coefficients.py)):
- ✅ **Auto-generates C++ coefficients**: Outputs ready-to-use .h file
- ✅ **Unified 1000 Hz sampling**: Same coefficients for all modes
- ✅ **Simplified implementation**: No compile-time flag needed

### Impact on Features

- **MAV**: Now reflects true EMG amplitude (DC drift removed by HPF)
- **WL**: More stable (noise reduced by LPF)
- **ZC**: More reliable frequency estimate (band-limited signal, no powerline)
- **SSC**: Better frequency content detection (clean bipolar signal)

### Validation

After filtering, verify signal quality:
```bash
# Upload new firmware
pio run -e real_time_inference -t upload && pio device monitor

# Check serial output for:
# - "Initializing DSP Filters" on startup
# - Filter configuration printout
# - Clean feature values (ZC/SSC should be stable, not erratic)
```

Validate filter consistency:
```bash
cd scripts_ai
python validate_filters.py
# Should show: "✅ Filters match within tolerance"
```

### Retraining REQUIRED

⚠️ **CRITICAL**: Models from 2D-CNN branch are **incompatible** with Sigma Project. You **MUST retrain**:

1. C++ now applies DSP filters during data collection
2. Python pipeline must match (already updated in dsp_filters.py)
3. Old models trained on raw noisy signals
4. New models trained on clean filtered signals

**Retrain steps:**
```bash
cd scripts_ai
python generate_filter_coefficients.py 50  # If not already done
python validate_filters.py  # Verify filter consistency
python feature_extraction.py data/training_data.csv
python train_test_model.py
cp models/mlp_td4_*_int8.h ../src/model.h
pio run -e real_time_inference -t upload
```

## Key Dependencies

Defined in [platformio.ini](platformio.ini):
- **Board:** `esp32-s3-devkitc-1` (ESP32-S3 with 16MB Flash, 8MB PSRAM)
- **Framework:** Arduino (built-in with ESP32 platform)
- **TensorFlowLite_ESP32** (local lib) - TensorFlow Lite Micro v2.1.1 for real-time inference
- **ESP32Servo** (v1.2.1) - Servo control for robotic hand
- **No external libraries needed** - All code uses Arduino.h built-in functions

Build flags:
- `-DBOARD_HAS_PSRAM` - Enable 8MB PSRAM support
- `-mfix-esp32-psram-cache-issue` - ESP32 PSRAM cache bug workaround
- `-DARDUINO_USB_CDC_ON_BOOT=1` - Enable USB Serial on boot
- `-DARDUINO_USB_MODE=1` - USB mode enabled

**Note:** ESP32-S3 uses different build flags than ESP32-C3. ADC2 works without special flags on S3.

## Development Notes

### DSP Filtering (Sigma Project Enhancement)
- **IIR biquad cascade**: Industry-standard Direct Form II Transposed structure
- **Literature-based design**: HPF (20Hz, Butterworth 4th-order), LPF (450Hz, Butterworth 4th-order), Notch (50/60Hz, Q=12.5)
- **Per-sensor state management**: 6 independent filter chains prevent cross-contamination
- **Minimal overhead**: ~480 bytes memory, <0.5% CPU @ 240MHz, ~12ms latency
- **Unified 1000 Hz sampling**: Same coefficients for all modes (data acquisition and real-time inference)
- **Python/C++ consistency**: Validated with [validate_filters.py](scripts_ai/validate_filters.py)
- **Auto-generated coefficients**: Use [generate_filter_coefficients.py](scripts_ai/generate_filter_coefficients.py) to regenerate

### TD4 Feature Extraction (Literature-Based)
- **Hudgins' TD4 features**: Gold standard for EMG pattern recognition (1993)
- **4 features per sensor**: MAV, WL, ZC, SSC
- **6 EMG sensors**: GPIO 4-7, 15-16 (24 total features)
- **MPU6050 REMOVED**: No longer used (EMG-only system)
- **Dual-stage processing**: DSP filters first, then DC offset removal, then TD4
- **Global normalization**: Preserves amplitude differences between gestures
- **Thresholds**: ZC=15.0 ADC, SSC=15.0 ADC (raw ADC units, noise-reduced)
- **Feature parity critical**: C++ extraction must match Python training code exactly

### Code Architecture Notes
- **Filter state buffers**: `filter_states[NUM_SENSORS]` in [filters.cpp](src/filters.cpp):98 - per-sensor biquad states
- **Raw sensor buffers**: `raw_sensor_data[6][250]` - 250 filtered ADC samples per sensor
- **24 feature array**: Global `features[NUM_FEATURES]` in [functions.cpp](src/functions.cpp)
- **DSP filtering applied per-sample**: Each ADC read passes through `filter_sample()` before storage
- **DC offset removal**: Mean calculated per sensor, subtracted to center signal at 0
- **TFLite Int8 quantization**: Features quantized with `input->data.int8[i] = (int8_t)((features[i] / input->params.scale) + input->params.zero_point)`
- **Output dequantization**: `float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale`
- **Never use** `.f` float accessors with Int8 models - this will cause NaN values
- Serial output shows DSP filter status, TD4 features per sensor, and 11 class probabilities
- Duplicate gestures suppressed (only shows when gesture changes)
- Confidence threshold: 0.8 for gesture detection
- **Response time**: 250ms collection + ~12ms DSP + ~20ms inference = ~282ms total latency

### Model Architecture & Performance
- **Wide & Deep MLP**: 256→128→64 neurons with dropout regularization
- **Int8 quantization**: 75% size reduction, 2-3× faster inference on ESP32
- **Literature-proven**: TD4 features achieve 85-95% accuracy in EMG literature
- **DSP-enhanced**: Cleaner features should improve accuracy by 5-10%
- **Processor-friendly**: Simple time-domain features + IIR filters, no FFT
- **Expected inference time**: <20ms per classification on ESP32-S3 @ 240MHz

### Training Data & Workflow
- Training data collected with 6 EMG sensors (DSP filters enabled)
- DSP filters applied via [dsp_filters.py](scripts_ai/dsp_filters.py) during training
- TD4 features extracted via [feature_extraction.py](scripts_ai/feature_extraction.py)
- Models trained via [train_test_model.py](scripts_ai/train_test_model.py)
- Gesture reference images in `images_hand/` folder (used during data collection)
- Legacy tools: [scripts_ai/README.md](scripts_ai/README.md) describes old CNN pipeline

## Troubleshooting

### Compilation Issues
- **Error: "Model schema mismatch"**: Model was trained with wrong TensorFlow version. Retrain with TFLite converter.
- **AllocateTensors() failed**: Increase `kTensorArenaSize` in [main.cpp](src/main.cpp)
- **Input tensor size mismatch**: Model expects different feature count. Check model input shape matches `NUM_FEATURES` in [functions.h](src/functions.h)
- **Filter coefficient errors**: Regenerate with [generate_filter_coefficients.py](scripts_ai/generate_filter_coefficients.py)

### Runtime Issues
- **Low accuracy**: Retrain model with more data, check sensor placement, validate filters with [validate_filters.py](scripts_ai/validate_filters.py)
- **No gesture detection**: Lower confidence threshold in [main.cpp](src/main.cpp) line 129
- **Serial port errors**: Update `upload_port` and `monitor_port` in [platformio.ini](platformio.ini)
- **Erratic ZC/SSC values**: Check filter initialization, verify `filters_init()` called in setup()
- **Filter instability**: Verify coefficients match sampling rate, check for NaN/Inf in filter states

### Model Training Issues
- **Feature mismatch**: Ensure C++ and Python DSP filters + TD4 extraction produce identical features
- **Filter inconsistency**: Run [validate_filters.py](scripts_ai/validate_filters.py) to verify
- **Low accuracy with TD4**: Check sensor placement, collect more training data, verify DSP filters enabled
- **ZC/SSC always zero**: Verify DC offset removal is applied after filtering
- See comprehensive troubleshooting in [scripts_ai/README.md](scripts_ai/README.md#troubleshooting)

## References & Literature

### DSP Filtering
- **Oppenheim & Schafer (2009)**: "Discrete-Time Signal Processing" - IIR filter design fundamentals
- **Parks & Burrus (1987)**: "Digital Filter Design" - Biquad structures
- **De Luca (2002)**: "Surface Electromyography: Detection and Recording" - EMG frequency bands

### TD4 Features
- **Hudgins et al. (1993)**: "A New Strategy for Multifunction Myoelectric Control" - Original TD4 paper
- **Phinyomark et al. (2012)**: "Feature Reduction and Selection for EMG Signal Classification" - TD4 validation study

### Wide & Deep Architecture
- **Cheng et al. (2016)**: "Wide & Deep Learning for Recommender Systems" - Architecture foundation

### EMG Pattern Recognition
- **Atzori et al. (2014)**: "Electromyography data for non-invasive naturally-controlled robotic hand prostheses"
- **Oskoei & Hu (2007)**: "Myoelectric Control Systems—A Survey"

---

## Changelog

### 2025-11-26 - scripts_ai Folder Reorganization

**Major restructuring** of the Python ML pipeline for better organization and maintainability:

**New Folder Structure:**
- `critical/` - Core end-to-end pipeline scripts (feature_extraction.py, train_test_model.py)
- `filters/` - DSP filter tools (dsp_filters.py, generate_filter_coefficients.py)
- `validation/` - Quality control and testing scripts (4 validation scripts)
- `visualization/` - Signal quality visualization tools

**Documentation Added:**
- Main project [README.md](README.md) - Comprehensive project documentation
- [scripts_ai/README.md](scripts_ai/README.md) - Pipeline overview and workflow
- [scripts_ai/SCRIPT_EXPLANATIONS.md](scripts_ai/SCRIPT_EXPLANATIONS.md) - Detailed Turkish explanations
- [scripts_ai/QUICK_START.md](scripts_ai/QUICK_START.md) - 3-minute quick start guide
- Individual READMEs for each subfolder (critical/, filters/, validation/, visualization/)

**Benefits:**
- ✅ Clear separation of concerns (core pipeline vs. utilities)
- ✅ Easy to find critical scripts for end-to-end workflow
- ✅ Comprehensive Turkish documentation for each script
- ✅ Better maintainability and onboarding

**Migration:** All scripts moved to appropriate subfolders, paths updated in this document.
