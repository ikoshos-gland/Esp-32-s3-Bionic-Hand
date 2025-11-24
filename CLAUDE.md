# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time gesture classification system for **ESP32-S3 DevKitC-1** using TensorFlow Lite Micro. The system uses **6 MyoWare EMG sensors** to classify 11 hand gestures for prosthetic control with **literature-validated TD4 features** (Hudgins et al. 1993).

**Gestures recognized:** Rest, Fist, Open, Point, Victory, OK, ThumbUp, ThumbDn, Grasp, Pinch, WristFlex

## Hardware Configuration

- **Board:** ESP32-S3 DevKitC-1 (N16R8 variant)
  - 16MB Flash, 8MB PSRAM (Octal OPI mode)
  - Dual-core Xtensa LX7 @ 240MHz
- **Serial Port:** COM7 (115200 baud) - Update in [platformio.ini](platformio.ini) if different
- **MPU6050:** NOT USED (IMU removed from system - EMG-only)
- **ADC Pins (6 EMG Sensors):**
  - GPIO4  → MW1 (EMG Sensor 1) - ADC1_CH3
  - GPIO5  → MW2 (EMG Sensor 2) - ADC1_CH4
  - GPIO6  → MW3 (EMG Sensor 3) - ADC1_CH5
  - GPIO7  → MW4 (EMG Sensor 4) - ADC1_CH6
  - GPIO15 → MW5 (EMG Sensor 5) - ADC2_CH4
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
   - No TFLite model (faster compilation, smaller binary)
   - Streams raw ADC values over serial at 1000 Hz
   - Used with `scripts_ai/training_data_collection.py`

2. **`real_time_inference`** - For real-time gesture classification
   - Includes TFLite Micro model
   - Classifies gestures using TD4 features
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

### Signal Processing Pipeline

The system processes sensor data through a **REFACTORED** literature-based TD4 pipeline that operates on **RAW signals** with DC offset removal:

1. **Raw Data Collection** (`prelim_collection()` in [functions.cpp](src/functions.cpp))
   - Collects **250ms window** of **raw ADC data** at 1000Hz sampling rate
   - **250 raw samples per sensor** (250ms × 1000Hz = 250 samples)
   - Reads **6 MyoWare EMG sensors** (GPIO 4-7, 15-16)
   - **No RMS preprocessing** (previous approach broke ZC/SSC features)
   - Stores in `raw_sensor_data[6][250]` buffers
   - **Response time: 250ms** (4× faster than previous 1000ms)

2. **DC Offset Removal & Feature Extraction** (`extract_features_from_raw()` in [functions.cpp](src/functions.cpp))
   - **STEP 1: Calculate mean** (DC offset) for each sensor
   - **STEP 2: Center signal** by subtracting mean (creates bipolar signal centered at 0)
   - **STEP 3: Extract TD4 features** on centered signal:
     - **MAV** (Mean Absolute Value): Average signal amplitude
     - **WL** (Waveform Length): Signal complexity measure
     - **ZC** (Zero Crossings): Frequency estimate (threshold: 15.0 ADC units)
     - **SSC** (Slope Sign Changes): Frequency content (threshold: 15.0 ADC units)
   - **STEP 4: Global normalization** using ADC_MAX = 4095 (preserves amplitude info)
   - **Output:** float array[24] for TFLite model input
   - **CRITICAL:** ZC/SSC now work correctly (were always 0 with old RMS approach)

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

### File Structure

**Source Files:**
- [main.cpp](src/main.cpp) - **Real-time inference**: Arduino setup/loop, TFLite integration, gesture detection
- [data_acquisition.cpp](src/data_acquisition.cpp) - **Data collection**: Streams raw EMG data at 1000 Hz for training
- [functions.cpp](src/functions.cpp) - **RAW signal processing**, DC offset removal, TD4 feature extraction
- [functions.h](src/functions.h) - Function declarations, pin definitions, constants (`NUM_FEATURES = 24`, `ADC_MAX = 4095`)
- [model.h](src/model.h) - Auto-generated TFLite model as C++ byte array (Wide & Deep MLP, Int8)
- ~~modeldata.h~~ - **DELETED** (obsolete, replaced by model.h)
- ~~modeldata.cpp~~ - **DELETED** (obsolete, replaced by model.h auto-generation)

**Build Configuration:**
- [platformio.ini](platformio.ini) - Defines 2 environments: `data_acquisition` and `real_time_inference`

**Documentation:**
- [ADVANCED_FEATURES_PROPOSAL.md](ADVANCED_FEATURES_PROPOSAL.md) - Proposal for 57-114 feature expansion (not currently implemented)
- ~~Model/~~ - Removed (obsolete backup folder)

**Libraries:**
- [lib/TensorFlowLite_ESP32/](lib/TensorFlowLite_ESP32/) - Custom TFLite Micro port for ESP32
  - Based on TensorFlow v2.1.1
  - Modified from upstream: .cc → .cpp, include paths adjusted
  - Uses experimental micro API

**AI Training Pipeline:**
- [scripts_ai/](scripts_ai/) - Python ML pipeline for TD4 feature extraction and MLP training
  - [feature_extraction.py](scripts_ai/feature_extraction.py) - TD4 feature extraction (MAV, WL, ZC, SSC)
  - [train_test_model.py](scripts_ai/train_test_model.py) - Wide & Deep MLP training with Int8 quantization
  - See [scripts_ai/README.md](scripts_ai/README.md) for legacy pipeline documentation
- `data/features/` - TD4 features NPZ files (auto-generated from feature_extraction.py)
- `models/` - TFLite models (Int8 quantized) and C headers (auto-generated)
- `plots/` - Training curves and confusion matrices (auto-generated)

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

## Training & Deploying Models

### TD4 Feature Extraction & MLP Training Pipeline

The project uses a **literature-based** approach with Hudgins' TD4 features and Wide & Deep MLP.

**Training workflow:**
```bash
cd scripts_ai

# 1. Collect training data from ESP32 (6 EMG sensors)
python training_data_collection.py
# Output: data/training_data_TIMESTAMP.csv

# 2. Extract TD4 features (MAV, WL, ZC, SSC)
python feature_extraction.py data/training_data_TIMESTAMP.csv
# Output: data/features/training_data_TIMESTAMP_td4_features.npz

# 3. Train Wide & Deep MLP with Int8 quantization
python train_test_model.py
# Output: models/mlp_td4_TIMESTAMP_int8.h

# 4. Copy generated model.h to src/
cp models/mlp_td4_TIMESTAMP_int8.h src/model.h

# 5. Build and upload to ESP32
pio run -t upload
```

### Model Architecture

**Input:** 24 TD4 features (4 features × 6 EMG sensors)
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
3. Replace [model.h](src/model.h) with new auto-generated .h file
4. Verify `NUM_FEATURES = 24` in [functions.h](src/functions.h)
5. Adjust `kTensorArenaSize` in [main.cpp](src/main.cpp) if needed

## Preprocessing Alignment (2025-11-24 CRITICAL UPDATE)

**🚨 BREAKING CHANGE**: The preprocessing pipeline was **REFACTORED** to fix critical Zero Crossing (ZC) and Slope Sign Change (SSC) feature bugs.

### The Problem

**Previous (BROKEN) Approach:**
```
Raw ADC → RMS windows → Per-window normalize [0,1] → TD4 features
```

**Why it failed:**
- ZC/SSC features require **bipolar signals** (centered at 0) to detect crossings
- RMS (Root Mean Square) is **always positive** (never crosses zero)
- Per-window normalization to [0,1] also stays positive
- **Result:** ZC and SSC were **always 0** (no crossings detected)
- Per-window normalization destroyed amplitude information between gestures

### The Solution

**New (CORRECT) Approach:**
```
Raw ADC → DC offset removal → TD4 features → Global normalization
```

| Component | Old (Broken) | **New (Fixed)** |
|-----------|--------------|-----------------|
| **Window Size** | 1000ms | **250ms** (4× faster response) |
| **Preprocessing** | RMS windows | **Raw samples with DC removal** |
| **Signal Type** | Unipolar (RMS ≥ 0) | **Bipolar (centered at 0)** |
| **Normalization** | Per-window [0,1] | **Global ADC [0,4095]** |
| **ZC/SSC Features** | Always 0 ❌ | **Working correctly** ✅ |
| **Array Size** | `mw_arr[20]` RMS | `raw_sensor_data[6][250]` |

### Changes Made

**C++ Side** ([functions.cpp](src/functions.cpp), [functions.h](src/functions.h)):
- ✅ **Removed RMS preprocessing** entirely
- ✅ Changed to **raw ADC sampling**: 250 samples @ 1000Hz = 250ms window
- ✅ **Added DC offset removal**: Calculate mean, subtract to center signal at 0
- ✅ **Fixed ZC/SSC**: Now work on centered bipolar signals
- ✅ **Changed normalization**: Per-window → Global (ADC_MAX = 4095)
- ✅ **Updated thresholds**: 0.01 (normalized) → 15.0 (ADC units)
- ✅ **Increased speed**: 1000ms → 250ms window (4× faster)

**Python Side** ([feature_extraction.py](scripts_ai/feature_extraction.py)):
- ✅ **Already refactored** to match C++ approach
- ✅ Uses raw ADC values (250ms window)
- ✅ Applies DC offset removal (mean subtraction)
- ✅ Calculates TD4 on centered signals
- ✅ Uses global normalization (ADC_MAX = 4095)

### Why This Matters

**Before (Broken):**
```python
# Python
rms_values = apply_rms(raw_adc)  # [2000, 2100, 2050, ...] (always positive)
normalized = minmax_normalize(rms_values)  # [0.0, 1.0, 0.5, ...] (still positive)
zc = count_zero_crossings(normalized)  # 0 (no crossings possible!)
```

**After (Fixed):**
```python
# Python
mean = np.mean(raw_adc)  # 2048 (DC offset)
centered = raw_adc - mean  # [-48, +52, +2, ...] (bipolar signal)
zc = count_zero_crossings(centered)  # 12 (crossings detected!)
```

**Impact on Features:**
- **MAV**: Now correctly preserves amplitude differences (global norm)
- **WL**: Reflects true signal complexity
- **ZC**: Now detects frequency content (was always 0)
- **SSC**: Now detects slope changes (was always 0)

### Validation

After retraining, verify features are extracted correctly:
```bash
# Upload new firmware
pio run -t upload && pio device monitor

# Check serial output for:
# - ZC values: should be 5-20 for active gestures (not 0)
# - SSC values: should be 5-20 for active gestures (not 0)
# - MAV varies between gestures (amplitude preserved)
```

### Retraining REQUIRED

⚠️ **CRITICAL**: Old models are **completely incompatible**. You **MUST retrain**:

1. Python pipeline was refactored (already done)
2. C++ pipeline was refactored (just completed)
3. Both now use **raw signals with DC removal**
4. Old models used broken RMS approach

**Retrain steps:**
```bash
cd scripts_ai
python feature_extraction.py data/training_data.csv
python train_test_model.py
cp models/mlp_td4_*_int8.h ../src/model.h
pio run -t upload
```

See [RETRAINING_GUIDE.md](RETRAINING_GUIDE.md) for complete instructions.

## Key Dependencies

Defined in [platformio.ini](platformio.ini):
- **Board:** `esp32-s3-devkitc-1` (ESP32-S3 with 16MB Flash, 8MB PSRAM)
- **Framework:** Arduino (built-in with ESP32 platform)
- **TensorFlowLite_ESP32** (local lib) - TensorFlow Lite Micro v2.1.1 for real-time inference
- **No external libraries needed** - All code uses Arduino.h built-in functions

Build flags:
- `-DBOARD_HAS_PSRAM` - Enable 8MB PSRAM support
- `-mfix-esp32-psram-cache-issue` - ESP32 PSRAM cache bug workaround
- `-DARDUINO_USB_CDC_ON_BOOT=1` - Enable USB Serial on boot
- `-DARDUINO_USB_MODE=1` - USB mode enabled

**Note:** ESP32-S3 uses different build flags than ESP32-C3. ADC2 works without special flags on S3.

## Development Notes

### TD4 Feature Extraction (Literature-Based)
- **Hudgins' TD4 features**: Gold standard for EMG pattern recognition (1993)
- **4 features per sensor**: MAV, WL, ZC, SSC
- **6 EMG sensors**: GPIO 4-7, 15-16 (24 total features)
- **MPU6050 REMOVED**: No longer used (EMG-only system)
- **Raw signal processing**: DC offset removal creates bipolar signals for ZC/SSC
- **Global normalization**: Preserves amplitude differences between gestures
- **Thresholds**: ZC=15.0 ADC, SSC=15.0 ADC (raw ADC units, noise-reduced)
- **Feature parity critical**: C++ extraction must match Python training code exactly

### Code Architecture Notes
- **Raw sensor buffers**: `raw_sensor_data[6][250]` - 250 raw ADC samples per sensor
- **24 feature array**: Global `features[NUM_FEATURES]` in [functions.cpp](src/functions.cpp)
- **DC offset removal**: Mean calculated per sensor, subtracted to center signal at 0
- **TFLite Int8 quantization**: Features quantized with `input->data.int8[i] = (int8_t)((features[i] / input->params.scale) + input->params.zero_point)`
- **Output dequantization**: `float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale`
- **Never use** `.f` float accessors with Int8 models - this will cause NaN values
- Serial output shows TD4 features per sensor and 11 class probabilities
- Duplicate gestures suppressed (only shows when gesture changes)
- Confidence threshold: 0.8 for gesture detection
- **Response time**: 250ms collection + ~20ms inference = ~270ms total latency

### Model Architecture & Performance
- **Wide & Deep MLP**: 256→128→64 neurons with dropout regularization
- **Int8 quantization**: 75% size reduction, 2-3× faster inference on ESP32
- **Literature-proven**: TD4 features achieve 85-95% accuracy in EMG literature
- **Processor-friendly**: Simple time-domain features, no FFT or complex math
- **Expected inference time**: <20ms per classification on ESP32-S3 @ 240MHz

### Training Data & Workflow
- Training data collected with 6 EMG sensors (update data collection scripts)
- TD4 features extracted via [feature_extraction.py](scripts_ai/feature_extraction.py)
- Models trained via [train_test_model.py](scripts_ai/train_test_model.py)
- Gesture reference images in `images_hand/` folder (used during data collection)
- Legacy tools: [scripts_ai/README.md](scripts_ai/README.md) describes old CNN pipeline

## Troubleshooting

### Compilation Issues
- **Error: "Model schema mismatch"**: Model was trained with wrong TensorFlow version. Retrain with TFLite converter.
- **AllocateTensors() failed**: Increase `kTensorArenaSize` in [main.cpp](src/main.cpp)
- **Input tensor size mismatch**: Model expects different feature count. Check model input shape matches `NUM_FEATURES` in [functions.h](src/functions.h)

### Runtime Issues
- **Low accuracy**: Retrain model with more data, or check sensor placement
- **No gesture detection**: Lower confidence threshold in [main.cpp](src/main.cpp) line 129
- **Serial port errors**: Update `upload_port` and `monitor_port` in [platformio.ini](platformio.ini)

### Model Training Issues
- **Feature mismatch**: Ensure C++ and Python TD4 extraction produce identical features
- **Low accuracy with TD4**: Check sensor placement, collect more training data per gesture
- **ZC/SSC always zero**: Verify normalization is applied before feature extraction
- See comprehensive troubleshooting in [scripts_ai/README.md](scripts_ai/README.md#troubleshooting)

## References & Literature

### TD4 Features
- **Hudgins et al. (1993)**: "A New Strategy for Multifunction Myoelectric Control" - Original TD4 paper
- **Phinyomark et al. (2012)**: "Feature Reduction and Selection for EMG Signal Classification" - TD4 validation study

### Wide & Deep Architecture
- **Cheng et al. (2016)**: "Wide & Deep Learning for Recommender Systems" - Architecture foundation

### EMG Pattern Recognition
- **Atzori et al. (2014)**: "Electromyography data for non-invasive naturally-controlled robotic hand prostheses"
- **Oskoei & Hu (2007)**: "Myoelectric Control Systems—A Survey"
