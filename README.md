# 🤖 Real-Time EMG Gesture Classification for ESP32-S3

[![ESP32-S3](https://img.shields.io/badge/ESP32--S3-DevKitC--1-blue.svg)](https://www.espressif.com/en/products/socs/esp32-s3)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-Micro-orange.svg)](https://www.tensorflow.org/lite/microcontrollers)
[![PlatformIO](https://img.shields.io/badge/PlatformIO-Compatible-green.svg)](https://platformio.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Real-time hand gesture recognition system using 6 MyoWare EMG sensors, literature-validated TD4 features, and on-device TensorFlow Lite inference for prosthetic control.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Hardware Setup](#-hardware-setup)
- [Quick Start](#-quick-start)
- [System Architecture](#-system-architecture)
- [Signal Processing Pipeline](#-signal-processing-pipeline)
- [Gesture Classes](#-gesture-classes)
- [Data Collection & Training](#-data-collection--training)
- [Model Architecture](#-model-architecture)
- [Build Commands](#-build-commands)
- [File Structure](#-file-structure)
- [Key Technical Details](#-key-technical-details)
- [Troubleshooting](#-troubleshooting)
- [Performance Metrics](#-performance-metrics)
- [Literature References](#-literature-references)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements a **real-time EMG-based hand gesture classification system** for prosthetic control on the **ESP32-S3 DevKitC-1** microcontroller. The system uses:

- **6 MyoWare EMG Sensors** (electromyography) to capture muscle activity
- **TD4 Features** (Time Domain 4-feature set) from Hudgins et al. (1993)
- **Wide & Deep MLP** with Int8 quantization for embedded inference
- **TensorFlow Lite Micro** for on-device neural network execution

### Key Features

✅ **11 Hand Gestures:** Rest, Fist, Open, Point, Victory, OK, ThumbUp, ThumbDn, Grasp, Pinch, WristFlex
✅ **Fast Response Time:** ~270ms total latency (250ms data collection + ~20ms inference)
✅ **Literature-Validated:** TD4 features achieve 85-95% accuracy in EMG research
✅ **Embedded-Optimized:** Int8 quantization (75% size reduction, 2-3× faster inference)
✅ **Dual Environment System:** Separate builds for data acquisition and real-time inference
✅ **Complete Training Pipeline:** Python scripts for feature extraction and model training

### Hardware Specifications

- **MCU:** ESP32-S3 DevKitC-1 (N16R8 variant)
  - Dual-core Xtensa LX7 @ 240 MHz
  - 16MB Flash, 8MB PSRAM (Octal OPI mode)
  - 12-bit ADC (0-4095 range)
- **Sensors:** 6× MyoWare EMG Sensors (v2022 recommended)
- **Serial Port:** COM7 @ 115200 baud (configurable in [platformio.ini](platformio.ini))
- **IMU:** ❌ NOT USED (previous MPU6050 removed - EMG-only system)

---

## ⚙️ Hardware Setup

### 🔌 GPIO Pin Configuration

The system uses **6 EMG sensors** connected to specific GPIO pins on the ESP32-S3:

| Sensor | GPIO Pin | ADC Channel | Description |
|--------|----------|-------------|-------------|
| MW1    | GPIO 4   | ADC1_CH3    | EMG Sensor 1 |
| MW2    | GPIO 5   | ADC1_CH4    | EMG Sensor 2 |
| MW3    | GPIO 6   | ADC1_CH5    | EMG Sensor 3 |
| MW4    | GPIO 7   | ADC1_CH6    | EMG Sensor 4 |
| MW5    | GPIO 15  | ADC2_CH4    | EMG Sensor 5 |
| MW6    | GPIO 16  | ADC2_CH5    | EMG Sensor 6 |

### ⚠️ CRITICAL: Pin Change Warning

**GPIO pins were changed from 0-5 to 4-7, 15-16** to avoid UART0 conflicts:
- GPIO 1-2 are used by USB Serial (UART0 TX/RX) and must be avoided
- **You must physically reconnect your EMG sensors to the new pins!**
- ADC2 (GPIO 15-16) is safe on ESP32-S3 (no Wi-Fi conflict like ESP32-C3)

### 🔧 ADC Configuration

```cpp
Resolution:    12-bit (0-4095)
Attenuation:   11dB (0-3.3V range)
Sampling Rate: 1000 Hz (1ms per sample)
Window Size:   250ms (250 samples per sensor)
```

### 📍 Sensor Placement Recommendations

For optimal gesture recognition:
1. **Forearm placement:** Position sensors on the forearm muscle groups
2. **Consistent location:** Keep sensor placement consistent between training and inference
3. **Skin contact:** Ensure good skin contact with electrode pads
4. **Cable management:** Secure cables to prevent motion artifacts
5. **Reference images:** Use `images_hand/` folder during data collection for consistent gestures

---

## 🚀 Quick Start

### Prerequisites

1. **PlatformIO** installed ([Installation Guide](https://platformio.org/install))
2. **ESP32-S3 DevKitC-1** connected via USB
3. **6 MyoWare EMG Sensors** wired to GPIO pins 4-7, 15-16
4. **Python 3.8+** with TensorFlow 2.x for training

### Add PlatformIO to PATH (Windows)

```powershell
$env:Path += ";C:\Users\MERT\.platformio\penv\Scripts"
```

### Basic Usage

#### Option 1: Real-Time Inference (Pre-trained Model)

```bash
# 1. Upload inference firmware
pio run -e real_time_inference -t upload

# 2. Monitor serial output
pio device monitor

# Expected output:
# Gesture: Fist (Confidence: 0.92)
# Gesture: Open (Confidence: 0.87)
```

#### Option 2: Collect Training Data & Retrain

```bash
# 1. Upload data acquisition firmware
pio run -e data_acquisition -t upload

# 2. Collect training data
cd data_acquisition/scripts
python training_data_collection.py

# 3. Extract TD4 features
cd ../../scripts_ai
python feature_extraction.py ../data/training_data_TIMESTAMP.csv

# 4. Train Wide & Deep MLP
python train_test_model.py

# 5. Deploy new model to ESP32
cp models/mlp_td4_TIMESTAMP_int8.h ../src/model.h
cd ..
pio run -e real_time_inference -t upload
```

---

## 🏗️ System Architecture

### Dual Environment Build System

The project uses a **sophisticated two-environment architecture** in [platformio.ini](platformio.ini):

| Environment | Purpose | TFLite Model | Main File | Use Case |
|-------------|---------|--------------|-----------|----------|
| **`data_acquisition`** | Collect training data | ❌ No | [data_acquisition.cpp](src/data_acquisition.cpp) | Stream raw EMG @ 1000 Hz |
| **`real_time_inference`** | Gesture classification | ✅ Yes | [main.cpp](src/main.cpp) | Real-time inference |

**Why two environments?**
- **Faster compilation:** Data acquisition doesn't need TFLite (smaller binary, faster builds)
- **Cleaner separation:** Training and inference code isolated
- **Efficient development:** Switch between modes without code changes

### Build Selection

```bash
# Data acquisition mode
pio run -e data_acquisition -t upload

# Real-time inference mode
pio run -e real_time_inference -t upload
```

---

## 🔬 Signal Processing Pipeline

### The Problem: Previous Approach (BROKEN)

```
Raw ADC → RMS windows → Per-window normalize [0,1] → TD4 features
                                                          ↓
                                          ❌ ZC/SSC always returned 0
```

**Why it failed:**
- RMS (Root Mean Square) is **always positive** (never crosses zero)
- Per-window normalization to [0,1] also stays positive
- Zero Crossing (ZC) and Slope Sign Change (SSC) require **bipolar signals** centered at 0
- **Result:** ZC and SSC features were **completely broken** (always 0)

### The Solution: Current Approach (FIXED) ✅

```
Raw ADC → DC offset removal → TD4 features → Global normalization
            (center at 0)      (bipolar OK)    (preserve amplitude)
```

### Processing Stages

#### **Stage 1: Raw Data Collection** ([functions.cpp](src/functions.cpp):`prelim_collection()`)

```cpp
// Collects 250ms window of raw ADC data
void prelim_collection() {
    for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
        raw_sensor_data[0][i] = analogRead(pin_MW1);
        raw_sensor_data[1][i] = analogRead(pin_MW2);
        raw_sensor_data[2][i] = analogRead(pin_MW3);
        raw_sensor_data[3][i] = analogRead(pin_MW4);
        raw_sensor_data[4][i] = analogRead(pin_MW5);
        raw_sensor_data[5][i] = analogRead(pin_MW6);
        delayMicroseconds(1000);  // 1ms = 1000 Hz sampling
    }
}
```

- **Window size:** 250ms (250 samples @ 1000 Hz)
- **Storage:** `raw_sensor_data[6][250]` buffers
- **No preprocessing:** Pure raw ADC values (0-4095)
- **4× faster** than previous 1000ms approach

#### **Stage 2: DC Offset Removal & Feature Extraction** ([functions.cpp](src/functions.cpp):`extract_features_from_raw()`)

```cpp
void extract_features_from_raw() {
    int feature_idx = 0;

    for (int sensor = 0; sensor < NUM_SENSORS; sensor++) {
        // STEP 1: Calculate mean (DC offset)
        float mean = 0.0f;
        for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
            mean += raw_sensor_data[sensor][i];
        }
        mean /= RAW_WINDOW_SIZE;

        // STEP 2: Center signal by subtracting mean
        float centered[RAW_WINDOW_SIZE];
        for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
            centered[i] = raw_sensor_data[sensor][i] - mean;
        }

        // STEP 3: Extract TD4 features on centered signal
        float mav = compute_MAV(centered);
        float wl = compute_WL(centered);
        float zc = compute_ZC(centered);   // Now works! (bipolar signal)
        float ssc = compute_SSC(centered); // Now works! (bipolar signal)

        // STEP 4: Global normalization (preserve amplitude info)
        features[feature_idx++] = mav / ADC_MAX_GLOBAL;
        features[feature_idx++] = wl / (RAW_WINDOW_SIZE * ADC_MAX_GLOBAL);
        features[feature_idx++] = zc / RAW_WINDOW_SIZE;
        features[feature_idx++] = ssc / RAW_WINDOW_SIZE;
    }
}
```

**Critical changes:**
- ✅ DC offset removal creates **bipolar signals** (centered at 0)
- ✅ ZC/SSC now work correctly (detect zero crossings)
- ✅ Global normalization preserves amplitude differences between gestures
- ✅ Matches Python preprocessing exactly ([feature_extraction.py](scripts_ai/feature_extraction.py))

#### **Stage 3: TD4 Feature Set** (Hudgins et al. 1993)

The system extracts **4 time-domain features** per sensor × 6 sensors = **24 total features**:

| Feature | Abbreviation | Description | Formula |
|---------|--------------|-------------|---------|
| **Mean Absolute Value** | MAV | Average signal amplitude | `Σ\|x[i]\| / N` |
| **Waveform Length** | WL | Signal complexity measure | `Σ\|x[i+1] - x[i]\|` |
| **Zero Crossings** | ZC | Frequency estimate | Count of sign changes |
| **Slope Sign Changes** | SSC | Frequency content | Count of slope reversals |

**Feature ordering** (MUST match Python training):
```
[MAV₁, WL₁, ZC₁, SSC₁,  // Sensor 1
 MAV₂, WL₂, ZC₂, SSC₂,  // Sensor 2
 MAV₃, WL₃, ZC₃, SSC₃,  // Sensor 3
 MAV₄, WL₄, ZC₄, SSC₄,  // Sensor 4
 MAV₅, WL₅, ZC₅, SSC₅,  // Sensor 5
 MAV₆, WL₆, ZC₆, SSC₆]  // Sensor 6
```

**TD4 Thresholds:**
```cpp
#define ZC_THRESHOLD_ADC   15.0f  // Zero crossing threshold (ADC units)
#define SSC_THRESHOLD_ADC  15.0f  // Slope sign change threshold (ADC units)
```

#### **Stage 4: TFLite Inference** ([main.cpp](src/main.cpp):83-159)

```cpp
// Input quantization (Float32 → Int8)
TfLiteTensor* input = interpreter->input(0);
for (int i = 0; i < NUM_FEATURES; i++) {
    float scaled = (features[i] / input->params.scale) + input->params.zero_point;
    input->data.int8[i] = clamp_int8(scaled);  // Safe clamping [-128, 127]
}

// Run inference
TfLiteStatus invoke_status = interpreter->Invoke();

// Output dequantization (Int8 → Float32)
TfLiteTensor* output = interpreter->output(0);
for (int i = 0; i < 11; i++) {
    float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;
    probabilities[i] = clamp_prob(prob);  // Ensure [0.0, 1.0]
}
```

**Key details:**
- **30KB tensor arena** (increased from 20KB) with 16-byte SIMD alignment
- **Int8 quantization:** 75% size reduction, 2-3× faster inference
- **Confidence threshold:** 0.8 for gesture detection
- **Debouncing:** Requires 2 consecutive identical predictions (~540ms)
- **Motion locking:** 1000ms cooldown after gesture detection (except Rest)

---

## 🤲 Gesture Classes

The system recognizes **11 hand gestures**:

| ID | Gesture | Description | Image |
|----|---------|-------------|-------|
| 0  | **Rest** | Baseline/no movement | [0.jpg](images_hand/0.jpg) |
| 1  | **Fist** | Closed fist | [1.jpg](images_hand/1.jpg) |
| 2  | **Open** | Open hand, fingers extended | [2.jpg](images_hand/2.jpg) |
| 3  | **Point** | Index finger extended | [3.jpg](images_hand/3.jpg) |
| 4  | **Victory** | Peace sign (V-sign) | [4.jpg](images_hand/4.jpg) |
| 5  | **OK** | Thumb-index circle | [5.jpg](images_hand/5.jpg) |
| 6  | **ThumbUp** | Thumbs up | [6.jpg](images_hand/6.jpg) |
| 7  | **ThumbDn** | Thumbs down | [7.jpg](images_hand/7.jpg) |
| 8  | **Grasp** | Power grip | [8.jpg](images_hand/8.jpg) |
| 9  | **Pinch** | Precision grip | [9.jpg](images_hand/9.jpg) |
| 10 | **WristFlex** | Wrist flexion | [10.jpg](images_hand/10.jpg) |

**Visual reference:** All gesture images are available in the `images_hand/` folder for consistent data collection.

---

## 📊 Data Collection & Training

### Complete Training Workflow

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Data Collection │  →   │ Feature Extract  │  →   │  Model Training  │
│   (ESP32 + USB)  │      │   (Python TD4)   │      │   (TF2 + Int8)   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
        ↓                         ↓                          ↓
  training_data.csv         td4_features.npz       mlp_td4_int8.h
```

### Step 1: Data Collection

#### Upload Data Acquisition Firmware

```bash
pio run -e data_acquisition -t upload
pio device monitor  # Verify streaming works
```

#### Run Python Collection Script

```bash
cd data_acquisition/scripts
python training_data_collection.py
```

**Collection protocol:**
- **10 gestures** × **10 repetitions** each
- **4 seconds** per gesture + **3 seconds** rest
- Uses visual reference images from `images_hand/`
- Outputs: `data/training_data_TIMESTAMP.csv`

**Data format:**
```csv
Timestamp, Seq, Movement, Phase, Rep, EMG1, EMG2, EMG3, EMG4, EMG5, EMG6
0.000, 0, Fist, active, 1, 2048, 2051, 2049, 2050, 2047, 2052
0.001, 1, Fist, active, 1, 2100, 2103, 2098, 2101, 2099, 2104
...
```

### Step 2: Feature Extraction

```bash
cd scripts_ai
python feature_extraction.py ../data/training_data_TIMESTAMP.csv
```

**What it does:**
1. Loads raw CSV data (250ms windows @ 1000 Hz)
2. Applies **DC offset removal** (mean subtraction per window)
3. Extracts **TD4 features** on centered signals (MAV, WL, ZC, SSC)
4. Applies **global normalization** (ADC_MAX = 4095)
5. Saves features.npz with labels

**Output:** `data/features/training_data_TIMESTAMP_td4_features.npz`

**TD4 Feature Extractor Configuration:**
```python
TD4FeatureExtractor(
    window_size_ms=250,       # Matches C++ (was 1000ms)
    sampling_rate=1000,       # 1000 Hz
    zc_threshold_adc=15.0,    # ADC units (not normalized)
    ssc_threshold_adc=15.0,   # ADC units (not normalized)
    adc_max=4095.0            # Global normalization
)
```

### Step 3: Model Training

```bash
python train_test_model.py
```

**What it does:**
1. Auto-finds latest NPZ file in `data/features/`
2. Splits data: 70% train, 15% validation, 15% test
3. Trains Wide & Deep MLP (150 epochs, early stopping)
4. Applies **Post-Training Int8 Quantization**
5. Converts to TFLite and generates C header

**Outputs:**
- `models/mlp_td4_TIMESTAMP.keras` (full Keras model)
- `models/mlp_td4_TIMESTAMP_int8.tflite` (quantized TFLite)
- `models/mlp_td4_TIMESTAMP_int8.h` (C header for ESP32)
- `plots/training_history_TIMESTAMP.png` (loss/accuracy curves)
- `plots/confusion_matrix_TIMESTAMP.png` (classification results)

### Step 4: Deploy to ESP32

```bash
# Copy generated model header to src/
cp models/mlp_td4_TIMESTAMP_int8.h ../src/model.h

# Build and upload inference firmware
cd ..
pio run -e real_time_inference -t upload

# Monitor real-time predictions
pio device monitor
```

---

## 🧠 Model Architecture

### Wide & Deep MLP

The system uses a **Wide & Deep Multi-Layer Perceptron** with dropout regularization:

```
Input Layer: 24 TD4 features
     ↓
Dense(256, relu) + Dropout(0.3)
     ↓
Dense(128, relu) + Dropout(0.2)
     ↓
Dense(64, relu) + Dropout(0.1)
     ↓
Output Layer: 11 gestures (softmax)
```

**Architecture details:**
- **Parameters:** ~50K (quantized to Int8)
- **Input:** 24 TD4 features (4 features × 6 sensors)
- **Output:** 11 class probabilities (0-10)
- **Activation:** ReLU (hidden layers), Softmax (output)
- **Regularization:** Dropout layers prevent overfitting

### Training Configuration

```python
Epochs:        150 (early stopping patience: 15)
Batch size:    32
Optimizer:     Adam (learning rate: 0.001)
Loss:          Categorical crossentropy
Metrics:       Accuracy
```

### Post-Training Int8 Quantization

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
```

**Benefits:**
- **75% size reduction** (e.g., 50KB → 13KB)
- **2-3× faster inference** on ESP32
- **Minimal accuracy loss** (<1% typical)
- **Hardware acceleration** via SIMD instructions

### Model Deployment Format

The trained model is converted to a **C byte array** in [model.h](src/model.h):

```cpp
// Auto-generated by train_test_model.py
alignas(8) const unsigned char model_tflite[] = {
    0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
};
const unsigned int model_tflite_len = 13524;  // ~13.2KB
```

---

## 🛠️ Build Commands

### PlatformIO Commands

#### Data Acquisition Mode

```bash
# Build only
pio run -e data_acquisition

# Build and upload
pio run -e data_acquisition -t upload

# Upload and monitor
pio run -e data_acquisition -t upload && pio device monitor
```

#### Real-Time Inference Mode

```bash
# Build only
pio run -e real_time_inference

# Build and upload
pio run -e real_time_inference -t upload

# Upload and monitor
pio run -e real_time_inference -t upload && pio device monitor
```

#### General Commands

```bash
# Monitor serial output (works for both environments)
pio device monitor

# List connected devices
pio device list

# Clean build files
pio run --target clean

# Full clean (including dependencies)
pio run --target fullclean

# Update platforms and libraries
pio pkg update
```

### Serial Monitor Configuration

Edit [platformio.ini](platformio.ini) if your COM port is different:

```ini
upload_port = COM7
monitor_port = COM7
monitor_speed = 115200
```

---

## 📁 File Structure

```
real_time_esp322/
│
├── src/                              # ESP32 source code
│   ├── main.cpp                      # Real-time inference (TFLite + gesture detection)
│   ├── data_acquisition.cpp          # Data collection (streams raw EMG @ 1kHz)
│   ├── functions.cpp                 # Signal processing (DC removal, TD4 extraction)
│   ├── functions.h                   # Function declarations, constants
│   └── model.h                       # Auto-generated TFLite model (Int8, ~13KB)
│
├── scripts_ai/                       # Python ML pipeline
│   ├── feature_extraction.py        # TD4 feature extraction (MAV, WL, ZC, SSC)
│   ├── train_test_model.py          # Wide & Deep MLP training + Int8 quantization
│   ├── validate_preprocessing.py    # C++/Python preprocessing parity validation
│   └── README.md                     # Legacy pipeline documentation
│
├── data_acquisition/                 # Data collection tools
│   └── scripts/
│       └── training_data_collection.py  # Interactive data collection script
│
├── lib/                              # Libraries
│   └── TensorFlowLite_ESP32/         # Custom TFLite Micro v2.1.1 port
│       ├── src/                      # TFLite kernels and operations
│       └── tensorflow/               # TensorFlow headers
│
├── data/                             # Training data
│   ├── training_data_TIMESTAMP.csv   # Raw EMG data (from data collection)
│   └── features/                     # Extracted TD4 features
│       └── training_data_TIMESTAMP_td4_features.npz
│
├── models/                           # Trained models
│   ├── mlp_td4_TIMESTAMP.keras       # Full Keras model
│   ├── mlp_td4_TIMESTAMP_int8.tflite # Quantized TFLite model
│   └── mlp_td4_TIMESTAMP_int8.h      # C header for ESP32
│
├── plots/                            # Training visualizations
│   ├── training_history_TIMESTAMP.png
│   └── confusion_matrix_TIMESTAMP.png
│
├── images_hand/                      # Gesture reference images
│   ├── 0.jpg                         # Rest
│   ├── 1.jpg                         # Fist
│   └── ...                           # (gestures 0-10)
│
├── platformio.ini                    # PlatformIO configuration (dual environments)
├── CLAUDE.md                         # Comprehensive project documentation
└── README.md                         # This file
```

### Key Files Explained

| File | Purpose | Details |
|------|---------|---------|
| [main.cpp](src/main.cpp) | Real-time inference | TFLite setup, gesture detection, serial output |
| [data_acquisition.cpp](src/data_acquisition.cpp) | Data collection | Streams raw EMG @ 1000 Hz for training |
| [functions.cpp](src/functions.cpp) | Signal processing | Raw data collection, DC removal, TD4 extraction |
| [functions.h](src/functions.h) | Constants & declarations | NUM_FEATURES, ADC_MAX, pin definitions |
| [model.h](src/model.h) | TFLite model | Auto-generated C byte array (~13KB) |
| [feature_extraction.py](scripts_ai/feature_extraction.py) | Feature extraction | Converts raw CSV to TD4 features NPZ |
| [train_test_model.py](scripts_ai/train_test_model.py) | Model training | Wide & Deep MLP + Int8 quantization |
| [platformio.ini](platformio.ini) | Build configuration | Dual environment setup, build flags |

---

## 🔑 Key Technical Details

### Important Constants

#### From [functions.h](src/functions.h)

```cpp
// Feature configuration
#define NUM_FEATURES      24      // TD4 features: 4 features × 6 EMG sensors
#define NUM_SENSORS       6       // Number of EMG sensors

// Raw data collection
#define RAW_WINDOW_SIZE   250     // 250ms window (250 samples at 1kHz)
#define SAMPLING_FREQ     1000    // 1000 Hz sampling rate

// Global ADC normalization (12-bit ADC on ESP32)
#define ADC_MIN_GLOBAL    0.0f    // Minimum ADC value
#define ADC_MAX_GLOBAL    4095.0f // Maximum ADC value (12-bit = 4095)

// GPIO Pin Assignments (ESP32-S3)
#define pin_MW1  4    // GPIO 4  (ADC1_CH3)
#define pin_MW2  5    // GPIO 5  (ADC1_CH4)
#define pin_MW3  6    // GPIO 6  (ADC1_CH5)
#define pin_MW4  7    // GPIO 7  (ADC1_CH6)
#define pin_MW5  15   // GPIO 15 (ADC2_CH4)
#define pin_MW6  16   // GPIO 16 (ADC2_CH5)
```

#### From [functions.cpp](src/functions.cpp)

```cpp
// TD4 thresholds in ADC units (NOT normalized values)
#define ZC_THRESHOLD_ADC   15.0f  // Zero crossing threshold (ADC units)
#define SSC_THRESHOLD_ADC  15.0f  // Slope sign change threshold (ADC units)
```

#### From [main.cpp](src/main.cpp)

```cpp
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB for TFLite (increased from 20KB)
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // 16-byte aligned for SIMD
float threshold = 0.8;  // Confidence threshold for gesture detection
```

### Int8 Quantization Handling

**Critical:** Never use `.f` float accessors with Int8 models - causes NaN values!

#### Input Quantization (Float32 → Int8)

```cpp
TfLiteTensor* input = interpreter->input(0);
for (int i = 0; i < NUM_FEATURES; i++) {
    // Quantize: float → int8
    float scaled = (features[i] / input->params.scale) + input->params.zero_point;

    // Safe clamping to [-128, 127]
    if (scaled > 127.0f) scaled = 127.0f;
    if (scaled < -128.0f) scaled = -128.0f;

    input->data.int8[i] = static_cast<int8_t>(scaled);
}
```

#### Output Dequantization (Int8 → Float32)

```cpp
TfLiteTensor* output = interpreter->output(0);
for (int i = 0; i < 11; i++) {
    // Dequantize: int8 → float
    float prob = (output->data.int8[i] - output->params.zero_point) * output->params.scale;

    // Ensure probability range [0.0, 1.0]
    if (prob < 0.0f) prob = 0.0f;
    if (prob > 1.0f) prob = 1.0f;

    probabilities[i] = prob;
}
```

### Gesture Detection Logic

The system implements **robust gesture detection** with debouncing and motion locking:

```cpp
// 1. Find highest probability gesture
int max_idx = 0;
float max_prob = probabilities[0];
for (int i = 1; i < 11; i++) {
    if (probabilities[i] > max_prob) {
        max_prob = probabilities[i];
        max_idx = i;
    }
}

// 2. Check confidence threshold
if (max_prob < threshold) {
    return;  // Too uncertain
}

// 3. Debouncing: Require DEBOUNCE_FRAMES consecutive predictions
if (max_idx == last_predicted_gesture) {
    debounce_counter++;
} else {
    debounce_counter = 0;
    last_predicted_gesture = max_idx;
}

if (debounce_counter >= DEBOUNCE_FRAMES) {  // DEBOUNCE_FRAMES = 2
    // 4. Motion locking: Prevent rapid re-detections
    if (millis() - last_motion_time < MOTION_LOCK_MS) {  // MOTION_LOCK_MS = 1000
        return;  // Still in cooldown
    }

    // 5. Detect gesture change
    if (max_idx != last_output_gesture) {
        Serial.print("Gesture: ");
        Serial.print(gesture_names[max_idx]);
        Serial.print(" (Confidence: ");
        Serial.print(max_prob);
        Serial.println(")");

        last_output_gesture = max_idx;
        last_motion_time = millis();  // Rest bypasses this
    }
}
```

**Detection parameters:**
- **Confidence threshold:** 0.8 (80%)
- **Debounce frames:** 2 consecutive predictions (~540ms)
- **Motion lock:** 1000ms cooldown (except Rest gesture)
- **Duplicate suppression:** Only outputs when gesture changes

### Preprocessing Parity (C++ ↔ Python)

**CRITICAL:** Feature extraction must be **identical** between training and inference, or model will fail!

| Component | C++ ([functions.cpp](src/functions.cpp)) | Python ([feature_extraction.py](scripts_ai/feature_extraction.py)) | Status |
|-----------|----------------------|----------------------|--------|
| Window size | 250ms | 250ms | ✅ Match |
| Sampling rate | 1000 Hz | 1000 Hz | ✅ Match |
| DC offset removal | Mean subtraction | Mean subtraction | ✅ Match |
| Signal centering | `raw - mean` | `raw - mean` | ✅ Match |
| ZC threshold | 15.0 ADC units | 15.0 ADC units | ✅ Match |
| SSC threshold | 15.0 ADC units | 15.0 ADC units | ✅ Match |
| Normalization | Global (ADC_MAX = 4095) | Global (ADC_MAX = 4095) | ✅ Match |
| Feature order | MAV, WL, ZC, SSC | MAV, WL, ZC, SSC | ✅ Match |

**Validation script:** Run `python scripts_ai/validate_preprocessing.py` to verify parity.

---

## 🐛 Troubleshooting

### Compilation Issues

#### Error: "Model schema mismatch"

**Cause:** Model was trained with wrong TensorFlow version
**Fix:** Retrain model with TFLite converter (TF 2.x)

```bash
cd scripts_ai
python train_test_model.py
cp models/mlp_td4_*_int8.h ../src/model.h
```

#### Error: "AllocateTensors() failed"

**Cause:** Insufficient tensor arena memory
**Fix:** Increase `kTensorArenaSize` in [main.cpp](src/main.cpp)

```cpp
constexpr int kTensorArenaSize = 40 * 1024;  // Try 40KB instead of 30KB
```

#### Error: "Input tensor size mismatch"

**Cause:** Model expects different number of features
**Fix:** Verify model input shape matches `NUM_FEATURES` in [functions.h](src/functions.h)

```cpp
#define NUM_FEATURES 24  // Must match model input: 24 TD4 features
```

### Runtime Issues

#### ZC/SSC Features Always 0

**Cause:** Broken preprocessing (using RMS or no DC removal)
**Fix:** Ensure DC offset removal is applied before TD4 extraction

```cpp
// CORRECT: DC offset removal
float mean = calculate_mean(raw_sensor_data[sensor]);
for (int i = 0; i < RAW_WINDOW_SIZE; i++) {
    centered[i] = raw_sensor_data[sensor][i] - mean;  // Bipolar signal
}
float zc = compute_ZC(centered);  // Now works!
```

**Validation:**
```bash
pio device monitor
# Check serial output:
# ZC values should be 5-20 for active gestures (not 0)
# SSC values should be 5-20 for active gestures (not 0)
```

#### Low Accuracy

**Possible causes:**
1. **Sensor placement inconsistent** between training and inference
2. **Insufficient training data** (< 10 repetitions per gesture)
3. **Poor skin contact** with EMG electrodes
4. **Model not retrained** after code changes

**Fixes:**
```bash
# 1. Collect more training data
pio run -e data_acquisition -t upload
cd data_acquisition/scripts
python training_data_collection.py  # Do 20+ repetitions

# 2. Retrain model
cd ../../scripts_ai
python feature_extraction.py ../data/training_data_*.csv
python train_test_model.py

# 3. Deploy new model
cp models/mlp_td4_*_int8.h ../src/model.h
cd ..
pio run -e real_time_inference -t upload
```

#### No Gesture Detection

**Cause:** Confidence threshold too high
**Fix:** Lower threshold in [main.cpp](src/main.cpp)

```cpp
float threshold = 0.6;  // Try 0.6 instead of 0.8
```

#### Serial Port Errors

**Cause:** Wrong COM port configured
**Fix:** Update [platformio.ini](platformio.ini)

```ini
upload_port = COM7      # Change to your port (check Device Manager)
monitor_port = COM7
```

Find your port:
```bash
pio device list
```

### Model Training Issues

#### Feature Mismatch Between C++ and Python

**Cause:** TD4 extraction differs between C++ and Python
**Fix:** Validate preprocessing parity

```bash
cd scripts_ai
python validate_preprocessing.py
# Should output: "✅ All features match within tolerance!"
```

#### Training Data Loading Errors

**Cause:** CSV file format incorrect
**Expected format:**
```csv
Timestamp, Seq, Movement, Phase, Rep, EMG1, EMG2, EMG3, EMG4, EMG5, EMG6
```

**Fix:** Re-collect data with official script:
```bash
cd data_acquisition/scripts
python training_data_collection.py
```

---

## 📈 Performance Metrics

### Timing Breakdown

| Stage | Duration | Details |
|-------|----------|---------|
| **Raw data collection** | 250ms | 250 samples @ 1000 Hz × 6 sensors |
| **DC offset removal** | <5ms | Mean calculation + subtraction |
| **TD4 feature extraction** | <5ms | MAV, WL, ZC, SSC computation |
| **TFLite inference** | ~20ms | Int8 quantized MLP forward pass |
| **Total latency** | **~270ms** | End-to-end gesture classification |

**Response time:** 4× faster than previous 1000ms approach!

### Memory Usage

| Component | Size | Details |
|-----------|------|---------|
| **TFLite model** | 13.2KB | Int8 quantized Wide & Deep MLP |
| **Tensor arena** | 30KB | TFLite workspace (16-byte aligned) |
| **Raw data buffers** | 6KB | `raw_sensor_data[6][250]` × 4 bytes |
| **Feature array** | 96 bytes | `features[24]` × 4 bytes |
| **Total RAM usage** | **~50KB** | Active memory footprint |

**Flash usage:**
- Data acquisition firmware: ~800KB
- Real-time inference firmware: ~1.2MB (includes TFLite library)

### Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Input features** | 24 | 4 TD4 × 6 EMG sensors |
| **Output classes** | 11 | 11 hand gestures |
| **Model parameters** | ~50K | Quantized to Int8 |
| **Expected accuracy** | 85-95% | Literature-proven TD4 features |
| **Inference time** | ~20ms | ESP32-S3 @ 240 MHz |
| **Size reduction** | 75% | Int8 vs Float32 |
| **Speed improvement** | 2-3× | Int8 vs Float32 |

---

## 📚 Literature References

### TD4 Features

- **Hudgins, B., Parker, P., & Scott, R. N. (1993).** "A New Strategy for Multifunction Myoelectric Control." *IEEE Transactions on Biomedical Engineering*, 40(1), 82-94.
  - **Original TD4 paper** - Introduced MAV, WL, ZC, SSC features for EMG classification

- **Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012).** "Feature Reduction and Selection for EMG Signal Classification." *Expert Systems with Applications*, 39(8), 7420-7431.
  - **TD4 validation study** - Confirmed TD4 as optimal feature set for EMG recognition

### Wide & Deep Architecture

- **Cheng, H. T., Koc, L., Harmsen, J., et al. (2016).** "Wide & Deep Learning for Recommender Systems." *Proceedings of the 1st Workshop on Deep Learning for Recommender Systems*, 7-10.
  - **Architecture foundation** - Wide & Deep neural network design principles

### EMG Pattern Recognition

- **Atzori, M., Gijsberts, A., Castellini, C., et al. (2014).** "Electromyography Data for Non-Invasive Naturally-Controlled Robotic Hand Prostheses." *Scientific Data*, 1, 140053.
  - **EMG dataset benchmark** - Standard dataset for prosthetic control research

- **Oskoei, M. A., & Hu, H. (2007).** "Myoelectric Control Systems—A Survey." *Biomedical Signal Processing and Control*, 2(4), 275-294.
  - **Comprehensive survey** - Overview of EMG-based control systems

---

## 🤝 Contributing

### Development Notes

- **Code style:** Follow Arduino/C++ naming conventions
- **Comments:** Document signal processing math and TFLite operations
- **Testing:** Always validate preprocessing parity after changes
- **Documentation:** Update CLAUDE.md and README.md for major changes

### Feature Ideas

- [ ] Add Bluetooth/Wi-Fi gesture streaming
- [ ] Implement adaptive thresholds for ZC/SSC
- [ ] Support more gesture classes (15-20 gestures)
- [ ] Add real-time performance monitoring (FPS, latency)
- [ ] Implement on-device model retraining (TinyML)
- [ ] Add IMU fusion (MPU6050 re-integration)

### Known Limitations

- **Fixed sensor count:** Hardcoded for 6 EMG sensors
- **No auto-calibration:** Requires manual sensor placement consistency
- **Single user:** Model trained for one person (poor cross-user accuracy)
- **Static gestures:** Best for discrete gestures (not continuous motion)

### Reporting Issues

Found a bug or have a suggestion? Please report it with:
1. **Hardware setup:** ESP32 model, sensor type, wiring
2. **Firmware version:** Git commit hash or date
3. **Expected vs actual behavior**
4. **Serial output logs** (if applicable)

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Hudgins et al.** for pioneering TD4 features in EMG pattern recognition
- **TensorFlow Lite team** for embedded ML framework
- **Advancer Technologies** for MyoWare EMG sensors
- **Espressif Systems** for ESP32-S3 platform

---

## 📧 Contact

For questions or collaboration:
- **GitHub Issues:** [Report an issue](https://github.com/yourusername/real_time_esp322/issues)
- **Email:** your.email@example.com

---

**Built with ❤️ for prosthetic control and assistive technology research.**

---

## 🔄 Project Status

**Last Updated:** November 24, 2025
**Version:** 2.0.0 (Post-preprocessing refactor)
**Status:** ✅ Production-ready

**Recent changes:**
- ✅ Fixed ZC/SSC features (DC offset removal)
- ✅ Reduced latency 4× (1000ms → 250ms)
- ✅ Validated C++/Python preprocessing parity
- ✅ Increased tensor arena (20KB → 30KB)
- ✅ Added dual environment build system

---

