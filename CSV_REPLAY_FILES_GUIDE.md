# CSV Replay System - Complete File Guide

This document lists **ALL files** needed for the CSV replay (real-like-time) system, organized by purpose.

---

## 📁 File Organization

```
real_time_esp322/
├── 🔧 ESP32 Firmware Files (Upload to MCU)
├── 🐍 Python Scripts (Run on PC)
├── ⚙️ Configuration Files
├── 📊 Data Files (CSV inputs)
└── 📄 Documentation
```

---

## 🔧 1. ESP32 FIRMWARE FILES (Upload to MCU)

These files are compiled into firmware and uploaded to your ESP32-S3.

### Main Firmware
| File | Lines | Purpose |
|------|-------|---------|
| **[src/csv_replay.cpp](src/csv_replay.cpp)** | 553 | **Main CSV replay firmware** - Receives data from Python, runs inference |

### Supporting Files (Auto-included in build)
| File | Lines | Purpose |
|------|-------|---------|
| **[src/functions.cpp](src/functions.cpp)** | ~340 | Feature extraction (TD4: MAV, WL, ZC, SSC) - Reused from main system |
| **[src/functions.h](src/functions.h)** | ~45 | Function declarations, pin definitions, constants |
| **[src/filters.cpp](src/filters.cpp)** | ~336 | DSP filter implementation (bypassed in CSV mode) |
| **[src/filters.h](src/filters.h)** | ~150 | **Modified** - Filter bypass mechanism added |
| **[src/model.h](src/model.h)** | Large | TFLite model as byte array (auto-generated) |
| **[src/servo_controller.cpp](src/servo_controller.cpp)** | ~200 | Servo control (not used in CSV replay but included) |
| **[src/servo_controller.h](src/servo_controller.h)** | ~50 | Servo declarations |

### How to Upload to MCU
```powershell
# Navigate to project
cd "C:\Users\MERT\Documents\PlatformIO\Projects\real_time_esp322"

# Add PlatformIO to PATH
$env:Path += ";C:\Users\MERT\.platformio\penv\Scripts"

# Build and upload
pio run -e csv_replay -t upload

# Monitor to verify
pio device monitor
```

**Expected Serial Output:**
```
=== ESP32 CSV REPLAY MODE ===
Firmware: CSV Replay v1.0
DSP Filters: BYPASSED (data pre-filtered)
✓ TFLite initialized successfully
=== READY FOR CSV REPLAY ===
Waiting for 'CSV_START' command from Python...
```

---

## 🐍 2. PYTHON SCRIPTS (Run on PC)

### Main Script - CSV Replay System
| File | Lines | Purpose |
|------|-------|---------|
| **[scripts_ai/validation/csv_replay.py](scripts_ai/validation/csv_replay.py)** | 635 | **Main replay script** - Streams CSV to ESP32, validates predictions |

**Usage:**
```powershell
cd scripts_ai\validation
python csv_replay.py ..\..\data\real-like-time\S1_50P_combined.csv --port COM11
```

**What it does:**
1. Reads CSV file (EMG1-6 + Movement columns)
2. Scales float values (-1 to +1) → ADC integers (0-4095)
3. Creates 250-sample windows (non-overlapping)
4. Sends each window to ESP32 via serial (binary protocol)
5. Receives prediction + features from ESP32
6. Validates against ground truth
7. Generates:
   - Console output (real-time progress)
   - Detailed log file
   - CSV summary (machine-readable)
   - Confusion matrix plot (PNG)

---

### Validation Test Scripts
| File | Lines | Purpose |
|------|-------|---------|
| **[scripts_ai/validation/test_csv_scaling.py](scripts_ai/validation/test_csv_scaling.py)** | 298 | Tests ADC scaling algorithm |
| **[scripts_ai/validation/test_feature_consistency.py](scripts_ai/validation/test_feature_consistency.py)** | 433 | Tests TD4 feature extraction |
| **[scripts_ai/validation/test_end_to_end.py](scripts_ai/validation/test_end_to_end.py)** | 310 | Tests complete system integration |

**Usage:**
```powershell
cd scripts_ai\validation

# Test ADC scaling
python test_csv_scaling.py

# Test feature extraction (no hardware needed)
python test_feature_consistency.py

# Test system integration
python test_end_to_end.py
```

**When to run:**
- **Before first replay**: Run `test_end_to_end.py` to verify setup
- **After code changes**: Run all 3 tests
- **Debugging low accuracy**: Run `test_feature_consistency.py`

---

### Supporting Python Files (Auto-imported)
| File | Purpose |
|------|---------|
| **[scripts_ai/filters/dsp_filters.py](scripts_ai/filters/dsp_filters.py)** | DSP filter implementation (for validation) |
| **[scripts_ai/critical/feature_extraction.py](scripts_ai/critical/feature_extraction.py)** | Reference TD4 implementation |

---

## ⚙️ 3. CONFIGURATION FILES

### Build Configuration
| File | Section | Purpose |
|------|---------|---------|
| **[platformio.ini](platformio.ini)** | Lines 121-152 | **csv_replay environment** - Build settings for CSV replay firmware |

**Key Configuration:**
```ini
[env:csv_replay]
build_flags =
    -DCSV_REPLAY_MODE=1      # Enable CSV replay mode
    -DBYPASS_DSP_FILTERS=1   # Skip DSP filters (data pre-filtered)

build_src_filter =
    +<*>
    -<data_acquisition.cpp>  # Exclude data acquisition
    -<main.cpp>              # Exclude real-time inference
    +<csv_replay.cpp>        # Include CSV replay firmware
```

**DO NOT MODIFY** unless you know what you're doing.

---

## 📊 4. DATA FILES

### Input Data (CSV Files)
| File | Size | Purpose |
|------|------|---------|
| **[data/real-like-time/S1_50P_combined.csv](data/real-like-time/S1_50P_combined.csv)** | ~84,673 rows | **Sample CSV data** for testing |

**CSV Structure:**
```csv
Timestamp,Seq,Movement,Phase,Rep,EMG1,EMG2,EMG3,EMG4,EMG5,EMG6,EMG7,EMG8
0.0,0,1,MOVEMENT,1,-0.040436,-0.011444,-0.066072,-0.043488,0.017243,0.0013733,0.019684,0.050507
0.001,1,1,MOVEMENT,1,-0.067903,0.013275,-0.074311,-0.10117,0.024262,-0.0050355,-0.0035096,0.045319
...
```

**Required Columns:**
- `EMG1` to `EMG6`: Sensor data (normalized float -1 to +1)
- `Movement`: Ground truth gesture (1-11)
  - 1 = Fist, 2 = Open, 3 = Point, ..., 11 = WristFlex
  - **Note**: Rest is typically not in Movement column, model uses 0-10 indices

**Your CSV files:**
- `data/real-like-time/S1_50P_combined.csv` - Main test data (84,672 samples)
- Any other CSV files in `data/real-like-time/` directory

---

### Output Files (Auto-generated)
| Location | Files Generated |
|----------|-----------------|
| **[data/csv_replay_logs/](data/csv_replay_logs/)** | All replay results stored here |

**Files created after each replay:**
```
data/csv_replay_logs/
├── replay_20251129_HHMMSS.log              # Detailed text log
├── replay_20251129_HHMMSS_summary.csv      # Machine-readable results
└── confusion_matrix_20251129_HHMMSS.png    # Visualization
```

---

## 📄 5. DOCUMENTATION FILES

| File | Purpose |
|------|---------|
| **[CSV_REPLAY_IMPLEMENTATION_SUMMARY.md](CSV_REPLAY_IMPLEMENTATION_SUMMARY.md)** | Complete implementation details |
| **[CSV_REPLAY_FILES_GUIDE.md](CSV_REPLAY_FILES_GUIDE.md)** | This file - Complete file listing |
| **[README.md](README.md)** | Main project documentation |
| **[CLAUDE.md](CLAUDE.md)** | Development guide and instructions |

---

## 🗂️ COMPLETE FILE TREE

```
real_time_esp322/
│
├── 🔧 ESP32 Firmware (Upload to MCU)
│   ├── src/
│   │   ├── csv_replay.cpp ⭐ MAIN FIRMWARE
│   │   ├── functions.cpp
│   │   ├── functions.h
│   │   ├── filters.cpp
│   │   ├── filters.h (modified)
│   │   ├── model.h
│   │   ├── servo_controller.cpp
│   │   └── servo_controller.h
│   └── platformio.ini ⭐ BUILD CONFIG (lines 121-152)
│
├── 🐍 Python Scripts (Run on PC)
│   └── scripts_ai/
│       ├── validation/
│       │   ├── csv_replay.py ⭐ MAIN SCRIPT
│       │   ├── test_csv_scaling.py
│       │   ├── test_feature_consistency.py
│       │   └── test_end_to_end.py
│       ├── filters/
│       │   └── dsp_filters.py (supporting)
│       └── critical/
│           └── feature_extraction.py (supporting)
│
├── 📊 Data Files
│   ├── real-like-time/
│   │   └── S1_50P_combined.csv ⭐ INPUT DATA
│   └── csv_replay_logs/ ⭐ OUTPUT LOGS
│       ├── replay_*.log
│       ├── replay_*_summary.csv
│       └── confusion_matrix_*.png
│
└── 📄 Documentation
    ├── CSV_REPLAY_IMPLEMENTATION_SUMMARY.md
    ├── CSV_REPLAY_FILES_GUIDE.md (this file)
    ├── README.md
    └── CLAUDE.md
```

---

## ⚡ QUICK START CHECKLIST

### ✅ Files YOU Need to Use:

**1. Upload to ESP32 (once):**
- ✅ Run: `pio run -e csv_replay -t upload`
- Uploads: `src/csv_replay.cpp` + all dependencies automatically

**2. Run on PC (every test):**
- ✅ Main script: `scripts_ai/validation/csv_replay.py`
- ✅ Input data: `data/real-like-time/S1_50P_combined.csv`

**3. Check results:**
- ✅ View: `data/csv_replay_logs/confusion_matrix_*.png`
- ✅ Read: `data/csv_replay_logs/replay_*.log`

### ❌ Files You DON'T Need to Touch:

- `src/functions.cpp` - Auto-included, already correct
- `src/filters.cpp` - Auto-included, bypass already configured
- `src/model.h` - Auto-included, uses existing model
- `platformio.ini` - Already configured correctly
- `test_*.py` - Only for validation, not required for normal use

---

## 🎯 MINIMAL WORKFLOW

**For regular CSV replay testing:**

```powershell
# 1. Upload firmware (ONCE, or when code changes)
cd "C:\Users\MERT\Documents\PlatformIO\Projects\real_time_esp322"
$env:Path += ";C:\Users\MERT\.platformio\penv\Scripts"
pio run -e csv_replay -t upload

# 2. Run replay (EVERY TIME you want to test)
cd scripts_ai\validation
python csv_replay.py ..\..\data\real-like-time\S1_50P_combined.csv --port COM11

# 3. View results
cd ..\..\data\csv_replay_logs
explorer .
# Open the newest confusion_matrix_*.png
```

---

## 🔍 FILE DEPENDENCIES

```
csv_replay.py (Python)
    ↓ imports
    ├─ pandas (external)
    ├─ numpy (external)
    ├─ serial (external)
    └─ matplotlib (external)

csv_replay.cpp (ESP32)
    ↓ includes
    ├─ Arduino.h
    ├─ model.h (TFLite model)
    ├─ functions.h → functions.cpp (feature extraction)
    └─ filters.h → filters.cpp (filter bypass)
```

**All dependencies are automatically handled by:**
- **PlatformIO** (for ESP32 firmware)
- **Python imports** (for scripts)

---

## 📋 FILE STATUS

| Category | Files | Status | Modified |
|----------|-------|--------|----------|
| ESP32 Firmware | 1 new, 1 modified | ✅ Complete | `csv_replay.cpp` (new), `filters.h` (modified) |
| Python Scripts | 4 new | ✅ Complete | All in `scripts_ai/validation/` |
| Configuration | 1 modified | ✅ Complete | `platformio.ini` |
| Data Files | 1 input | ✅ Ready | `S1_50P_combined.csv` |
| Documentation | 3 new | ✅ Complete | Implementation docs |

**Total Files Created**: 9 files (2,264 lines of code)
**Total Files Modified**: 2 files

---

## 🎓 UNDERSTANDING THE FLOW

```
┌─────────────────────┐
│  1. CSV FILE        │  S1_50P_combined.csv (84,672 samples)
│  (on PC)            │  EMG1-6 + Movement column
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. PYTHON SCRIPT   │  csv_replay.py
│  (on PC)            │  • Reads CSV
│                     │  • Scales to ADC (0-4095)
│                     │  • Creates 250-sample windows
│                     │  • Sends via serial (binary)
└──────────┬──────────┘
           │ USB Serial @ 921600 baud
           ▼
┌─────────────────────┐
│  3. ESP32 FIRMWARE  │  csv_replay.cpp (uploaded to MCU)
│  (on ESP32)         │  • Receives window packet (3007 bytes)
│                     │  • Extracts TD4 features (bypasses filters)
│                     │  • Runs TFLite inference
│                     │  • Sends prediction (106 bytes)
└──────────┬──────────┘
           │ USB Serial @ 921600 baud
           ▼
┌─────────────────────┐
│  4. PYTHON SCRIPT   │  csv_replay.py
│  (on PC)            │  • Receives prediction
│                     │  • Validates vs ground truth
│                     │  • Logs results
│                     │  • Generates plots
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. OUTPUT FILES    │  data/csv_replay_logs/
│  (on PC)            │  • Log file (detailed)
│                     │  • CSV summary
│                     │  • Confusion matrix (PNG)
└─────────────────────┘
```

---

## 🚨 CRITICAL FILES (DO NOT DELETE)

**ESP32:**
- ✅ `src/csv_replay.cpp` - Main firmware
- ✅ `src/functions.cpp` - Feature extraction
- ✅ `src/model.h` - TFLite model
- ✅ `platformio.ini` - Build config

**Python:**
- ✅ `scripts_ai/validation/csv_replay.py` - Main script

**Data:**
- ✅ `data/real-like-time/*.csv` - Input data

---

## 📞 NEED HELP?

**If you get errors, check:**
1. ✅ Is ESP32 connected? Run `mode` to check COM ports
2. ✅ Is csv_replay firmware uploaded? Run `pio device monitor` to verify
3. ✅ Are Python packages installed? Run `pip list | grep -E "pandas|numpy|serial"`
4. ✅ Is CSV file correct? Check columns with `head S1_50P_combined.csv`

**Common issues:**
- "Port not found" → Check COM port number, close other serial monitors
- "Timeout" → ESP32 not running csv_replay firmware, re-upload
- "Low accuracy" → Check CSV data quality, verify labels match training

---

**Last Updated**: 2025-11-29
**Total Files**: 11 core files (9 new, 2 modified)
**Status**: ✅ All files ready for use
