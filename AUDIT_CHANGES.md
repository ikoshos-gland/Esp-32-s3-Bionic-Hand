# Audit fixes (branch `audit-fixes`, based on `functions_real_time_v2`)

This branch applies the seven corrections from the September 2026 code audit
(see `bionic-hand-mulakat/denetim.html`). Every item below names the file,
the change and how it was verified.

## 1. Servos never moved (critical)

`servoController.update()` was only called inside `#ifdef REAL_TIME_INFERENCE_MODE`
in `functions.cpp`, and the flag had been dropped from `platformio.ini` on
30 Nov 2025. Fixed by removing the ifdef dependency: `main.cpp` calls
`update()` at the top of `loop()` and defines `sampling_idle_hook()`, which
`prelim_collection()` calls while waiting for the next window. The flag is
defined again in `platformio.ini` for documentation only.

## 2. Deployed model trained on corrupted data (critical, needs hardware)

`include/model.h` is still the 26 Nov 00:33 model (md5 `d89284ef...`). Its
training CSV has uint16 wrap-around in 87 % of rows, so the model cannot work
on the device. There is no clean 11-class recording in the repository, so the
model could not be retrained here. What changed so this cannot happen again:

- `feature_extraction.py` refuses CSVs with values > 4095 (wrap-around).
- `train_test_model.py` aborts when any MAV or WL feature exceeds 0.5 (the
  physical maximum after global normalisation).
- `baseline_lda_cv.py` runs the same sanity check before scoring.
- `model_meta.h` records the training data and a note; the firmware prints
  them at boot. The current header is marked as a placeholder.

Recording clean data requires the oscilloscope check first (see 6).

## 3. Class order trap (critical)

`train_test_model.py` used alphabetical order unless `--strict` was given;
the firmware hard-codes Rest-first. Now:

- strict (fixed) order is the default; `--allow-auto-order` is research only.
- `export_model_header.py` writes `include/model_meta.h` with
  `GESTURE_NAMES[]`, `REST_CLASS_INDEX`, `MODEL_NUM_FEATURES`,
  `MODEL_NUM_CLASSES`, md5 and provenance. `model.h` includes it.
- `main.cpp` and `servo_controller.cpp` use `GESTURE_NAMES` from the header;
  their private copies were deleted.
- `functions.h` and `servo_controller.h` contain `static_assert`s on feature
  count and class count, and `setup()` compares the TFLite tensor shapes with
  the header at boot.

## 4. Training data hygiene

- Phase filter is a whitelist (`HAREKET`, `DINLENME*`); `HAZIRLANIYOR...`,
  `HAZIRLIK`, `PREPARATION` and unknown phases are dropped.
- The first 1000 ms of every recording is dropped (`--drop-first-ms`), and
  `--onset-skip-ms` can drop the first N ms after each phase change.
- Rep numbers are kept in the NPZ; `train_test_model.py --cv` runs
  leave-one-rep-out cross-validation; `baseline_lda_cv.py` gives the LDA and
  logistic-regression reference numbers without TensorFlow.

## 5. Decision state machine (`main.cpp`)

- A low-confidence frame resets the debounce counter.
- `REST_FALLBACK_FRAMES` (4 ≈ 1 s) low-confidence frames emit REST, so the
  hand opens instead of staying frozen and the same gesture can be
  re-issued afterwards.
- REST bypasses the motion lock and never starts one; only movements do.
- Comments and constants agree (`CONFIDENCE_THRESHOLD 0.8`).

## 6. Continuous sampling (`functions.cpp`)

An `esp_timer` (1000 µs period, task dispatch) reads the six ADC channels,
runs the IIR filters and writes into a 1024-sample ring buffer per sensor.
Filters never see a gap, inference and serial printing no longer steal
sampling time, and `prelim_collection()` only waits for `INFERENCE_HOP`
new samples. Overruns and the worst callback duration are reported every
10 s. `data_acquisition.cpp` keeps a drift-free `+=` schedule with resync.

## 7. Filter equivalence

- `dsp_filters.py` uses causal `sosfilt` (the old `sosfiltfilt` was
  zero-phase and squared the magnitude response, which the device cannot
  reproduce).
- `validate_filters.py` now parses `include/filter_coefficients_50hz.h`,
  runs a float32 transliteration of the C++ biquad cascade with those
  coefficients and compares it sample by sample with scipy, checks pole
  stability, frequency response, and TD4 equivalence between the Python
  extractor and a transliteration of `extract_features_from_raw()`.
- `filter_coefficients_50hz.h` gained an include guard and
  `FILTER_SAMPLING_RATE_HZ`; the 2000 Hz set is behind
  `FILTER_SAMPLING_2000HZ` instead of `DATA_ACQUISITION_MODE`, so defining
  the acquisition flag can no longer switch the filters to the wrong rate.

## Other

- `platformio.ini`: one `ESP32Servo` version; `functions.cpp` and TFLite
  excluded from the acquisition build.
- Docs: Int8 / 85-95 % / 2000 Hz claims corrected.

## How to verify

```
python scripts_ai/validation/validate_filters.py
python scripts_ai/critical/feature_extraction.py data/training_data_20251130_152555.csv --output scripts_ai/data/features
python scripts_ai/critical/baseline_lda_cv.py scripts_ai/data/features/training_data_20251130_152555_td4_features.npz
python -m platformio run -e real_time_inference -e data_acquisition
```

## Verification results (2026-09-16, this branch)

Firmware (PlatformIO, espressif32 / Arduino core, ESP32Servo 3.0.9):

| environment          | result  | RAM                | Flash                |
|----------------------|---------|--------------------|----------------------|
| data_acquisition     | SUCCESS | 20,068 B (6.1 %)   | 273,953 B (8.7 %)    |
| real_time_inference  | SUCCESS | 89,252 B (27.2 %)  | 614,033 B (19.5 %)   |

Python:

- `validate_filters.py`: 15/15 checks pass. Header coefficients equal the
  scipy design to 5e-11; float32 C++ biquad cascade vs scipy `sosfilt`
  differ by at most 0.003 ADC counts; all poles inside the unit circle;
  TD4 Python vs C++ transliteration differ by < 5e-8 on continuous and on
  integer (threshold-tie) data.
- `feature_extraction.py` on the 30 Nov recording: 3171 preparation rows
  dropped, first 1000 ms dropped, 1696 windows, Rest balanced to 385,
  final 1349 x 24 with rep numbers.
- `feature_extraction.py` on the 26 Nov recording: refused
  (`values > 4095` in all six channels, 65k to 620k rows per channel).
- `baseline_lda_cv.py` on the 26 Nov NPZ: refused by the sanity check
  (MAV up to 7.1, physically impossible).
- `baseline_lda_cv.py` on the 30 Nov NPZ (Rest, Fist, Open, Point):

  | classifier      | leave-one-rep-out | random 80/20 (leaky) |
  |-----------------|-------------------|----------------------|
  | LDA             | 0.359 +- 0.130    | 0.430                |
  | LogReg          | 0.359 +- 0.130    | 0.430                |
  | MLP 256-128-64  | 0.352 +- 0.089    | 0.504                |

  Chance is 0.25. The signal, not the classifier, is the bottleneck: median
  MAV per class is 4 to 6 ADC counts on every channel. `validate_noise_data.py`
  flags all six channels as LOW VARIANCE (30 to 83). Fix the electrode chain
  before collecting the 11-class dataset.

Not verified here (needs hardware): esp_timer sampler jitter, servo motion,
ADC read time inside the 1 ms period. The firmware prints sampler
statistics every 10 s for that purpose.
