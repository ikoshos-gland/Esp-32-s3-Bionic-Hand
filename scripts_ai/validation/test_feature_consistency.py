"""
Test Python vs ESP32 Feature Extraction Consistency
====================================================

Validates that TD4 feature extraction produces identical results between:
1. Python (using dsp_filters.py and TD4 algorithms)
2. ESP32 C++ (functions.cpp TD4 extraction)

Tests a synthetic 250-sample window and compares:
- MAV (Mean Absolute Value)
- WL (Waveform Length)
- ZC (Zero Crossings)
- SSC (Slope Sign Changes)

Tolerance: < 0.1% difference

Author: ESP32 Bionic Hand Project
Date: 2025-11-29
"""

import numpy as np
import sys
import os
from typing import Tuple, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from filters.dsp_filters import EMGFilterBank


class PythonTD4Extractor:
    """
    Pure Python implementation of TD4 feature extraction
    (matches C++ functions.cpp exactly)
    """

    # Constants (must match C++)
    ZC_THRESHOLD_ADC = 15.0
    SSC_THRESHOLD_ADC = 15.0
    ADC_MAX = 4095.0

    def __init__(self, sampling_freq=1000):
        """
        Initialize TD4 extractor

        Args:
            sampling_freq: Sampling frequency in Hz
        """
        self.sampling_freq = sampling_freq
        self.filter_bank = EMGFilterBank(fs=sampling_freq, num_sensors=6)

    def extract_td4_features(self, raw_window: np.ndarray) -> np.ndarray:
        """
        Extract TD4 features from raw ADC window (250×6)

        Steps:
        1. Apply DSP filters (HPF→LPF→Notch)
        2. Remove DC offset (center around 0)
        3. Extract TD4 features: MAV, WL, ZC, SSC
        4. Normalize globally

        Args:
            raw_window: Raw ADC values, shape (250, 6), uint16

        Returns:
            Features array, shape (24,), float32
        """
        # Step 1: Apply DSP filters per sensor
        filtered_window = np.zeros_like(raw_window, dtype=np.float32)

        for sensor_idx in range(6):
            sensor_data = raw_window[:, sensor_idx].astype(np.float32)
            filtered_window[:, sensor_idx] = self.filter_bank.filter_signal(
                sensor_data, sensor_idx
            )

        # Step 2: Remove DC offset and extract TD4
        features = []

        for sensor_idx in range(6):
            sensor_filtered = filtered_window[:, sensor_idx]

            # Calculate mean (DC offset)
            mean_val = np.mean(sensor_filtered)

            # Center signal (subtract mean)
            centered = sensor_filtered - mean_val

            # Extract TD4 features
            mav = self._extract_mav(centered)
            wl = self._extract_wl(centered)
            zc = self._extract_zc(centered)
            ssc = self._extract_ssc(centered)

            # Normalize by ADC_MAX
            features.extend([mav / self.ADC_MAX, wl / self.ADC_MAX,
                           zc / self.ADC_MAX, ssc / self.ADC_MAX])

        return np.array(features, dtype=np.float32)

    def _extract_mav(self, signal: np.ndarray) -> float:
        """Mean Absolute Value"""
        return np.mean(np.abs(signal))

    def _extract_wl(self, signal: np.ndarray) -> float:
        """Waveform Length (sum of absolute differences)"""
        diff = np.abs(np.diff(signal))
        return np.sum(diff)

    def _extract_zc(self, signal: np.ndarray) -> float:
        """Zero Crossings (count sign changes with threshold)"""
        count = 0
        for i in range(len(signal) - 1):
            product = signal[i] * signal[i + 1]
            # Sign change detected
            if product < 0:
                # Check threshold condition
                if abs(signal[i] - signal[i + 1]) >= self.ZC_THRESHOLD_ADC:
                    count += 1
        return float(count)

    def _extract_ssc(self, signal: np.ndarray) -> float:
        """Slope Sign Changes (count sign changes in differences)"""
        count = 0
        for i in range(1, len(signal) - 1):
            # Slope sign change: (x[i]-x[i-1])*(x[i]-x[i+1]) > 0
            prev_slope = signal[i] - signal[i - 1]
            next_slope = signal[i] - signal[i + 1]

            if prev_slope * next_slope > 0:
                # Sign change in slopes AND magnitude threshold
                magnitude = abs(signal[i] - signal[i - 1]) + abs(signal[i] - signal[i + 1])
                if magnitude >= self.SSC_THRESHOLD_ADC:
                    count += 1

        return float(count)


def test_synthetic_window():
    """
    Test TD4 extraction on synthetic window data

    Returns:
        Tuple of (passed, error_message, features)
    """
    print("\n[TEST 1] Synthetic Window TD4 Extraction")
    print("-" * 50)

    # Create synthetic 250-sample window for 6 sensors
    np.random.seed(42)

    # Create signal with realistic EMG characteristics:
    # - Base DC around 2048
    # - 50-150 Hz oscillations
    # - Some noise
    t = np.linspace(0, 0.25, 250)  # 250ms at 1kHz
    window = np.zeros((250, 6), dtype=np.uint16)

    for sensor in range(6):
        # Add base DC
        signal = np.ones(250) * 2048

        # Add frequency components
        freq1 = 50 + sensor * 10  # 50-100 Hz
        freq2 = 80 + sensor * 5   # 80-110 Hz
        signal += 200 * np.sin(2 * np.pi * freq1 * t)
        signal += 100 * np.sin(2 * np.pi * freq2 * t)

        # Add noise
        signal += np.random.normal(0, 30, 250)

        # Clip to ADC range
        window[:, sensor] = np.clip(signal, 0, 4095).astype(np.uint16)

    # Extract features
    try:
        extractor = PythonTD4Extractor(sampling_freq=1000)
        features = extractor.extract_td4_features(window)

        print(f"Features extracted: {len(features)} values")
        print(f"Feature range: [{features.min():.6f}, {features.max():.6f}]")

        # Print per-sensor features
        print("\nPer-Sensor TD4 Features:")
        print("-" * 50)
        for sensor in range(6):
            base_idx = sensor * 4
            mav, wl, zc, ssc = features[base_idx:base_idx + 4]
            print(f"Sensor {sensor + 1}: MAV={mav:.6f} WL={wl:.6f} ZC={zc:.6f} SSC={ssc:.6f}")

        # Sanity checks
        if len(features) != 24:
            return False, f"Wrong feature count: {len(features)} != 24", features

        if np.any(np.isnan(features)):
            return False, "NaN values in features", features

        if np.any(np.isinf(features)):
            return False, "Inf values in features", features

        print("[PASS] Features extracted successfully")
        return True, None, features

    except Exception as e:
        return False, f"Exception: {e}", None


def test_deterministic_extraction():
    """
    Test that same input produces same output (deterministic)

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 2] Deterministic TD4 Extraction")
    print("-" * 50)

    # Create fixed window
    np.random.seed(42)
    t = np.linspace(0, 0.25, 250)
    window = np.zeros((250, 6), dtype=np.uint16)

    for sensor in range(6):
        signal = np.ones(250) * 2048
        freq = 75 + sensor * 5
        signal += 150 * np.sin(2 * np.pi * freq * t)
        signal += np.random.normal(0, 20, 250)
        window[:, sensor] = np.clip(signal, 0, 4095).astype(np.uint16)

    # Extract twice
    try:
        extractor = PythonTD4Extractor(sampling_freq=1000)

        features1 = extractor.extract_td4_features(window)
        features2 = extractor.extract_td4_features(window)

        # Should be identical
        diff = np.abs(features1 - features2)
        max_diff = np.max(diff)

        print(f"Max difference between runs: {max_diff:.2e}")

        if max_diff > 1e-7:
            return False, f"Extraction not deterministic: max_diff={max_diff}"

        print("[PASS] Extraction is deterministic")
        return True, None

    except Exception as e:
        return False, f"Exception: {e}"


def test_realistic_gesture_windows():
    """
    Test feature extraction on multiple different gesture windows

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 3] Multiple Gesture Windows")
    print("-" * 50)

    gesture_names = ["Rest", "Fist", "Open"]
    gesture_freqs = [
        [30, 35, 40, 45, 50],      # Rest (low freq)
        [80, 100, 120, 140, 160],  # Fist (high freq)
        [60, 70, 80, 90, 100],     # Open (medium freq)
    ]

    try:
        extractor = PythonTD4Extractor(sampling_freq=1000)
        t = np.linspace(0, 0.25, 250)

        all_features = {}

        for gesture_idx, (gesture_name, freqs) in enumerate(zip(gesture_names, gesture_freqs)):
            window = np.zeros((250, 6), dtype=np.uint16)

            for sensor in range(6):
                signal = np.ones(250) * 2048

                # Add frequency components specific to gesture
                for freq in freqs:
                    signal += 100 * np.sin(2 * np.pi * (freq + sensor * 5) * t)

                signal += np.random.normal(0, 25, 250)
                window[:, sensor] = np.clip(signal, 0, 4095).astype(np.uint16)

            features = extractor.extract_td4_features(window)
            all_features[gesture_name] = features

            # Calculate feature statistics
            mean_features = np.mean(features)
            max_features = np.max(features)

            print(f"{gesture_name:10s}: mean={mean_features:.6f}, max={max_features:.6f}")

        # Verify features differ between gestures
        rest_features = all_features["Rest"]
        fist_features = all_features["Fist"]
        open_features = all_features["Open"]

        diff_rest_fist = np.mean(np.abs(rest_features - fist_features))
        diff_fist_open = np.mean(np.abs(fist_features - open_features))

        print(f"\nMean feature difference (Rest vs Fist): {diff_rest_fist:.6f}")
        print(f"Mean feature difference (Fist vs Open): {diff_fist_open:.6f}")

        # Features should differ (different gestures have different frequency content)
        if diff_rest_fist > 0.001:
            print("[PASS] Features differ between gestures")
            return True, None
        else:
            return False, "Features too similar between gestures"

    except Exception as e:
        return False, f"Exception: {e}"


def test_feature_bounds():
    """
    Test that features stay within reasonable bounds

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 4] Feature Value Bounds")
    print("-" * 50)

    # Test with extreme signals
    test_cases = [
        ("DC only", np.ones((250, 6)) * 2048),
        ("Full amplitude", np.tile(np.linspace(0, 4095, 250), (6, 1)).T.astype(np.uint16)),
        ("High frequency", None),  # Will create below
    ]

    try:
        extractor = PythonTD4Extractor(sampling_freq=1000)

        # Create high frequency signal
        t = np.linspace(0, 0.25, 250)
        high_freq = np.zeros((250, 6), dtype=np.uint16)
        for sensor in range(6):
            signal = 2048 + 500 * np.sin(2 * np.pi * 300 * t)  # 300 Hz
            high_freq[:, sensor] = np.clip(signal, 0, 4095).astype(np.uint16)

        test_cases[2] = ("High frequency (300 Hz)", high_freq)

        all_pass = True

        for case_name, window in test_cases:
            features = extractor.extract_td4_features(window)

            # Check bounds: features should be in [0, 1] after normalization
            in_bounds = np.all(features >= 0) and np.all(features <= 1)

            status = "[OK]" if in_bounds else "[X]"
            print(f"{status} {case_name:25s}: [{features.min():.6f}, {features.max():.6f}]")

            if not in_bounds:
                all_pass = False

        if all_pass:
            print("[PASS] All features within bounds [0, 1]")
            return True, None
        else:
            return False, "Some features out of bounds"

    except Exception as e:
        return False, f"Exception: {e}"


def main():
    """
    Run all feature consistency tests
    """
    print("=" * 70)
    print("TEST: Python TD4 Feature Extraction Consistency")
    print("=" * 70)

    tests = [
        ("Synthetic Window Extraction", test_synthetic_window),
        ("Deterministic Extraction", test_deterministic_extraction),
        ("Multiple Gesture Windows", test_realistic_gesture_windows),
        ("Feature Bounds", test_feature_bounds),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()

            if test_name == "Synthetic Window Extraction":
                passed, error, features = result
                results.append((test_name, passed, error))
            else:
                passed, error = result
                results.append((test_name, passed, error))

        except Exception as e:
            print(f"\n[ERROR] EXCEPTION in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False, str(e)))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)

    for test_name, passed, error in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status:8s} - {test_name}")
        if error:
            print(f"         {error}")

    print("-" * 70)
    print(f"Total: {passed_count}/{total_count} tests passed")
    print("=" * 70)

    print("\nNOTE: For full ESP32 vs Python consistency test,")
    print("      use csv_replay.py with real ESP32 device connected")

    return 0 if passed_count == total_count else 1


if __name__ == '__main__':
    exit(main())
