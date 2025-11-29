"""
Test ADC Scaling Algorithm - CSV Replay System
==============================================

Validates the ADC scaling algorithm used in csv_replay.py to ensure:
1. Normalized EMG values (-1.0 to +1.0) scale correctly to ADC range (0-4095)
2. Zero value maps to 2048 (DC offset center)
3. Range is symmetric around 2048
4. Edge cases are handled properly

Formula: ADC = (EMG + 1.0) × 2047.5

Author: ESP32 Bionic Hand Project
Date: 2025-11-29
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from validation.csv_replay import CSVReplaySystem


def test_zero_scaling():
    """
    Test that zero normalized EMG value maps to 2048

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 1] Zero Scaling (0 -> 2048)")
    print("-" * 50)

    # Create dummy system (don't need CSV or serial)
    system = CSVReplaySystem.__new__(CSVReplaySystem)
    system.ADC_MIN = 0
    system.ADC_MAX = 4095

    # Test zero
    emg_zero = np.array([[0.0]])
    adc = system.scale_to_adc(emg_zero)

    expected = 2048
    actual = adc[0, 0]

    print(f"Input:    0.0 (normalized)")
    print(f"Expected: {expected} (ADC units)")
    print(f"Actual:   {actual} (ADC units)")

    if actual == expected:
        print("[PASS] Zero scales correctly to 2048")
        return True, None
    else:
        error = f"Zero should map to 2048, got {actual}"
        print(f"[FAIL] {error}")
        return False, error


def test_range_scaling():
    """
    Test that full EMG range (-1.0 to +1.0) maps to ADC range (0-4095)

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 2] Range Scaling (-1.0 to +1.0 -> 0 to 4095)")
    print("-" * 50)

    system = CSVReplaySystem.__new__(CSVReplaySystem)
    system.ADC_MIN = 0
    system.ADC_MAX = 4095

    # Test edge cases
    test_cases = [
        (-1.0, 0, "Min: -1.0 -> 0"),
        (-0.5, 1024, "Quarter: -0.5 -> ~1024"),
        (0.0, 2048, "Center: 0.0 -> 2048"),
        (0.5, 3072, "Quarter: 0.5 -> ~3072"),
        (1.0, 4095, "Max: 1.0 -> 4095"),
    ]

    all_pass = True
    for emg_val, expected_adc, description in test_cases:
        emg_array = np.array([[emg_val]])
        adc = system.scale_to_adc(emg_array)
        actual = adc[0, 0]

        # Allow ±1 tolerance for rounding
        tolerance = 1
        is_correct = abs(actual - expected_adc) <= tolerance

        status = "[OK]" if is_correct else "[X]"
        print(f"{status} {description:25s} (got {actual}, expected ~{expected_adc})")

        if not is_correct:
            all_pass = False

    if all_pass:
        print("[PASS] Full range scales correctly")
        return True, None
    else:
        error = "Some range values out of tolerance"
        print(f"[FAIL] {error}")
        return False, error


def test_symmetry():
    """
    Test that scaling is symmetric around 2048

    For example: -0.5 → 1024 and +0.5 → 3072 (distance from 2048 is equal)

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 3] Symmetry Around 2048")
    print("-" * 50)

    system = CSVReplaySystem.__new__(CSVReplaySystem)
    system.ADC_MIN = 0
    system.ADC_MAX = 4095

    # Test symmetric points
    test_offsets = [-0.8, -0.5, -0.2, 0.2, 0.5, 0.8]
    all_pass = True

    for offset in test_offsets:
        neg_emg = np.array([[-offset]])
        pos_emg = np.array([[offset]])

        neg_adc = system.scale_to_adc(neg_emg)[0, 0]
        pos_adc = system.scale_to_adc(pos_emg)[0, 0]

        # Check symmetry: distance from 2048 should be equal
        dist_neg = abs(neg_adc - 2048)
        dist_pos = abs(pos_adc - 2048)

        is_symmetric = abs(dist_neg - dist_pos) <= 1  # Allow 1 unit tolerance
        status = "[OK]" if is_symmetric else "[X]"

        print(f"{status} Offset ~{offset:0.1f}: -{offset}->{neg_adc} (+{int(dist_neg)}), "
              f"+{offset}->{pos_adc} (+{int(dist_pos)})")

        if not is_symmetric:
            all_pass = False

    if all_pass:
        print("[PASS] Scaling is symmetric")
        return True, None
    else:
        error = "Scaling not symmetric around 2048"
        print(f"[FAIL] {error}")
        return False, error


def test_clipping():
    """
    Test that values outside [-1, +1] are clipped to [0, 4095]

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 4] Clipping of Out-of-Range Values")
    print("-" * 50)

    system = CSVReplaySystem.__new__(CSVReplaySystem)
    system.ADC_MIN = 0
    system.ADC_MAX = 4095

    # Test out-of-range values
    test_cases = [
        (-2.0, 0, "Negative overflow"),
        (-1.5, 0, "Negative overflow"),
        (1.5, 4095, "Positive overflow"),
        (2.0, 4095, "Positive overflow"),
    ]

    all_pass = True
    for emg_val, expected_adc, description in test_cases:
        emg_array = np.array([[emg_val]])
        adc = system.scale_to_adc(emg_array)
        actual = adc[0, 0]

        is_correct = actual == expected_adc
        status = "[OK]" if is_correct else "[X]"

        print(f"{status} {description:20s}: {emg_val:0.1f} -> {actual} (expected {expected_adc})")

        if not is_correct:
            all_pass = False

    if all_pass:
        print("[PASS] Clipping works correctly")
        return True, None
    else:
        error = "Clipping failed for out-of-range values"
        print(f"[FAIL] {error}")
        return False, error


def test_batch_scaling():
    """
    Test scaling of a full 250-sample window (250×6 array)

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 5] Batch Scaling (250x6 Window)")
    print("-" * 50)

    system = CSVReplaySystem.__new__(CSVReplaySystem)
    system.ADC_MIN = 0
    system.ADC_MAX = 4095

    # Create synthetic window: 250 samples × 6 sensors
    # Mix of random values
    np.random.seed(42)
    emg_window = np.random.uniform(-1.0, 1.0, (250, 6))

    # Scale
    adc_window = system.scale_to_adc(emg_window)

    # Verify shape
    if adc_window.shape != (250, 6):
        print(f"[FAIL] Shape mismatch. Expected (250, 6), got {adc_window.shape}")
        return False, "Shape mismatch"

    # Verify all values in range
    min_adc = adc_window.min()
    max_adc = adc_window.max()

    print(f"Window shape: {adc_window.shape}")
    print(f"ADC min: {min_adc}, max: {max_adc}")
    print(f"Values in range [0, 4095]: {0 <= min_adc and max_adc <= 4095}")

    if min_adc >= 0 and max_adc <= 4095:
        print("[PASS] Batch scaling valid")
        return True, None
    else:
        error = f"Values out of range: [{min_adc}, {max_adc}]"
        print(f"[FAIL] {error}")
        return False, error


def main():
    """
    Run all ADC scaling tests
    """
    print("=" * 70)
    print("TEST: ADC Scaling Algorithm (csv_replay.py)")
    print("=" * 70)

    tests = [
        ("Zero Scaling", test_zero_scaling),
        ("Range Scaling", test_range_scaling),
        ("Symmetry", test_symmetry),
        ("Clipping", test_clipping),
        ("Batch Scaling", test_batch_scaling),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed, error = test_func()
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

    return 0 if passed_count == total_count else 1


if __name__ == '__main__':
    exit(main())
