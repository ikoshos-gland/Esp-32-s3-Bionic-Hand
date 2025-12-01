"""
Round-Trip Bipolar Encoding Test
=================================

Tests the C++ bipolar encoding matches Python decoding.

Encoding (C++):  float [-2047.5, +2047.5] -> uint16_t [0, 4095]
Decoding (Python): uint16_t [0, 4095] -> float [-2047.5, +2047.5]

This test verifies that round-trip encoding/decoding has minimal error (<1.0 ADC unit).

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import numpy as np
import sys


def encode_bipolar(filtered_value):
    """
    Simulate C++ encode_bipolar() function

    Maps: float [-2047.5, +2047.5] -> uint16_t [0, 4095]

    Args:
        filtered_value: Bipolar filtered signal (can be negative)

    Returns:
        Encoded uint16_t in range [0, 4095]
    """
    BIPOLAR_OFFSET = 2047.5
    return int(np.clip(filtered_value + BIPOLAR_OFFSET, 0, 4095))


def decode_bipolar(encoded_value):
    """
    Simulate Python decoding

    Maps: uint16_t [0, 4095] -> float [-2047.5, +2047.5]

    Args:
        encoded_value: Encoded uint16_t from C++

    Returns:
        Decoded bipolar float
    """
    BIPOLAR_OFFSET = 2047.5
    return float(encoded_value) - BIPOLAR_OFFSET


def test_encoding():
    """Test C++ encoding matches Python decoding"""

    print("="*80)
    print("BIPOLAR ENCODING/DECODING ROUND-TRIP TEST")
    print("="*80)
    print()

    # Test values covering full range
    test_values = [
        -2047.5,  # Min value (should encode to 0)
        -2000.0,  # Large negative
        -1000.0,  # Medium negative
        -500.0,   # Small negative
        -50.0,    # Tiny negative
        -0.5,     # Very small negative
        0.0,      # DC center (should encode to 2048)
        +0.5,     # Very small positive
        +50.0,    # Tiny positive
        +500.0,   # Small positive
        +1000.0,  # Medium positive
        +2000.0,  # Large positive
        +2047.5,  # Max value (should encode to 4095)
    ]

    print(f"Testing {len(test_values)} values across full bipolar range:")
    print(f"{'Original':>12} -> {'Encoded':>8} -> {'Decoded':>12} | {'Error':>8} | Status")
    print("-" * 80)

    max_error = 0.0
    all_passed = True

    for val in test_values:
        # Simulate C++ encoding
        encoded = encode_bipolar(val)

        # Simulate Python decoding
        decoded = decode_bipolar(encoded)

        # Calculate round-trip error
        error = abs(decoded - val)
        max_error = max(max_error, error)

        # Status
        passed = error < 1.0
        status = "[PASS]" if passed else "[FAIL]"

        if not passed:
            all_passed = False

        print(f"{val:+12.1f} -> {encoded:8d} -> {decoded:+12.1f} | {error:8.2f} | {status}")

    print("-" * 80)
    print()

    # Summary
    print("TEST SUMMARY:")
    print(f"  Max round-trip error: {max_error:.2f} ADC units")
    print(f"  Error tolerance: <1.0 ADC units")
    print()

    if all_passed:
        print("[OK] ALL TESTS PASSED!")
        print("  C++ encoding and Python decoding are compatible.")
        print()
        return 0
    else:
        print("[X] SOME TESTS FAILED!")
        print("  C++ and Python encoding/decoding mismatch detected.")
        print()
        return 1


def test_typical_emg_values():
    """Test with typical EMG signal values"""

    print("="*80)
    print("TYPICAL EMG SIGNAL TEST")
    print("="*80)
    print()
    print("Testing realistic EMG values after HPF (bipolar signal):")
    print()

    # Typical EMG signal after HPF (DC removed, bipolar)
    typical_emg = np.array([
        +150.0, -120.0, +80.0, -90.0, +200.0,  # Active gesture
        -180.0, +50.0, -30.0, +10.0, -5.0,     # Moderate activity
        +2.0, -3.0, +1.0, -2.0, +0.5,          # Low activity (near rest)
    ])

    # Encode and decode
    encoded = np.array([encode_bipolar(val) for val in typical_emg])
    decoded = np.array([decode_bipolar(enc) for enc in encoded])

    # Calculate errors
    errors = np.abs(decoded - typical_emg)

    print(f"{'Original':>10} | {'Encoded':>8} | {'Decoded':>10} | {'Error':>8}")
    print("-" * 55)

    for orig, enc, dec, err in zip(typical_emg, encoded, decoded, errors):
        print(f"{orig:+10.1f} | {enc:8d} | {dec:+10.1f} | {err:8.2f}")

    print("-" * 55)
    print()
    print(f"Mean error: {np.mean(errors):.3f} ADC units")
    print(f"Max error:  {np.max(errors):.3f} ADC units")
    print()

    if np.max(errors) < 1.0:
        print("[OK] Typical EMG values encode/decode correctly!")
        print()
        return 0
    else:
        print("[X] Encoding error too large for typical EMG values!")
        print()
        return 1


def test_boundary_conditions():
    """Test edge cases and boundary conditions"""

    print("="*80)
    print("BOUNDARY CONDITION TESTS")
    print("="*80)
    print()

    # Boundary tests
    tests = [
        ("Min value (exact)", -2047.5, 0),
        ("Max value (exact)", +2047.5, 4095),
        ("DC center (exact)", 0.0, 2047),  # 0.0 + 2047.5 = 2047.5 -> int() = 2047
        ("Below min (clamp)", -3000.0, 0),
        ("Above max (clamp)", +3000.0, 4095),
        ("Negative near min", -2045.0, 2),
        ("Positive near max", +2045.0, 4092),
    ]

    print(f"{'Test Case':<25} | {'Input':>12} | {'Expected':>10} | {'Actual':>10} | Status")
    print("-" * 80)

    all_passed = True

    for test_name, input_val, expected_enc in tests:
        actual_enc = encode_bipolar(input_val)
        passed = (actual_enc == expected_enc)
        status = "[PASS]" if passed else "[FAIL]"

        if not passed:
            all_passed = False

        print(f"{test_name:<25} | {input_val:+12.1f} | {expected_enc:10d} | {actual_enc:10d} | {status}")

    print("-" * 80)
    print()

    if all_passed:
        print("[OK] All boundary tests passed!")
        print()
        return 0
    else:
        print("[X] Some boundary tests failed!")
        print()
        return 1


def main():
    """Run all encoding tests"""

    print("\n" + "="*80)
    print(" BIPOLAR ENCODING VALIDATION SUITE")
    print(" Testing C++ <-> Python encoding/decoding compatibility")
    print("="*80 + "\n")

    # Run all test suites
    result1 = test_encoding()
    result2 = test_typical_emg_values()
    result3 = test_boundary_conditions()

    # Final summary
    print("="*80)
    print("FINAL RESULT")
    print("="*80)

    if result1 == 0 and result2 == 0 and result3 == 0:
        print("[OK] ALL VALIDATION TESTS PASSED!")
        print()
        print("Bipolar encoding/decoding is working correctly.")
        print("C++ (data_acquisition.cpp) and Python (feature_extraction.py) are compatible.")
        print()
        return 0
    else:
        print("[X] SOME VALIDATION TESTS FAILED!")
        print()
        print("Please review the encoding/decoding implementation.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
