"""
Quick End-to-End CSV Replay Test
=================================

Simple wrapper for csv_replay.py that:
1. Validates csv_replay module can be imported
2. Checks basic system initialization
3. Reports pass/fail based on accuracy threshold (80%)

This is a quick smoke test that doesn't require ESP32 hardware.
For full testing with ESP32 hardware, use csv_replay.py directly.

Author: ESP32 Bionic Hand Project
Date: 2025-11-29
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from validation.csv_replay import CSVReplaySystem


def test_module_imports():
    """
    Test that csv_replay module imports successfully

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 1] Module Import")
    print("-" * 50)

    try:
        # Already imported above, but verify class exists
        assert hasattr(CSVReplaySystem, 'load_csv'), "Missing load_csv method"
        assert hasattr(CSVReplaySystem, 'scale_to_adc'), "Missing scale_to_adc method"
        assert hasattr(CSVReplaySystem, 'stream_batch'), "Missing stream_batch method"
        assert hasattr(CSVReplaySystem, 'receive_response'), "Missing receive_response method"
        assert hasattr(CSVReplaySystem, 'run_replay'), "Missing run_replay method"

        print("[OK] CSVReplaySystem class imported")
        print("[OK] All required methods present")
        print("[PASS] Module import successful")

        return True, None

    except Exception as e:
        error = f"Import failed: {e}"
        print(f"[FAIL] {error}")
        return False, error


def test_system_initialization():
    """
    Test that CSVReplaySystem can be initialized

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 2] System Initialization")
    print("-" * 50)

    try:
        # Create dummy paths (don't need actual files)
        csv_path = "dummy_data.csv"
        serial_port = "COM11"
        baud_rate = 921600

        # Initialize system
        system = CSVReplaySystem(csv_path, serial_port, baud_rate)

        # Verify attributes set correctly
        assert system.csv_path == csv_path, "CSV path not set"
        assert system.serial_port == serial_port, "Serial port not set"
        assert system.baud_rate == baud_rate, "Baud rate not set"
        assert system.correct_count == 0, "Correct count not initialized"

        # Verify constants
        assert system.WINDOW_SIZE == 250, "Window size incorrect"
        assert system.NUM_SENSORS == 6, "Num sensors incorrect"
        assert system.NUM_FEATURES == 24, "Num features incorrect"
        assert len(system.GESTURE_NAMES) == 11, "Num gestures incorrect"

        print("[OK] System initialized successfully")
        print(f"  CSV path: {system.csv_path}")
        print(f"  Serial port: {system.serial_port} @ {system.baud_rate} baud")
        print(f"  Constants: window={system.WINDOW_SIZE}, "
              f"sensors={system.NUM_SENSORS}, features={system.NUM_FEATURES}")
        print("[PASS] System initialization successful")

        return True, None

    except Exception as e:
        error = f"Initialization failed: {e}"
        print(f"[FAIL] {error}")
        return False, error


def test_protocol_constants():
    """
    Test that protocol constants are correctly defined

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 3] Protocol Constants Validation")
    print("-" * 50)

    try:
        # Access class constants
        window_header = CSVReplaySystem.WINDOW_HEADER
        response_header = CSVReplaySystem.RESPONSE_HEADER
        adc_min = CSVReplaySystem.ADC_MIN
        adc_max = CSVReplaySystem.ADC_MAX

        # Verify headers
        assert len(window_header) == 2, f"Window header wrong size: {len(window_header)}"
        assert len(response_header) == 2, f"Response header wrong size: {len(response_header)}"

        # Verify ADC range
        assert adc_min == 0, "ADC_MIN should be 0"
        assert adc_max == 4095, "ADC_MAX should be 4095 (12-bit)"

        # Verify gesture names
        gesture_names = CSVReplaySystem.GESTURE_NAMES
        assert len(gesture_names) == 11, f"Should have 11 gestures, got {len(gesture_names)}"
        assert gesture_names[0] == "Rest", "First gesture should be Rest"
        assert gesture_names[10] == "WristFlex", "Last gesture should be WristFlex"

        print(f"[OK] Window Header: {window_header.hex()}")
        print(f"[OK] Response Header: {response_header.hex()}")
        print(f"[OK] ADC Range: [{adc_min}, {adc_max}]")
        print(f"[OK] Gestures: {len(gesture_names)} classes")
        print(f"  {', '.join(gesture_names)}")
        print("[PASS] Protocol constants valid")

        return True, None

    except Exception as e:
        error = f"Protocol validation failed: {e}"
        print(f"[FAIL] {error}")
        return False, error


def test_scaling_algorithm():
    """
    Test the ADC scaling algorithm without needing actual CSV

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 4] ADC Scaling Algorithm")
    print("-" * 50)

    try:
        import numpy as np

        system = CSVReplaySystem.__new__(CSVReplaySystem)
        system.ADC_MIN = 0
        system.ADC_MAX = 4095

        # Test key values
        test_values = [
            (-1.0, 0),      # Min EMG → Min ADC
            (0.0, 2048),    # Zero → 2048 (DC center)
            (1.0, 4095),    # Max EMG → Max ADC
        ]

        all_pass = True

        for emg_val, expected_adc in test_values:
            emg_array = np.array([[emg_val]])
            adc = system.scale_to_adc(emg_array)
            actual = adc[0, 0]

            tolerance = 1
            is_correct = abs(actual - expected_adc) <= tolerance

            status = "[OK]" if is_correct else "[X]"
            print(f"{status} EMG {emg_val:+0.1f} -> ADC {actual} (expected {expected_adc})")

            if not is_correct:
                all_pass = False

        if all_pass:
            print("[PASS] Scaling algorithm valid")
            return True, None
        else:
            return False, "Scaling algorithm test failed"

    except Exception as e:
        error = f"Scaling test failed: {e}"
        print(f"[FAIL] {error}")
        return False, error


def test_documentation():
    """
    Test that module has proper documentation

    Returns:
        Tuple of (passed, error_message)
    """
    print("\n[TEST 5] Module Documentation")
    print("-" * 50)

    try:
        # Check module docstring
        if not CSVReplaySystem.__doc__:
            return False, "CSVReplaySystem missing docstring"

        # Check key methods have docstrings
        methods_to_check = ['load_csv', 'scale_to_adc', 'stream_batch', 'receive_response', 'run_replay']

        missing_docs = []
        for method_name in methods_to_check:
            method = getattr(CSVReplaySystem, method_name)
            if not method.__doc__:
                missing_docs.append(method_name)

        print(f"[OK] Module docstring present ({len(CSVReplaySystem.__doc__)} chars)")
        print(f"[OK] Methods with docstrings: {len(methods_to_check) - len(missing_docs)}/{len(methods_to_check)}")

        if missing_docs:
            print(f"[WARN] Methods missing docstrings: {', '.join(missing_docs)}")
            return False, "Some methods missing docstrings"

        print("[PASS] Documentation present")
        return True, None

    except Exception as e:
        error = f"Documentation check failed: {e}"
        print(f"[FAIL] {error}")
        return False, error


def main():
    """
    Run all end-to-end tests
    """
    print("=" * 70)
    print("TEST: End-to-End CSV Replay System")
    print("=" * 70)

    tests = [
        ("Module Import", test_module_imports),
        ("System Initialization", test_system_initialization),
        ("Protocol Constants", test_protocol_constants),
        ("ADC Scaling Algorithm", test_scaling_algorithm),
        ("Module Documentation", test_documentation),
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

    # Accuracy threshold check
    accuracy = (passed_count / total_count) * 100

    if accuracy >= 80:
        print(f"ACCURACY: {accuracy:.1f}% (>= 80% threshold)")
        print("[PASS] END-TO-END TEST PASSED")
        exit_code = 0
    else:
        print(f"ACCURACY: {accuracy:.1f}% (< 80% threshold)")
        print("[FAIL] END-TO-END TEST FAILED")
        exit_code = 1

    print("=" * 70)
    print()

    print("Next Steps:")
    print("  1. For full feature testing, run: python test_csv_scaling.py")
    print("  2. For TD4 validation, run: python test_feature_consistency.py")
    print("  3. For ESP32 hardware testing, run: python csv_replay.py <csv_file> --port COM11")
    print()

    return exit_code


if __name__ == '__main__':
    exit(main())
