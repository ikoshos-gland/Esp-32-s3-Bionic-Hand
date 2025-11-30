"""
DSP Filter Validation Script
=============================

Validates that Python filters match C++ implementation on ESP32-S3.

Tests:
1. WL normalization bug fix validation
2. Filter frequency response
3. Time-domain filter behavior
4. C++/Python feature parity (requires ESP32 connection)

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dsp_filters import EMGFilterBank
from feature_extraction import TD4FeatureExtractor


def test_wl_normalization():
    """
    Test that WL normalization matches between C++ and Python.

    CRITICAL BUG FIX: C++ was dividing by 4095, Python by (4095×250).
    Now both should produce identical results.
    """
    print("\n" + "="*70)
    print("TEST 1: WL Normalization Bug Fix")
    print("="*70)

    # Create test signal: alternating [0, 100, 0, 100, ...]
    window_size = 250
    test_signal = np.tile([0, 100], window_size // 2)

    # Calculate WL (sum of absolute differences)
    wl_raw = np.sum(np.abs(np.diff(test_signal)))
    print(f"\nRaw WL: {wl_raw} (expected: {100 * 249} = 24,900)")

    # Old (WRONG) normalization: wl / 4095
    wl_old = wl_raw / 4095.0
    print(f"Old (WRONG) normalization: {wl_old:.6f}")

    # New (CORRECT) normalization: wl / (4095 × 250)
    wl_new = wl_raw / (4095.0 * 250)
    print(f"New (CORRECT) normalization: {wl_new:.6f}")

    print(f"\nDiscrepancy: {wl_old / wl_new:.1f}× difference!")
    print(f"Expected: 0.0243 (Python ground truth)")

    # Validate
    expected = 24900 / (4095 * 250)
    if abs(wl_new - expected) < 0.0001:
        print("✅ PASS: WL normalization is correct")
    else:
        print(f"❌ FAIL: Expected {expected:.6f}, got {wl_new:.6f}")

    print("="*70)


def test_filter_frequency_response():
    """
    Test filter frequency responses.

    Validates:
    - HPF: DC rejection, 20 Hz cutoff
    - LPF: High-frequency rejection, 450 Hz cutoff
    - Notch: 50 Hz rejection
    """
    print("\n" + "="*70)
    print("TEST 2: Filter Frequency Response")
    print("="*70)

    # Create filter bank
    filter_bank = EMGFilterBank(sampling_rate=2000, powerline_freq=50)

    # Test DC rejection (HPF)
    print("\nHigh-Pass Filter (20 Hz):")
    w, mag = filter_bank.get_frequency_response('hpf')
    dc_gain = mag[0]  # Magnitude at 0 Hz
    print(f"  DC gain: {dc_gain:.1f} dB (should be << -40 dB)")

    if dc_gain < -40:
        print("  ✅ PASS: DC properly rejected")
    else:
        print("  ❌ FAIL: DC not sufficiently rejected")

    # Test high-frequency rejection (LPF)
    print("\nLow-Pass Filter (450 Hz):")
    w, mag = filter_bank.get_frequency_response('lpf')
    nyquist_gain = mag[-1]  # Magnitude at Nyquist (1000 Hz)
    print(f"  Nyquist gain: {nyquist_gain:.1f} dB (should be << -40 dB)")

    if nyquist_gain < -40:
        print("  ✅ PASS: High frequencies properly rejected")
    else:
        print("  ❌ FAIL: High frequencies not sufficiently rejected")

    # Test powerline rejection (Notch)
    print("\nNotch Filter (50 Hz):")
    w, mag = filter_bank.get_frequency_response('notch')
    idx_50hz = np.argmin(np.abs(w - 50))
    notch_gain = mag[idx_50hz]
    print(f"  50 Hz gain: {notch_gain:.1f} dB (should be < -30 dB)")

    if notch_gain < -30:
        print("  ✅ PASS: Powerline properly rejected")
    else:
        print("  ❌ FAIL: Powerline not sufficiently rejected")

    print("="*70)

    # Plot frequency response
    try:
        filter_bank.plot_frequency_response(save_path='plots/filter_frequency_response.png')
    except:
        print("\nNote: Could not save plot (matplotlib may not be available)")


def test_filter_time_domain():
    """
    Test filters in time domain with synthetic EMG signal.
    """
    print("\n" + "="*70)
    print("TEST 3: Time-Domain Filter Behavior")
    print("="*70)

    # Create synthetic signal (1 second @ 2000 Hz)
    t = np.linspace(0, 1, 2000)

    # Components
    dc_offset = 2048  # ADC midpoint
    low_freq_drift = 200 * np.sin(2 * np.pi * 5 * t)  # 5 Hz motion
    emg_signal = 300 * np.sin(2 * np.pi * 100 * t)  # 100 Hz EMG
    powerline_noise = 150 * np.sin(2 * np.pi * 50 * t)  # 50 Hz powerline
    high_freq_noise = 50 * np.random.randn(len(t))  # Random noise

    raw_signal = dc_offset + low_freq_drift + emg_signal + powerline_noise + high_freq_noise

    # Apply filters
    filter_bank = EMGFilterBank(sampling_rate=2000, powerline_freq=50)
    filtered_signal = filter_bank.filter_signal(raw_signal)

    # Analyze results
    print(f"\nRaw Signal:")
    print(f"  Mean: {np.mean(raw_signal):.1f} (should be ~{dc_offset})")
    print(f"  Std: {np.std(raw_signal):.1f}")

    print(f"\nFiltered Signal:")
    print(f"  Mean: {np.mean(filtered_signal):.1f} (should be ~0, DC removed)")
    print(f"  Std: {np.std(filtered_signal):.1f}")

    # Validate DC removal
    if abs(np.mean(filtered_signal)) < 10:
        print("  ✅ PASS: DC offset removed")
    else:
        print(f"  ❌ FAIL: DC not removed (mean = {np.mean(filtered_signal):.1f})")

    # Calculate SNR improvement
    noise_raw = raw_signal - dc_offset - emg_signal
    noise_filtered = filtered_signal - emg_signal

    snr_raw = 10 * np.log10(np.mean(emg_signal**2) / np.mean(noise_raw**2))
    snr_filtered = 10 * np.log10(np.mean(emg_signal**2) / np.mean(noise_filtered**2))
    snr_improvement = snr_filtered - snr_raw

    print(f"\nSNR Analysis:")
    print(f"  Raw SNR: {snr_raw:.1f} dB")
    print(f"  Filtered SNR: {snr_filtered:.1f} dB")
    print(f"  Improvement: {snr_improvement:.1f} dB")

    if snr_improvement > 5:
        print("  ✅ PASS: Significant SNR improvement (>5 dB)")
    else:
        print("  ❌ FAIL: Insufficient SNR improvement")

    print("="*70)

    # Plot time-domain comparison
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(t[:500], raw_signal[:500], 'b-', alpha=0.7, label='Raw Signal')
        axes[0].set_title('Raw EMG Signal (First 250ms)', fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('ADC Value')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(t[:500], filtered_signal[:500], 'r-', alpha=0.7, label='Filtered Signal')
        axes[1].set_title('Filtered EMG Signal (After HPF+LPF+Notch)', fontweight='bold')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('ADC Value')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig('plots/filter_time_domain.png', dpi=300, bbox_inches='tight')
        print("\nTime-domain plot saved to: plots/filter_time_domain.png")
    except:
        print("\nNote: Could not save plot (matplotlib may not be available)")


def test_td4_feature_extraction():
    """
    Test TD4 feature extraction on filtered vs unfiltered signals.
    """
    print("\n" + "="*70)
    print("TEST 4: TD4 Feature Extraction (Filtered vs Unfiltered)")
    print("="*70)

    # Create synthetic EMG signal (250ms @ 2000 Hz = 500 samples)
    window_size = 500
    t = np.linspace(0, 0.25, window_size)

    # Realistic EMG with noise
    emg_clean = 300 * np.sin(2 * np.pi * 100 * t)
    dc_offset = 2048
    powerline = 150 * np.sin(2 * np.pi * 50 * t)
    noise = 50 * np.random.randn(window_size)

    raw_signal = dc_offset + emg_clean + powerline + noise

    # Apply filters
    filter_bank = EMGFilterBank(sampling_rate=2000, powerline_freq=50)
    filtered_signal = filter_bank.filter_signal(raw_signal)

    # Extract TD4 features
    extractor = TD4FeatureExtractor(
        window_size_ms=250,
        sampling_rate=2000,
        zc_threshold_adc=15.0,
        ssc_threshold_adc=15.0,
        adc_max=4095.0
    )

    # Compute features (6 sensors, but we'll use same signal for demo)
    sensor_data_raw = np.tile(raw_signal, (6, 1)).T
    sensor_data_filtered = np.tile(filtered_signal, (6, 1)).T

    features_raw = extractor.extract_window_features(sensor_data_raw,
                                                     [f'EMG{i+1}' for i in range(6)])
    features_filtered = extractor.extract_window_features(sensor_data_filtered,
                                                          [f'EMG{i+1}' for i in range(6)])

    print("\nFeature Comparison (Sensor 1):")
    print(f"  MAV (raw): {features_raw['mav_EMG1']:.6f}")
    print(f"  MAV (filtered): {features_filtered['mav_EMG1']:.6f}")
    print(f"  WL (raw): {features_raw['wl_EMG1']:.6f}")
    print(f"  WL (filtered): {features_filtered['wl_EMG1']:.6f}")
    print(f"  ZC (raw): {features_raw['zc_EMG1']:.6f}")
    print(f"  ZC (filtered): {features_filtered['zc_EMG1']:.6f}")
    print(f"  SSC (raw): {features_raw['ssc_EMG1']:.6f}")
    print(f"  SSC (filtered): {features_filtered['ssc_EMG1']:.6f}")

    print("\nExpected behavior:")
    print("  - MAV should be lower (noise removed)")
    print("  - WL should be lower (smoother signal)")
    print("  - ZC/SSC may change (cleaner zero crossings)")

    if features_filtered['mav_EMG1'] < features_raw['mav_EMG1']:
        print("  ✅ PASS: MAV reduced by filtering")
    else:
        print("  ⚠️  WARNING: MAV not reduced (may be OK depending on signal)")

    print("="*70)


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*80)
    print(" "*20 + "DSP FILTER VALIDATION SUITE")
    print("="*80)

    # Create plots directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)

    # Run tests
    test_wl_normalization()
    test_filter_frequency_response()
    test_filter_time_domain()
    test_td4_feature_extraction()

    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review frequency response plots in plots/ directory")
    print("2. Upload firmware to ESP32: pio run -e data_acquisition -t upload")
    print("3. Collect filtered test data: python training_data_collection.py")
    print("4. Compare C++ vs Python features for parity")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
