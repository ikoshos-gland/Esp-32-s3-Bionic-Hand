"""
DSP Filter Bank for EMG Signal Processing
==========================================

Python implementation of IIR filters matching the C++ implementation
on ESP32-S3. Ensures identical preprocessing between training and inference.

Filters:
- High-Pass: 20 Hz, 4th-order Butterworth (removes DC + motion artifacts)
- Low-Pass: 450 Hz, 4th-order Butterworth (removes electronic noise)
- Notch: 50/60 Hz, 2nd-order IIR (removes powerline interference)

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Union, Tuple


class EMGFilterBank:
    """
    EMG Signal Filter Bank

    Implements cascaded IIR filters for EMG signal conditioning.
    Uses SciPy's Second-Order Sections (SOS) format for numerical stability.

    Example:
        >>> filter_bank = EMGFilterBank(sampling_rate=2000, powerline_freq=50)
        >>> filtered_signal = filter_bank.filter_signal(raw_signal)
        >>> filtered_df = filter_bank.filter_dataframe(df, sensor_columns)
    """

    def __init__(self,
                 sampling_rate: int = 2000,
                 powerline_freq: int = 50,
                 enable_hpf: bool = True,
                 enable_lpf: bool = True,
                 enable_notch: bool = True,
                 enable_ma: bool = False):
        """
        Initialize EMG filter bank.

        Args:
            sampling_rate: Sampling frequency in Hz (1000 or 2000)
            powerline_freq: Powerline frequency in Hz (50 or 60)
            enable_hpf: Enable high-pass filter (20 Hz)
            enable_lpf: Enable low-pass filter (450 Hz)
            enable_notch: Enable notch filter (50/60 Hz)
            enable_ma: Enable moving average (disabled by default)
        """
        self.sampling_rate = sampling_rate
        self.powerline_freq = powerline_freq
        self.enable_hpf = enable_hpf
        self.enable_lpf = enable_lpf
        self.enable_notch = enable_notch
        self.enable_ma = enable_ma

        # Design filters
        self._design_filters()

        print(f"EMG Filter Bank Initialized:")
        print(f"  Sampling Rate: {sampling_rate} Hz")
        print(f"  High-Pass (20 Hz): {'ENABLED' if enable_hpf else 'DISABLED'}")
        print(f"  Low-Pass (450 Hz): {'ENABLED' if enable_lpf else 'DISABLED'}")
        print(f"  Notch ({powerline_freq} Hz): {'ENABLED' if enable_notch else 'DISABLED'}")
        print(f"  Moving Average: {'ENABLED' if enable_ma else 'DISABLED'}")

    def _design_filters(self):
        """Design all IIR filters using SciPy."""
        # High-Pass Filter: 20 Hz, 4th-order Butterworth
        if self.enable_hpf:
            self.sos_hpf = signal.butter(4, 20, 'high',
                                         fs=self.sampling_rate,
                                         output='sos')

        # Low-Pass Filter: 450 Hz, 4th-order Butterworth
        if self.enable_lpf:
            self.sos_lpf = signal.butter(4, 450, 'low',
                                         fs=self.sampling_rate,
                                         output='sos')

        # Notch Filter: 50/60 Hz, Q=12.5, 2nd-order
        if self.enable_notch:
            b_notch, a_notch = signal.iirnotch(self.powerline_freq, Q=12.5,
                                               fs=self.sampling_rate)
            # Convert to SOS format for consistency
            self.sos_notch = np.array([[b_notch[0], b_notch[1], b_notch[2],
                                       a_notch[0], a_notch[1], a_notch[2]]])

    def filter_signal(self, signal_data: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Apply filter cascade to a signal array.

        Filter order: HPF → LPF → Notch → (MA)

        Args:
            signal_data: Input signal (1D or 2D array)
            axis: Axis along which to filter (default: -1, last axis)

        Returns:
            Filtered signal (same shape as input)
        """
        filtered = signal_data.copy()

        # High-Pass Filter (20 Hz)
        if self.enable_hpf:
            filtered = signal.sosfiltfilt(self.sos_hpf, filtered, axis=axis)

        # Low-Pass Filter (450 Hz)
        if self.enable_lpf:
            filtered = signal.sosfiltfilt(self.sos_lpf, filtered, axis=axis)

        # Notch Filter (50/60 Hz)
        if self.enable_notch:
            filtered = signal.sosfiltfilt(self.sos_notch, filtered, axis=axis)

        # Moving Average (optional)
        if self.enable_ma:
            filtered = self._apply_moving_average(filtered, axis=axis)

        return filtered

    def _apply_moving_average(self, signal_data: np.ndarray,
                              window_size: int = 5,
                              axis: int = -1) -> np.ndarray:
        """
        Apply moving average filter.

        Args:
            signal_data: Input signal
            window_size: MA window size (default: 5)
            axis: Axis along which to filter

        Returns:
            Smoothed signal
        """
        # Use convolution for moving average
        kernel = np.ones(window_size) / window_size

        if signal_data.ndim == 1:
            return np.convolve(signal_data, kernel, mode='same')
        else:
            # Apply along specified axis
            return np.apply_along_axis(
                lambda x: np.convolve(x, kernel, mode='same'),
                axis=axis,
                arr=signal_data
            )

    def filter_dataframe(self,
                        df: pd.DataFrame,
                        sensor_columns: list) -> pd.DataFrame:
        """
        Apply filters to EMG sensor columns in a DataFrame.

        Args:
            df: Input DataFrame with EMG data
            sensor_columns: List of sensor column names to filter

        Returns:
            DataFrame with filtered sensor data (original df is not modified)
        """
        # Create copy to avoid modifying original
        df_filtered = df.copy()

        # Extract sensor data as 2D array (samples × sensors)
        sensor_data = df[sensor_columns].values

        # Apply filters along time axis (axis=0)
        filtered_data = self.filter_signal(sensor_data, axis=0)

        # Replace sensor columns with filtered data
        df_filtered[sensor_columns] = filtered_data

        return df_filtered

    def get_frequency_response(self, filter_type: str = 'cascade') -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate frequency response of filters.

        Args:
            filter_type: 'hpf', 'lpf', 'notch', or 'cascade'

        Returns:
            Tuple of (frequencies, magnitude in dB)
        """
        # Frequency points
        w = np.linspace(0, self.sampling_rate / 2, 2048)

        if filter_type == 'cascade':
            # Combined response of all enabled filters
            h_total = np.ones_like(w, dtype=complex)

            if self.enable_hpf:
                _, h = signal.sosfreqz(self.sos_hpf, worN=w, fs=self.sampling_rate)
                h_total *= h

            if self.enable_lpf:
                _, h = signal.sosfreqz(self.sos_lpf, worN=w, fs=self.sampling_rate)
                h_total *= h

            if self.enable_notch:
                _, h = signal.sosfreqz(self.sos_notch, worN=w, fs=self.sampling_rate)
                h_total *= h

            mag_db = 20 * np.log10(np.abs(h_total) + 1e-10)
            return w, mag_db

        elif filter_type == 'hpf' and self.enable_hpf:
            _, h = signal.sosfreqz(self.sos_hpf, worN=w, fs=self.sampling_rate)
            mag_db = 20 * np.log10(np.abs(h) + 1e-10)
            return w, mag_db

        elif filter_type == 'lpf' and self.enable_lpf:
            _, h = signal.sosfreqz(self.sos_lpf, worN=w, fs=self.sampling_rate)
            mag_db = 20 * np.log10(np.abs(h) + 1e-10)
            return w, mag_db

        elif filter_type == 'notch' and self.enable_notch:
            _, h = signal.sosfreqz(self.sos_notch, worN=w, fs=self.sampling_rate)
            mag_db = 20 * np.log10(np.abs(h) + 1e-10)
            return w, mag_db

        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

    def plot_frequency_response(self, save_path: str = None):
        """
        Plot frequency response of all filters.

        Args:
            save_path: Optional path to save plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Error: matplotlib not installed. Install with: pip install matplotlib")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'EMG Filter Bank Frequency Response ({self.sampling_rate} Hz)',
                     fontsize=14, fontweight='bold')

        # High-Pass Filter
        if self.enable_hpf:
            w, mag = self.get_frequency_response('hpf')
            axes[0, 0].plot(w, mag, 'b-', linewidth=2)
            axes[0, 0].axhline(-3, color='r', linestyle='--', label='-3dB')
            axes[0, 0].axvline(20, color='g', linestyle='--', label='20 Hz cutoff')
            axes[0, 0].set_title('High-Pass Filter (20 Hz)', fontweight='bold')
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Magnitude (dB)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].set_xlim(0, 100)

        # Low-Pass Filter
        if self.enable_lpf:
            w, mag = self.get_frequency_response('lpf')
            axes[0, 1].plot(w, mag, 'b-', linewidth=2)
            axes[0, 1].axhline(-3, color='r', linestyle='--', label='-3dB')
            axes[0, 1].axvline(450, color='g', linestyle='--', label='450 Hz cutoff')
            axes[0, 1].set_title('Low-Pass Filter (450 Hz)', fontweight='bold')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Magnitude (dB)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            axes[0, 1].set_xlim(300, self.sampling_rate / 2)

        # Notch Filter
        if self.enable_notch:
            w, mag = self.get_frequency_response('notch')
            axes[1, 0].plot(w, mag, 'b-', linewidth=2)
            axes[1, 0].axhline(-40, color='r', linestyle='--', label='-40dB')
            axes[1, 0].axvline(self.powerline_freq, color='g', linestyle='--',
                              label=f'{self.powerline_freq} Hz')
            axes[1, 0].set_title(f'Notch Filter ({self.powerline_freq} Hz)', fontweight='bold')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Magnitude (dB)')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_xlim(self.powerline_freq - 10, self.powerline_freq + 10)

        # Cascade (Combined Response)
        w, mag = self.get_frequency_response('cascade')
        axes[1, 1].plot(w, mag, 'b-', linewidth=2, label='Cascade')
        axes[1, 1].axhline(-3, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Complete Filter Cascade', fontweight='bold')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude (dB)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, self.sampling_rate / 2)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Frequency response plot saved to: {save_path}")
        else:
            plt.show()


def test_filters():
    """Test the filter bank with synthetic signals."""
    print("\n" + "="*70)
    print("Testing EMG Filter Bank")
    print("="*70)

    # Create filter bank
    filter_bank = EMGFilterBank(sampling_rate=2000, powerline_freq=50)

    # Generate test signal (1 second)
    t = np.linspace(0, 1, 2000)

    # Synthetic EMG: DC + low freq drift + EMG band + 50Hz noise + high freq noise
    dc_offset = 2048  # ADC midpoint
    low_freq_drift = 200 * np.sin(2 * np.pi * 5 * t)  # 5 Hz motion artifact
    emg_signal = 300 * np.sin(2 * np.pi * 100 * t)  # 100 Hz EMG
    powerline_noise = 150 * np.sin(2 * np.pi * 50 * t)  # 50 Hz powerline
    high_freq_noise = 50 * np.random.randn(len(t))  # Random high freq noise

    raw_signal = dc_offset + low_freq_drift + emg_signal + powerline_noise + high_freq_noise

    # Apply filters
    filtered_signal = filter_bank.filter_signal(raw_signal)

    print(f"\nRaw Signal Statistics:")
    print(f"  Mean: {np.mean(raw_signal):.1f} (should be ~{dc_offset})")
    print(f"  Std: {np.std(raw_signal):.1f}")
    print(f"  Range: [{np.min(raw_signal):.1f}, {np.max(raw_signal):.1f}]")

    print(f"\nFiltered Signal Statistics:")
    print(f"  Mean: {np.mean(filtered_signal):.1f} (DC removed)")
    print(f"  Std: {np.std(filtered_signal):.1f}")
    print(f"  Range: [{np.min(filtered_signal):.1f}, {np.max(filtered_signal):.1f}]")

    # Calculate SNR improvement
    signal_power = np.mean(emg_signal**2)
    noise_power_raw = np.mean((raw_signal - dc_offset - emg_signal)**2)
    noise_power_filtered = np.mean((filtered_signal - emg_signal)**2)

    snr_raw = 10 * np.log10(signal_power / noise_power_raw)
    snr_filtered = 10 * np.log10(signal_power / noise_power_filtered)

    print(f"\nSNR Analysis:")
    print(f"  Raw SNR: {snr_raw:.1f} dB")
    print(f"  Filtered SNR: {snr_filtered:.1f} dB")
    print(f"  SNR Improvement: {snr_filtered - snr_raw:.1f} dB")

    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    test_filters()
