"""
EMG DSP filter bank (Python twin of src/filters.cpp)
====================================================

Filters:
- High-pass: 20 Hz, 4th-order Butterworth (2 biquads)
- Low-pass: 450 Hz, 4th-order Butterworth (2 biquads)
- Notch: 50/60 Hz, Q = 12.5 (1 biquad)

IMPORTANT: the firmware filters are CAUSAL single-pass IIR biquads
(Direct Form II Transposed, zero initial state). This module therefore uses
scipy.signal.sosfilt (causal), never sosfiltfilt. Zero-phase forward-backward
filtering would square the magnitude response and remove the phase delay,
producing training data the device can never reproduce.

`biquad_cascade_reference()` is a literal transliteration of the C++ code and
is used by validation/validate_filters.py to prove sample-level equivalence.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy import signal


class EMGFilterBank:
    """Causal IIR filter cascade matching the ESP32 firmware."""

    def __init__(self,
                 sampling_rate: int = 1000,
                 powerline_freq: int = 50,
                 enable_hpf: bool = True,
                 enable_lpf: bool = True,
                 enable_notch: bool = True,
                 enable_ma: bool = False,
                 verbose: bool = True):
        self.sampling_rate = sampling_rate
        self.powerline_freq = powerline_freq
        self.enable_hpf = enable_hpf
        self.enable_lpf = enable_lpf
        self.enable_notch = enable_notch
        self.enable_ma = enable_ma
        self._design_filters()
        if verbose:
            print("EMG Filter Bank (causal, matches C++):")
            print(f"  Sampling Rate: {sampling_rate} Hz")
            print(f"  HPF 20 Hz: {'ENABLED' if enable_hpf else 'DISABLED'}")
            print(f"  LPF 450 Hz: {'ENABLED' if enable_lpf else 'DISABLED'}")
            print(f"  Notch {powerline_freq} Hz: {'ENABLED' if enable_notch else 'DISABLED'}")

    # ------------------------------------------------------------------ design
    def _design_filters(self):
        fs = self.sampling_rate
        self.sos_hpf = signal.butter(4, 20, 'high', fs=fs, output='sos') if self.enable_hpf else None
        lpf_cut = min(450, 0.45 * fs)   # keep the cutoff below Nyquist for any fs
        self.sos_lpf = signal.butter(4, lpf_cut, 'low', fs=fs, output='sos') if self.enable_lpf else None
        if self.enable_notch:
            b, a = signal.iirnotch(self.powerline_freq, Q=12.5, fs=fs)
            self.sos_notch = np.array([[b[0], b[1], b[2], a[0], a[1], a[2]]])
        else:
            self.sos_notch = None

    def cascade_sos(self) -> np.ndarray:
        """All enabled sections in firmware order (HPF, LPF, Notch)."""
        parts = [s for s in (self.sos_hpf, self.sos_lpf, self.sos_notch) if s is not None]
        return np.vstack(parts) if parts else np.empty((0, 6))

    # --------------------------------------------------------------- filtering
    def filter_signal(self, signal_data: np.ndarray, axis: int = -1) -> np.ndarray:
        """Causal filtering with zero initial state (same as a freshly reset device)."""
        filtered = np.asarray(signal_data, dtype=np.float64).copy()
        for sos in (self.sos_hpf, self.sos_lpf, self.sos_notch):
            if sos is not None:
                filtered = signal.sosfilt(sos, filtered, axis=axis)
        if self.enable_ma:
            filtered = self._apply_moving_average(filtered, axis=axis)
        return filtered

    def _apply_moving_average(self, signal_data: np.ndarray, window_size: int = 5, axis: int = -1) -> np.ndarray:
        # Causal moving average (the firmware version is causal too)
        kernel = np.ones(window_size) / window_size

        def ma(x):
            return signal.lfilter(kernel, [1.0], x)

        if signal_data.ndim == 1:
            return ma(signal_data)
        return np.apply_along_axis(ma, axis=axis, arr=signal_data)

    def filter_dataframe(self, df: pd.DataFrame, sensor_columns: list) -> pd.DataFrame:
        df_filtered = df.copy()
        df_filtered[sensor_columns] = self.filter_signal(df[sensor_columns].values, axis=0)
        return df_filtered

    # ---------------------------------------------------------------- analysis
    def get_frequency_response(self, filter_type: str = 'cascade') -> Tuple[np.ndarray, np.ndarray]:
        w = np.linspace(0, self.sampling_rate / 2, 2048)
        sel = {'hpf': self.sos_hpf, 'lpf': self.sos_lpf, 'notch': self.sos_notch, 'cascade': self.cascade_sos()}[filter_type]
        if sel is None or len(sel) == 0:
            return w, np.zeros_like(w)
        _, h = signal.sosfreqz(sel, worN=w, fs=self.sampling_rate)
        mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
        return w, mag_db

    def plot_frequency_response(self, save_path: str = None):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f'EMG Filter Bank Frequency Response ({self.sampling_rate} Hz, causal)', fontweight='bold')
        for ax, (name, title) in zip(axes.flat, [('hpf', 'High-pass 20 Hz'), ('lpf', 'Low-pass 450 Hz'),
                                                  ('notch', f'Notch {self.powerline_freq} Hz'), ('cascade', 'Cascade')]):
            w, mag = self.get_frequency_response(name)
            ax.plot(w, mag)
            ax.set_title(title); ax.set_xlabel('Hz'); ax.set_ylabel('dB'); ax.grid(alpha=.3)
            ax.set_ylim(-80, 5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        return fig


# ----------------------------------------------------------------------------
# Literal C++ reference (Direct Form II Transposed, float32 arithmetic)
# ----------------------------------------------------------------------------
def biquad_cascade_reference(x: np.ndarray, stages) -> np.ndarray:
    """
    Run a cascade of biquads exactly like src/filters.cpp::biquad_process().

    stages: iterable of (b0, b1, b2, a1, a2) with a0 normalised to 1.
    Arithmetic is done in float32 to mirror the ESP32.
    """
    f32 = np.float32
    coeffs = [tuple(f32(c) for c in s) for s in stages]
    z = [[f32(0), f32(0)] for _ in coeffs]
    y = np.empty(len(x), dtype=np.float32)
    for n, xn in enumerate(np.asarray(x, dtype=np.float32)):
        s = xn
        for k, (b0, b1, b2, a1, a2) in enumerate(coeffs):
            out = f32(b0 * s + z[k][0])
            z[k][0] = f32(b1 * s - a1 * out + z[k][1])
            z[k][1] = f32(b2 * s - a2 * out)
            s = out
        y[n] = s
    return y


def sos_to_stages(sos: np.ndarray):
    """Convert scipy SOS rows [b0 b1 b2 a0 a1 a2] to (b0, b1, b2, a1, a2) with a0 = 1."""
    stages = []
    for b0, b1, b2, a0, a1, a2 in sos:
        stages.append((b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0))
    return stages


if __name__ == '__main__':
    fb = EMGFilterBank(sampling_rate=1000, powerline_freq=50)
    t = np.arange(0, 1, 1 / 1000)
    x = 2048 + 300 * np.sin(2 * np.pi * 100 * t) + 150 * np.sin(2 * np.pi * 50 * t)
    y = fb.filter_signal(x)
    print(f"raw mean {x.mean():.1f} -> filtered mean {y[200:].mean():.2f}, std {y[200:].std():.1f}")
