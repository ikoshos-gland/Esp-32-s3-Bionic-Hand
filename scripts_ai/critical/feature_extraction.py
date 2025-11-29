"""
TD4 Feature Extraction for 8-Channel EMG Data
==============================================

Adapted from Hudgins' TD4 (Time-Domain 4) feature set for 8 EMG sensors.

Features (per EMG sensor):
- MAV (Mean Absolute Value): Average signal amplitude
- WL (Waveform Length): Signal complexity measure
- ZC (Zero Crossings): Frequency estimate
- SSC (Slope Sign Changes): Frequency content indicator

Total features: 4 features × 8 EMG sensors = 32 features

References:
- Hudgins et al. (1993): "A New Strategy for Multifunction Myoelectric Control"
- Phinyomark et al. (2012): "Feature Reduction and Selection for EMG Signal Classification"
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import signal as sp_signal

# ADC constants - must match C++ embedded system (functions.h lines 21, 26-27)
ADC_MAX_GLOBAL = 4095.0  # 12-bit ADC on ESP32-S3
ZC_THRESHOLD_ADC = 15.0   # Zero crossing threshold (ADC units)
SSC_THRESHOLD_ADC = 15.0  # Slope sign change threshold (ADC units)


class SimpleBandpassFilter:
    """Simple bandpass filter for EMG signal preprocessing"""

    def __init__(self, lowcut=20, highcut=450, fs=1000, order=4):
        """
        Initialize bandpass filter

        Args:
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # Design Butterworth bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = sp_signal.butter(order, [low, high], btype='band')

    def filter_signal(self, data):
        """Apply bandpass filter to signal"""
        return sp_signal.filtfilt(self.b, self.a, data)


class TD4FeatureExtractor:
    """
    TD4 Feature Extractor for 8-channel EMG signals
    """

    def __init__(self,
                 window_size_ms: int = 250,
                 overlap_ms: int = 125,
                 sampling_rate: int = 1000,
                 zc_threshold: float = 15.0 / 4095.0,  # ≈ 0.00366 (matches C++ effective threshold)
                 ssc_threshold: float = 15.0 / 4095.0,  # ≈ 0.00366 (matches C++ effective threshold)
                 apply_bandpass: bool = True):
        """
        Initialize TD4 feature extractor

        Args:
            window_size_ms: Analysis window size in milliseconds
            overlap_ms: Overlap between consecutive windows in milliseconds
            sampling_rate: Sampling frequency in Hz
            zc_threshold: Threshold for zero crossing (normalized)
            ssc_threshold: Threshold for slope sign change (normalized)
            apply_bandpass: Apply bandpass filter (20-450 Hz)
        """
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms
        self.sampling_rate = sampling_rate
        self.zc_threshold = zc_threshold
        self.ssc_threshold = ssc_threshold
        self.apply_bandpass = apply_bandpass

        # Calculate window parameters
        self.window_samples = int((window_size_ms / 1000) * sampling_rate)
        self.overlap_samples = int((overlap_ms / 1000) * sampling_rate)
        self.hop_samples = self.window_samples - self.overlap_samples

        # Initialize bandpass filter
        if apply_bandpass:
            self.bandpass = SimpleBandpassFilter(lowcut=20, highcut=450, fs=sampling_rate)

        print(f"TD4 Feature Extractor initialized (8-channel):")
        print(f"  Window size: {window_size_ms}ms ({self.window_samples} samples)")
        print(f"  Overlap: {overlap_ms}ms ({self.overlap_samples} samples)")
        print(f"  Hop size: {self.hop_samples} samples")
        print(f"  ZC threshold: {zc_threshold:.6f} (matches C++ 15.0 ADC units / 4095)")
        print(f"  SSC threshold: {ssc_threshold:.6f} (matches C++ 15.0 ADC units / 4095)")
        print(f"  Bandpass filter: {'Enabled (20-450 Hz)' if apply_bandpass else 'Disabled'}")

    def compute_mav(self, signal: np.ndarray) -> float:
        """
        Mean Absolute Value (MAV)
        MAV = (1/N) × Σ|x[i]|
        """
        return np.mean(np.abs(signal))

    def compute_wl(self, signal: np.ndarray) -> float:
        """
        Waveform Length (WL)
        WL = Σ|x[i+1] - x[i]|
        """
        return np.sum(np.abs(np.diff(signal)))

    def compute_zc(self, signal: np.ndarray, threshold: float) -> int:
        """
        Zero Crossings (ZC)
        Counts zero crossings with threshold
        """
        # Center signal
        centered = signal - np.mean(signal)

        # Calculate product of consecutive samples
        products = centered[:-1] * centered[1:]

        # Calculate absolute differences
        diffs = np.abs(centered[:-1] - centered[1:])

        # Count crossings
        zc_count = np.sum((products < 0) & (diffs >= threshold))

        return int(zc_count)

    def compute_ssc(self, signal: np.ndarray, threshold: float) -> int:
        """
        Slope Sign Changes (SSC)
        Counts slope changes with threshold
        """
        if len(signal) < 3:
            return 0

        # Center signal
        centered = signal - np.mean(signal)

        # Calculate slopes
        left_slope = centered[1:-1] - centered[:-2]
        right_slope = centered[1:-1] - centered[2:]

        # Calculate products
        products = left_slope * right_slope

        # Count sign changes
        ssc_count = np.sum(products >= threshold)

        return int(ssc_count)

    def extract_window_features(self, window: np.ndarray, sensor_names: List[str]) -> Dict[str, float]:
        """
        Extract TD4 features from a single window

        Args:
            window: EMG data window (samples × channels)
            sensor_names: List of sensor column names

        Returns:
            Dictionary of feature values
        """
        features = {}

        for idx, sensor in enumerate(sensor_names):
            signal = window[:, idx]

            # Apply bandpass filter if enabled
            if self.apply_bandpass:
                signal = self.bandpass.filter_signal(signal)

            # Compute TD4 features
            mav = self.compute_mav(signal)
            wl = self.compute_wl(signal)
            zc = self.compute_zc(signal, self.zc_threshold)
            ssc = self.compute_ssc(signal, self.ssc_threshold)

            # CRITICAL FIX: Normalize features to match C++ functions.cpp lines 316-322
            # C++ uses GLOBAL normalization by ADC range (ADC_MAX_GLOBAL = 4095)
            mav_norm = mav / ADC_MAX_GLOBAL  # Match C++ line 316
            wl_norm = wl / (ADC_MAX_GLOBAL * len(signal))  # Match C++ line 317 (4095 × 250)
            zc_norm = float(zc) / len(signal)  # Match C++ line 320
            ssc_norm = float(ssc) / len(signal)  # Match C++ line 321

            # Store features
            features[f'mav_{sensor}'] = mav_norm
            features[f'wl_{sensor}'] = wl_norm
            features[f'zc_{sensor}'] = zc_norm
            features[f'ssc_{sensor}'] = ssc_norm

        return features

    def extract_features_from_dataframe(self,
                                       df: pd.DataFrame,
                                       sensor_columns: List[str],
                                       label_column: str = 'Movement') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract TD4 features from a pandas DataFrame using sliding windows

        Args:
            df: DataFrame with EMG data
            sensor_columns: List of EMG sensor column names (should be 8)
            label_column: Name of the label/class column

        Returns:
            Tuple of (features_array, labels_array, feature_names)
        """
        print(f"\nProcessing DataFrame: {len(df)} samples...")
        print(f"Sensors: {sensor_columns}")

        features_list = []
        labels_list = []
        window_info_list = []

        # Generate feature names
        feature_names = []
        for sensor in sensor_columns:
            feature_names.extend([
                f'mav_{sensor}',
                f'wl_{sensor}',
                f'zc_{sensor}',
                f'ssc_{sensor}'
            ])

        # Sliding window extraction
        n_windows = 0
        for start_idx in range(0, len(df) - self.window_samples + 1, self.hop_samples):
            end_idx = start_idx + self.window_samples

            # Extract window
            window_df = df.iloc[start_idx:end_idx]
            window_data = window_df[sensor_columns].values

            # Get label (most common label in window)
            try:
                window_label = window_df[label_column].mode()[0]
                window_rep = window_df['Rep'].mode()[0] if 'Rep' in window_df.columns else 0
                window_timestamp = window_df['Timestamp'].iloc[0] if 'Timestamp' in window_df.columns else 0
            except (IndexError, KeyError):
                continue

            # Extract features
            window_features = self.extract_window_features(window_data, sensor_columns)

            # Convert to ordered list
            feature_vector = [window_features[name] for name in feature_names]

            features_list.append(feature_vector)
            labels_list.append(window_label)
            window_info_list.append({
                'label': window_label,
                'rep': window_rep,
                'timestamp': window_timestamp,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            n_windows += 1

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        print(f"\nExtracted {n_windows} windows")
        print(f"Feature shape: {features_array.shape}")
        print(f"Unique labels: {np.unique(labels_array)}")

        # Show class distribution
        unique_labels, label_counts = np.unique(labels_array, return_counts=True)
        print(f"\nClass distribution:")
        for label, count in zip(unique_labels, label_counts):
            percentage = (count / len(labels_array)) * 100
            print(f"  Class {label}: {count} windows ({percentage:.1f}%)")

        return features_array, labels_array, feature_names, window_info_list


def extract_features_from_csv(csv_path: str,
                              sensor_columns: List[str] = None,
                              apply_bandpass: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Extract TD4 features from a CSV file

    Args:
        csv_path: Path to input CSV file
        sensor_columns: List of 8 EMG sensor column names
        apply_bandpass: Apply bandpass filter (20-450 Hz)

    Returns:
        Tuple of (features, labels, feature_names, dataframe)
    """
    # Default sensor columns for 8 EMG sensors
    if sensor_columns is None:
        sensor_columns = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']

    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Verify sensor columns exist
    for col in sensor_columns:
        if col not in df.columns:
            raise ValueError(f"Sensor column '{col}' not found in CSV")

    # Initialize extractor
    # Match C++ thresholds (functions.h: ZC/SSC_THRESHOLD_ADC = 15.0)
    ADC_MAX = 4095.0
    extractor = TD4FeatureExtractor(
        window_size_ms=250,
        overlap_ms=125,
        sampling_rate=1000,
        zc_threshold=15.0 / ADC_MAX,  # ≈ 0.00366 (matches C++)
        ssc_threshold=15.0 / ADC_MAX,  # ≈ 0.00366 (matches C++)
        apply_bandpass=apply_bandpass
    )

    # Extract features
    features, labels, feature_names, window_info = extractor.extract_features_from_dataframe(
        df, sensor_columns, label_column='Movement'
    )

    return features, labels, feature_names, df


if __name__ == "__main__":
    # Example usage
    csv_path = "S1_50P_combined.csv"

    print("="*70)
    print("TD4 FEATURE EXTRACTION - 8 Channel EMG")
    print("="*70)

    features, labels, feature_names, df = extract_features_from_csv(
        csv_path,
        sensor_columns=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8'],
        apply_bandpass=True
    )

    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"Features shape: {features.shape}")
    print(f"Total features per window: {len(feature_names)}")
    print(f"Total windows: {len(features)}")
