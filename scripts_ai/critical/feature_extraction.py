"""
TD4 Feature Extraction for ESP32-S3 Bionic Hand
================================================

Implements Hudgins' TD4 (Time-Domain 4) feature set - the "Gold Standard"
for EMG pattern recognition from literature.

Features (per EMG sensor):
- MAV (Mean Absolute Value): Average signal amplitude
- WL (Waveform Length): Signal complexity measure
- ZC (Zero Crossings): Frequency estimate
- SSC (Slope Sign Changes): Frequency content indicator

Total features: 4 features × 6 EMG sensors = 24 features

References:
- Hudgins et al. (1993): "A New Strategy for Multifunction Myoelectric Control"
- Phinyomark et al. (2012): "Feature Reduction and Selection for EMG Signal Classification"

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import os


class TD4FeatureExtractor:
    """
    TD4 Feature Extractor for 6-channel EMG signals

    Extracts literature-validated time-domain features optimized for
    real-time embedded systems (ESP32-S3).
    """

    def __init__(self,
                 window_size_ms: int = 250,
                 overlap_ms: int = 125,
                 sampling_rate: int = 2000,
                 zc_threshold_adc: float = 15.0,
                 ssc_threshold_adc: float = 15.0,
                 adc_max: float = 4095.0):
        """
        Initialize TD4 feature extractor

        REFACTORED: Now processes RAW signals (no RMS preprocessing)

        Args:
            window_size_ms: Analysis window size in milliseconds (CHANGED: 250ms to match C++)
            overlap_ms: Overlap between consecutive windows in milliseconds
            sampling_rate: Sampling frequency in Hz
            zc_threshold_adc: Threshold for zero crossing in ADC units (CHANGED: was normalized)
            ssc_threshold_adc: Threshold for slope sign change in ADC units (CHANGED: was normalized)
            adc_max: Maximum ADC value for global normalization (12-bit = 4095)
        """
        self.window_size_ms = window_size_ms
        self.overlap_ms = overlap_ms
        self.sampling_rate = sampling_rate
        self.zc_threshold_adc = zc_threshold_adc
        self.ssc_threshold_adc = ssc_threshold_adc
        self.adc_max = adc_max

        # Calculate window parameters
        self.window_samples = int((window_size_ms / 1000) * sampling_rate)
        self.overlap_samples = int((overlap_ms / 1000) * sampling_rate)
        self.hop_samples = self.window_samples - self.overlap_samples

        print(f"TD4 Feature Extractor initialized (RAW signal processing):")
        print(f"  Window size: {window_size_ms}ms ({self.window_samples} samples)")
        print(f"  Overlap: {overlap_ms}ms ({self.overlap_samples} samples)")
        print(f"  Hop size: {self.hop_samples} samples")
        print(f"  ZC threshold: {zc_threshold_adc} ADC units")
        print(f"  SSC threshold: {ssc_threshold_adc} ADC units")
        print(f"  ADC max (global norm): {adc_max}")

    def compute_mav(self, signal: np.ndarray) -> float:
        """
        Mean Absolute Value (MAV)

        MAV = (1/N) × Σ|x[i]|

        Represents the average amplitude of the signal.
        Most commonly used EMG feature in literature.

        Args:
            signal: Input EMG signal window

        Returns:
            MAV feature value
        """
        return np.mean(np.abs(signal))

    def compute_wl(self, signal: np.ndarray) -> float:
        """
        Waveform Length (WL)

        WL = Σ|x[i+1] - x[i]|

        Measures the complexity of the EMG signal.
        Related to signal frequency content and amplitude.

        Args:
            signal: Input EMG signal window

        Returns:
            WL feature value
        """
        return np.sum(np.abs(np.diff(signal)))

    def compute_zc(self, signal: np.ndarray, threshold: float = 15.0) -> int:
        """
        Zero Crossings (ZC) - Raw Signal Version

        ZC = Σ sgn(x[i] × x[i+1] < 0) with threshold

        Counts the number of times the centered signal crosses zero.
        CRITICAL: Signal must be centered (DC offset removed) for this to work!

        Args:
            signal: Centered EMG signal window (mean-subtracted)
            threshold: Minimum difference in ADC units for valid crossing

        Returns:
            ZC count
        """
        # Calculate product of consecutive samples
        products = signal[:-1] * signal[1:]

        # Calculate absolute differences (threshold condition)
        diffs = np.abs(signal[:-1] - signal[1:])

        # Count crossings where product is negative and difference exceeds threshold
        zc_count = np.sum((products < 0) & (diffs >= threshold))

        return int(zc_count)

    def compute_ssc(self, signal: np.ndarray, threshold: float = 15.0) -> int:
        """
        Slope Sign Changes (SSC) - Raw Signal Version

        SSC = Σ sgn((x[i] - x[i-1]) × (x[i] - x[i+1]) >= threshold)

        Counts the number of times the signal slope changes sign.
        CRITICAL: Signal must be centered (DC offset removed) for this to work!

        Args:
            signal: Centered EMG signal window (mean-subtracted)
            threshold: Minimum product value in ADC units squared

        Returns:
            SSC count
        """
        if len(signal) < 3:
            return 0

        # Calculate slopes
        left_slope = signal[1:-1] - signal[:-2]   # x[i] - x[i-1]
        right_slope = signal[1:-1] - signal[2:]   # x[i] - x[i+1]

        # Calculate products
        products = left_slope * right_slope

        # Count sign changes where product exceeds threshold
        ssc_count = np.sum(products >= threshold)

        return int(ssc_count)

    def extract_window_features(self, window: np.ndarray, sensor_names: List[str]) -> Dict[str, float]:
        """
        Extract TD4 features from raw ADC signals

        REFACTORED: Now processes RAW signals (no RMS preprocessing)

        Process (matching C++ exactly):
        1. Calculate DC offset (mean) for each sensor
        2. Center signal by subtracting mean
        3. Calculate TD4 features on centered signal
        4. Apply GLOBAL normalization (using ADC max)

        Args:
            window: EMG data window (samples × channels) - raw ADC values
            sensor_names: List of sensor column names

        Returns:
            Dictionary of feature values
        """
        features = {}

        for idx, sensor in enumerate(sensor_names):
            raw_signal = window[:, idx]

            # STEP 1: Calculate DC offset (mean)
            mean_val = np.mean(raw_signal)

            # STEP 2: Center signal (remove DC offset)
            centered_signal = raw_signal - mean_val

            # STEP 3: Calculate TD4 features on centered signal
            mav = self.compute_mav(centered_signal)
            wl = self.compute_wl(centered_signal)
            zc = self.compute_zc(centered_signal, self.zc_threshold_adc)
            ssc = self.compute_ssc(centered_signal, self.ssc_threshold_adc)

            # STEP 4: Apply GLOBAL normalization (preserves amplitude differences)
            mav_normalized = mav / self.adc_max
            wl_normalized = wl / (self.adc_max * len(centered_signal))
            zc_normalized = float(zc) / len(centered_signal)  # Normalize to rate
            ssc_normalized = float(ssc) / len(centered_signal)  # Normalize to rate

            # STEP 5: Store features (order MUST match C++!)
            features[f'mav_{sensor}'] = mav_normalized
            features[f'wl_{sensor}'] = wl_normalized
            features[f'zc_{sensor}'] = zc_normalized
            features[f'ssc_{sensor}'] = ssc_normalized

        return features

    def extract_features_from_dataframe(self,
                                       df: pd.DataFrame,
                                       sensor_columns: List[str],
                                       label_column: str = 'Movement',
                                       apply_filters: bool = True,
                                       powerline_freq: int = 50) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract TD4 features from a pandas DataFrame using sliding windows

        PIPELINE:
        1. Apply DSP filters (HPF + LPF + Notch) - ADDED 2025
        2. Filter HAZIRLIK at DataFrame level (BEFORE windowing)
        3. Balance "Rest" class after windowing

        Args:
            df: DataFrame with EMG data
            sensor_columns: List of EMG sensor column names (should be 6)
            label_column: Name of the label/class column
            apply_filters: Apply DSP filters (HPF+LPF+Notch) to match C++ (default: True)
            powerline_freq: Powerline frequency 50 or 60 Hz (default: 50)

        Returns:
            Tuple of (features_array, labels_array, feature_names)
        """
        assert len(sensor_columns) == 6, f"Expected 6 EMG sensors, got {len(sensor_columns)}"

        print(f"\nProcessing DataFrame: {len(df)} raw samples...")

        # ============================================================================
        # STEP 0: APPLY DSP FILTERS (NEW - Matches C++ preprocessing)
        # ============================================================================
        if apply_filters:
            print(f"\n{'='*60}")
            print("APPLYING DSP FILTERS (Matching C++ ESP32 preprocessing)")
            print(f"{'='*60}")

            from dsp_filters import EMGFilterBank

            filter_bank = EMGFilterBank(
                sampling_rate=self.sampling_rate,
                powerline_freq=powerline_freq,
                enable_hpf=True,   # High-pass 20 Hz
                enable_lpf=True,   # Low-pass 450 Hz
                enable_notch=True, # Notch 50/60 Hz
                enable_ma=False    # Moving average disabled
            )

            df = filter_bank.filter_dataframe(df, sensor_columns)

            print(f"  Filters applied to {len(sensor_columns)} EMG channels")
            print(f"  Signal conditioned: DC removed, noise reduced, powerline rejected")
            print(f"{'='*60}\n")

        # ============================================================================
        # STEP 1: FILTER HAZIRLIK DATA (BEFORE windowing!)
        # ============================================================================
        if 'Phase' in df.columns:
            print(f"\n{'='*60}")
            print("CLEANING DATA: Removing 'HAZIRLIK' (preparation) phase")
            print(f"{'='*60}")
            
            initial_count = len(df)
            
            # Filter out rows where Phase contains 'HAZIRLIK' (case-insensitive)
            df = df[~df['Phase'].str.upper().str.contains('HAZIRLIK', na=False)]
            
            # Also filter out 'PREPARATION' (English variant)
            df = df[~df['Phase'].str.upper().str.contains('PREPARATION', na=False)]
            
            removed_count = initial_count - len(df)
            print(f"  Removed {removed_count} raw samples (Phase='HAZIRLIK')")
            print(f"  Remaining raw samples: {len(df)}")
            print(f"{'='*60}\n")
        else:
            print("\n⚠️  WARNING: 'Phase' column not found. Skipping HAZIRLIK filtering!")
            print("   This is OK if your data doesn't have preparation phases.\n")

        # ============================================================================
        # STEP 2: EXTRACT FEATURES FROM WINDOWS
        # ============================================================================
        print(f"Extracting windows from {len(df)} samples...")
        print(f"Sensors: {sensor_columns}")

        features_list = []
        labels_list = []

        # Generate feature names
        feature_names = []
        for sensor in sensor_columns:
            feature_names.extend([
                f'mav_{sensor}',
                f'wl_{sensor}',
                f'zc_{sensor}',
                f'ssc_{sensor}'
            ])

        # Reset index after filtering (CRITICAL!)
        df = df.reset_index(drop=True)

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
            except IndexError:
                continue  # Skip empty windows if any

            # Extract features
            window_features = self.extract_window_features(window_data, sensor_columns)

            # Convert to ordered list matching feature_names
            feature_vector = [window_features[name] for name in feature_names]

            features_list.append(feature_vector)
            labels_list.append(window_label)
            n_windows += 1

        features_array = np.array(features_list)
        labels_array = np.array(labels_list)

        print(f"\nExtracted {n_windows} windows")
        print(f"Feature shape: {features_array.shape}")
        
        # ============================================================================
        # STEP 3: BALANCE 'Rest' CLASS (AFTER windowing)
        # ============================================================================
        print(f"\n{'='*60}")
        print("BALANCING CLASSES: Equalizing 'Rest' with other gestures")
        print(f"{'='*60}")
        
        # Get class distribution
        unique_labels, label_counts = np.unique(labels_array, return_counts=True)
        
        print(f"\nClass distribution (before balancing):")
        for label, count in zip(unique_labels, label_counts):
            percentage = (count / len(labels_array)) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        # Find "Rest" class (could be in different formats)
        rest_variants = ['Rest', 'REST', 'rest', 'Dinlenme', 'DINLENME']
        rest_label = None
        for variant in rest_variants:
            if variant in unique_labels:
                rest_label = variant
                break
        
        if rest_label is not None:
            # Calculate median count of non-Rest classes
            non_rest_counts = [count for label, count in zip(unique_labels, label_counts) 
                             if label != rest_label]
            
            if len(non_rest_counts) > 0:
                # Target: 1.2x median of other classes (slight buffer is good)
                target_count = int(np.median(non_rest_counts) * 1.2)
                current_rest_count = label_counts[unique_labels == rest_label][0]
                
                print(f"\nBalancing '{rest_label}' class:")
                print(f"  Current count: {current_rest_count}")
                print(f"  Target count: ~{target_count} (1.2× median of others)")
                
                # Only downsample if Rest is actually dominant
                if current_rest_count > target_count:
                    # Get indices of Rest and non-Rest samples
                    rest_indices = np.where(labels_array == rest_label)[0]
                    non_rest_indices = np.where(labels_array != rest_label)[0]
                    
                    # Randomly subsample Rest class to target count
                    np.random.seed(42)  # For reproducibility
                    rest_indices_sampled = np.random.choice(rest_indices, target_count, replace=False)
                    
                    # Combine sampled Rest with all non-Rest samples
                    final_indices = np.concatenate([rest_indices_sampled, non_rest_indices])
                    final_indices = np.sort(final_indices)
                    
                    features_array = features_array[final_indices]
                    labels_array = labels_array[final_indices]
                    
                    print(f"  ✅ Downsampled '{rest_label}' from {current_rest_count} to {target_count}")
                    print(f"  Total samples after balancing: {len(labels_array)}")
                else:
                    print(f"  ✅ '{rest_label}' already balanced (no downsampling needed)")
        
        # Show final class distribution
        print(f"\nFinal Class Distribution:")
        unique_labels_final, label_counts_final = np.unique(labels_array, return_counts=True)
        for label, count in zip(unique_labels_final, label_counts_final):
            percentage = (count / len(labels_array)) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nFinal feature shape: {features_array.shape}")
        print(f"Total features per window: {len(feature_names)}")
        print(f"{'='*60}\n")

        return features_array, labels_array, feature_names

    def normalize_features(self, features: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Min-Max normalization to [0, 1] range

        Args:
            features: Feature array (n_samples × n_features)

        Returns:
            Tuple of (normalized_features, normalization_params)
        """
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)

        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0

        normalized = (features - min_vals) / range_vals

        params = {
            'min_vals': min_vals,
            'max_vals': max_vals,
            'range_vals': range_vals
        }

        print(f"\nNormalization applied:")
        print(f"  Min values: {min_vals[:4]}... (showing first 4)")
        print(f"  Max values: {max_vals[:4]}... (showing first 4)")

        return normalized, params


def extract_features_from_csv(csv_path: str,
                              output_dir: str = 'scripts_ai/data/features',
                              sensor_columns: List[str] = None,
                              apply_filters: bool = True,
                              powerline_freq: int = 50) -> str:
    """
    Extract TD4 features from a CSV file and save to NPZ format

    Args:
        csv_path: Path to input CSV file
        output_dir: Directory to save extracted features
        sensor_columns: List of 6 EMG sensor column names
        apply_filters: Apply DSP filters (HPF+LPF+Notch) to match C++ (default: True)
        powerline_freq: Powerline frequency 50 or 60 Hz (default: 50)

    Returns:
        Path to saved NPZ file
    """
    # Default sensor columns for 6 EMG sensors
    if sensor_columns is None:
        sensor_columns = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6']

    # Load data
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")

    # Verify sensor columns exist
    for col in sensor_columns:
        if col not in df.columns:
            raise ValueError(f"Sensor column '{col}' not found in CSV")

    # Initialize extractor (UPDATED: 250ms window, raw signal processing to match C++)
    extractor = TD4FeatureExtractor(
        window_size_ms=250,
        overlap_ms=125,
        sampling_rate=2000,
        zc_threshold_adc=15.0,
        ssc_threshold_adc=15.0,
        adc_max=4095.0
    )

    # Extract features (now uses raw signal processing with DC offset removal + DSP filters)
    features, labels, feature_names = extractor.extract_features_from_dataframe(
        df, sensor_columns, label_column='Movement',
        apply_filters=apply_filters, powerline_freq=powerline_freq
    )

    # Features are already globally normalized during extraction
    # This matches C++ behavior: global ADC normalization preserves amplitude info
    features_normalized = features
    norm_params = {'note': 'Global ADC normalization applied during extraction', 'adc_max': 4095.0}

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_dir, f'{csv_basename}_td4_features.npz')

    # Save to NPZ
    np.savez_compressed(
        output_path,
        features=features_normalized,
        labels=labels,
        feature_names=feature_names,
        norm_params_note=norm_params['note'],
        norm_params_adc_max=norm_params['adc_max'],
        sensor_columns=sensor_columns
    )

    print(f"\nFeatures saved to: {output_path}")
    print(f"Features shape: {features_normalized.shape}")
    print(f"Unique labels: {np.unique(labels)}")

    return output_path


if __name__ == "__main__":
    """
    Example usage
    """
    import sys

    if len(sys.argv) < 2:
        print("Usage: python feature_extraction.py <path_to_csv>")
        print("\nExample:")
        print("  python feature_extraction.py data/training_data_20251123.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Extract features
    output_path = extract_features_from_csv(
        csv_path,
        output_dir='scripts_ai/data/features',
        sensor_columns=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6']
    )

    print("\n" + "="*60)
    print("TD4 Feature extraction complete!")
    print(f"Output: {output_path}")
    print("="*60)
