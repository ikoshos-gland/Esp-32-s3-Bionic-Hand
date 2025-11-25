#!/usr/bin/env python3
"""
Dummy EMG Data Generator for Pipeline Testing
==============================================

Generates synthetic EMG data for 11 gestures to test the training pipeline.
This is NOT real EMG data - just for verifying the scripts work correctly.

Output: data/dummy_training_data.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Gesture definitions (must match real system)
GESTURE_NAMES = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]

# Configuration
SAMPLING_RATE = 1000  # Hz
SAMPLES_PER_GESTURE = 1000  # 1 second of data per gesture
NUM_SENSORS = 6
ADC_MIN = 0
ADC_MAX = 4095

# EMG signal characteristics (synthetic)
# Rest: low activation, narrow range
# Active gestures: higher activation with unique patterns per gesture
GESTURE_PARAMS = {
    'Rest': {
        'mean': [2000, 2010, 2005, 2000, 2015, 2010],  # Low, near baseline
        'std': [20, 25, 22, 20, 23, 21],  # Small variance
        'freq': 5  # Hz - slow drift
    },
    'Fist': {
        'mean': [2800, 2900, 2850, 2820, 2880, 2860],  # High activation
        'std': [150, 160, 155, 150, 158, 152],  # Moderate variance
        'freq': 45  # Hz - muscle fiber firing rate
    },
    'Open': {
        'mean': [2600, 2700, 2650, 2620, 2680, 2640],  # Medium-high
        'std': [120, 130, 125, 120, 128, 122],
        'freq': 40
    },
    'Point': {
        'mean': [2400, 2900, 2300, 2350, 2850, 2320],  # Asymmetric (finger isolate)
        'std': [100, 150, 95, 105, 148, 98],
        'freq': 35
    },
    'Victory': {
        'mean': [2500, 2800, 2400, 2450, 2780, 2420],  # Two-finger pattern
        'std': [110, 140, 105, 115, 138, 108],
        'freq': 38
    },
    'OK': {
        'mean': [2450, 2750, 2500, 2480, 2730, 2510],  # Thumb-index coordination
        'std': [105, 135, 110, 108, 133, 112],
        'freq': 37
    },
    'ThumbUp': {
        'mean': [2550, 2650, 2800, 2570, 2630, 2780],  # Thumb emphasis
        'std': [115, 125, 145, 118, 123, 142],
        'freq': 36
    },
    'ThumbDn': {
        'mean': [2580, 2680, 2750, 2600, 2660, 2730],  # Similar to ThumbUp
        'std': [118, 128, 140, 120, 126, 138],
        'freq': 36
    },
    'Grasp': {
        'mean': [2700, 2850, 2750, 2720, 2830, 2740],  # Strong, uniform activation
        'std': [140, 155, 145, 142, 153, 143],
        'freq': 42
    },
    'Pinch': {
        'mean': [2350, 2700, 2400, 2380, 2680, 2420],  # Precision grip pattern
        'std': [95, 135, 100, 98, 133, 102],
        'freq': 34
    },
    'WristFlex': {
        'mean': [2900, 2950, 2500, 2920, 2930, 2480],  # Forearm emphasis
        'std': [160, 165, 110, 162, 163, 108],
        'freq': 44
    }
}


def generate_emg_signal(gesture_name, num_samples):
    """
    Generate synthetic EMG signal for a gesture.

    Combines:
    - DC offset (mean activation level)
    - Low-frequency drift (simulates motion artifacts)
    - High-frequency noise (simulates muscle fiber activity)
    - Clipping to ADC range [0, 4095]

    Args:
        gesture_name: Name of gesture (must be in GESTURE_PARAMS)
        num_samples: Number of samples to generate

    Returns:
        numpy array of shape (num_samples, 6) with integer ADC values
    """
    params = GESTURE_PARAMS[gesture_name]
    means = np.array(params['mean'])
    stds = np.array(params['std'])
    freq = params['freq']

    # Time vector
    t = np.arange(num_samples) / SAMPLING_RATE

    # Initialize signal array
    signal = np.zeros((num_samples, NUM_SENSORS))

    for sensor_idx in range(NUM_SENSORS):
        # DC component (mean activation)
        dc = means[sensor_idx]

        # Low-frequency drift (0.5-2 Hz) - simulates motion artifacts
        drift = 30 * np.sin(2 * np.pi * 1.5 * t + np.random.rand() * 2 * np.pi)

        # High-frequency component (muscle fiber firing)
        # Multiple frequency components for realism
        hf = stds[sensor_idx] * 0.3 * np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
        hf += stds[sensor_idx] * 0.2 * np.sin(2 * np.pi * (freq * 2) * t + np.random.rand() * 2 * np.pi)

        # Random noise (thermal noise, quantization noise)
        noise = np.random.normal(0, stds[sensor_idx] * 0.5, num_samples)

        # Combine components
        signal[:, sensor_idx] = dc + drift + hf + noise

    # Clip to ADC range and convert to integers
    signal = np.clip(signal, ADC_MIN, ADC_MAX).astype(int)

    return signal


def generate_dummy_dataset():
    """
    Generate complete dummy dataset with all 11 gestures.

    CSV format:
    Timestamp,Seq,Movement,Phase,Rep,EMG1,EMG2,EMG3,EMG4,EMG5,EMG6
    """
    print("=" * 70)
    print("Dummy EMG Data Generator")
    print("=" * 70)
    print(f"Gestures: {len(GESTURE_NAMES)}")
    print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
    print(f"Total samples: {len(GESTURE_NAMES) * SAMPLES_PER_GESTURE}")
    print(f"Sampling rate: {SAMPLING_RATE} Hz")
    print()

    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent.parent / 'data'
    data_dir.mkdir(exist_ok=True)

    # Prepare CSV data
    all_data = []
    seq_counter = 0

    for gesture_idx, gesture_name in enumerate(GESTURE_NAMES):
        print(f"Generating {gesture_name:12s} ... ", end='', flush=True)

        # Generate EMG signal
        emg_signal = generate_emg_signal(gesture_name, SAMPLES_PER_GESTURE)

        # Create DataFrame rows
        for sample_idx in range(SAMPLES_PER_GESTURE):
            timestamp = seq_counter / SAMPLING_RATE

            row = {
                'Timestamp': f"{timestamp:.4f}",
                'Seq': seq_counter,
                'Movement': gesture_name,
                'Phase': 'HAREKET',  # Turkish for "movement"
                'Rep': 1,  # Single repetition for simplicity
                'EMG1': emg_signal[sample_idx, 0],
                'EMG2': emg_signal[sample_idx, 1],
                'EMG3': emg_signal[sample_idx, 2],
                'EMG4': emg_signal[sample_idx, 3],
                'EMG5': emg_signal[sample_idx, 4],
                'EMG6': emg_signal[sample_idx, 5],
            }

            all_data.append(row)
            seq_counter += 1

        print(f"OK ({SAMPLES_PER_GESTURE} samples)")

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    output_path = data_dir / 'dummy_training_data.csv'
    df.to_csv(output_path, index=False)

    print()
    print("=" * 70)
    print(f"OK Dataset saved: {output_path}")
    print(f"  Total samples: {len(df):,}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    print()
    print("Next steps:")
    print(f"  1. Extract features: python scripts_ai/feature_extraction.py {output_path}")
    print(f"  2. Train model: python scripts_ai/train_test_model.py")
    print("=" * 70)

    return output_path


if __name__ == '__main__':
    generate_dummy_dataset()
