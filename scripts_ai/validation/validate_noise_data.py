"""
Validate EMG Data Quality - Detect Noise and ADC Overflow
===========================================================

This script analyzes EMG CSV data to check for:
1. ADC overflow (values > 4095 for 12-bit ADCs)
2. Low variance (disconnected sensors or pure noise)
3. Gesture separability (ability to distinguish between gestures)

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import List, Tuple


def check_adc_overflow(df: pd.DataFrame, emg_cols: List[str]) -> Tuple[bool, dict]:
    """
    Check for ADC overflow (values exceeding 12-bit range)
    
    Args:
        df: DataFrame with EMG data
        emg_cols: List of EMG column names
        
    Returns:
        Tuple of (has_overflow, stats_dict)
    """
    print("\n" + "="*80)
    print("ADC OVERFLOW CHECK")
    print("="*80)
    
    has_overflow = False
    stats = {}
    
    for col in emg_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        mean_val = df[col].mean()
        var_val = df[col].var()
        std_val = df[col].std()
        
        stats[col] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'variance': var_val,
            'std': std_val
        }
        
        print(f"\n{col}:")
        print(f"  Min:      {min_val:8.1f}")
        print(f"  Max:      {max_val:8.1f}")
        print(f"  Mean:     {mean_val:8.1f}")
        print(f"  Std Dev:  {std_val:8.1f}")
        print(f"  Variance: {var_val:10.1f}")
        
        # Check for overflow (12-bit ADC max = 4095)
        if max_val > 4095:
            print(f"  ❌ OVERFLOW DETECTED! Max should be ≤ 4095 (12-bit ADC)")
            print(f"     This indicates DSP filter output isn't clamped properly")
            has_overflow = True
        else:
            print(f"  ✅ Valid range (within 12-bit ADC limits)")
        
        # Check for suspicious values near uint16_t limits
        if max_val > 60000:
            print(f"  🚨 CRITICAL: Values near uint16_t max (65535) detected!")
            print(f"     This is likely DSP filter negative values wrapping around")
            has_overflow = True
    
    return has_overflow, stats


def check_variance(df: pd.DataFrame, emg_cols: List[str], 
                   threshold_variance: float = 100.0) -> Tuple[bool, dict]:
    """
    Check for low variance (disconnected sensors or pure noise)
    
    Args:
        df: DataFrame with EMG data
        emg_cols: List of EMG column names
        threshold_variance: Minimum acceptable variance
        
    Returns:
        Tuple of (has_low_variance, results_dict)
    """
    print("\n" + "="*80)
    print("VARIANCE CHECK (Detecting Disconnected Sensors)")
    print("="*80)
    print(f"Threshold: Variance should be > {threshold_variance}")
    
    has_low_variance = False
    results = {}
    
    for col in emg_cols:
        var_val = df[col].var()
        results[col] = var_val
        
        print(f"\n{col}: Variance = {var_val:10.1f}", end="")
        
        if var_val < threshold_variance:
            print(f"  ⚠️  LOW VARIANCE - Possible disconnected sensor or noise")
            has_low_variance = True
        else:
            print(f"  ✅ OK")
    
    return has_low_variance, results


def check_gesture_separability(df: pd.DataFrame, emg_cols: List[str], 
                               movement_col: str = 'Movement') -> dict:
    """
    Check if gestures are distinguishable by their EMG patterns
    
    Args:
        df: DataFrame with EMG data
        emg_cols: List of EMG column names
        movement_col: Column name containing gesture labels
        
    Returns:
        Dictionary of separability metrics
    """
    print("\n" + "="*80)
    print("GESTURE SEPARABILITY CHECK")
    print("="*80)
    
    if movement_col not in df.columns:
        print(f"⚠️  Column '{movement_col}' not found. Skipping separability check.")
        return {}
    
    gestures = df[movement_col].unique()
    print(f"\nGestures found: {list(gestures)}")
    
    separability = {}
    
    for gesture in gestures:
        gesture_df = df[df[movement_col] == gesture]
        
        print(f"\n{gesture} ({len(gesture_df)} samples):")
        
        gesture_stats = {}
        for col in emg_cols:
            mean = gesture_df[col].mean()
            std = gesture_df[col].std()
            gesture_stats[col] = {'mean': mean, 'std': std}
            
            print(f"  {col}: μ={mean:7.1f} σ={std:6.1f}")
        
        separability[gesture] = gesture_stats
    
    # Calculate between-gesture variance (good) vs within-gesture variance (bad)
    print("\n" + "-"*80)
    print("Signal-to-Noise Ratio (Between-Gesture / Within-Gesture Variance)")
    print("-"*80)
    
    for col in emg_cols:
        # Between-gesture variance (mean differences)
        gesture_means = [separability[g][col]['mean'] for g in gestures]
        between_var = np.var(gesture_means)
        
        # Within-gesture variance (average std across gestures)
        gesture_stds = [separability[g][col]['std'] for g in gestures]
        within_var = np.mean([s**2 for s in gesture_stds])
        
        # SNR = between / within (higher is better)
        if within_var > 0:
            snr = between_var / within_var
            print(f"{col}: SNR = {snr:6.2f}", end="")
            
            if snr < 1.0:
                print(f"  ⚠️  Low separability (gestures overlap)")
            elif snr < 5.0:
                print(f"  ⚙️  Moderate separability")
            else:
                print(f"  ✅ Good separability")
        else:
            print(f"{col}: SNR = N/A (zero variance)")
    
    return separability


def validate_emg_data(csv_path: str, 
                     emg_cols: List[str] = None,
                     threshold_variance: float = 100.0) -> bool:
    """
    Main validation function
    
    Args:
        csv_path: Path to CSV file
        emg_cols: List of EMG column names (default: EMG1-EMG6)
        threshold_variance: Minimum acceptable variance
        
    Returns:
        True if data is valid, False otherwise
    """
    if emg_cols is None:
        emg_cols = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6']
    
    print("\n" + "="*80)
    print(f"VALIDATING: {os.path.basename(csv_path)}")
    print("="*80)
    
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    except Exception as e:
        print(f"\n❌ ERROR loading CSV: {e}")
        return False
    
    # Verify all EMG columns exist
    missing_cols = [col for col in emg_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ ERROR: Missing columns: {missing_cols}")
        print(f"   Available columns: {df.columns.tolist()}")
        return False
    
    # Run checks
    has_overflow, overflow_stats = check_adc_overflow(df, emg_cols)
    has_low_variance, variance_results = check_variance(df, emg_cols, threshold_variance)
    separability_results = check_gesture_separability(df, emg_cols)
    
    # Final verdict
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    is_valid = True
    
    if has_overflow:
        print("\n❌ FAILED: ADC overflow detected")
        print("   → Fix: Update firmware to clamp DSP filter outputs to [0, 4095]")
        print("   → File: data_acquisition.cpp and functions.cpp")
        is_valid = False
    else:
        print("\n✅ PASSED: No ADC overflow detected")
    
    if has_low_variance:
        print("\n⚠️  WARNING: Low variance detected in some sensors")
        print("   → Possible causes:")
        print("     - Disconnected sensor(s)")
        print("     - Electrodes not attached to skin")
        print("     - Pure noise data (no muscle activity)")
        # Low variance is a warning, not a failure
    else:
        print("\n✅ PASSED: All sensors show adequate variance")
    
    if not separability_results:
        print("\n⚠️  WARNING: Could not check gesture separability (no Movement column)")
    else:
        print("\n✅ INFO: Gesture separability metrics calculated (see above)")
    
    print("\n" + "="*80)
    if is_valid:
        print("✅ DATA IS VALID - Safe to use for training")
    else:
        print("❌ DATA IS INVALID - Do not use for training!")
        print("\n   Next steps:")
        print("   1. Fix firmware issues (ADC overflow clamping)")
        print("   2. Re-upload firmware to ESP32")
        print("   3. Collect new training data")
        print("   4. Re-run this validation script")
    print("="*80 + "\n")
    
    return is_valid


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_noise_data.py <path_to_csv>")
        print("\nExample:")
        print("  python scripts_ai/validate_noise_data.py data/training_data_20251126.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        sys.exit(1)
    
    is_valid = validate_emg_data(csv_path)
    
    sys.exit(0 if is_valid else 1)
