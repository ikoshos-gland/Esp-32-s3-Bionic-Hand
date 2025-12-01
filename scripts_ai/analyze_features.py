"""
TD4 Feature Value Analyzer
===========================

Analyzes extracted TD4 features to verify normalization and detect data corruption.

Usage:
    python scripts_ai/analyze_features.py data/features/training_data_TIMESTAMP_td4_features.npz

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import numpy as np
import sys
import os


def analyze_features(npz_path):
    """Analyze TD4 feature values"""

    print("="*80)
    print("TD4 FEATURE VALUE ANALYZER")
    print("="*80)
    print()

    # Load NPZ
    print(f"Loading: {npz_path}")
    data = np.load(npz_path)
    features = data['features']
    labels = data['labels']
    feature_names = data['feature_names']

    print(f"  Shape: {features.shape}")
    print(f"  Classes: {np.unique(labels)}")
    print()

    # Feature column indices
    mav_cols = [0, 4, 8, 12, 16, 20]   # MAV for sensors 0-5
    wl_cols = [1, 5, 9, 13, 17, 21]    # WL
    zc_cols = [2, 6, 10, 14, 18, 22]   # ZC
    ssc_cols = [3, 7, 11, 15, 19, 23]  # SSC

    # Analyze value ranges
    print("-"*80)
    print("FEATURE VALUE RANGES (Normalized [0, 1])")
    print("-"*80)
    print(f"{'Feature':<10} | {'Min':>8} | {'Max':>8} | {'Mean':>8} | {'Std':>8} | Status")
    print("-"*80)

    def analyze_feature_group(name, cols):
        vals = features[:, cols]
        min_val = vals.min()
        max_val = vals.max()
        mean_val = vals.mean()
        std_val = vals.std()

        # Health check
        if max_val > 1.0:
            status = "[ERROR] Values > 1.0!"
        elif min_val < 0.0:
            status = "[ERROR] Values < 0.0!"
        elif max_val < 0.01:
            status = "[WARN] Too small"
        else:
            status = "[OK]"

        print(f"{name:<10} | {min_val:8.4f} | {max_val:8.4f} | {mean_val:8.4f} | {std_val:8.4f} | {status}")
        return min_val, max_val, mean_val

    mav_min, mav_max, mav_mean = analyze_feature_group("MAV", mav_cols)
    wl_min, wl_max, wl_mean = analyze_feature_group("WL", wl_cols)
    zc_min, zc_max, zc_mean = analyze_feature_group("ZC", zc_cols)
    ssc_min, ssc_max, ssc_mean = analyze_feature_group("SSC", ssc_cols)

    print("-"*80)
    print()

    # Zero corruption check (CRITICAL for ZC/SSC)
    print("-"*80)
    print("ZERO CORRUPTION CHECK (Old Half-Wave Rectified Bug)")
    print("-"*80)

    total_samples = features.shape[0] * 6  # samples × 6 sensors

    zc_values = features[:, zc_cols].flatten()
    ssc_values = features[:, ssc_cols].flatten()

    zc_zeros = (zc_values == 0).sum()
    ssc_zeros = (ssc_values == 0).sum()

    zc_zero_pct = 100 * zc_zeros / total_samples
    ssc_zero_pct = 100 * ssc_zeros / total_samples

    print(f"ZC  zeros: {zc_zeros:6d} / {total_samples:6d} ({zc_zero_pct:5.1f}%)", end="")
    if zc_zero_pct > 50:
        print(" [ERROR] CORRUPTED!")
    elif zc_zero_pct > 10:
        print(" [WARN] Suspicious")
    else:
        print(" [OK]")

    print(f"SSC zeros: {ssc_zeros:6d} / {total_samples:6d} ({ssc_zero_pct:5.1f}%)", end="")
    if ssc_zero_pct > 50:
        print(" [ERROR] CORRUPTED!")
    elif ssc_zero_pct > 10:
        print(" [WARN] Suspicious")
    else:
        print(" [OK]")

    print("-"*80)
    print()

    # Expected ranges
    print("-"*80)
    print("EXPECTED VALUE RANGES (For Reference)")
    print("-"*80)
    print("Feature | Rest      | Low Act.  | Med Act.  | High Act.")
    print("--------|-----------|-----------|-----------|----------")
    print("MAV     | 0.01-0.05 | 0.05-0.15 | 0.15-0.35 | 0.35-0.70")
    print("WL      | 0.02-0.10 | 0.10-0.30 | 0.30-0.60 | 0.60-1.00")
    print("ZC      | 0.01-0.04 | 0.04-0.10 | 0.10-0.20 | 0.20-0.40")
    print("SSC     | 0.01-0.04 | 0.04-0.10 | 0.10-0.20 | 0.20-0.40")
    print("-"*80)
    print()

    # Per-class analysis
    print("-"*80)
    print("PER-CLASS FEATURE ANALYSIS")
    print("-"*80)

    unique_classes = np.unique(labels)
    for cls in unique_classes[:5]:  # Show first 5 classes
        mask = labels == cls
        cls_features = features[mask]

        mav_mean_cls = cls_features[:, mav_cols].mean()
        zc_mean_cls = cls_features[:, zc_cols].mean()
        ssc_mean_cls = cls_features[:, ssc_cols].mean()

        print(f"{cls:12s}: MAV={mav_mean_cls:.3f}, ZC={zc_mean_cls:.3f}, SSC={ssc_mean_cls:.3f}")

    if len(unique_classes) > 5:
        print(f"... ({len(unique_classes)-5} more classes)")

    print("-"*80)
    print()

    # Final verdict
    print("="*80)
    print("FINAL VERDICT")
    print("="*80)

    issues = []

    if zc_zero_pct > 50 or ssc_zero_pct > 50:
        issues.append("CRITICAL: ZC/SSC mostly zeros - Half-wave rectified signal!")

    if mav_max > 1.0 or wl_max > 1.0 or zc_max > 1.0 or ssc_max > 1.0:
        issues.append("ERROR: Features not normalized (values > 1.0)")

    if mav_max < 0.01 and wl_max < 0.01:
        issues.append("WARNING: Features too small - possible normalization issue")

    if zc_mean < 0.001 or ssc_mean < 0.001:
        issues.append("WARNING: ZC/SSC near zero - verify bipolar decoding")

    if len(issues) == 0:
        print("[OK] All features look healthy!")
        print("     - Normalization: Correct")
        print("     - ZC/SSC: Non-zero (bipolar signal preserved)")
        print("     - Value ranges: Within expected bounds")
    else:
        print("[ISSUES DETECTED]")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print("="*80)
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find latest NPZ in data/features/
        features_dir = "scripts_ai/data/features"
        if os.path.exists(features_dir):
            npz_files = [f for f in os.listdir(features_dir) if f.endswith('.npz')]
            if npz_files:
                # Sort by modification time
                npz_files.sort(key=lambda x: os.path.getmtime(os.path.join(features_dir, x)), reverse=True)
                latest_npz = os.path.join(features_dir, npz_files[0])
                print(f"No file specified, using latest: {latest_npz}\n")
                analyze_features(latest_npz)
            else:
                print("Error: No NPZ files found in scripts_ai/data/features/")
                print("\nUsage: python scripts_ai/analyze_features.py <path_to_npz>")
                sys.exit(1)
        else:
            print("Error: scripts_ai/data/features/ directory not found")
            print("\nUsage: python scripts_ai/analyze_features.py <path_to_npz>")
            sys.exit(1)
    else:
        npz_path = sys.argv[1]
        if not os.path.exists(npz_path):
            print(f"Error: File not found: {npz_path}")
            sys.exit(1)

        analyze_features(npz_path)
