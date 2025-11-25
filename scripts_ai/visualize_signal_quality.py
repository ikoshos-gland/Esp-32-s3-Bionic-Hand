"""
EMG Signal Quality Visualization and Analysis
==============================================

Analyzes and visualizes the quality of EMG signals from training data
to identify potential issues affecting model performance.

Author: ESP32 Bionic Hand Project
Date: 2025-11-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def calculate_signal_quality_metrics(signal):
    """
    Calculate quality metrics for an EMG signal
    
    Args:
        signal: 1D EMG signal array
        
    Returns:
        Dictionary of quality metrics
    """
    # Remove DC offset
    signal_centered = signal - np.mean(signal)
    
    # RMS (Root Mean Square) - signal strength
    rms = np.sqrt(np.mean(signal_centered**2))
    
    # Signal variance
    variance = np.var(signal_centered)
    
    # Peak-to-peak amplitude
    peak_to_peak = np.max(signal) - np.min(signal)
    
    # Zero crossing rate (frequency estimate)
    zero_crossings = np.sum(np.diff(np.sign(signal_centered)) != 0)
    zcr = zero_crossings / len(signal)
    
    # Signal-to-Noise Ratio estimate (using median absolute deviation)
    mad = np.median(np.abs(signal_centered - np.median(signal_centered)))
    snr_estimate = rms / (mad * 1.4826) if mad > 0 else 0  # 1.4826 is normalization factor
    
    return {
        'rms': rms,
        'variance': variance,
        'peak_to_peak': peak_to_peak,
        'zero_crossing_rate': zcr,
        'snr_estimate': snr_estimate
    }


def visualize_signal_samples(df, sensor_columns, output_dir='scripts_ai/data/plots/signal_quality'):
    """
    Visualize sample signals for each gesture
    
    Args:
        df: DataFrame with EMG data
        sensor_columns: List of EMG sensor column names
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter to include both HAREKET and DINLENME phases (exclude only HAZIRLIK)
    if 'Phase' in df.columns:
        # Check what phase values exist
        phase_values = df['Phase'].unique()
        print(f"Phase values in data: {phase_values}")
        
        # CRITICAL: Include BOTH phases:
        # - HAREKET (active movements like Fist, Open, etc.)
        # - DINLENME (rest periods - Rest movement is here!)
        # Exclude only HAZIRLIK (preparation)
        df_movement = df[~df['Phase'].str.upper().str.contains('HAZIRLIK', na=False)]
        
        print(f"Filtered out HAZIRLIK: {len(df_movement)} samples remaining")
        print(f"  (Includes both HAREKET and DINLENME phases)")
    else:
        df_movement = df
        print("No 'Phase' column found, using all data")
    
    # Get unique movements
    unique_movements = sorted(df_movement['Movement'].unique())
    print(f"\nFound {len(unique_movements)} movements: {unique_movements}")
    
    # ============================================================================
    # 1. Plot sample signals for each movement (all 6 channels)
    # ============================================================================
    print("\n" + "="*60)
    print("Generating signal samples for each movement...")
    print("="*60)
    
    sample_duration = 2000  # 1 second at 2000 Hz
    
    for movement in unique_movements:
        movement_data = df_movement[df_movement['Movement'] == movement]
        
        if len(movement_data) < sample_duration:
            print(f"⚠️  Skipping {movement}: Not enough samples ({len(movement_data)} < {sample_duration})")
            continue
        
        # Take a random sample
        start_idx = np.random.randint(0, len(movement_data) - sample_duration)
        sample_data = movement_data.iloc[start_idx:start_idx + sample_duration]
        
        # Create figure with 6 subplots (one per sensor)
        fig, axes = plt.subplots(6, 1, figsize=(14, 10), sharex=True)
        fig.suptitle(f'EMG Signals - {movement} (1 second sample)', fontsize=16, fontweight='bold')
        
        time = np.arange(sample_duration) / 2000  # Convert to seconds
        
        for idx, sensor in enumerate(sensor_columns):
            signal = sample_data[sensor].values
            
            # Calculate metrics
            metrics = calculate_signal_quality_metrics(signal)
            
            # Plot signal
            axes[idx].plot(time, signal, linewidth=0.5, alpha=0.8)
            axes[idx].set_ylabel(f'{sensor}\n[ADC]', fontsize=10)
            axes[idx].grid(True, alpha=0.3)
            
            # Add metrics as text
            metrics_text = f"RMS: {metrics['rms']:.1f} | Var: {metrics['variance']:.0f} | P2P: {metrics['peak_to_peak']:.0f}"
            axes[idx].text(0.02, 0.95, metrics_text, transform=axes[idx].transAxes,
                          fontsize=8, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Set y-axis limits with some margin
            y_margin = (np.max(signal) - np.min(signal)) * 0.1
            axes[idx].set_ylim(np.min(signal) - y_margin, np.max(signal) + y_margin)
        
        axes[-1].set_xlabel('Time [s]', fontsize=12)
        plt.tight_layout()
        
        output_path = Path(output_dir) / f'signal_sample_{movement}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_path.name}")
    
    # ============================================================================
    # 2. Compare all movements side-by-side (EMG1 only for clarity)
    # ============================================================================
    print("\n" + "="*60)
    print("Generating movement comparison (EMG1)...")
    print("="*60)
    
    n_movements = len(unique_movements)
    fig, axes = plt.subplots(n_movements, 1, figsize=(14, 2*n_movements), sharex=True)
    if n_movements == 1:
        axes = [axes]
    
    fig.suptitle('Movement Comparison - EMG1 Channel (1 second samples)', fontsize=16, fontweight='bold')
    
    for idx, movement in enumerate(unique_movements):
        movement_data = df_movement[df_movement['Movement'] == movement]
        
        if len(movement_data) < sample_duration:
            continue
        
        start_idx = np.random.randint(0, len(movement_data) - sample_duration)
        signal = movement_data.iloc[start_idx:start_idx + sample_duration]['EMG1'].values
        
        metrics = calculate_signal_quality_metrics(signal)
        
        time = np.arange(sample_duration) / 2000
        axes[idx].plot(time, signal, linewidth=0.7, alpha=0.9, color=f'C{idx}')
        axes[idx].set_ylabel(movement, fontsize=11, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        
        # Add metrics
        metrics_text = f"RMS: {metrics['rms']:.1f} | SNR Est: {metrics['snr_estimate']:.1f}"
        axes[idx].text(0.98, 0.95, metrics_text, transform=axes[idx].transAxes,
                      fontsize=9, verticalalignment='top', horizontalalignment='right',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    axes[-1].set_xlabel('Time [s]', fontsize=12)
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'movement_comparison_emg1.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    
    # ============================================================================
    # 3. Signal quality metrics summary
    # ============================================================================
    print("\n" + "="*60)
    print("Calculating signal quality metrics for all movements...")
    print("="*60)
    
    metrics_summary = []
    
    for movement in unique_movements:
        movement_data = df_movement[df_movement['Movement'] == movement]
        
        if len(movement_data) < sample_duration:
            continue
        
        # Take multiple random samples for better statistics
        n_samples = min(5, len(movement_data) // sample_duration)
        
        for sensor in sensor_columns:
            sensor_metrics = []
            
            for _ in range(n_samples):
                start_idx = np.random.randint(0, len(movement_data) - sample_duration)
                signal = movement_data.iloc[start_idx:start_idx + sample_duration][sensor].values
                metrics = calculate_signal_quality_metrics(signal)
                sensor_metrics.append(metrics)
            
            # Average metrics
            avg_metrics = {
                'Movement': movement,
                'Sensor': sensor,
                'RMS': np.mean([m['rms'] for m in sensor_metrics]),
                'Variance': np.mean([m['variance'] for m in sensor_metrics]),
                'Peak-to-Peak': np.mean([m['peak_to_peak'] for m in sensor_metrics]),
                'Zero Crossing Rate': np.mean([m['zero_crossing_rate'] for m in sensor_metrics]),
                'SNR Estimate': np.mean([m['snr_estimate'] for m in sensor_metrics])
            }
            metrics_summary.append(avg_metrics)
    
    metrics_df = pd.DataFrame(metrics_summary)
    
    # Print summary table
    print("\nSignal Quality Metrics (averaged across 5 samples):")
    print("-" * 100)
    print(metrics_df.to_string(index=False))
    
    # Save to CSV
    csv_path = Path(output_dir) / 'signal_quality_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics saved to: {csv_path}")
    
    # ============================================================================
    # 4. Plot metrics comparison across movements
    # ============================================================================
    print("\n" + "="*60)
    print("Generating metrics comparison plots...")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Signal Quality Metrics Comparison Across Movements', fontsize=16, fontweight='bold')
    
    # RMS comparison
    rms_pivot = metrics_df.pivot(index='Movement', columns='Sensor', values='RMS')
    rms_pivot.plot(kind='bar', ax=axes[0, 0], legend=True)
    axes[0, 0].set_title('RMS (Signal Strength)', fontweight='bold')
    axes[0, 0].set_ylabel('RMS [ADC units]')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend(title='Sensor', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Variance comparison
    var_pivot = metrics_df.pivot(index='Movement', columns='Sensor', values='Variance')
    var_pivot.plot(kind='bar', ax=axes[0, 1], legend=True)
    axes[0, 1].set_title('Variance', fontweight='bold')
    axes[0, 1].set_ylabel('Variance [ADC units²]')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].legend(title='Sensor', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Peak-to-Peak comparison
    p2p_pivot = metrics_df.pivot(index='Movement', columns='Sensor', values='Peak-to-Peak')
    p2p_pivot.plot(kind='bar', ax=axes[1, 0], legend=True)
    axes[1, 0].set_title('Peak-to-Peak Amplitude', fontweight='bold')
    axes[1, 0].set_ylabel('Amplitude [ADC units]')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].legend(title='Sensor', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # SNR comparison
    snr_pivot = metrics_df.pivot(index='Movement', columns='Sensor', values='SNR Estimate')
    snr_pivot.plot(kind='bar', ax=axes[1, 1], legend=True)
    axes[1, 1].set_title('SNR Estimate', fontweight='bold')
    axes[1, 1].set_ylabel('SNR')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].legend(title='Sensor', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {output_path.name}")
    
    # ============================================================================
    # 5. Check for separability between movements
    # ============================================================================
    print("\n" + "="*60)
    print("Analyzing movement separability...")
    print("="*60)
    
    # Calculate average RMS per movement (all sensors combined)
    avg_rms_per_movement = metrics_df.groupby('Movement')['RMS'].mean().sort_values()
    
    print("\nAverage RMS per Movement (across all sensors):")
    for movement, rms in avg_rms_per_movement.items():
        print(f"  {movement:12s}: {rms:6.2f} ADC units")
    
    # Check if Rest has significantly lower RMS than active movements
    if 'Rest' in avg_rms_per_movement.index:
        rest_rms = avg_rms_per_movement['Rest']
        other_rms = avg_rms_per_movement.drop('Rest')
        
        if rest_rms < other_rms.mean() * 0.5:
            print(f"\n✅ GOOD: Rest RMS ({rest_rms:.1f}) is clearly lower than active movements (avg: {other_rms.mean():.1f})")
        else:
            print(f"\n⚠️  WARNING: Rest RMS ({rest_rms:.1f}) is similar to active movements (avg: {other_rms.mean():.1f})")
            print("   This suggests Rest periods may contain unwanted muscle activity!")
    
    # Check RMS variance across movements
    rms_std = avg_rms_per_movement.std()
    rms_mean = avg_rms_per_movement.mean()
    coefficient_of_variation = rms_std / rms_mean
    
    print(f"\nMovement Separability Analysis:")
    print(f"  RMS Coefficient of Variation: {coefficient_of_variation:.2f}")
    if coefficient_of_variation < 0.3:
        print("  ⚠️  LOW separability - movements have similar signal strengths!")
        print("      This makes classification difficult.")
    elif coefficient_of_variation < 0.6:
        print("  ⚡ MODERATE separability - movements are somewhat distinguishable.")
    else:
        print("  ✅ GOOD separability - movements have distinct signal strengths!")
    
    print("\n" + "="*60)
    print("Signal quality analysis complete!")
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_signal_quality.py <path_to_csv>")
        print("\nExample:")
        print("  python visualize_signal_quality.py data_acquisition/data/training_data_S3_20251125_002634.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    print("="*60)
    print("EMG Signal Quality Visualization")
    print("="*60)
    print(f"\nLoading data from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    sensor_columns = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6']
    
    # Verify sensor columns exist
    for col in sensor_columns:
        if col not in df.columns:
            print(f"❌ Error: Sensor column '{col}' not found in CSV")
            sys.exit(1)
    
    visualize_signal_samples(df, sensor_columns)
    
    print("\n✅ All visualizations saved to: plots/signal_quality/")
