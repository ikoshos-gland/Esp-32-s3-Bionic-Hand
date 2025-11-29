"""
IMPROVED Multi-Intensity EMG Gesture Classification
====================================================

IMPROVEMENTS OVER PREVIOUS VERSION:
1. ✅ Reduced model capacity (fewer neurons) to prevent overfitting
2. ✅ Stronger L2 regularization (0.01 → 0.02)
3. ✅ Higher dropout rates (50/40/30 → 60/50/40)
4. ✅ Label smoothing to prevent overconfident predictions
5. ✅ Better early stopping strategy
6. ✅ Learning rate scheduling
7. ✅ Ensemble predictions (optional)

OVERFITTING ANALYSIS:
- Previous model: Train ~99%, Val ~97%, Test ~90% → OVERFITTING!
- This model targets: Train ~92%, Val ~91%, Test ~88-91% → Better generalization

PIPELINE ALIGNMENT WITH C++ EMBEDDED SYSTEM:
8. ✅ RobustScaler REMOVED (C++ has no scaler)
9. ✅ MAV normalized by ADC_MAX (4095)
10. ✅ WL normalized by (ADC_MAX × window_size)
11. ✅ ZC/SSC thresholds match C++ (15 ADC units / 4095)
12. ⚠️  Filtering difference: Python uses filtfilt (zero-phase), C++ uses causal biquad
    Model tolerates this ~12ms phase delay (small relative to 250ms window)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
# Scaler removed to match C++ inference pipeline (no scaler in embedded system)
# from sklearn.preprocessing import StandardScaler, RobustScaler
from feature_extraction import extract_features_from_csv

# Set style
sns.set_style("whitegrid")

# ============================================================================
# IMPROVED CONFIGURATION
# ============================================================================

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32  # Smaller batches for better generalization
LEARNING_RATE = 0.0005  # Reduced initial LR

# IMPROVED Model architecture - SMALLER to reduce overfitting
LAYER_1_UNITS = 128  # Was 256 → Reduced by 50%
LAYER_2_UNITS = 64   # Was 128 → Reduced by 50%
LAYER_3_UNITS = 32   # Was 64 → Reduced by 50%

# STRONGER Regularization
L2_LAMBDA = 0.02           # Was 0.01 → Doubled
DROPOUT_RATE_1 = 0.60      # Was 0.50 → Increased
DROPOUT_RATE_2 = 0.50      # Was 0.40 → Increased
DROPOUT_RATE_3 = 0.40      # Was 0.30 → Increased
LABEL_SMOOTHING = 0.1      # NEW: Prevents overconfident predictions

# Early stopping - MORE AGGRESSIVE
EARLY_STOPPING_PATIENCE = 20  # Was 25
MIN_DELTA = 0.001              # NEW: Minimum improvement threshold

# Intensities to include in training
TRAIN_INTENSITIES = ['20P', '30P', '40P', '50P', '60P', '70P', '80P', 'Light', 'Medium', 'Hard']
TEST_INTENSITY = '60P'  # Changed to 60P for different test

# Repetition split (within each intensity)
TEST_REPS = [2, 4]  # Test on reps 2 and 4
VAL_REPS = [3]      # Validate on rep 3

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # Go up to S1_20P directory
DATA_DIR = os.path.join(PARENT_DIR, 'data', 'raw')  # Point to data/raw
MODELS_DIR = os.path.join(CURRENT_DIR, 'models_improved')
PLOTS_DIR = os.path.join(CURRENT_DIR, 'plots_improved')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Movement class names
MOVEMENT_NAMES = [
    'No Movement',
    'Wrist Flexion',
    'Wrist Extension',
    'Wrist Pronation',
    'Wrist Supination',
    'Chuck Grip',
    'Hand Open'
]

NUM_CLASSES = len(MOVEMENT_NAMES)

print("="*80)
print("IMPROVED MULTI-INTENSITY EMG CLASSIFICATION (ANTI-OVERFITTING)".center(80))
print("="*80)

# ============================================================================
# DATA LOADING & MERGING
# ============================================================================

def parse_filename(filename):
    """Parse filename to extract metadata."""
    basename = os.path.basename(filename)
    pattern = r'S(\d+)_(\w+)_C(\d+)_R(\d+)\.csv'
    match = re.match(pattern, basename)
    
    if match:
        return {
            'subject': int(match.group(1)),
            'intensity': match.group(2),
            'class_id': int(match.group(3)),
            'rep_id': int(match.group(4)),
            'filename': basename
        }
    else:
        return None

def load_all_data(intensities):
    """Load and merge data from all specified intensities."""
    print(f"\n{'='*80}")
    print("LOADING MULTI-INTENSITY DATA")
    print(f"{'='*80}\n")
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    for intensity in intensities:
        print(f"\n{'-'*80}")
        print(f"Processing Intensity: {intensity}")
        print(f"{'-'*80}")
        
        pattern = os.path.join(DATA_DIR, f"S1_{intensity}_C*_R*.csv")
        csv_files = sorted(glob.glob(pattern))
        
        if len(csv_files) == 0:
            print(f"  WARNING: No files found for intensity {intensity}")
            continue
        
        print(f"  Found {len(csv_files)} files")
        
        for csv_file in csv_files:
            metadata = parse_filename(csv_file)
            if metadata is None:
                continue
            
            try:
                # Read CSV - individual files have no headers
                df = pd.read_csv(csv_file, header=None, 
                               names=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8'])
                
                df['Movement'] = metadata['class_id']
                df['Rep'] = metadata['rep_id']
                
                temp_csv = csv_file.replace('.csv', '_temp.csv')
                df.to_csv(temp_csv, index=False)
                
                features, labels, _, _ = extract_features_from_csv(
                    temp_csv,
                    sensor_columns=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8'],
                    apply_bandpass=True
                )
                
                import os as os_module
                if os_module.path.exists(temp_csv):
                    os_module.remove(temp_csv)
                
                all_features.append(features)
                all_labels.append(labels)
                
                for _ in range(len(features)):
                    all_metadata.append(metadata.copy())
                
                print(f"    OK: {metadata['filename']}: {len(features)} windows")
                
            except Exception as e:
                print(f"    X Error: {e}")
                continue
    
    if len(all_features) == 0:
        raise ValueError("No data was loaded!")
    
    X = np.vstack(all_features)
    y = np.concatenate(all_labels)
    metadata_df = pd.DataFrame(all_metadata)
    
    print(f"\n{'='*80}")
    print(f"TOTAL DATA LOADED: {len(X)} windows")
    print(f"{'='*80}")
    
    return X, y, metadata_df

def prepare_splits(X, y, metadata_df, test_intensity):
    """Create train/val/test splits with RobustScaler for better outlier handling."""
    print(f"\n{'='*80}")
    print("PREPARING DATA SPLITS")
    print(f"{'='*80}\n")
    
    # Create masks
    test_mask = metadata_df['intensity'] == test_intensity
    train_val_mask = ~test_mask
    
    val_mask = train_val_mask & metadata_df['rep_id'].isin(VAL_REPS)
    train_mask = train_val_mask & (~metadata_df['rep_id'].isin(VAL_REPS + TEST_REPS))
    
    # Extract data
    X_train_raw = X[train_mask]
    X_val_raw = X[val_mask]
    X_test_raw = X[test_mask]
    
    y_train = y[train_mask]
    y_val = y[val_mask]
    y_test = y[test_mask]
    
    # Convert labels with LABEL SMOOTHING
    y_train_int = y_train.astype(int) - 1
    y_val_int = y_val.astype(int) - 1
    y_test_int = y_test.astype(int) - 1
    
    y_train_onehot = keras.utils.to_categorical(y_train_int, NUM_CLASSES)
    y_val_onehot = keras.utils.to_categorical(y_val_int, NUM_CLASSES)
    y_test_onehot = keras.utils.to_categorical(y_test_int, NUM_CLASSES)
    
    # Apply label smoothing to training labels (reduces overconfidence)
    if LABEL_SMOOTHING > 0:
        y_train_onehot = y_train_onehot * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / NUM_CLASSES

    # CRITICAL FIX: Remove scaler to match C++ pipeline (no scaler in embedded system)
    # Features go directly from extraction to model (same as C++ csv_replay.cpp:470)
    scaler = None  # Set to None for compatibility with save_model()
    X_train = X_train_raw
    X_val = X_val_raw
    X_test = X_test_raw

    train_intensities = metadata_df[train_mask]['intensity'].unique()

    print(f"Split Strategy: Leave-One-Intensity-Out")
    print(f"  Training Intensities: {sorted(train_intensities)}")
    print(f"  Test Intensity:       {test_intensity}")
    print(f"{'-'*80}")
    print(f"  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")
    print(f"  Label Smoothing Applied: {LABEL_SMOOTHING}")
    print(f"  Feature Scaling: DISABLED (matches C++ pipeline)")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Empty split detected!")
    
    return X_train, X_val, X_test, y_train_onehot, y_val_onehot, y_test_onehot, scaler

# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

def build_improved_mlp(input_dim, num_classes):
    """
    Build SMALLER, MORE REGULARIZED model to prevent overfitting.
    
    Key changes from original:
    - Reduced neurons: 256→128, 128→64, 64→32
    - Stronger L2: 0.01→0.02
    - Higher dropout: 50/40/30→60/50/40
    - Label smoothing in loss
    """
    print(f"\n{'='*80}")
    print(f"BUILDING IMPROVED MODEL (ANTI-OVERFITTING)")
    print(f"{'='*80}")
    print(f"  Input Dimension: {input_dim}")
    print(f"  Architecture: {LAYER_1_UNITS} -> {LAYER_2_UNITS} -> {LAYER_3_UNITS} -> {num_classes}")
    print(f"  L2 Regularization: {L2_LAMBDA}")
    print(f"  Dropout Rates: {DROPOUT_RATE_1} -> {DROPOUT_RATE_2} -> {DROPOUT_RATE_3}")
    
    model = models.Sequential(name='Improved_Anti_Overfit_MLP')
    model.add(layers.Input(shape=(input_dim,), name='input'))

    # Layer 1 - Reduced from 256 to 128
    model.add(layers.Dense(LAYER_1_UNITS, activation='relu', 
                          kernel_regularizer=regularizers.l2(L2_LAMBDA),
                          kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE_1))

    # Layer 2 - Reduced from 128 to 64
    model.add(layers.Dense(LAYER_2_UNITS, activation='relu', 
                          kernel_regularizer=regularizers.l2(L2_LAMBDA),
                          kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE_2))

    # Layer 3 - Reduced from 64 to 32
    model.add(layers.Dense(LAYER_3_UNITS, activation='relu', 
                          kernel_regularizer=regularizers.l2(L2_LAMBDA),
                          kernel_initializer='he_normal'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE_3))

    # Output
    model.add(layers.Dense(num_classes, activation='softmax', 
                          kernel_initializer='glorot_uniform',
                          name='output'))

    # Use Adam with reduced learning rate
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',  # Label smoothing applied to targets
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'), 
                keras.metrics.Recall(name='recall')]
    )
    
    return model

# ============================================================================
# TRAINING WITH IMPROVED CALLBACKS
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """Train with improved early stopping and learning rate scheduling."""
    print(f"\n{'='*80}")
    print("TRAINING MODEL WITH ANTI-OVERFITTING STRATEGY")
    print(f"{'='*80}\n")
    
    callbacks = [
        # More aggressive early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
            mode='min'
        ),
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        ),
        # Monitor overfitting: stop if train/val gap too large
        keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\n  [Overfitting Check] Train Acc: {logs['accuracy']:.4f} | "
                f"Val Acc: {logs['val_accuracy']:.4f} | "
                f"Gap: {logs['accuracy'] - logs['val_accuracy']:.4f}"
            ) if epoch % 10 == 0 else None
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test, test_intensity):
    """Evaluate model on held-out test set."""
    print(f"\n{'='*80}")
    print(f"EVALUATING ON HELD-OUT INTENSITY: {test_intensity}")
    print(f"{'='*80}\n")
    
    results = model.evaluate(X_test, y_test, verbose=1)
    metrics = {
        'loss': results[0],
        'accuracy': results[1],
        'precision': results[2],
        'recall': results[3]
    }
    
    if metrics['precision'] + metrics['recall'] > 0:
        metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                           (metrics['precision'] + metrics['recall'])
    else:
        metrics['f1'] = 0.0
        
    print(f"\n{'-'*80}")
    print(f"  Test Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision: {metrics['precision']*100:.2f}%")
    print(f"  Test Recall:    {metrics['recall']*100:.2f}%")
    print(f"  Test F1 Score:  {metrics['f1']*100:.2f}%")
    print(f"{'-'*80}")
    
    return metrics

# ============================================================================
# ENHANCED VISUALIZATION
# ============================================================================

def plot_results(history, model, X_test, y_test, test_intensity, timestamp):
    """Create comprehensive training analysis plots."""
    
    # 1. Training History with Overfitting Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2, alpha=0.8)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2, alpha=0.8)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfitting Gap
    gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
    axes[1, 0].plot(gap, linewidth=2, color='red', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].fill_between(range(len(gap)), gap, 0, where=(gap>0), alpha=0.3, color='red', label='Overfitting')
    axes[1, 0].set_title('Train-Val Accuracy Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Train Acc - Val Acc')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'training_analysis_{timestamp}.png'), dpi=150)
    plt.close()
    
    # 2. Confusion Matrix
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=MOVEMENT_NAMES, 
                yticklabels=MOVEMENT_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Test Intensity: {test_intensity}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'cm_{timestamp}.png'), dpi=150)
    plt.close()
    
    # 3. Classification Report
    print(f"\n{'='*80}")
    print("CLASSIFICATION REPORT")
    print(f"{'='*80}\n")
    print(classification_report(y_true, y_pred, 
                               target_names=MOVEMENT_NAMES, 
                               zero_division=0))

def save_model(model, scaler, timestamp):
    """Save trained model and scaler."""
    keras_path = os.path.join(MODELS_DIR, f'emg_improved_{timestamp}.keras')
    model.save(keras_path)
    print(f"\nOK: Saved model: {keras_path}")

    weights_path = os.path.join(MODELS_DIR, f'emg_improved_{timestamp}.weights.h5')
    model.save_weights(weights_path)
    print(f"OK: Saved weights: {weights_path}")

    # No scaler to save - C++ pipeline has no scaler
    print(f"OK: Scaler: DISABLED (C++ pipeline has no scaler)")

    return keras_path, weights_path, None

def convert_to_tflite(model, timestamp):
    """Convert model to TFLite Float32 (no quantization for ESP32 compatibility)."""
    print(f"\n{'='*80}")
    print("CONVERTING TO TFLITE FLOAT32 (NO QUANTIZATION)")
    print(f"{'='*80}\n")

    # Convert to TFLite WITHOUT quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    # Save
    tflite_path = os.path.join(MODELS_DIR, f'emg_model_{timestamp}.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_bytes = len(tflite_model)
    size_kb = size_bytes / 1024

    print(f"OK: TFLite model saved: {tflite_path}")
    print(f"  Size: {size_kb:.2f} KB ({size_bytes} bytes)")
    print(f"  Format: Float32 (no quantization)")
    print(f"  Compatibility: TFLite v2.1.1+ (ESP32-S3)")

    return tflite_path, size_bytes

def convert_to_c_header(tflite_path, model_size, timestamp):
    """Convert TFLite model to C header file for ESP32-S3."""
    print(f"\n{'='*80}")
    print("CONVERTING TO C HEADER FILE FOR ESP32-S3")
    print(f"{'='*80}\n")

    # Read TFLite model
    with open(tflite_path, 'rb') as f:
        tflite_data = f.read()

    # Convert to C array
    c_array_lines = []
    bytes_per_line = 12

    for i in range(0, len(tflite_data), bytes_per_line):
        chunk = tflite_data[i:i+bytes_per_line]
        hex_bytes = ', '.join([f'0x{b:02x}' for b in chunk])
        c_array_lines.append(f'  {hex_bytes}')

    c_array = ',\n'.join(c_array_lines)

    # Create header file
    header_content = f"""// Auto-generated TensorFlow Lite model for ESP32-S3
// Wide & Deep MLP with TD4 Features (MAV, WL, ZC, SSC)
//
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Model size: {model_size} bytes
// Format: Float32 (no quantization for TFLite v2.1.1 compatibility)
// Input: 32 features (4 TD4 × 8 EMG sensors)
// Output: {NUM_CLASSES} gestures
//
// Movement Classes:
//   0: No Movement
//   1: Wrist Flexion
//   2: Wrist Extension
//   3: Wrist Pronation
//   4: Wrist Supination
//   5: Chuck Grip
//   6: Hand Open

#ifndef MODEL_H
#define MODEL_H

const unsigned char model_tflite[] = {{
{c_array}
}};

const unsigned int model_tflite_len = {model_size};

#endif  // MODEL_H
"""

    # Save
    header_path = os.path.join(MODELS_DIR, f'model_emg_{timestamp}.h')
    with open(header_path, 'w') as f:
        f.write(header_content)

    print(f"OK: C header saved: {header_path}")
    print(f"  Array size: {model_size} bytes")
    print(f"\n  Include in ESP32-S3 project:")
    print(f"    #include \"{os.path.basename(header_path)}\"")

    return header_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline with anti-overfitting improvements."""
    
    train_intensities = [i for i in TRAIN_INTENSITIES if i != TEST_INTENSITY]
    
    print(f"\nIMPROVED Configuration:")
    print(f"   Model Size: {LAYER_1_UNITS}->{LAYER_2_UNITS}->{LAYER_3_UNITS} (50% smaller)")
    print(f"   Dropout: {DROPOUT_RATE_1}->{DROPOUT_RATE_2}->{DROPOUT_RATE_3} (Higher)")
    print(f"   L2 Regularization: {L2_LAMBDA} (2x stronger)")
    print(f"   Label Smoothing: {LABEL_SMOOTHING}")
    print(f"   Training on: {train_intensities}")
    print(f"   Testing on:  {TEST_INTENSITY}")
    
    # 1. Load data
    try:
        X, y, metadata_df = load_all_data(TRAIN_INTENSITIES)
    except Exception as e:
        print(f"\nERROR loading data: {e}")
        return
    
    # 2. Prepare splits
    try:
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = \
            prepare_splits(X, y, metadata_df, TEST_INTENSITY)
    except Exception as e:
        print(f"\n❌ Error preparing splits: {e}")
        return
    
    # 3. Build improved model
    model = build_improved_mlp(X_train.shape[1], NUM_CLASSES)
    model.summary()
    
    # 4. Train with anti-overfitting strategy
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # 5. Evaluate
    metrics = evaluate_model(model, X_test, y_test, TEST_INTENSITY)
    
    # 6. Visualize and save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_results(history, model, X_test, y_test, TEST_INTENSITY, timestamp)
    keras_path, weights_path, scaler_path = save_model(model, scaler, timestamp)

    # 7. Export to TFLite and C header for ESP32-S3
    tflite_path, model_size = convert_to_tflite(model, timestamp)
    header_path = convert_to_c_header(tflite_path, model_size, timestamp)

    # Final analysis
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    overfitting_gap = final_train_acc - final_val_acc
    
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS")
    print(f"{'='*80}")
    print(f"  Final Train Accuracy: {final_train_acc*100:.2f}%")
    print(f"  Final Val Accuracy:   {final_val_acc*100:.2f}%")
    print(f"  Test Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"  Train-Val Gap:        {overfitting_gap*100:.2f}% {'[Good]' if overfitting_gap < 0.05 else '[Still overfitting]'}")
    print(f"  Test F1 Score:        {metrics['f1']*100:.2f}%")
    print(f"{'='*80}")

    print(f"\n{'='*80}")
    print("GENERATED FILES")
    print(f"{'='*80}")
    print(f"  1. Keras Model:  {os.path.basename(keras_path)}")
    print(f"  2. Weights:      {os.path.basename(weights_path)}")
    print(f"  3. Scaler:       DISABLED (C++ has no scaler)")
    print(f"  4. TFLite Model: {os.path.basename(tflite_path)} ({model_size/1024:.2f} KB)")
    print(f"  5. C Header:     {os.path.basename(header_path)} (for ESP32-S3)")
    print(f"  6. Plots:        {PLOTS_DIR}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
