"""
Wide & Deep MLP Training for ESP32-S3 Bionic Hand with TD4 Features
=====================================================================

Implements a literature-based Wide & Deep Multi-Layer Perceptron architecture
optimized for EMG gesture classification on embedded systems.

Architecture:
- Input: 24 TD4 features (4 features × 6 EMG sensors)
- Wide & Deep layers: 256 → 128 → 64 neurons
- Dropout regularization for robustness
- Post-Training Int8 Quantization for ESP32 deployment

Features:
- Hudgins' TD4 features (MAV, WL, ZC, SSC)
- Optimized for TensorFlow Lite Micro
- Automatic C header generation for ESP32-S3

References:
- Cheng et al. (2016): "Wide & Deep Learning for Recommender Systems"
- Atzori et al. (2014): "Electromyography data for non-invasive naturally-controlled robotic hand prostheses"

Author: ESP32 Bionic Hand Project
Date: 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
from glob import glob
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

# Training parameters
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Model architecture
LAYER_1_UNITS = 256
LAYER_2_UNITS = 128
LAYER_3_UNITS = 64
DROPOUT_RATE_1 = 0.3
DROPOUT_RATE_2 = 0.2
DROPOUT_RATE_3 = 0.1

# Early stopping
EARLY_STOPPING_PATIENCE = 15

# Paths - Organized under scripts_ai/data/
MODELS_DIR = 'scripts_ai/data/models/'
PLOTS_DIR = 'scripts_ai/data/plots/'
FEATURES_DIR = 'scripts_ai/data/features/'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Gesture mapping
# CRITICAL: Gesture order MUST match data collection script and C++ code!
# This order is used for label encoding and MUST NOT be changed.
# Alphabetical sorting causes label mismatch between training and inference!
GESTURE_NAMES_FIXED = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]
GESTURE_NAMES = None  # Will be populated in load_npz_features()
NUM_GESTURES = None  # Will be populated in load_npz_features()


# ============================================================================
# DATA LOADING
# ============================================================================

def find_latest_features_npz():
    """
    Find the latest TD4 features NPZ file

    Returns:
        str: Path to latest NPZ file or None
    """
    npz_files = glob(os.path.join(FEATURES_DIR, '*_td4_features.npz'))

    if not npz_files:
        print("❌ No *_td4_features.npz files found in data/features/")
        print("   Run feature_extraction.py first!")
        return None

    latest = max(npz_files, key=os.path.getctime)
    return latest


def load_npz_features(npz_path):
    """
    Load TD4 features from NPZ file

    CRITICAL FIX: Uses FIXED gesture order (GESTURE_NAMES_FIXED) instead of
    alphabetical sorting to match data collection script and C++ code.
    This prevents label mismatch between training and ESP32 inference!

    Args:
        npz_path: Path to NPZ file

    Returns:
        Tuple of (features, labels, feature_names, num_classes)
    """
    global GESTURE_NAMES, NUM_GESTURES

    print(f"\n{'='*80}")
    print(f"Loading TD4 features from NPZ")
    print(f"{'='*80}\n")

    data = np.load(npz_path, allow_pickle=True)

    features = data['features']
    labels = data['labels']
    feature_names = data['feature_names']
    sensor_columns = data['sensor_columns']

    print(f"File: {os.path.basename(npz_path)}")
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Sensors: {sensor_columns}")

    # ============================================================================
    # CRITICAL FIX: Use FIXED gesture order (matches data collection & C++ code)
    # ============================================================================
    # Verify all expected gestures are present in the data
    unique_labels_in_data = np.unique(labels)

    print(f"\n{'='*60}")
    print(f"VERIFYING GESTURE NAMES:")
    print(f"{'='*60}")

    missing_gestures = []
    for gesture in GESTURE_NAMES_FIXED:
        if gesture not in unique_labels_in_data:
            missing_gestures.append(gesture)
            print(f"  ⚠️  WARNING: '{gesture}' not found in data!")

    extra_gestures = []
    for gesture in unique_labels_in_data:
        if gesture not in GESTURE_NAMES_FIXED:
            extra_gestures.append(gesture)
            print(f"  ⚠️  WARNING: '{gesture}' found in data but not in GESTURE_NAMES_FIXED!")

    if missing_gestures or extra_gestures:
        print(f"\n❌ ERROR: Gesture name mismatch detected!")
        print(f"   Expected: {GESTURE_NAMES_FIXED}")
        print(f"   Found in data: {list(unique_labels_in_data)}")
        raise ValueError("Gesture names in data do not match GESTURE_NAMES_FIXED!")

    # Use fixed order (matches C++ and data collection)
    GESTURE_NAMES = GESTURE_NAMES_FIXED
    NUM_GESTURES = len(GESTURE_NAMES)

    print(f"\n{'='*60}")
    print(f"GESTURE NAMES (FIXED ORDER - Matches C++ & Data Collection):")
    print(f"{'='*60}")
    for idx, label in enumerate(GESTURE_NAMES):
        print(f"  [{idx}] {label}")
    print(f"{'='*60}")

    print(f"\nClass distribution:")
    for label in GESTURE_NAMES:
        count = np.sum(labels == label)
        percentage = (count / len(labels)) * 100
        print(f"  {label}: {count} samples ({percentage:.1f}%)")

    print(f"\nTotal classes: {NUM_GESTURES}")

    return features, labels, feature_names, NUM_GESTURES


def prepare_data(features, labels, num_classes):
    """
    Prepare data for training

    CRITICAL FIX: Uses GESTURE_NAMES (fixed order) for label mapping
    instead of alphabetical np.unique() to match C++ inference order.

    Args:
        features: Feature array (n_samples × n_features)
        labels: Label array (n_samples,)
        num_classes: Number of classes

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print(f"\n{'='*80}")
    print("Preparing data splits")
    print(f"{'='*80}\n")

    # Convert labels to integers using FIXED gesture order
    # This ensures index 0 = 'Rest', index 1 = 'Fist', etc. (matches C++)
    label_map = {label: idx for idx, label in enumerate(GESTURE_NAMES)}
    labels_int = np.array([label_map[label] for label in labels])

    print(f"Label mapping (FIXED ORDER - matches C++):")
    for label, idx in label_map.items():
        print(f"  '{label}' → {idx}")

    # One-hot encode labels
    labels_onehot = keras.utils.to_categorical(labels_int, num_classes)

    # Split: train + temp (for validation + test)
    # CRITICAL FIX: Disable shuffle to prevent temporal data leakage
    # Consecutive windows from same gesture should NOT be split between train/test
    # This was causing artificially high accuracy (~37%) even on pure noise data
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels_onehot,
        test_size=(VALIDATION_SPLIT + TEST_SPLIT),
        random_state=42,
        shuffle=False,  # FIXED: Preserve temporal ordering to prevent leakage
        stratify=None   # FIXED: Cannot use stratify when shuffle=False
    )

    # Split temp into validation and test (also without shuffle)
    val_ratio = VALIDATION_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=(1 - val_ratio),
        random_state=42,
        shuffle=False  # FIXED: Maintain temporal ordering
    )

    print(f"Data splits:")
    print(f"  Training:   {X_train.shape[0]} samples ({(1-VALIDATION_SPLIT-TEST_SPLIT)*100:.0f}%)")
    print(f"  Validation: {X_val.shape[0]} samples ({VALIDATION_SPLIT*100:.0f}%)")
    print(f"  Test:       {X_test.shape[0]} samples ({TEST_SPLIT*100:.0f}%)")
    print(f"\nFeature dimension: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_wide_deep_mlp(input_dim, num_classes):
    """
    Build Wide & Deep MLP for EMG classification

    Architecture inspired by Wide & Deep learning (Cheng et al., 2016)
    but adapted for time-series classification.

    Layers:
        Input (24) → Dense(256) → Dropout(0.3) →
        Dense(128) → Dropout(0.2) →
        Dense(64) → Dropout(0.1) →
        Output(num_classes)

    Args:
        input_dim: Number of input features (should be 24 for TD4)
        num_classes: Number of output classes (11 gestures)

    Returns:
        Compiled Keras model
    """
    print(f"\n{'='*80}")
    print("Building Wide & Deep MLP Architecture")
    print(f"{'='*80}\n")

    model = models.Sequential(name='Wide_Deep_MLP')

    # Input layer
    model.add(layers.Input(shape=(input_dim,), name='input'))

    # Wide & Deep Layer 1: 256 neurons
    model.add(layers.Dense(
        LAYER_1_UNITS,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_256'
    ))
    model.add(layers.Dropout(DROPOUT_RATE_1, name='dropout_1'))

    # Deep Layer 2: 128 neurons
    model.add(layers.Dense(
        LAYER_2_UNITS,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_128'
    ))
    model.add(layers.Dropout(DROPOUT_RATE_2, name='dropout_2'))

    # Deep Layer 3: 64 neurons
    model.add(layers.Dense(
        LAYER_3_UNITS,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_64'
    ))
    model.add(layers.Dropout(DROPOUT_RATE_3, name='dropout_3'))

    # Output layer
    model.add(layers.Dense(
        num_classes,
        activation='softmax',
        name='output'
    ))

    print("Model architecture:")
    model.summary()

    # Calculate total parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")

    return model


def compile_model(model):
    """
    Compile model with optimizer and loss function

    Args:
        model: Keras model

    Returns:
        Compiled model
    """
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]
    )

    print(f"\nModel compiled:")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Loss: categorical_crossentropy")
    print(f"  Metrics: accuracy, precision, recall")

    return model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the model with callbacks

    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Training history
    """
    print(f"\n{'='*80}")
    print("Training Wide & Deep MLP")
    print(f"{'='*80}\n")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"Training parameters:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print()

    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Training completed in {training_time:.1f}s ({training_time/60:.1f} min)")
    print(f"{'='*80}")

    return history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set

    Args:
        model: Trained Keras model
        X_test, y_test: Test data

    Returns:
        Test metrics dictionary
    """
    print(f"\n{'='*80}")
    print("Evaluating on test set")
    print(f"{'='*80}\n")

    test_results = model.evaluate(X_test, y_test, verbose=1)

    metrics = {
        'loss': test_results[0],
        'accuracy': test_results[1],
        'precision': test_results[2],
        'recall': test_results[3]
    }

    print(f"\nTest Results:")
    print(f"  Loss:      {metrics['loss']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    return metrics


def plot_training_history(history, save_path):
    """
    Plot and save training history

    Args:
        history: Keras training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  OK Saved training history: {save_path}")


def plot_confusion_matrix(model, X_test, y_test, save_path):
    """
    Plot and save confusion matrix

    Args:
        model: Trained model
        X_test, y_test: Test data
        save_path: Path to save plot
    """
    # Predict
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GESTURE_NAMES,
                yticklabels=GESTURE_NAMES)
    plt.title('Confusion Matrix - TD4 Wide & Deep MLP', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  OK Saved confusion matrix: {save_path}")

    # Classification report
    print("\nClassification Report:")
    # Get unique labels present in test set
    unique_labels = sorted(set(y_true) | set(y_pred))
    target_names_present = [GESTURE_NAMES[i] for i in unique_labels]
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names_present, zero_division=0))


# ============================================================================
# MODEL EXPORT
# ============================================================================

def convert_to_tflite_float32(model, output_path):
    """
    Convert model to TFLite with Float32 (NO quantization)

    This ensures maximum compatibility with older TFLite libraries on ESP32.
    Model will be larger but more compatible.

    Args:
        model: Trained Keras model
        output_path: Path to save TFLite model

    Returns:
        Size of model in KB
    """
    print(f"\n{'='*80}")
    print("Converting to TFLite Float32 (No Quantization)")
    print(f"{'='*80}\n")

    # Convert to TFLite WITHOUT quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # NO quantization - use Float32 for maximum ESP32 compatibility
    # This works with TFLite v2.1.1 on ESP32 without operator version issues
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    # Float32 input/output (default, but explicit for clarity)
    # No need to set inference_input_type or inference_output_type

    tflite_model = converter.convert()

    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024

    print(f"✅ Float32 TFLite model saved: {output_path}")
    print(f"  Size: {size_kb:.2f} KB")
    print(f"  Format: Float32 (no quantization)")
    print(f"  Compatibility: TFLite v2.1.1+ (ESP32-S3)")

    return size_kb


def convert_to_c_header(tflite_path, output_path):
    """
    Convert TFLite model to C header file for ESP32-S3

    Args:
        tflite_path: Path to TFLite model
        output_path: Path to save .h file
    """
    print(f"\nConverting to C header file for ESP32-S3...")

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
// Model size: {len(tflite_data)} bytes
// Format: Float32 (no quantization for TFLite v2.1.1 compatibility)
// Input: 24 features (4 TD4 × 6 EMG sensors)
// Output: {NUM_GESTURES} gestures

#ifndef MODEL_H
#define MODEL_H

const unsigned char model_tflite[] = {{
{c_array}
}};

const unsigned int model_tflite_len = {len(tflite_data)};

#endif  // MODEL_H
"""

    # Save
    with open(output_path, 'w') as f:
        f.write(header_content)

    print(f"OK C header saved: {output_path}")
    print(f"  Array size: {len(tflite_data)} bytes")
    print(f"\n  Include in ESP32-S3 project:")
    print(f"    #include \"{os.path.basename(output_path)}\"")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main training pipeline
    """
    print("\n" + "="*80)
    print("ESP32-S3 Bionic Hand - Wide & Deep MLP Training".center(80))
    print("TD4 Features (MAV, WL, ZC, SSC) - Literature-Based Approach".center(80))
    print("="*80)

    # Find and load features
    npz_path = find_latest_features_npz()
    if npz_path is None:
        return

    features, labels, feature_names, num_classes = load_npz_features(npz_path)

    # Verify feature count
    expected_features = 24  # 4 TD4 features × 6 sensors
    if features.shape[1] != expected_features:
        print(f"\n⚠️  WARNING: Expected {expected_features} features, got {features.shape[1]}")
        print("   Make sure you're using 6 EMG sensors with TD4 features")

    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        features, labels, num_classes
    )

    # Build model
    model = build_wide_deep_mlp(X_train.shape[1], num_classes)
    model = compile_model(model)

    # Train
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Plot results
    print(f"\n{'='*80}")
    print("Generating visualizations")
    print(f"{'='*80}\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    plot_training_history(
        history,
        os.path.join(PLOTS_DIR, f'training_history_{timestamp}.png')
    )

    plot_confusion_matrix(
        model,
        X_test, y_test,
        os.path.join(PLOTS_DIR, f'confusion_matrix_{timestamp}.png')
    )

    # Export models
    print(f"\n{'='*80}")
    print("Exporting models")
    print(f"{'='*80}\n")

    # Save Keras model
    keras_model_path = os.path.join(MODELS_DIR, f'mlp_td4_{timestamp}.keras')
    model.save(keras_model_path)
    print(f"OK Keras model saved: {keras_model_path}")

    # Convert to Float32 TFLite (no quantization for ESP32 compatibility)
    tflite_path = os.path.join(MODELS_DIR, f'mlp_td4_{timestamp}_float32.tflite')
    model_size_kb = convert_to_tflite_float32(model, tflite_path)

    # Convert to C header
    header_path = os.path.join(MODELS_DIR, f'mlp_td4_{timestamp}_float32.h')
    convert_to_c_header(tflite_path, header_path)

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!".center(80))
    print(f"{'='*80}\n")

    print(f"Model Performance:")
    print(f"  Test Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"  Test Precision: {metrics['precision']:.4f}")
    print(f"  Test Recall:    {metrics['recall']:.4f}")

    print(f"\nGenerated Files:")
    print(f"  1. Keras model:     {keras_model_path}")
    print(f"  2. TFLite (Float32): {tflite_path} ({model_size_kb:.2f} KB)")
    print(f"  3. C header:        {header_path}")
    print(f"  4. Training plots:  {PLOTS_DIR}")

    print(f"\nNext Steps:")
    print(f"  1. Copy {os.path.basename(header_path)} to src/")
    print(f"  2. Update ESP32-S3 code to use TD4 features")
    print(f"  3. Ensure feature extraction matches training (MAV, WL, ZC, SSC)")
    print(f"  4. Upload and test on hardware")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
