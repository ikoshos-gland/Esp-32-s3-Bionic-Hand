"""
CSV Replay System for ESP32 EMG Model Validation
=================================================

Streams pre-recorded EMG data from CSV files to ESP32 for offline model
validation without requiring physical sensors.

Features:
- Loads CSV data (EMG1-6 columns + Movement labels)
- Scales normalized floats (-1, +1) to 12-bit ADC integers (0-4095)
- Streams 250-sample windows to ESP32 via serial binary protocol
- Receives predictions and features from ESP32
- Validates against ground truth labels
- Generates accuracy metrics and confusion matrix

Protocol:
- Window Packet (Python → ESP32): 3007 bytes
  [Header(2) + GroundTruth(1) + WindowSize(2) + Data(3000) + Checksum(2)]
- Response Packet (ESP32 → Python): 106 bytes
  [Header(2) + Predicted(1) + Confidence(4) + Features(96) + GT_Echo(1) + Checksum(2)]

Author: ESP32 Bionic Hand Project
Date: 2025-11-29
"""

import numpy as np
import pandas as pd
import serial
import struct
import time
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class CSVReplaySystem:
    """
    CSV Replay System for offline ESP32 model validation

    Streams pre-recorded EMG data to ESP32 and validates model predictions
    against ground truth labels from CSV files.
    """

    # Protocol constants - Real-Time Streaming Mode
    SAMPLE_HEADER = bytes([0xAA, 0x55])  # Sample packet header
    RESPONSE_HEADER = bytes([0xB5, 0x6B])  # Response packet header
    WINDOW_SIZE = 250  # Samples per window
    NUM_SENSORS = 8  # Using all 8 EMG sensors for model validation
    NUM_FEATURES = 32  # 4 TD4 features × 8 sensors

    # Gesture names (MUST match model exactly!)
    GESTURE_NAMES = [
        "No Movement", "Wrist Flexion", "Wrist Extension",
        "Wrist Pronation", "Wrist Supination", "Chuck Grip", "Hand Open"
    ]

    # ADC constants
    ADC_MIN = 0
    ADC_MAX = 4095

    def __init__(self, csv_path: str, serial_port: str, baud_rate: int = 921600):
        """
        Initialize CSV Replay System

        Args:
            csv_path: Path to CSV file containing EMG data
            serial_port: Serial port for ESP32 (e.g., 'COM11')
            baud_rate: Serial baud rate (default: 921600)
        """
        self.csv_path = csv_path
        self.serial_port = serial_port
        self.baud_rate = baud_rate

        # Data storage
        self.emg_data = None
        self.ground_truth_labels = None
        self.num_windows = 0

        # Serial connection
        self.ser = None

        # Results storage
        self.predictions = []
        self.confidences = []
        self.features_log = []
        self.ground_truths = []
        self.correct_count = 0

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.path.dirname(csv_path), "..", "csv_replay_logs")
        os.makedirs(log_dir, exist_ok=True)

        self.log_file = os.path.join(log_dir, f"replay_{timestamp}.log")
        self.csv_summary_file = os.path.join(log_dir, f"replay_{timestamp}_summary.csv")
        self.confusion_matrix_file = os.path.join(log_dir, f"confusion_matrix_{timestamp}.png")

        print("=" * 70)
        print("CSV Replay System Initialized")
        print("=" * 70)
        print(f"CSV File: {csv_path}")
        print(f"Serial Port: {serial_port} @ {baud_rate} baud")
        print(f"Log File: {self.log_file}")
        print(f"CSV Summary: {self.csv_summary_file}")
        print(f"Confusion Matrix: {self.confusion_matrix_file}")
        print("=" * 70)
        print()

    def load_csv(self) -> bool:
        """
        Load CSV file and extract EMG data and ground truth labels

        CSV Format:
        - Columns: Timestamp, Seq, Movement, Phase, Rep, EMG1-EMG8
        - EMG values: Normalized floats (-1.0 to +1.0)
        - Movement labels: Integer 1-7 (maps to 0-6 for model)

        Returns:
            True if successful, False otherwise
        """
        try:
            print("Loading CSV file...")
            df = pd.read_csv(self.csv_path)

            # Extract EMG1-8 columns (using all 8 sensors for model validation)
            emg_columns = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']

            # Verify columns exist
            missing_cols = [col for col in emg_columns if col not in df.columns]
            if missing_cols:
                print(f"ERROR: Missing EMG columns: {missing_cols}")
                return False

            if 'Movement' not in df.columns:
                print("ERROR: Missing 'Movement' column")
                return False

            # Extract data
            self.emg_data = df[emg_columns].values  # Shape: (samples, 8)

            # Convert Movement labels (1-7) to model indices (0-6)
            # CSV Movement: 1=No Movement, 2=Wrist Flexion, ..., 7=Hand Open
            # Model indices: 0=No Movement, 1=Wrist Flexion, ..., 6=Hand Open
            self.ground_truth_labels = (df['Movement'].values - 1).astype(np.int8)

            # Calculate number of non-overlapping windows
            total_samples = len(self.emg_data)
            self.num_windows = total_samples // self.WINDOW_SIZE

            print(f"OK: CSV loaded successfully")
            print(f"  Total samples: {total_samples:,}")
            print(f"  EMG sensors: {len(emg_columns)}")
            print(f"  Non-overlapping windows: {self.num_windows}")
            print(f"  Movement labels range: {df['Movement'].min()}-{df['Movement'].max()}")
            print(f"  ESP32 indices range: {self.ground_truth_labels.min()}-{self.ground_truth_labels.max()}")
            print(f"  EMG value range: [{self.emg_data.min():.3f}, {self.emg_data.max():.3f}]")
            print()

            return True

        except Exception as e:
            print(f"ERROR loading CSV: {e}")
            return False

    def scale_to_adc(self, emg_normalized: np.ndarray) -> np.ndarray:
        """
        Scale normalized EMG values to 12-bit ADC range

        Formula: ADC = (EMG + 1.0) × 2047.5

        This preserves:
        - DC offset around 2048 (realistic for ESP32 ADC)
        - Bipolar signal characteristics
        - Amplitude relationships for MAV/WL features
        - Valid ZC/SSC threshold behavior (15 ADC units)

        Args:
            emg_normalized: Normalized EMG values (-1.0 to +1.0)

        Returns:
            ADC values (0-4095) as uint16
        """
        # Scale from [-1, +1] to [0, 4095]
        adc_float = (emg_normalized + 1.0) * 2047.5

        # Clip to valid range and convert to uint16
        adc_int = np.clip(adc_float, self.ADC_MIN, self.ADC_MAX).astype(np.uint16)

        return adc_int

    def send_sample(self, sample_data: np.ndarray, ground_truth: int) -> bool:
        """
        Send a single EMG sample to ESP32 via serial (simulates real-time ADC)

        Sample Packet Format (21 bytes):
        - Header (2 bytes): 0xAA, 0x55
        - Sensors (16 bytes): 8 sensors × 2 bytes (uint16, little-endian)
        - Ground Truth (1 byte): 0-6
        - Checksum (2 bytes): sum(sensor_bytes) % 65536 (little-endian uint16)

        Args:
            sample_data: Single sample data (8,) in ADC units (uint16)
            ground_truth: Ground truth gesture label (0-6)

        Returns:
            True if sample sent successfully
        """
        try:
            # Build packet
            packet = bytearray()

            # Header
            packet.extend(self.SAMPLE_HEADER)

            # Sensors (8 × uint16, little-endian)
            sensor_bytes = bytearray()
            for sensor_idx in range(self.NUM_SENSORS):
                adc_value = sample_data[sensor_idx]
                sensor_bytes.extend(struct.pack('<H', adc_value))

            packet.extend(sensor_bytes)

            # Ground truth (uint8)
            packet.append(ground_truth)

            # Checksum (sum of sensor bytes % 65536)
            checksum = sum(sensor_bytes) % 65536
            packet.extend(struct.pack('<H', checksum))

            # Verify packet size (21 bytes)
            expected_size = 2 + 16 + 1 + 2
            if len(packet) != expected_size:
                print(f"ERROR: Sample packet size mismatch ({len(packet)} != {expected_size})")
                return False

            # Send packet
            self.ser.write(packet)
            self.ser.flush()

            return True

        except Exception as e:
            print(f"ERROR sending sample: {e}")
            return False

    def receive_response(self, timeout: float = 2.0) -> Optional[Dict]:
        """
        Receive and parse response packet from ESP32

        Response Packet Format (138 bytes):
        - Header (2 bytes): 0xB5, 0x6B
        - Predicted Gesture (1 byte): 0-6 or 255 (no prediction)
        - Confidence (4 bytes): Float32
        - Features (128 bytes): 32 × Float32 (TD4 features for 8 sensors)
        - Ground Truth Echo (1 byte): Echo of sent ground truth
        - Checksum (2 bytes): Validation

        Args:
            timeout: Timeout in seconds

        Returns:
            Dictionary with response data, or None if error/timeout
        """
        try:
            # Set timeout
            self.ser.timeout = timeout

            # Read header
            header = self.ser.read(2)
            if len(header) != 2:
                print("ERROR: Timeout reading response header")
                return None

            if header != self.RESPONSE_HEADER:
                print(f"ERROR: Invalid response header: {header.hex()}")
                return None

            # Read predicted gesture (uint8)
            pred_byte = self.ser.read(1)
            if len(pred_byte) != 1:
                print("ERROR: Timeout reading predicted gesture")
                return None
            predicted_gesture = pred_byte[0]

            # Read confidence (float32)
            conf_bytes = self.ser.read(4)
            if len(conf_bytes) != 4:
                print("ERROR: Timeout reading confidence")
                return None
            confidence = struct.unpack('<f', conf_bytes)[0]

            # Read features (24 × float32)
            features = []
            for i in range(self.NUM_FEATURES):
                feat_bytes = self.ser.read(4)
                if len(feat_bytes) != 4:
                    print(f"ERROR: Timeout reading feature {i}")
                    return None
                feature_val = struct.unpack('<f', feat_bytes)[0]
                features.append(feature_val)

            # Read ground truth echo (uint8)
            gt_echo_byte = self.ser.read(1)
            if len(gt_echo_byte) != 1:
                print("ERROR: Timeout reading ground truth echo")
                return None
            ground_truth_echo = gt_echo_byte[0]

            # Read checksum (uint16)
            checksum_bytes = self.ser.read(2)
            if len(checksum_bytes) != 2:
                print("ERROR: Timeout reading checksum")
                return None
            received_checksum = struct.unpack('<H', checksum_bytes)[0]

            # Calculate expected checksum (sum of all data bytes before checksum)
            # Data: pred(1) + conf(4) + features(128) + gt_echo(1) = 134 bytes
            data_for_checksum = bytearray()
            data_for_checksum.append(predicted_gesture)
            data_for_checksum.extend(conf_bytes)
            for feat in features:
                data_for_checksum.extend(struct.pack('<f', feat))
            data_for_checksum.append(ground_truth_echo)

            expected_checksum = sum(data_for_checksum) % 65536

            if received_checksum != expected_checksum:
                print(f"WARNING: Checksum mismatch (recv={received_checksum}, expected={expected_checksum})")
                # Continue anyway for debugging

            # Handle no prediction (255 = no prediction)
            if predicted_gesture == 255:
                predicted_gesture = -1

            return {
                'predicted_gesture': predicted_gesture,
                'confidence': confidence,
                'features': features,
                'ground_truth_echo': ground_truth_echo,
                'checksum_valid': (received_checksum == expected_checksum)
            }

        except Exception as e:
            print(f"ERROR receiving response: {e}")
            return None

    def open_serial(self) -> bool:
        """
        Open serial connection to ESP32

        Returns:
            True if successful
        """
        try:
            print(f"Opening serial port {self.serial_port}...")
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baud_rate,
                timeout=2.0,
                write_timeout=2.0
            )

            # Increase buffer sizes to prevent overflow
            self.ser.set_buffer_size(rx_size=8192, tx_size=8192)

            # Wait for ESP32 to stabilize
            time.sleep(2)

            # Flush buffers
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            print("OK: Serial port opened successfully")
            print()
            return True

        except Exception as e:
            print(f"ERROR opening serial port: {e}")
            return False

    def close_serial(self):
        """Close serial connection"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Serial port closed")

    def run_replay(self) -> bool:
        """
        Main replay loop - stream all windows and validate predictions

        Returns:
            True if replay completed successfully
        """
        try:
            # Open log files
            log_f = open(self.log_file, 'w')
            csv_f = open(self.csv_summary_file, 'w')

            # Write CSV header
            feature_headers = []
            for sensor in range(1, self.NUM_SENSORS + 1):
                feature_headers.extend([f"MAV{sensor}", f"WL{sensor}", f"ZC{sensor}", f"SSC{sensor}"])

            csv_header = "Window,GroundTruth,Predicted,Confidence,Correct," + ",".join(feature_headers) + "\n"
            csv_f.write(csv_header)

            # Write log header
            log_f.write("=" * 80 + "\n")
            log_f.write("CSV Replay Log\n")
            log_f.write(f"CSV File: {self.csv_path}\n")
            log_f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_f.write("=" * 80 + "\n\n")

            print("=" * 70)
            print("Starting Replay...")
            print("=" * 70)
            print()

            # Process each window
            for window_idx in range(self.num_windows):
                # Extract window
                start_idx = window_idx * self.WINDOW_SIZE
                end_idx = start_idx + self.WINDOW_SIZE

                # Get normalized EMG data for this window
                emg_window_norm = self.emg_data[start_idx:end_idx, :]  # (250, 8)

                # Get ground truth (majority vote in window)
                gt_window = self.ground_truth_labels[start_idx:end_idx]
                ground_truth = np.bincount(gt_window).argmax()  # Most common label

                # Scale to ADC
                emg_window_adc = self.scale_to_adc(emg_window_norm)

                # DIAGNOSTIC: Log first window details (Python side)
                if window_idx == 0:
                    print("\n=== PYTHON DIAGNOSTIC: First Window ===")
                    print(f"Normalized EMG (first 5 samples, all sensors):")
                    for i in range(min(5, len(emg_window_norm))):
                        print(f"  Sample {i}: {emg_window_norm[i]}")
                    print(f"\nScaled ADC (first 5 samples, all sensors):")
                    for i in range(min(5, len(emg_window_adc))):
                        print(f"  Sample {i}: {emg_window_adc[i]}")
                    print("=== END PYTHON DIAGNOSTIC ===\n")

                # Send samples one-by-one (simulates real-time ADC)
                for sample_idx in range(self.WINDOW_SIZE):
                    sample = emg_window_adc[sample_idx, :]  # (8,)
                    
                    if not self.send_sample(sample, ground_truth):
                        print(f"ERROR: Failed to send sample {sample_idx} of window {window_idx}")
                        break
                    
                    # Small delay to simulate real-time sampling rate
                    # At 1000 Hz sampling: 1ms between samples
                    time.sleep(0.001)  # 1ms delay

                # After 250 samples sent, ESP32 should send response
                response = self.receive_response(timeout=5.0)
                if response is None:
                    print(f"ERROR: Failed to receive response for window {window_idx}")
                    continue

                # Validate ground truth echo
                if response['ground_truth_echo'] != ground_truth:
                    print(f"WARNING: Ground truth mismatch (sent={ground_truth}, echo={response['ground_truth_echo']})")

                # Extract results
                predicted = response['predicted_gesture']
                confidence = response['confidence']
                features = response['features']

                # DIAGNOSTIC: Log first window response (Python side)
                if window_idx == 0:
                    print("\n=== PYTHON DIAGNOSTIC: ESP32 Response ===")
                    print(f"Ground truth: {ground_truth}")
                    print(f"Predicted: {predicted}")
                    print(f"Confidence: {confidence:.6f}")
                    print(f"\nFeatures received from ESP32 (all 32):")
                    for i in range(len(features)):
                        sensor_num = i // 4 + 1
                        feature_type = ['MAV', 'WL', 'ZC', 'SSC'][i % 4]
                        print(f"  Sensor{sensor_num}_{feature_type}: {features[i]:.8f}")
                    print("=== END PYTHON DIAGNOSTIC ===\n")

                # Check correctness
                is_correct = (predicted == ground_truth)
                if is_correct:
                    self.correct_count += 1

                # Store results
                self.predictions.append(predicted)
                self.confidences.append(confidence)
                self.features_log.append(features)
                self.ground_truths.append(ground_truth)

                # Console output
                gt_name = self.GESTURE_NAMES[ground_truth] if 0 <= ground_truth < 7 else "UNKNOWN"
                pred_name = self.GESTURE_NAMES[predicted] if 0 <= predicted < 7 else "NO_PRED"
                status = "OK" if is_correct else "X"

                print(f"Window {window_idx:3d}/{self.num_windows}: GT={gt_name:10s} PRED={pred_name:10s} ({confidence:.2f}) {status}")

                # Log to file
                log_f.write(f"Window {window_idx}\n")
                log_f.write(f"  Ground Truth: {gt_name} ({ground_truth})\n")
                log_f.write(f"  Predicted: {pred_name} ({predicted})\n")
                log_f.write(f"  Confidence: {confidence:.4f}\n")
                log_f.write(f"  Correct: {is_correct}\n")
                log_f.write(f"  Features: {', '.join([f'{f:.3f}' for f in features])}\n")
                log_f.write("\n")

                # Write to CSV summary
                csv_line = f"{window_idx},{ground_truth},{predicted},{confidence:.4f},{int(is_correct)}"
                csv_line += "," + ",".join([f"{f:.6f}" for f in features]) + "\n"
                csv_f.write(csv_line)

                # Flush periodically
                if window_idx % 10 == 0:
                    log_f.flush()
                    csv_f.flush()

            # Close log files
            log_f.close()
            csv_f.close()

            # Print summary
            print()
            print("=" * 70)
            print("REPLAY COMPLETE")
            print("=" * 70)
            accuracy = (self.correct_count / self.num_windows) * 100 if self.num_windows > 0 else 0
            print(f"Total Windows: {self.num_windows}")
            print(f"Correct: {self.correct_count}")
            print(f"Incorrect: {self.num_windows - self.correct_count}")
            print(f"Accuracy: {accuracy:.2f}%")
            print("=" * 70)
            print()

            # Generate confusion matrix
            self.generate_confusion_matrix()

            return True

        except Exception as e:
            print(f"ERROR during replay: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_confusion_matrix(self):
        """
        Generate and save confusion matrix visualization
        """
        try:
            print("Generating confusion matrix...")

            # Build confusion matrix
            num_classes = len(self.GESTURE_NAMES)
            cm = np.zeros((num_classes, num_classes), dtype=np.int32)

            for gt, pred in zip(self.ground_truths, self.predictions):
                if 0 <= gt < num_classes and 0 <= pred < num_classes:
                    cm[gt, pred] += 1

            # Create figure
            plt.figure(figsize=(12, 10))

            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=self.GESTURE_NAMES,
                yticklabels=self.GESTURE_NAMES,
                cbar_kws={'label': 'Count'}
            )

            plt.title('Confusion Matrix - CSV Replay Validation', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Gesture', fontsize=12)
            plt.ylabel('Ground Truth Gesture', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            # Save figure
            plt.savefig(self.confusion_matrix_file, dpi=300, bbox_inches='tight')
            print(f"OK: Confusion matrix saved to: {self.confusion_matrix_file}")

            plt.close()

            # Print per-gesture accuracy
            print()
            print("Per-Gesture Accuracy:")
            print("-" * 50)
            for i, gesture_name in enumerate(self.GESTURE_NAMES):
                total = cm[i, :].sum()
                correct = cm[i, i]
                accuracy = (correct / total * 100) if total > 0 else 0
                print(f"  {gesture_name:12s}: {correct:3d}/{total:3d} = {accuracy:5.1f}%")
            print("-" * 50)
            print()

        except Exception as e:
            print(f"ERROR generating confusion matrix: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Main entry point
    """
    import argparse

    parser = argparse.ArgumentParser(description='CSV Replay System for ESP32 EMG Model Validation')
    parser.add_argument('csv_file', type=str, help='Path to CSV file with EMG data')
    parser.add_argument('--port', type=str, default='COM11', help='Serial port (default: COM11)')
    parser.add_argument('--baud', type=int, default=921600, help='Baud rate (default: 921600)')

    args = parser.parse_args()

    # Verify CSV file exists
    if not os.path.exists(args.csv_file):
        print(f"ERROR: CSV file not found: {args.csv_file}")
        return 1

    # Create replay system
    replay = CSVReplaySystem(args.csv_file, args.port, args.baud)

    # Load CSV
    if not replay.load_csv():
        print("ERROR: Failed to load CSV file")
        return 1

    # Open serial connection
    if not replay.open_serial():
        print("ERROR: Failed to open serial connection")
        return 1

    try:
        # Run replay
        success = replay.run_replay()

        if success:
            print("OK: Replay completed successfully")
            return 0
        else:
            print("X: Replay failed")
            return 1

    finally:
        # Always close serial
        replay.close_serial()


if __name__ == '__main__':
    exit(main())
