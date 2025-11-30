"""
ESP32-S3 Sampling Rate Test Script
Records exactly 4 seconds of data to verify Hz
"""
import serial
import struct
import time
import csv
from datetime import datetime
import os

# Configuration
PORT = 'COM11'
BAUD = 921600
RECORD_DURATION = 4  # seconds

# File path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def parse_packet(data):
    """Parse 16-byte binary packet: Header(2) + Seq(2) + 6xADC(2)"""
    if len(data) != 16 or data[0] != 0xA5 or data[1] != 0x5A:
        return None

    unpacked = struct.unpack('<H HHHHHH', data[2:])  # Little endian
    return {
        'seq': unpacked[0],
        'emg': unpacked[1:]  # Tuple of 6
    }

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f'test_sampling_{timestamp}.csv')

    print(f"=== ESP32-S3 Sampling Rate Test ===")
    print(f"Port: {PORT}, Baud: {BAUD}")
    print(f"Recording duration: {RECORD_DURATION} seconds")
    print(f"Output file: {filename}")
    print()

    try:
        # Open serial connection
        ser = serial.Serial(PORT, BAUD, timeout=1)
        print("✓ Serial port opened")
        time.sleep(2)  # Wait for ESP32 reset

        ser.reset_input_buffer()
        ser.write(b'S')  # Start streaming
        print("✓ Streaming started")
        time.sleep(0.5)  # Let buffer fill

        # Open CSV file
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Timestamp', 'Seq', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6'])

        # Record data
        buffer = bytearray()
        packet_count = 0
        start_time = time.time()
        record_start = start_time

        print(f"\n⏱️  Recording for {RECORD_DURATION} seconds...")

        while (time.time() - record_start) < RECORD_DURATION:
            if ser.in_waiting:
                buffer.extend(ser.read(ser.in_waiting))

                while len(buffer) >= 16:
                    # Find header
                    idx = buffer.find(b'\xA5\x5A')
                    if idx == -1:
                        buffer = bytearray()
                        break

                    if idx > 0:
                        buffer = buffer[idx:]

                    if len(buffer) < 16:
                        break

                    packet_data = buffer[:16]
                    buffer = buffer[16:]

                    parsed = parse_packet(packet_data)
                    if parsed:
                        packet_count += 1
                        ts = time.time() - start_time

                        csv_writer.writerow([
                            f"{ts:.4f}",
                            parsed['seq'],
                            *parsed['emg']
                        ])

                        # Progress indicator every 100 packets
                        if packet_count % 100 == 0:
                            elapsed = time.time() - record_start
                            print(f"  Packets: {packet_count}, Elapsed: {elapsed:.1f}s", end='\r')

        # Stop streaming
        ser.write(b'E')
        ser.close()
        csv_file.close()

        # Calculate results
        actual_duration = time.time() - record_start
        avg_hz = packet_count / actual_duration

        print(f"\n\n=== Results ===")
        print(f"✓ Recording complete!")
        print(f"  Total packets: {packet_count}")
        print(f"  Actual duration: {actual_duration:.2f} seconds")
        print(f"  Average sampling rate: {avg_hz:.1f} Hz")
        print(f"  Expected: ~2000 Hz")
        print(f"  Deviation: {abs(2000 - avg_hz):.1f} Hz")
        print(f"\n📁 Data saved: {filename}")
        print(f"\n💡 Tip: Open the CSV and check:")
        print(f"   - Count rows (should be ~8000 for 4 seconds at 2kHz)")
        print(f"   - Check timestamps increment by ~0.0005 seconds (0.5ms)")

    except serial.SerialException as e:
        print(f"\n❌ Serial Error: {e}")
        print(f"   Make sure ESP32 is connected to {PORT}")
        print(f"   and data_acquisition firmware is uploaded")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print()

if __name__ == '__main__':
    main()
