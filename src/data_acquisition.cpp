/*
 * ESP32-S3 (N16R8) - Bionic Hand: 6-channel EMG data acquisition
 * ================================================================
 *
 * Streams FILTERED, BIPOLAR-ENCODED samples at 1000 Hz over USB serial for
 * training. The signal path is identical to the inference firmware:
 *
 *   raw ADC (0-4095) -> HPF 20 Hz -> LPF 450 Hz -> Notch 50 Hz
 *                    -> encode_bipolar(): float -> uint16 (value + 2047.5)
 *
 * Python (feature_extraction.py) decodes with  value - 2047.5.
 *
 * Packet (16 bytes, little-endian): A5 5A | seq u16 | 6 x u16
 * Commands: 'S' start streaming, 'E' stop.
 *
 * Do not mix recordings from firmware without bipolar encoding (values wrap
 * to 65535). scripts_ai/validation/validate_noise_data.py detects that.
 */

#include <Arduino.h>
#include "filters.h"
#include "functions.h"   // pins, encode_bipolar(), SAMPLING_FREQ

#define INTERVAL_US SAMPLE_PERIOD_US

static const int sensor_pins[NUM_SENSORS] = {pin_MW1, pin_MW2, pin_MW3, pin_MW4, pin_MW5, pin_MW6};

static unsigned long next_sample_time = 0;
static bool streaming = false;

#pragma pack(push, 1)
struct DataPacket {
  uint8_t  header[2] = {0xA5, 0x5A};
  uint16_t seq;
  uint16_t mw[NUM_SENSORS];
};
#pragma pack(pop)
static_assert(sizeof(DataPacket) == 16, "DataPacket must be 16 bytes (Python unpacks '<H HHHHHH')");

static DataPacket packet;
static uint16_t packet_seq = 0;

void setup() {
  Serial.begin(921600);
  delay(1000);

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);
  for (int s = 0; s < NUM_SENSORS; s++) {
    pinMode(sensor_pins[s], INPUT);
    (void)analogRead(sensor_pins[s]);
  }
  filters_init();

  Serial.println("ESP32-S3 data acquisition ready (filtered + bipolar encoded)");
  Serial.printf("Sampling %d Hz, %d channels, 16-byte packets. 'S' start, 'E' stop\n",
                SAMPLING_FREQ, NUM_SENSORS);
}

static void read_sensors() {
  for (int s = 0; s < NUM_SENSORS; s++) {
    float filtered = filter_sample(s, (float)analogRead(sensor_pins[s]));
    packet.mw[s] = encode_bipolar(filtered);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      streaming = true;
      packet_seq = 0;
      next_sample_time = micros();
      while (Serial.available()) Serial.read();
    } else if (cmd == 'E') {
      streaming = false;
    }
  }

  if (!streaming) return;

  // Drift-free schedule: the deadline advances by exactly INTERVAL_US per
  // sample. If the loop was blocked for many periods, resynchronise instead of
  // emitting a burst of catch-up samples.
  unsigned long now = micros();
  if ((long)(now - next_sample_time) < 0) return;
  if ((unsigned long)(now - next_sample_time) > 10UL * INTERVAL_US) next_sample_time = now;
  next_sample_time += INTERVAL_US;

  read_sensors();
  packet.seq = packet_seq++;
  Serial.write((uint8_t*)&packet, sizeof(DataPacket));
}
