/*
ESP32-S3-N16R8 - Bionic Hand Project
6-Channel EMG Data Acquisition for TD4 Training
=============================================

This file is compiled when using the data_acquisition environment:
  pio run -e data_acquisition -t upload

Purpose: Collect raw EMG data from 6 MyoWare sensors and stream over serial
Protocol: Binary packets (16 bytes) at 1000 Hz sampling rate
Output: Used by scripts_ai/training_data_collection.py for model training
*/

#include <Arduino.h>
#include "filters.h"

// ESP32-S3 Pin Definitions (Must match functions.h)
#define pin_MW1 4   // GPIO 4  - ADC1_CH3
#define pin_MW2 5   // GPIO 5  - ADC1_CH4
#define pin_MW3 6   // GPIO 6  - ADC1_CH5
#define pin_MW4 7   // GPIO 7  - ADC1_CH6
#define pin_MW5 15  // GPIO 15 - ADC2_CH4
#define pin_MW6 16  // GPIO 16 - ADC2_CH5

// Sampling Configuration (1000 Hz - unified with real-time inference)
#define FREQUENCY 1000
// Standard interval for 1000 Hz sampling (UNIFIED with functions.h)
#define INTERVAL_US 1000  // 1000μs = 1ms = 1000Hz

unsigned long last_sample_time = 0;
bool streaming = false;

// Binary Packet Structure (16 bytes total)
// Python side expects: Header(2) + Seq(2) + 6xEMG(12)
#pragma pack(push, 1)
struct DataPacket {
  uint8_t header[2] = {0xA5, 0x5A};  // Sync bytes
  uint16_t seq;                       // Sequence number
  uint16_t mw1;                       // EMG sensor 1 (0-4095)
  uint16_t mw2;                       // EMG sensor 2
  uint16_t mw3;                       // EMG sensor 3
  uint16_t mw4;                       // EMG sensor 4
  uint16_t mw5;                       // EMG sensor 5
  uint16_t mw6;                       // EMG sensor 6
};
#pragma pack(pop)

DataPacket packet;
uint16_t packet_seq = 0;

void setup_ADC() {
  // Configure GPIO pins as analog inputs
  pinMode(pin_MW1, INPUT);
  pinMode(pin_MW2, INPUT);
  pinMode(pin_MW3, INPUT);
  pinMode(pin_MW4, INPUT);
  pinMode(pin_MW5, INPUT);
  pinMode(pin_MW6, INPUT);

  // ESP32-S3 ADC Configuration
  analogReadResolution(12);        // 12-bit resolution (0-4095)
  analogSetAttenuation(ADC_11db);  // 0-3.1V range (suitable for 3.3V sensors)
}

void setup(void) {
  // High baud rate for 2000 Hz @ 6 channels @ 16 bytes/packet = ~256 kbps
  Serial.begin(921600);
  delay(1000);  // Wait for USB serial to initialize

  setup_ADC();

  // Initialize DSP filters (HPF + LPF + Notch)
  filters_init();

  Serial.println("ESP32-S3 Data Acquisition Ready");
  Serial.println("Commands: 'S' = Start streaming, 'E' = End streaming");
  Serial.println("Sampling: 1000 Hz @ 921600 baud, 6 channels, 12-bit ADC (with DSP filtering)");
}

void readSensors() {
  // Read all 6 EMG channels and apply DSP filters
  // CRITICAL FIX: DSP filters (HPF) can output NEGATIVE values (DC offset removal)
  // Casting negative float to uint16_t causes OVERFLOW (e.g., -2047 → 63489)
  // Solution: Clamp filtered output to valid 12-bit ADC range [0, 4095]
  
  float filtered1 = filter_sample(0, (float)analogRead(pin_MW1));
  float filtered2 = filter_sample(1, (float)analogRead(pin_MW2));
  float filtered3 = filter_sample(2, (float)analogRead(pin_MW3));
  float filtered4 = filter_sample(3, (float)analogRead(pin_MW4));
  float filtered5 = filter_sample(4, (float)analogRead(pin_MW5));
  float filtered6 = filter_sample(5, (float)analogRead(pin_MW6));
  
  // Clamp to 12-bit ADC range [0, 4095] to prevent uint16_t overflow
  packet.mw1 = (uint16_t)constrain(filtered1, 0.0f, 4095.0f);
  packet.mw2 = (uint16_t)constrain(filtered2, 0.0f, 4095.0f);
  packet.mw3 = (uint16_t)constrain(filtered3, 0.0f, 4095.0f);
  packet.mw4 = (uint16_t)constrain(filtered4, 0.0f, 4095.0f);
  packet.mw5 = (uint16_t)constrain(filtered5, 0.0f, 4095.0f);
  packet.mw6 = (uint16_t)constrain(filtered6, 0.0f, 4095.0f);
}

void loop() {
  // Check for serial commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      streaming = true;
      packet_seq = 0;
      last_sample_time = micros();  // Initialize to current time to prevent catch-up
      // Clear input buffer
      while (Serial.available()) Serial.read();
    } else if (cmd == 'E') {
      streaming = false;
    }
  }

  // Precision timing using micros() instead of blocking delay()
  if (streaming) {
    unsigned long current_time = micros();
    if (current_time - last_sample_time >= INTERVAL_US) {
      // Use += to prevent timing drift from execution overhead
      last_sample_time += INTERVAL_US;

      readSensors();
      packet.seq = packet_seq++;

      // Send binary packet (16 bytes)
      Serial.write((uint8_t*)&packet, sizeof(DataPacket));
    }
  }
}
