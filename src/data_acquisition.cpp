/*
 * ESP32-S3 (N16R8) - Biyonik El Projesi
 * 6 Kanallı EMG Veri Toplama (TD4 Eğitimi İçin)
 * 
 * UPDATED 2025: Now applies IDENTICAL DSP filtering as real-time inference
 * ============================================================================
 * 
 * DATA PIPELINE (matches real-time inference exactly):
 * 1. Read raw ADC (12-bit, 0-4095)
 * 2. Apply DSP filter cascade: HPF (20Hz) → LPF (450Hz) → Notch (50/60Hz)
 * 3. Encode bipolar filtered signal: float [-2047.5, +2047.5] → uint16_t [0, 4095]
 * 4. Transmit encoded data via serial (binary protocol)
 * 
 * Python side MUST decode: uint16_t [0, 4095] → float [-2047.5, +2047.5]
 * 
 * WHY THIS MATTERS:
 * - Training data MUST match inference data characteristics
 * - Model learns on filtered signals, so training needs filtered signals
 * - Bipolar encoding preserves negative values from HPF
 * 
 * CRITICAL: Do NOT mix data from old (unfiltered) and new (filtered) firmware!
 */

#include <Arduino.h>

// DSP Filters (SAME as real-time inference)
#include "filters.h"

// Bipolar encoding for filtered signals (SAME as real-time inference)
#include "functions.h"

// ESP32-S3 Pin Tanımları (Inference kodunla uyumlu)
// UYARI: GPIO 15 ve 16 ADC2'dir. WiFi kullanıldığında çalışmazlar.
// Bu kodda WiFi kapalı olduğu için sorun yok.
#define pin_MW1 4   // ADC1_CH3
#define pin_MW2 5   // ADC1_CH4
#define pin_MW3 6   // ADC1_CH5
#define pin_MW4 7   // ADC1_CH6
#define pin_MW5 15  // ADC2_CH4
#define pin_MW6 16  // ADC2_CH5

// Sampling Ayarları (UNIFIED 1000Hz - matches inference exactly)
#define FREQUENCY 1000 
#define INTERVAL_US (1000000 / FREQUENCY)

unsigned long last_sample_time = 0;
bool streaming = false;

// Binary Paket Yapısı (16 byte total)
// UPDATED: Sensor values are now FILTERED + BIPOLAR ENCODED
// Header: 0xA5 0x5A (2 bytes)
// Seq: uint16_t (2 bytes)
// MW1-6: uint16_t [0, 4095] - bipolar encoded filtered signals (12 bytes)
//
// Decoding in Python: decoded_float = uint16_value - 2047.5
#pragma pack(push, 1)
struct DataPacket {
  uint8_t header[2] = {0xA5, 0x5A};
  uint16_t seq;
  uint16_t mw1;
  uint16_t mw2;
  uint16_t mw3;
  uint16_t mw4;
  uint16_t mw5;
  uint16_t mw6;
};
#pragma pack(pop)

DataPacket packet;
uint16_t packet_seq = 0;

void setup_ADC() {
  // Pin modları
  pinMode(pin_MW1, INPUT);
  pinMode(pin_MW2, INPUT);
  pinMode(pin_MW3, INPUT);
  pinMode(pin_MW4, INPUT);
  pinMode(pin_MW5, INPUT);
  pinMode(pin_MW6, INPUT);

  // ESP32-S3 ADC Ayarları
  // 12-bit çözünürlük (0-4095)
  analogReadResolution(12);
  
  // 11dB zayıflatma (0 - 3.1V arası okuma sağlar, sensörler 3.3V ise uygundur)
  analogSetAttenuation(ADC_11db);

  // ============================================================================
  // DSP Filter Initialization (CRITICAL - SAME as real-time inference)
  // ============================================================================
  // Initialize IIR filter cascade for all 6 sensors:
  //   HPF (20 Hz)   - Removes DC drift and motion artifacts
  //   LPF (450 Hz)  - Removes high-frequency noise
  //   Notch (50 Hz) - Removes powerline interference
  //
  // MUST be called BEFORE data collection starts!
  filters_init();
  
  Serial.println("✅ DSP Filters initialized (HPF + LPF + Notch)");
  Serial.println("   Training data will match inference signal characteristics");
}

void setup(void) {
  // S3 Native USB kullanıyorsan Serial, UART kullanıyorsan Serial0 olabilir.
  Serial.begin(921600);
  delay(1000); // USB'nin oturması için bekle

  setup_ADC();

  // Seri portun hazır olduğunu bildir
  // (Python scripti bu mesajı beklemiyor ama debug için iyi)
}

void readSensors() {
  // ============================================================================
  // SIGNAL PROCESSING PIPELINE (matches real-time inference EXACTLY)
  // ============================================================================
  // For each sensor:
  //   1. Read raw ADC value (12-bit: 0-4095)
  //   2. Apply DSP filter cascade: HPF → LPF → Notch
  //   3. Encode bipolar filtered signal: float [-2047.5, +2047.5] → uint16_t [0, 4095]
  //
  // This ensures training data has IDENTICAL characteristics to inference data.
  // ============================================================================

  // Sensor 1 (EMG1)
  uint16_t raw1 = analogRead(pin_MW1);
  float filtered1 = filter_sample(0, (float)raw1);  // Apply DSP filters
  packet.mw1 = encode_bipolar(filtered1);           // Encode for transmission

  // Sensor 2 (EMG2)
  uint16_t raw2 = analogRead(pin_MW2);
  float filtered2 = filter_sample(1, (float)raw2);
  packet.mw2 = encode_bipolar(filtered2);

  // Sensor 3 (EMG3)
  uint16_t raw3 = analogRead(pin_MW3);
  float filtered3 = filter_sample(2, (float)raw3);
  packet.mw3 = encode_bipolar(filtered3);

  // Sensor 4 (EMG4)
  uint16_t raw4 = analogRead(pin_MW4);
  float filtered4 = filter_sample(3, (float)raw4);
  packet.mw4 = encode_bipolar(filtered4);

  // Sensor 5 (EMG5)
  uint16_t raw5 = analogRead(pin_MW5);
  float filtered5 = filter_sample(4, (float)raw5);
  packet.mw5 = encode_bipolar(filtered5);

  // Sensor 6 (EMG6)
  uint16_t raw6 = analogRead(pin_MW6);
  float filtered6 = filter_sample(5, (float)raw6);
  packet.mw6 = encode_bipolar(filtered6);
}

void loop() {
  // Komut kontrolü
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'S') {
      streaming = true;
      packet_seq = 0;
      // Buffer temizle
      while (Serial.available()) Serial.read();
    } else if (cmd == 'E') {
      streaming = false;
    }
  }

  // Zamanlayıcı (Blocking delay yerine micros farkı)
  if (streaming) {
    unsigned long current_time = micros();
    if (current_time - last_sample_time >= INTERVAL_US) {
      last_sample_time = current_time;

      readSensors();
      packet.seq = packet_seq++;

      // Binary paketi gönder
      Serial.write((uint8_t*)&packet, sizeof(DataPacket));
    }
  }
}