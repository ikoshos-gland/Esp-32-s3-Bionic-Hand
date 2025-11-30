/*
ESP32-S3 (N16R8) - Biyonik El Projesi
6 Kanallı EMG Veri Toplama (TD4 Eğitimi İçin)
*/

#include <Arduino.h>

// ESP32-S3 Pin Tanımları
// Tüm sensörler ADC1 üzerinde (ADC2'den kaçınıldı)
#define pin_MW1 4   // GPIO 4  - ADC1_CH3
#define pin_MW2 5   // GPIO 5  - ADC1_CH4
#define pin_MW3 6   // GPIO 6  - ADC1_CH5
#define pin_MW4 7   // GPIO 7  - ADC1_CH6
#define pin_MW5 15   // GPIO 15  - ADC2_CH4
#define pin_MW6 16  // GPIO 16  - ADC2_CH5

// Sampling Ayarları (Feature Extraction ile uyumlu 1000Hz)
#define FREQUENCY 1000 
#define INTERVAL_US (1000000 / FREQUENCY)

unsigned long last_sample_time = 0;
bool streaming = false;

// Binary Paket Yapısı (16 byte - Değişmedi)
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
  // 6 Kanal okuma
  packet.mw1 = analogRead(pin_MW1);
  packet.mw2 = analogRead(pin_MW2);
  packet.mw3 = analogRead(pin_MW3);
  packet.mw4 = analogRead(pin_MW4);
  packet.mw5 = analogRead(pin_MW5);
  packet.mw6 = analogRead(pin_MW6);
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