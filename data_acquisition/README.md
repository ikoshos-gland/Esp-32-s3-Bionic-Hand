# ESP32-S3 Biyonik El - 6 Kanallı EMG Veri Toplama

Bu proje ESP32-S3-N16R8 mikrodenetleyici kullanarak 6 kanallı EMG (elektromiyografi) sensörlerinden veri toplamak için tasarlanmıştır.

## Donanım Gereksinimleri

- ESP32-S3 DevKitC-1 (N16R8 variant)
- 6x MyoWare EMG sensörü
- SD kart modülü (opsiyonel)

## Pin Bağlantıları

### EMG Sensörleri (ADC Pinleri)
- EMG 1: GPIO 4  (ADC1_CH3)
- EMG 2: GPIO 5  (ADC1_CH4)
- EMG 3: GPIO 6  (ADC1_CH5)
- EMG 4: GPIO 7  (ADC1_CH6)
- EMG 5: GPIO 15 (ADC2_CH4)
- EMG 6: GPIO 16 (ADC2_CH5)

### SD Kart (SPI - Opsiyonel)
- MISO: GPIO 6
- MOSI: GPIO 4
- SCK: GPIO 5
- CS: GPIO 1

## Kurulum

### 1. PlatformIO Kurulumu
```bash
# VSCode PlatformIO Extension yükleyin
# veya
pip install platformio
```

### 2. Projeyi Derleme ve Yükleme
```bash
# ESP32'ye firmware yükle
pio run --target upload

# Seri port monitörü aç
pio device monitor
```

### 3. Python Bağımlılıkları
```bash
pip install pyserial pillow matplotlib numpy
```

## Kullanım

### Veri Toplama

1. ESP32'yi bilgisayara bağlayın
2. Python scriptini çalıştırın:
```bash
python scripts/training_data_collection.py
```

3. Script otomatik olarak:
   - ESP32'ye bağlanır (COM9 - değiştirilebilir)
   - 10 farklı hareketi sırayla gösterir
   - Her hareket 5 saniye sürer, 3 saniye dinlenme
   - Her hareket 6 kez tekrarlanır
   - Verileri `data/training_data_YYYYMMDD_HHMMSS.csv` dosyasına kaydeder

### Veri Formatı

Binary paket yapısı (16 bytes):
```
Header: 0xA5 0x5A (2 bytes)
Sequence: uint16 (2 bytes)
EMG1-6: 6x uint16 (12 bytes)
```

CSV çıktısı:
```
Timestamp, Sequence, Movement, Phase, Repetition,
EMG1, EMG2, EMG3, EMG4, EMG5, EMG6
```

## Hareket Görselleri

`images_hand/` klasöründe 0.jpg - 10.jpg dosyaları bulunmalıdır:
- 0.jpg: Dinlenme pozisyonu
- 1.jpg - 10.jpg: Hareket görselleri

## Seri Port Komutları

ESP32 şu komutları destekler:
- `S`: Veri akışını başlat (Start streaming)
- `E`: Veri akışını durdur (End streaming)

## Özellikler

- 1000 Hz örnekleme frekansı
- Gerçek zamanlı grafik görüntüleme (6 kanal EMG)
- Otomatik hareket protokolü
- CSV veri kaydetme
- Hafif ve hızlı (sadece EMG, IMU yok)

## Lisans

Bu proje eğitim amaçlıdır.
