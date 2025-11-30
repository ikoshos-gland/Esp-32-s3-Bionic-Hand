# src/ Klasörü - Kaynak Dosyalar Dokümantasyonu

ESP32-S3 tabanlı Biyonik El Projesi'nin C++ kaynak kodlarını içeren dizin. Bu klasörde 5 adet temel dosya bulunur: ana program, veri toplama modülü, özellik çıkarma fonksiyonları, DSP filtre uygulamaları ve servo kontrol sistemi.

---

## 📂 Dosya Yapısı

```
src/
├── main.cpp               # Real-time inference: TFLite model ile jest sınıflandırma
├── data_acquisition.cpp   # Veri toplama: 6 EMG sensöründen eğitim verisi toplama
├── functions.cpp          # TD4 özellik çıkarma ve veri işleme fonksiyonları
├── filters.cpp            # DSP filtre uygulamaları (HPF, LPF, Notch)
└── servo_controller.cpp   # Robotik el için 6 servo motor kontrolü
```

---

## 1. main.cpp - Real-Time Inference (Gerçek Zamanlı Jest Sınıflandırma)

### 🎯 Amaç
TensorFlow Lite Micro modelini kullanarak 6 EMG sensöründen gelen sinyalleri gerçek zamanlı olarak işler ve 11 farklı el jestini sınıflandırır. Algılanan jestlere göre robotik eli kontrol eder.

### 🔧 Derleme Komutu
```bash
pio run -e real_time_inference -t upload
```

### 📊 Teknik Detaylar

#### Kullanılan Kütüphaneler
- **Arduino.h** - ESP32 temel fonksiyonları
- **TensorFlowLite_ESP32** - TFLite Micro v2.1.1 (lib/ klasöründe özel port)
- **model.h** - Otomatik oluşturulan TFLite model (Float32, 195KB)
- **functions.h** - TD4 özellik çıkarma fonksiyonları
- **filters.h** - DSP filtre fonksiyonları
- **servo_controller.h** - Servo kontrol sınıfı

#### Bellek Konfigürasyonu
```cpp
constexpr int kTensorArenaSize = 30 * 1024;  // 30KB tensor arena (20KB'den artırıldı)
alignas(16) uint8_t tensor_arena[kTensorArenaSize];  // 16-byte hizalı (SIMD için)
```

#### Model Yapısı
- **Giriş:** 24 TD4 özelliği (4 özellik × 6 EMG sensörü)
- **Mimari:** Wide & Deep MLP (256→128→64→11 nöron)
- **Çıkış:** 11 jest sınıfı olasılığı (Rest, Fist, Open, Point, Victory, OK, ThumbUp, ThumbDn, Grasp, Pinch, WristFlex)
- **Format:** Float32 (Int8 yerine - TFLite v2.1.1 uyumluluğu için)

### 🔄 İşlem Akışı

#### setup() Fonksiyonu (Satır 52-141)
1. **Seri Port Başlatma:** 921600 baud (yüksek hız)
2. **Bellek Diagnostiği:** Heap ve PSRAM kontrolü
3. **ADC Konfigürasyonu:** 11dB attenuation (0-3.3V → 0-4095)
4. **DSP Filtre Başlatma:** `filters_init()` çağrısı
5. **TFLite Model Yükleme:**
   - Model şema versiyonu kontrolü
   - Tensor tahsisi (AllocateTensors)
   - Başarısızlıkta HALT (sonsuz döngü + hata mesajı)
6. **Servo Kontrolcü Başlatma:** 6 servo bağlama ve home pozisyonuna gitme

#### loop() Fonksiyonu (Satır 145-258)
1. **Veri Toplama** (Satır 147)
   ```cpp
   float* features = prelim_collection();
   // 6 sensörden 250ms veri toplar (250 örnek @ 1000Hz)
   // DSP filtreleri uygular (HPF→LPF→Notch)
   // TD4 özelliklerini çıkarır (MAV, WL, ZC, SSC)
   ```

2. **TFLite Model Giriş Hazırlama** (Satır 150-153)
   ```cpp
   for (int i = 0; i < NUM_FEATURES; i++) {
     input->data.f[i] = features[i];  // Float32 doğrudan kopyalama
   }
   ```

3. **TFLite Inference** (Satır 173-177)
   ```cpp
   TfLiteStatus invoke_status = interpreter->Invoke();
   ```

4. **Çıkış İşleme** (Satır 188-196)
   - 11 sınıf olasılığını kontrol et
   - Eşik değeri: **0.8 (80% güven)** - Geçici olarak 0.5'e düşürüldü (test için)
   - En yüksek olasılıklı jesti seç

5. **Debouncing (Gürültü Filtreleme)** (Satır 198-215)
   ```cpp
   #define DEBOUNCE_FRAMES 2  // 2 ardışık tahmin gerekli (~540ms)
   ```
   - Aynı jestin 2 kez üst üste algılanmasını bekler
   - Anlık hataları filtreler

6. **Motion Locking (Hareket Kilitleme)** (Satır 221-256)
   ```cpp
   #define MOTION_LOCK_MS 1000  // 1000ms cooldown
   ```
   - Jest algılandıktan sonra 1 saniye bekleme süresi
   - Hızlı tekrar algılamayı önler
   - Rest (dinlenme) jesti kilidi atlar

7. **Servo Kontrolü** (Satır 242)
   ```cpp
   servoController.moveToGesture(this_predict);
   ```

### 🐛 Hata Yönetimi

#### Model Schema Uyumsuzluğu (Satır 83-91)
```cpp
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model schema mismatch!");
  while(1) { delay(1000); Serial.print("."); }  // HALT
}
```

#### Tensor Tahsis Hatası (Satır 107-120)
```cpp
if (allocate_status != kTfLiteOk) {
  Serial.println("❌ FATAL: TensorFlow Lite memory allocation failed");
  Serial.println("   - Increase kTensorArenaSize in main.cpp");
  while(1) { delay(1000); Serial.print("."); }  // HALT
}
```

### 📈 Performans Metrikleri
- **Toplam Gecikme:** ~282ms
  - Veri toplama: 250ms
  - DSP filtreleme: ~12ms
  - TFLite inference: ~20ms
- **CPU Kullanımı:** ~5.5% @ 240MHz
- **Bellek Kullanımı:**
  - Tensor arena: 30KB
  - Model: 195KB Flash
  - Stack: ~2KB

### 🎛️ Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `DEBOUNCE_FRAMES` | 2 | Ardışık tahmin sayısı |
| `MOTION_LOCK_MS` | 1000ms | Jest sonrası cooldown |
| `kTensorArenaSize` | 30KB | TFLite bellek alanı |
| `threshold` | 0.8 (80%) | Güven eşiği |

---

## 2. data_acquisition.cpp - Veri Toplama Modülü

### 🎯 Amaç
6 MyoWare EMG sensöründen **ham** (DSP filtrelenmiş) sinyalleri 1000 Hz örnekleme hızında toplayarak Python eğitim scripti ([training_data_collection.py](../scripts_ai/training_data_collection.py)) için veri akışı sağlar.

### 🔧 Derleme Komutu
```bash
pio run -e data_acquisition -t upload
```

### 📡 İletişim Protokolü

#### Binary Paket Yapısı (16 byte)
```cpp
#pragma pack(push, 1)
struct DataPacket {
  uint8_t header[2] = {0xA5, 0x5A};  // Senkronizasyon byte'ları
  uint16_t seq;                       // Sıra numarası
  uint16_t mw1;                       // EMG sensör 1 (0-4095)
  uint16_t mw2;                       // EMG sensör 2
  uint16_t mw3;                       // EMG sensör 3
  uint16_t mw4;                       // EMG sensör 4
  uint16_t mw5;                       // EMG sensör 5
  uint16_t mw6;                       // EMG sensör 6
};
#pragma pack(pop)
```

#### Seri Port Konfigürasyonu
```cpp
Serial.begin(921600);  // Yüksek baud rate (1000Hz × 6 kanal × 16 byte ≈ 96 KB/s)
```

### 🔄 İşlem Akışı

#### setup() Fonksiyonu (Satır 65-78)
1. **Seri Port Başlatma:** 921600 baud + 1s gecikme (USB Serial hazır olana kadar)
2. **ADC Konfigürasyonu:** 12-bit çözünürlük (0-4095), 11dB attenuation (0-3.1V)
3. **DSP Filtre Başlatma:** `filters_init()` - HPF, LPF, Notch filtreleri
4. **Komut Bekleme:** 'S' = Start streaming, 'E' = End streaming

#### readSensors() Fonksiyonu (Satır 80-100)
```cpp
// ÖNEMLİ: DSP filtreleri (HPF) NEGATİF değerler üretebilir (DC offset kaldırma)
// Negatif float'u uint16_t'ye cast etmek OVERFLOW'a sebep olur (-2047 → 63489)
// Çözüm: Filtrelenmiş çıkışı [0, 4095] aralığına sınırla

float filtered1 = filter_sample(0, (float)analogRead(pin_MW1));
packet.mw1 = (uint16_t)constrain(filtered1, 0.0f, 4095.0f);
```

**Kritik Düzeltme:** DSP filtreleri uygulandıktan sonra negatif değerler oluşabilir (yüksek geçiren filtre DC'yi kaldırır). Bu değerleri uint16_t'ye cast etmek taşmaya neden olur. `constrain()` ile [0, 4095] aralığına sınırlandırma yapılır.

#### loop() Fonksiyonu (Satır 102-131)
1. **Komut Dinleme:**
   - `'S'` → Streaming başlat, sıra numarasını sıfırla
   - `'E'` → Streaming durdur

2. **Hassas Zamanlama (micros() tabanlı):**
   ```cpp
   #define INTERVAL_US 1000  // 1000μs = 1ms = 1000Hz

   if (current_time - last_sample_time >= INTERVAL_US) {
     last_sample_time += INTERVAL_US;  // += ile timing drift önlenir
     readSensors();
     Serial.write((uint8_t*)&packet, sizeof(DataPacket));
   }
   ```

### 📊 Veri Formatı

#### Python Tarafında Beklenen Format
```python
header = b'\xA5\x5A'
seq = struct.unpack('<H', data[2:4])[0]
mw1, mw2, mw3, mw4, mw5, mw6 = struct.unpack('<6H', data[4:16])
```

### 🎛️ Önemli Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `FREQUENCY` | 1000 Hz | Örnekleme frekansı |
| `INTERVAL_US` | 1000 μs | Örnek aralığı |
| Baud Rate | 921600 | Seri port hızı |
| ADC Çözünürlük | 12-bit | 0-4095 aralığı |
| Attenuation | 11dB | 0-3.3V giriş aralığı |

### ⚠️ Bilinen Sorunlar ve Çözümler

#### 1. DSP Filtre Overflow (ÇÖZÜLDÜ)
**Sorun:** HPF negatif değerler üretir → uint16_t cast → 65535'e yakın değerler
**Çözüm:** `constrain(filtered, 0.0f, 4095.0f)` kullanımı (Satır 94-99)

#### 2. Timing Drift (ÇÖZÜLDÜ)
**Sorun:** `last_sample_time = micros()` kullanımı drift'e sebep olur
**Çözüm:** `last_sample_time += INTERVAL_US` (Satır 122)

---

## 3. functions.cpp - TD4 Özellik Çıkarma Motoru

### 🎯 Amaç
6 EMG sensöründen toplanan **ham sinyalleri** işleyerek Hudgins' TD4 (Time-Domain 4) özelliklerini çıkarır. Bu özellikler TFLite modeline girdi olarak verilir.

### 📚 Literatür Temeli
- **Hudgins et al. (1993)**: "A New Strategy for Multifunction Myoelectric Control"
- **Phinyomark et al. (2012)**: "Feature Reduction and Selection for EMG Signal Classification"
- TD4 özellikleri EMG literatüründe **%85-95 doğruluk** sağlayan "altın standart" yöntemdir.

### 🔬 TD4 Özellikleri (4 Özellik × 6 Sensör = 24 Toplam)

#### 1. MAV (Mean Absolute Value) - Ortalama Mutlak Değer
```cpp
MAV = (1/N) × Σ|x[i]|
```
- **Fiziksel Anlam:** Sinyal genliğinin ortalaması
- **Kullanım:** Kas aktivitesi şiddetini ölçer
- **Kod:** `compute_mav()` (Satır 74-80)

#### 2. WL (Waveform Length) - Dalga Formu Uzunluğu
```cpp
WL = Σ|x[i+1] - x[i]|
```
- **Fiziksel Anlam:** Sinyal karmaşıklığı
- **Kullanım:** Frekans içeriği ve genlik değişimi göstergesi
- **Kod:** `compute_wl()` (Satır 94-100)

#### 3. ZC (Zero Crossings) - Sıfır Geçişleri
```cpp
ZC = Σ sgn(x[i] × x[i+1] < -threshold)
```
- **Fiziksel Anlam:** Sinyalin sıfır eksenini kaç kez geçtiği
- **Kullanım:** Frekans tahmini (yüksek frekans → fazla geçiş)
- **Eşik:** 15.0 ADC birimi (gürültüden kaynaklı yanlış geçişleri önler)
- **Kod:** `compute_zc()` (Satır 120-134)

#### 4. SSC (Slope Sign Changes) - Eğim İşaret Değişimleri
```cpp
SSC = Σ sgn((x[i] - x[i-1]) × (x[i] - x[i+1]) >= threshold)
```
- **Fiziksel Anlam:** Sinyalin zirve/dip noktaları
- **Kullanım:** Frekans içeriği göstergesi
- **Eşik:** 15.0 ADC birimi (gürültü hassasiyetini azaltır)
- **Kod:** `compute_ssc()` (Satır 155-172)

### 🔄 Veri İşleme Pipeline'ı

#### Aşama 1: Ham Veri Toplama - prelim_collection() (Satır 192-234)
```cpp
Süreç:
1. RAW_WINDOW_SIZE (250) örnek topla (250ms @ 1000Hz)
2. Her örneği DSP filtrelerinden geçir:
   Raw ADC → HPF (20Hz) → LPF (450Hz) → Notch (50Hz) → Filtrelenmiş ADC
3. Filtrelenmiş değerleri [0, 4095] aralığına sınırla (overflow önleme)
4. raw_sensor_data[6][250] buffer'larında sakla
5. Her 10 örneğe servo kontrolcüyü güncelle (smooth hareket için)
```

**Kritik Değişiklik:** Artık **filtrelenmiş** ADC değerleri saklanıyor (önceden ham ADC)

#### Aşama 2: TD4 Özellik Çıkarma - extract_features_from_raw() (Satır 251-339)
```cpp
Her sensör için:
1. DC Offset Hesaplama:
   mean = Σ data[i] / N

2. Sinyali Merkezleme (0'a sabitleme):
   centered[i] = data[i] - mean
   // Bipolar sinyal oluşturur (±ADC aralığı)

3. TD4 Özelliklerini Hesaplama:
   - MAV: Σ|centered[i]| / N
   - WL:  Σ|centered[i+1] - centered[i]|
   - ZC:  Sıfır geçiş sayısı (threshold: 15 ADC)
   - SSC: Eğim işaret değişimi sayısı (threshold: 15 ADC)

4. Global Normalizasyon:
   MAV_norm = MAV / 4095
   WL_norm  = WL / (4095 × 250)
   ZC_norm  = ZC / 250
   SSC_norm = SSC / 250

5. Özellik Sırası (Python eğitimi ile uyumlu):
   [MAV₁, WL₁, ZC₁, SSC₁,  // Sensör 1
    MAV₂, WL₂, ZC₂, SSC₂,  // Sensör 2
    ...
    MAV₆, WL₆, ZC₆, SSC₆]  // Sensör 6
```

### 🔧 Kritik Düzeltmeler (2025-11-25 Sigma Projesi)

#### Önceki Yaklaşım (2D-CNN Branch - YANLIŞ)
```cpp
// RMS ön işleme → ZC/SSC her zaman 0
rms_window = sqrt(Σx²/N)  // HER ZAMAN POZİTİF
zc = count_zero_crossings(rms_window)  // ❌ HİÇBİR ZAMAN 0'DAN GEÇMEZ
```

#### Yeni Yaklaşım (Sigma Project - DOĞRU)
```cpp
// Ham sinyal + DC offset kaldırma → ZC/SSC doğru çalışır
centered = raw - mean  // BİPOLAR SİNYAL (-2047 ile +2047 arası)
zc = count_zero_crossings(centered)  // ✅ DOĞRU FREKANS TAHMİNİ
```

### 🎛️ Önemli Sabitler

```cpp
#define NUM_FEATURES     24     // TD4: 4 özellik × 6 sensör
#define NUM_SENSORS      6      // MyoWare EMG sensör sayısı
#define RAW_WINDOW_SIZE  250    // 250ms pencere (250 örnek @ 1kHz)
#define SAMPLING_FREQ    1000   // 1000 Hz örnekleme
#define ADC_MAX_GLOBAL   4095.0 // 12-bit ADC maksimum değer
#define ZC_THRESHOLD_ADC 15.0   // Sıfır geçiş eşiği (ADC birimi)
#define SSC_THRESHOLD_ADC 15.0  // Eğim değişimi eşiği (ADC birimi)
```

### 🔄 Servo Entegrasyonu (Real-Time Inference Modunda)

```cpp
#ifdef REAL_TIME_INFERENCE_MODE
  // Her 10 örneğe bir servo güncelle (~10ms aralık)
  if (i % 10 == 0) {
    servoController.update();
  }
#endif
```

### ⚠️ Önemli Uyarılar

1. **Model Yeniden Eğitimi Gerekli:** Sigma Projesi güncellemeleri sonrası eski modeller **uyumsuz**. Yeni DSP filtrelenmiş verilerle eğitim yapılmalı.

2. **Özellik Sırası Tutarlılığı:** Python eğitimi ve C++ inference arasında **tamamen aynı** özellik sırası kullanılmalı (Satır 326-329).

3. **Threshold Birimleri:** ZC ve SSC eşikleri **ADC birimi** cinsinden (normalize edilmiş değil). Python'da da aynı değerler kullanılmalı.

---

## 4. filters.cpp - DSP Filtre Motoru (Sigma Project Enhancement)

### 🎯 Amaç
EMG sinyallerini temizlemek için **gerçek zamanlı IIR (Infinite Impulse Response)** filtreler uygular. Gürültü, hareket artefaktları ve elektrik şebekesi parazitlerini kaldırır.

### 📚 Literatür Temeli
- **Oppenheim & Schafer (2009)**: "Discrete-Time Signal Processing"
- **De Luca (2002)**: "Surface Electromyography: Detection and Recording"
- **Parks & Burrus (1987)**: "Digital Filter Design"

### 🔬 Filtre Zinciri

```
Raw ADC → HPF (20Hz) → LPF (450Hz) → Notch (50/60Hz) → DC Removal → TD4
```

#### 1. HPF (High-Pass Filter) - Yüksek Geçiren Filtre
- **Kesim Frekansı:** 20 Hz
- **Derece:** 4th-order Butterworth (2 biquad kademesi)
- **Amaç:** DC drift ve hareket artefaktlarını (0-20 Hz) kaldırır
- **Kod:** `hpf_stage1`, `hpf_stage2` (Satır 230-232)

#### 2. LPF (Low-Pass Filter) - Alçak Geçiren Filtre
- **Kesim Frekansı:** 450 Hz
- **Derece:** 4th-order Butterworth (2 biquad kademesi)
- **Amaç:** Elektronik gürültüyü (>500 Hz) kaldırır
- **Kod:** `lpf_stage1`, `lpf_stage2` (Satır 237-239)

#### 3. Notch Filter - Çentik Filtre
- **Merkez Frekans:** 50 Hz (Avrupa/Asya) veya 60 Hz (Amerika)
- **Q Faktörü:** 12.5 (dar bant)
- **Derece:** 2nd-order (1 biquad)
- **Amaç:** Elektrik şebekesi parazitini kaldırır
- **Kod:** `notch` (Satır 244)

### 🏗️ Biquad Yapısı: Direct Form II Transposed

```
Transfer Function: H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (1 + a1*z⁻¹ + a2*z⁻²)

Update Equations:
  y[n] = b0*x[n] + z1
  z1   = b1*x[n] - a1*y[n] + z2
  z2   = b2*x[n] - a2*y[n]
```

**Neden Direct Form II Transposed?**
- ✅ En sayısal olarak kararlı IIR yapısı
- ✅ Katsayı kuantalama hatasını minimize eder
- ✅ Sadece 2 durum değişkeni (minimal bellek: ~480 byte toplam)
- ✅ Endüstri standardı (ARM CMSIS-DSP tarafından kullanılır)
- ✅ Toplam gecikme: ~12 ms

### 🔄 Fonksiyon Detayları

#### biquad_process() - Tek Biquad İşleme (Satır 36-45)
```cpp
float biquad_process(BiquadFilter* bq, float input) {
  float output = bq->b0 * input + bq->z1;
  bq->z1 = bq->b1 * input - bq->a1 * output + bq->z2;
  bq->z2 = bq->b2 * input - bq->a2 * output;
  return output;
}
```

#### filters_init() - Tüm Filtreleri Başlat (Satır 96-159)
```cpp
Süreç:
1. 6 sensörün her biri için:
   - HPF 2 kademe başlat (filter_coefficients_50hz.h'den katsayılar)
   - LPF 2 kademe başlat
   - Notch 1 kademe başlat
   - Hareketli ortalama buffer'ını sıfırla (eğer aktifse)

2. Konfigürasyonu seri porta yazdır:
   - Örnekleme hızı: 1000 Hz (tüm modlar için birleştirildi)
   - HPF: ENABLED/DISABLED
   - LPF: ENABLED/DISABLED
   - Notch: ENABLED/DISABLED
   - Moving Average: ENABLED/DISABLED
```

#### filters_reset() - Durum Değişkenlerini Sıfırla (Satır 167-200)
```cpp
// Jest değiştirirken veya uzun boşta kalma sonrası çağrılır
// Önceki sinyallerden kalan geçici yanıtı temizler
for (int i = 0; i < NUM_SENSORS; i++) {
  state->hpf_stage1.z1 = 0.0f;
  state->hpf_stage1.z2 = 0.0f;
  // ... tüm filtreler için
}
```

#### filter_sample() - Tam Filtre Cascade (Satır 218-253)
```cpp
float filter_sample(int sensor_index, float raw_sample) {
  // Geçersiz sensör indeksi → filtrelenmemiş sinyal döndür
  if (sensor_index < 0 || sensor_index >= NUM_SENSORS) {
    return raw_sample;
  }

  SensorFilterState* state = &filter_states[sensor_index];
  float signal = raw_sample;

  // Filtre zinciri:
  #if ENABLE_HPF
    signal = biquad_process(&state->hpf_stage1, signal);
    signal = biquad_process(&state->hpf_stage2, signal);
  #endif

  #if ENABLE_LPF
    signal = biquad_process(&state->lpf_stage1, signal);
    signal = biquad_process(&state->lpf_stage2, signal);
  #endif

  #if ENABLE_NOTCH
    signal = biquad_process(&state->notch, signal);
  #endif

  #if ENABLE_MOVING_AVG
    signal = moving_average(state, signal);
  #endif

  return signal;
}
```

### 📊 Performans Metrikleri

| Metrik | Değer | Açıklama |
|--------|-------|----------|
| **Bellek Kullanımı** | ~480 bytes | 360B durum + 120B katsayılar |
| **Gecikme** | ~12 ms | Toplam grup gecikmesi |
| **CPU Kullanımı** | <0.5% | @ 240 MHz ESP32-S3 |
| **İşlem Süresi** | ~5 μs/örnek | 6 sensör × 7 biquad = 42 biquad/ms |

### 🎛️ Konfigürasyon Makroları (filters.h)

```cpp
#define POWERLINE_FREQ_HZ 50    // 50Hz (Avrupa) veya 60Hz (Amerika)
#define ENABLE_HPF        1     // Yüksek geçiren filtre (20 Hz)
#define ENABLE_LPF        1     // Alçak geçiren filtre (450 Hz)
#define ENABLE_NOTCH      1     // Çentik filtre (50/60 Hz)
#define ENABLE_MOVING_AVG 0     // Hareketli ortalama (devre dışı, gecikme ekler)
#define NUM_SENSORS       6     // 6 EMG sensörü
#define MA_WINDOW_SIZE    5     // Hareketli ortalama penceresi (aktifse)
```

### 🔧 Filtre Katsayıları

Otomatik oluşturulur: [generate_filter_coefficients.py](../scripts_ai/filters/generate_filter_coefficients.py)

```bash
cd scripts_ai/filters
python generate_filter_coefficients.py 50  # 50Hz için
# Çıkış: ../../include/filter_coefficients_50hz.h
```

### ⚠️ Kritik Notlar

1. **Sensör Başına Bağımsız Durum:** Her sensörün kendi filtre durumu var → çapraz kontaminasyon önlenir

2. **Negatif Çıkış:** HPF DC'yi kaldırdığı için **negatif değerler** üretebilir. `data_acquisition.cpp` ve `functions.cpp`'de `constrain()` kullanılmalı.

3. **Python/C++ Tutarlılığı:** Eğitim ve inference'da aynı filtreler kullanılmalı. [validate_filters.py](../scripts_ai/validation/validate_filters.py) ile doğrulayın.

---

## 5. servo_controller.cpp - Robotik El Servo Kontrolcüsü

### 🎯 Amaç
6 servo motoru kullanarak robotik elin 11 farklı el jestini gerçekleştirmesini sağlar. Yumuşak geçişler için doğrusal interpolasyon (LERP) kullanır.

### 🤖 Donanım Konfigürasyonu

```cpp
const int servo_pins[NUM_SERVOS] = {10, 11, 12, 13, 14, 47};

Servo Sırası:
[0] Thumb_Rotation → GPIO 10  (Başparmak rotasyonu)
[1] Thumb_Flex     → GPIO 11  (Başparmak bükülmesi)
[2] Index          → GPIO 12  (İşaret parmağı)
[3] Middle         → GPIO 13  (Orta parmak)
[4] Ring           → GPIO 14  (Yüzük parmağı)
[5] Pinky          → GPIO 47  (Serçe parmak)
```

### 📐 Açı Konvansiyonu
- **0°** = Parmak tamamen açık (uzatılmış)
- **180°** = Parmak tamamen kapalı (yumruk)
- **Geçerli Aralık:** 0-180 derece (sınırlandırma otomatik)

### 🗺️ Jest-Açı Eşleme Tablosu

```cpp
const int gesture_angles[NUM_GESTURES][NUM_SERVOS] = {
  // [Thumb_Rot, Thumb_Flex, Index, Middle, Ring, Pinky]

  /* 0: Rest      */ {10,  10,   0,   0,   0,   0},  // Dinlenme (açık el)
  /* 1: Fist      */ {90, 160, 170, 170, 170, 170},  // Yumruk (kapalı)
  /* 2: Open      */ {10,  10,   0,   0,   0,   0},  // Açık el (Rest ile aynı)
  /* 3: Point     */ {90, 150,  10, 170, 170, 170},  // İşaret parmağı
  /* 4: Victory   */ {90, 160,  10,  10, 170, 170},  // V işareti
  /* 5: OK        */ {80, 120, 130,   0,   0,   0},  // Tamam işareti
  /* 6: ThumbUp   */ { 0,  10, 170, 170, 170, 170},  // Başparmak yukarı
  /* 7: ThumbDn   */ { 0, 160, 170, 170, 170, 170},  // Başparmak aşağı
  /* 8: Grasp     */ {60,  60,  70,  70,  70,  70},  // Kavrama (C-şekli)
  /* 9: Pinch     */ {90, 100, 110,  10,  10,  10},  // Tutma (başparmak-işaret)
  /* 10: WristFlex*/ {45,  45,  45,  45,  45,  45},  // Bilek fleksiyonu (orta)
};
```

**Not:** Rest ve Open aynı açılara sahip olsa da **ayrı** girdilerdir. Biri değiştirildiğinde diğeri etkilenmez.

### 🎮 Ana Fonksiyonlar

#### begin() - Servo Kontrolcüyü Başlat (Satır 85-100)
```cpp
void ServoController::begin() {
  // 6 servoyu GPIO pinlerine bağla
  for (int i = 0; i < NUM_SERVOS; i++) {
    int channel = servos[i].attach(servo_pins[i]);
    Serial.printf("✅ Servo %d attached to GPIO %d (channel %d)\n",
                  i, servo_pins[i], channel);
  }
}
```

**Önemli:** TFLite başlatıldıktan **SONRA** çağrılmalı (bellek tahsisi başarısını garantilemek için).

#### setHome() - Güvenli Home Pozisyonuna Git (Satır 103-117)
```cpp
const int home_angles[NUM_SERVOS] = {10, 10, 0, 0, 0, 0};  // Parmaklar açık

void ServoController::setHome() {
  // Anında hareket (interpolasyon yok)
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].write(home_angles[i]);
    current_positions[i] = home_angles[i];
    target_positions[i] = home_angles[i];
  }
  is_moving = false;
}
```

#### moveToGesture() - Jest Pozisyonuna Yumuşak Geçiş (Satır 120-149)
```cpp
void ServoController::moveToGesture(int gesture_id) {
  // Geçersiz ID kontrolü
  if (gesture_id < 0 || gesture_id >= NUM_GESTURES) {
    Serial.printf("ERROR: Invalid gesture ID: %d\n", gesture_id);
    return;
  }

  // Jest açılarını al
  const int* angles = gesture_angles[gesture_id];

  // Debug çıktısı (jest adı, servo hareketleri, delta gösterimi)
  Serial.printf("🎯 GESTURE DETECTED: %s (ID: %d)\n",
                gesture_names[gesture_id], gesture_id);
  for (int i = 0; i < NUM_SERVOS; i++) {
    int delta = angles[i] - current_positions[i];
    Serial.printf("  [%d] %-11s: %3d° → %3d° (Δ%+4d°)\n",
                 i, servo_names[i], current_positions[i], angles[i], delta);
  }

  // 500ms interpolasyonlu hareketi başlat
  moveToAngles(angles, DEFAULT_INTERPOLATION_DURATION_MS);
}
```

#### moveToAngles() - Özel Açılara Hareket (Satır 152-165)
```cpp
void ServoController::moveToAngles(const int angles[NUM_SERVOS],
                                   unsigned long duration_ms) {
  // Başlangıç pozisyonlarını kaydet (hedef değil, mevcut pozisyonlar)
  for (int i = 0; i < NUM_SERVOS; i++) {
    start_positions[i] = current_positions[i];
    target_positions[i] = clamp(angles[i], 0, 180);  // 0-180° sınırla
  }

  // Interpolasyon durumunu başlat
  interpolation_start_time = millis();
  interpolation_duration = duration_ms;
  is_moving = true;
}
```

#### update() - Non-Blocking Interpolasyon Güncelleme (Satır 168-197)
```cpp
void ServoController::update() {
  if (!is_moving) return;  // Hareket yoksa hiçbir şey yapma

  unsigned long elapsed = millis() - interpolation_start_time;

  // Hareket tamamlandı mı?
  if (elapsed >= interpolation_duration) {
    // Son pozisyona snap
    for (int i = 0; i < NUM_SERVOS; i++) {
      servos[i].write(target_positions[i]);
      current_positions[i] = target_positions[i];
    }
    is_moving = false;
    Serial.println("✅ Movement complete!\n");
    return;
  }

  // İlerleme hesapla (0.0 - 1.0)
  float progress = (float)elapsed / (float)interpolation_duration;

  // Tüm servoları LERP ile güncelle
  for (int i = 0; i < NUM_SERVOS; i++) {
    int delta = target_positions[i] - start_positions[i];
    int new_position = start_positions[i] + (int)(delta * progress);

    servos[i].write(new_position);
    current_positions[i] = new_position;
  }
}
```

**Kritik:** Bu fonksiyon **düzenli olarak çağrılmalı** (10-20ms aralıklarla):
- `loop()` içinde (basit kullanım)
- `prelim_collection()` içinde her 10 örneğe bir (önerilen - daha yumuşak)

### 🎛️ Önemli Parametreler

```cpp
#define NUM_SERVOS 6                              // 6 servo motor
#define NUM_GESTURES 11                           // 11 jest
#define DEFAULT_INTERPOLATION_DURATION_MS 500     // 500ms geçiş süresi
```

### 📊 Jest İsimleri (Sıra Kritik!)

```cpp
const char* gesture_names[] = {
  "Rest", "Fist", "Open", "Point", "Victory",
  "OK", "ThumbUp", "ThumbDn", "Grasp", "Pinch", "WristFlex"
};
```

**Uyarı:** Bu sıra **tüm dosyalarda aynı** olmalı:
- [main.cpp](main.cpp:218)
- [servo_controller.cpp](servo_controller.cpp:131)
- [training_data_collection.py](../data_acquisition/scripts/training_collection new.py)
- [train_test_model.py](../scripts_ai/critical/train_test_model.py)

### 🎬 Kullanım Örneği

```cpp
// setup() içinde
servoController.begin();         // Servoları başlat
servoController.setHome();       // Home pozisyonuna git

// loop() içinde
servoController.moveToGesture(1);  // Fist jestine geç (ID: 1)
servoController.update();          // Her döngüde çağır (non-blocking)

// Özel açılar
int custom_angles[6] = {90, 90, 45, 45, 45, 45};
servoController.moveToAngles(custom_angles, 1000);  // 1 saniyede geçiş
```

### 🔧 Jest Açılarını Değiştirme

**Örnek:** "Victory" (Jest 4) açılarını değiştirmek için:

1. [servo_controller.cpp](servo_controller.cpp) dosyasını aç
2. Satır 47'yi bul: `/* 4: Victory   */`
3. Açıları değiştir: `{90, 160, 10, 10, 170, 170}` → istediğiniz değerler
4. Kaydet ve yeniden derle: `pio run -e real_time_inference -t upload`

**Not:** Her jest **bağımsız** bir dizi girdisidir. Birini değiştirmek diğerlerini etkilemez.

---

## 📚 Dosyalar Arası İlişkiler

```
main.cpp (Real-Time Inference)
  ↓ includes
  ├─ functions.h → functions.cpp (TD4 feature extraction)
  │    ↓ includes
  │    └─ filters.h → filters.cpp (DSP filtering)
  │         ↓ includes
  │         └─ filter_coefficients_50hz.h (auto-generated)
  ├─ model.h (TFLite model - auto-generated)
  └─ servo_controller.h → servo_controller.cpp (robotic hand control)

data_acquisition.cpp (Data Collection)
  ↓ includes
  └─ filters.h → filters.cpp (DSP filtering - SAME AS INFERENCE)
       ↓ includes
       └─ filter_coefficients_50hz.h (auto-generated)
```

**Kritik:** Her iki mod da **aynı DSP filtrelerini** kullanır → eğitim ve inference tutarlılığı

---

## 🛠️ Geliştirme Notları

### ADC Konfigürasyonu (ESP32-S3)
```cpp
analogReadResolution(12);        // 12-bit çözünürlük (0-4095)
analogSetAttenuation(ADC_11db);  // 0-3.3V aralığı
```

### GPIO Pin Değişiklikleri (Önemli!)
**Eski:** GPIO 0-5 (❌ UART0 ile çakışma)
**Yeni:** GPIO 4-7, 15-16 (✅ Güvenli)

⚠️ **Fiziksel yeniden bağlantı gerekli!** Sensörleri yeni pinlere bağlayın.

### Watchdog Devre Dışı
```cpp
// main.cpp setup() içinde watchdog disable edilmiş
// TD4 özellik çıkarma 250ms sürüyor → watchdog timeout'una sebep oluyordu
```

### Bellek Optimizasyonu
- **Static buffer kullanımı:** `raw_sensor_data[6][250]` → heap overflow önleme
- **16-byte hizalama:** `alignas(16) uint8_t tensor_arena` → SIMD optimizasyonu
- **PSRAM kullanımı:** `BOARD_HAS_PSRAM` bayrağı aktif

---

## 🐛 Bilinen Sorunlar ve Çözümler

### 1. DSP Filtre Overflow (ÇÖZÜLDÜ ✅)
**Sorun:** HPF negatif değerler üretir → uint16_t cast → overflow
**Çözüm:** `constrain(filtered, 0.0f, 4095.0f)` kullanımı

### 2. ZC/SSC Her Zaman 0 (ÇÖZÜLDÜ ✅)
**Sorun:** RMS ön işleme bipolar sinyali bozuyor
**Çözüm:** Ham sinyal + DC offset kaldırma yaklaşımı

### 3. Model Schema Mismatch (ÇÖZÜLDÜ ✅)
**Sorun:** TFLite v2.2+ modelleri v2.1.1 kütüphanesiyle çalışmıyor
**Çözüm:** Float32 model kullanımı (Int8 yerine)

### 4. Tensor Arena OOM (ÇÖZÜLDÜ ✅)
**Sorun:** 20KB arena yetersiz
**Çözüm:** 30KB'ye artırma

---

## 📖 Referanslar

### Literatür
- **Hudgins et al. (1993)**: TD4 özellikleri orijinal makale
- **Phinyomark et al. (2012)**: TD4 doğrulama çalışması
- **Oppenheim & Schafer (2009)**: IIR filtre tasarımı temel kitabı
- **De Luca (2002)**: EMG frekans bantları

### Kod Standartları
- **Direct Form II Transposed:** ARM CMSIS-DSP standardı
- **LERP Interpolation:** Unity/Unreal Engine smoothing yaklaşımı

---

## 🔗 İlgili Dosyalar

**Python Eğitim Pipeline:**
- [scripts_ai/critical/feature_extraction.py](../scripts_ai/critical/feature_extraction.py) - TD4 özellik çıkarma (C++ ile uyumlu)
- [scripts_ai/critical/train_test_model.py](../scripts_ai/critical/train_test_model.py) - Model eğitimi ve TFLite dönüşümü
- [scripts_ai/filters/dsp_filters.py](../scripts_ai/filters/dsp_filters.py) - Python DSP filtre bankası (C++ ile eşleşir)
- [scripts_ai/validation/validate_filters.py](../scripts_ai/validation/validate_filters.py) - Python-C++ filtre tutarlılık testi

**Konfigürasyon:**
- [platformio.ini](../platformio.ini) - Build ortamları (data_acquisition, real_time_inference)
- [CLAUDE.md](../CLAUDE.md) - Proje genel dokümantasyonu

---

**Son Güncelleme:** 2025-11-30
**Proje:** ESP32-S3 Bionic Hand - Real-Time Gesture Classification
