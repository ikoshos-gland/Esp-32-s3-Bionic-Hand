# include/ Klasörü - Header Dosyalar Dokümantasyonu

ESP32-S3 Biyonik El Projesi'nin C/C++ header dosyalarını içeren dizin. Header dosyaları, kaynak dosyalar arasında paylaşılan fonksiyon bildirimleri, makro tanımları, veri yapıları ve sabitler içerir.

---

## 📂 Dosya Yapısı

```
include/
├── functions.h                    # TD4 özellik çıkarma fonksiyonları ve sabitler
├── filters.h                      # DSP filtre yapıları ve fonksiyon bildirimleri
├── filter_coefficients_50hz.h     # Otomatik oluşturulan filtre katsayıları
├── model.h                        # Otomatik oluşturulan TFLite model (C++ byte array)
└── README                         # PlatformIO varsayılan header bilgilendirmesi
```

**Not:** `servo_controller.h` header dosyası [lib/servo_controller.h](../lib/servo_controller.h) konumundadır (include/ değil).

---

## 1. functions.h - TD4 Özellik Çıkarma ve Sistem Sabitleri

### 🎯 Amaç
TD4 (Time-Domain 4) özellik çıkarma fonksiyonlarının bildirimlerini, GPIO pin tanımlarını ve sistem geneli sabitleri içerir. Bu dosya **tüm kaynak dosyalar tarafından** kullanılan merkezi konfigürasyon dosyasıdır.

### 📦 İçindekiler

#### Header Guard (Satır 1-2)
```cpp
#ifndef FUNCTIONS_H_
#define FUNCTIONS_H_
```
**Amaç:** Aynı header dosyasının birden fazla kez dahil edilmesini önler (multiple inclusion guard).

#### Kütüphane Dahilleri (Satır 4-6)
```cpp
#include <stdint.h>  // uint8_t, uint16_t gibi sabit boyutlu tipler
#include <cmath>     // fabs(), sqrt() gibi matematik fonksiyonları
```

#### Sistem Sabitleri (Satır 8-21)

##### 1. Özellik Konfigürasyonu
```cpp
#define NUM_FEATURES 24   // TD4 özellikleri: 4 özellik × 6 EMG sensörü
#define NUM_SENSORS  6    // MyoWare EMG sensör sayısı
```

**Kritik:** `NUM_FEATURES` değeri **Python eğitim pipeline ile uyumlu** olmalıdır:
- [feature_extraction.py](../scripts_ai/critical/feature_extraction.py) - TD4 çıkarımı
- [train_test_model.py](../scripts_ai/critical/train_test_model.py) - Model eğitimi

**TD4 Hesaplama:**
```
NUM_FEATURES = NUM_SENSORS × 4
             = 6 × 4
             = 24 özellik
```

##### 2. Veri Toplama Konfigürasyonu
```cpp
#define RAW_WINDOW_SIZE  250   // 250ms pencere (250 örnek @ 1kHz)
#define SAMPLING_FREQ    1000  // 1000 Hz örnekleme - Tüm modlar için birleştirildi
```

**Önceki Sorun (ÇÖZÜLDÜ):**
- `data_acquisition` 2000 Hz kullanıyordu
- `real_time_inference` 1000 Hz kullanıyordu
- **→ Tutarsızlık, filtre katsayıları hatalı**

**Şimdiki Durum (Sigma Project):**
- **Her iki mod da 1000 Hz kullanır** → Filtre katsayıları tutarlı

**Window Size Hesaplama:**
```
RAW_WINDOW_SIZE = SAMPLING_FREQ × Window_Duration
                = 1000 Hz × 0.25 s
                = 250 örnek
```

##### 3. Global ADC Normalizasyon (Satır 18-21)
```cpp
#define ADC_MIN_GLOBAL   0.0f      // 12-bit ADC minimum değer
#define ADC_MAX_GLOBAL   4095.0f   // 12-bit ADC maksimum değer (2^12 - 1)
```

**Kritik:** Python eğitim pipeline'ı ile **tamamen aynı** normalizasyon kullanılmalı:

**C++ Tarafı (functions.cpp):**
```cpp
mav_normalized = mav / ADC_MAX_GLOBAL;              // MAV: 0-1 aralığı
wl_normalized = wl / (ADC_MAX_GLOBAL * RAW_WINDOW_SIZE);  // WL: 0-1 aralığı
```

**Python Tarafı (feature_extraction.py):**
```python
mav_norm = mav / 4095.0
wl_norm = wl / (4095.0 * 250)
```

#### GPIO Pin Tanımları (Satır 23-32)

**ESP32-S3 Pin Atamaları (ADC Kanalları):**
```cpp
#define pin_MW1 4   // GPIO 4  (ADC1_CH3) - MyoWare EMG Sensor 1
#define pin_MW2 5   // GPIO 5  (ADC1_CH4) - MyoWare EMG Sensor 2
#define pin_MW3 6   // GPIO 6  (ADC1_CH5) - MyoWare EMG Sensor 3
#define pin_MW4 7   // GPIO 7  (ADC1_CH6) - MyoWare EMG Sensor 4
#define pin_MW5 15  // GPIO 15 (ADC2_CH4) - MyoWare EMG Sensor 5
#define pin_MW6 16  // GPIO 16 (ADC2_CH5) - MyoWare EMG Sensor 6
```

**⚠️ BÜYÜK DEĞİŞİKLİK (2025-11-25):**

| Önceki (YANLIŞ) | Yeni (DOĞRU) | Neden |
|----------------|-------------|-------|
| GPIO 0-5 | GPIO 4-7, 15-16 | **UART0 çakışması** |
| ADC1_CH0-5 | ADC1_CH3-6, ADC2_CH4-5 | **Seri port (GPIO1-2) önleme** |

**UART0 Çakışması Sorunu:**
- GPIO 1-2: ESP32-S3'te **UART0 TX/RX** için kullanılır
- USB Serial (Serial.begin()) UART0 kullanır
- GPIO 0-5 kullanımı seri portu **devre dışı bırakıyordu**
- **→ Upload başarısız, Serial Monitor çalışmıyor**

**Çözüm:**
- GPIO 4-7: ADC1 kanalları (güvenli)
- GPIO 15-16: ADC2 kanalları (ESP32-S3'te Wi-Fi olmadan güvenli)

**⚠️ FİZİKSEL YENIDEN BAĞLANTI GEREKLİ:**
Eski pinlerden (GPIO 0-5) yeni pinlere (GPIO 4-7, 15-16) sensörleri yeniden bağlamalısınız!

#### Fonksiyon Bildirimleri (Satır 37-56)

##### 1. Ana Pipeline Fonksiyonları
```cpp
float *prelim_collection();          // 250ms veri topla + DSP filtrele + TD4 çıkar
float *extract_features_from_raw();  // Ham veriden TD4 özellikleri çıkar
```

**İşlem Akışı:**
```
prelim_collection()
  ↓
  ├─ 6 sensörden 250 örnek topla (250ms @ 1kHz)
  ├─ Her örneği DSP filtrelerinden geçir (HPF→LPF→Notch)
  ├─ raw_sensor_data[6][250] buffer'ında sakla
  ↓
extract_features_from_raw()
  ↓
  ├─ DC offset kaldır (mean subtraction)
  ├─ TD4 özelliklerini hesapla (MAV, WL, ZC, SSC)
  ├─ Global normalizasyon uygula
  ↓
return features[24]  // TFLite modeline girdi
```

##### 2. TD4 Özellik Hesaplama Fonksiyonları
```cpp
float compute_mav(float data[], int len);               // MAV: Mean Absolute Value
float compute_wl(float data[], int len);                // WL: Waveform Length
int compute_zc(float data[], int len, float threshold); // ZC: Zero Crossings
int compute_ssc(float data[], int len, float threshold);// SSC: Slope Sign Changes
```

**Literatür Referansı:**
- **Hudgins et al. (1993)**: "A New Strategy for Multifunction Myoelectric Control"
- EMG literatüründe **%85-95 doğruluk** sağlayan "altın standart" özellikler

**Özellik Detayları:**

| Özellik | Formül | Açıklama | Çıkış Tipi |
|---------|--------|----------|-----------|
| **MAV** | `(1/N) × Σ|x[i]|` | Ortalama sinyal genliği | `float` |
| **WL** | `Σ|x[i+1] - x[i]|` | Sinyal karmaşıklığı | `float` |
| **ZC** | `Σ sgn(x[i] × x[i+1] < -t)` | Sıfır geçiş sayısı | `int` |
| **SSC** | `Σ sgn((x[i]-x[i-1]) × (x[i]-x[i+1]) >= t)` | Eğim işaret değişimi | `int` |

**Threshold Kullanımı (ZC & SSC):**
```cpp
// functions.cpp içinde tanımlı
#define ZC_THRESHOLD_ADC  15.0f   // ADC birimi (normalize edilmemiş)
#define SSC_THRESHOLD_ADC 15.0f   // ADC birimi (normalize edilmemiş)
```

**Neden threshold gerekli?**
- Gürültüden kaynaklı **yanlış geçişleri** önler
- ZC: Küçük gürültü dalgalanmaları sıfır geçişi sayılmaz
- SSC: Küçük titreşimler eğim değişimi sayılmaz

---

## 2. filters.h - DSP Filtre Kütüphanesi

### 🎯 Amaç
EMG sinyallerini temizlemek için kullanılan IIR (Infinite Impulse Response) filtre yapılarını, konfigürasyonları ve fonksiyon bildirimlerini tanımlar.

### 📊 Filtre Tasarımı (Sigma Project - 2025-11-25)

```
Raw ADC → HPF (20Hz) → LPF (450Hz) → Notch (50/60Hz) → DC Removal → TD4
```

### 📦 İçindekiler

#### Header Guard ve Dahiller (Satır 26-29)
```cpp
#ifndef FILTERS_H
#define FILTERS_H

#include <Arduino.h>
```

#### Konfigürasyon Makroları (Satır 34-48)

##### 1. Powerline Frekansı
```cpp
#define POWERLINE_FREQ_HZ 50  // 50 Hz (Avrupa/Asya) veya 60 Hz (Amerika)
```
**Kullanım:** Notch filtre için merkez frekans

##### 2. Filtre Enable/Disable Bayrakları
```cpp
#define ENABLE_HPF         1  // High-pass filter (20 Hz) - AÇIK
#define ENABLE_LPF         1  // Low-pass filter (450 Hz) - AÇIK
#define ENABLE_NOTCH       1  // Notch filter (50/60 Hz) - AÇIK
#define ENABLE_MOVING_AVG  0  // Moving average - KAPALI (gecikme ekler)
```

**Önerilen Konfigürasyon (Varsayılan):**
- ✅ HPF: AÇIK → DC drift ve hareket artefaktlarını kaldırır
- ✅ LPF: AÇIK → Elektronik gürültüyü kaldırır
- ✅ Notch: AÇIK → 50/60 Hz elektrik şebekesi parazitini kaldırır
- ❌ Moving Avg: KAPALI → Ek gecikme ekler (~25ms), gerekli değil

##### 3. Diğer Sabitler
```cpp
#define NUM_SENSORS        6   // 6 EMG sensörü
#define MA_WINDOW_SIZE     5   // Moving average pencere boyutu (aktifse)
```

#### Biquad Filtre Yapısı (Satır 54-74)

**Direct Form II Transposed Biquad:**
```cpp
typedef struct {
  float b0, b1, b2;  // Numerator coefficients (pay)
  float a1, a2;      // Denominator coefficients (payda, a0=1.0 normalize edilmiş)
  float z1, z2;      // State variables (durum değişkenleri, gecikme hattı)
} BiquadFilter;
```

**Transfer Fonksiyonu:**
```
H(z) = (b0 + b1*z⁻¹ + b2*z⁻²) / (1 + a1*z⁻¹ + a2*z⁻²)
```

**Güncelleme Denklemleri:**
```cpp
y[n] = b0*x[n] + z1
z1   = b1*x[n] - a1*y[n] + z2
z2   = b2*x[n] - a2*y[n]
```

**Neden Direct Form II Transposed?**

| Özellik | Avantaj |
|---------|---------|
| **Sayısal Kararlılık** | En kararlı IIR yapısı |
| **Katsayı Hatası** | Katsayı kuantalama hatasını minimize eder |
| **Bellek Kullanımı** | Sadece 2 durum değişkeni (z1, z2) |
| **Endüstri Standardı** | ARM CMSIS-DSP tarafından kullanılır |
| **Performans** | Tek döngüde hesaplanabilir |

**Bellek Kullanımı Hesaplama:**
```
1 BiquadFilter = 5 float × 4 byte = 20 byte

1 Sensör için:
  - HPF (2 kademe): 2 × 20 byte = 40 byte
  - LPF (2 kademe): 2 × 20 byte = 40 byte
  - Notch (1 kademe): 1 × 20 byte = 20 byte
  - Toplam: 100 byte/sensör

6 Sensör için:
  - 6 × 100 byte = 600 byte (state)
  - Katsayılar (paylaşımlı): ~120 byte
  - TOPLAM: ~720 byte
```

#### Sensör Filtre Durumu Yapısı (Satır 80-104)

```cpp
typedef struct {
  // High-Pass Filter (4th-order = 2 biquad stages)
  BiquadFilter hpf_stage1;
  BiquadFilter hpf_stage2;

  // Low-Pass Filter (4th-order = 2 biquad stages)
  BiquadFilter lpf_stage1;
  BiquadFilter lpf_stage2;

  // Notch Filter (2nd-order = 1 biquad)
  BiquadFilter notch;

  // Moving Average (optional)
  #if ENABLE_MOVING_AVG
    float ma_buffer[MA_WINDOW_SIZE];  // Circular buffer
    int ma_index;                     // Current index
    float ma_sum;                     // Running sum
  #endif
} SensorFilterState;
```

**Tasarım Prensibi:**
- **Her sensörün bağımsız durumu var** → Çapraz kontaminasyon önlenir
- **6 ayrı SensorFilterState nesnesi** → `filter_states[NUM_SENSORS]`

#### Global Değişkenler (Satır 110)
```cpp
extern SensorFilterState filter_states[NUM_SENSORS];  // filters.cpp'de tanımlı
```

**`extern` Kullanımı:**
- Değişkenin **başka bir dosyada tanımlandığını** belirtir
- `filter_states` asıl olarak [filters.cpp](../src/filters.cpp:19) içinde oluşturulur
- Diğer dosyalar bu header'ı dahil ederek erişim sağlar

#### Fonksiyon Bildirimleri (Satır 116-175)

##### 1. filters_init() - Tüm Filtreleri Başlat
```cpp
void filters_init();
```
**Ne Zaman Çağrılır:** `setup()` içinde bir kez
**Ne Yapar:**
1. 6 sensör için filtre katsayılarını yükler
2. Tüm durum değişkenlerini sıfırlar (z1=0, z2=0)
3. Konfigürasyonu Serial'a yazdırır

**Çağrı Yeri:**
- [main.cpp](../src/main.cpp:71) - Real-time inference
- [data_acquisition.cpp](../src/data_acquisition.cpp:73) - Veri toplama

##### 2. filters_reset() - Durum Değişkenlerini Sıfırla
```cpp
void filters_reset();
```
**Ne Zaman Çağrılır:**
- Jest değiştirirken
- Uzun boşta kalma sonrası
- Geçici yanıt (transient response) temizleme gerektiğinde

**Ne Yapar:**
1. Tüm biquad z1, z2 değerlerini sıfırlar
2. Moving average buffer'ını sıfırlar (aktifse)

##### 3. filter_sample() - Tek Örnek İşleme
```cpp
float filter_sample(int sensor_index, float raw_sample);
```
**Parametreler:**
- `sensor_index`: 0-5 arası sensör numarası
- `raw_sample`: Ham ADC değeri (0-4095)

**Dönüş:** Filtrelenmiş ADC değeri

**İşlem Akışı:**
```cpp
Raw ADC (0-4095)
  ↓
HPF Stage 1 → HPF Stage 2
  ↓
LPF Stage 1 → LPF Stage 2
  ↓
Notch Filter
  ↓
(Optional) Moving Average
  ↓
Filtered ADC (can be NEGATIVE due to HPF!)
```

**⚠️ Kritik Uyarı:**
HPF (High-Pass Filter) DC offset'i kaldırdığı için **negatif değerler** üretebilir!

**Çözüm (data_acquisition.cpp ve functions.cpp):**
```cpp
float filtered = filter_sample(0, raw_adc);
uint16_t clamped = (uint16_t)constrain(filtered, 0.0f, 4095.0f);
```

##### 4. biquad_process() - Tek Biquad İşleme (Internal)
```cpp
float biquad_process(BiquadFilter* bq, float input);
```
**Kullanım:** `filter_sample()` tarafından dahili olarak çağrılır
**Görünürlük:** Public olsa da doğrudan kullanılması önerilmez

##### 5. moving_average() - Hareketli Ortalama (Opsiyonel)
```cpp
#if ENABLE_MOVING_AVG
float moving_average(SensorFilterState* state, float input);
#endif
```
**Sadece ENABLE_MOVING_AVG=1 ise derlenir**

**Nasıl Çalışır:**
- Circular buffer (döngüsel tampon) kullanır
- Son N örneğin ortalamasını alır
- ~25ms ek gecikme ekler

**Neden varsayılan olarak kapalı?**
- IIR filtreleri zaten yeterli yumuşatma sağlıyor
- Ek gecikme gerçek zamanlı uygulamalarda kabul edilemez

### 🎛️ Filtre Özellikleri

| Filtre | Tür | Derece | Kesim/Merkez Frekans | Amaç |
|--------|-----|--------|---------------------|------|
| **HPF** | Butterworth | 4th-order | 20 Hz | DC drift, hareket artefaktları (0-20 Hz) |
| **LPF** | Butterworth | 4th-order | 450 Hz | Elektronik gürültü (>500 Hz) |
| **Notch** | IIR Notch | 2nd-order | 50/60 Hz (Q=12.5) | Elektrik şebekesi paraziti |

**EMG Frekans Bandı (Literatür):**
- **Kullanışlı EMG:** 20-450 Hz
- **DC Drift:** 0-20 Hz (hareket, ter) → HPF ile kaldırılır
- **Elektronik Gürültü:** >500 Hz → LPF ile kaldırılır
- **Powerline:** 50/60 Hz → Notch ile kaldırılır

### 📈 Performans Metrikleri

```
Bellek Kullanımı: ~720 byte toplam
CPU Kullanımı: <0.5% @ 240 MHz
İşlem Süresi: ~5 μs/örnek (6 sensör × 7 biquad)
Gecikme: ~12 ms (grup gecikmesi)
```

---

## 3. filter_coefficients_50hz.h - Otomatik Oluşturulan Filtre Katsayıları

### 🎯 Amaç
DSP filtrelerinin (HPF, LPF, Notch) biquad katsayılarını içerir. Python scripti tarafından otomatik oluşturulur.

### 🤖 Otomatik Üretim

**Oluşturma Scripti:**
```bash
cd scripts_ai/filters
python generate_filter_coefficients.py 50  # 50Hz için (Avrupa/Asya)
# veya
python generate_filter_coefficients.py 60  # 60Hz için (Amerika)

# Çıkış: ../../include/filter_coefficients_50hz.h
```

**Script Konumu:** [generate_filter_coefficients.py](../scripts_ai/filters/generate_filter_coefficients.py)

### 📦 İçerik Yapısı

#### 1. Dosya Başlığı (Satır 2-18)
```cpp
/*
 * Auto-Generated Filter Coefficients
 * Generated by: generate_filter_coefficients.py
 * Powerline Frequency: 50 Hz
 *
 * Filters:
 * - High-Pass: 20 Hz, 4th-order Butterworth (2 biquad stages)
 * - Low-Pass: 450 Hz, 4th-order Butterworth (2 biquad stages)
 * - Notch: 50 Hz, Q=12.5 (1 biquad)
 *
 * UNIFIED 1000 Hz SAMPLING:
 * Both data_acquisition and real_time_inference use 1000 Hz.
 */
```

#### 2. 1000 Hz Örnekleme Katsayıları (Satır 22-54)

**Preprocessor Koşulu:**
```cpp
#ifndef DATA_ACQUISITION_MODE
  // 1000 Hz katsayıları (varsayılan)
#endif
```

**Kullanım:** Her iki mod da (`data_acquisition` ve `real_time_inference`) **aynı katsayıları kullanır**

**HPF Katsayıları (Stage 1):**
```cpp
const float HPF_STAGE1_B0 = 0.8484752955f;
const float HPF_STAGE1_B1 = -1.6969505910f;
const float HPF_STAGE1_B2 = 0.8484752955f;
const float HPF_STAGE1_A1 = -1.7783134881f;
const float HPF_STAGE1_A2 = 0.7924474718f;
```

**HPF Katsayıları (Stage 2):**
```cpp
const float HPF_STAGE2_B0 = 1.0000000000f;
const float HPF_STAGE2_B1 = -2.0000000000f;
const float HPF_STAGE2_B2 = 1.0000000000f;
const float HPF_STAGE2_A1 = -1.8934156010f;
const float HPF_STAGE2_A2 = 0.9084644129f;
```

**LPF Katsayıları (Stage 1 & 2):**
```cpp
const float LPF_STAGE1_B0 = 0.6620158372f;
const float LPF_STAGE1_B1 = 1.3240316744f;
const float LPF_STAGE1_B2 = 0.6620158372f;
const float LPF_STAGE1_A1 = 1.4796742169f;
const float LPF_STAGE1_A2 = 0.5558215433f;

const float LPF_STAGE2_B0 = 1.0000000000f;
const float LPF_STAGE2_B1 = 2.0000000000f;
const float LPF_STAGE2_B2 = 1.0000000000f;
const float LPF_STAGE2_A1 = 1.7009643319f;
const float LPF_STAGE2_A2 = 0.7884997398f;
```

**Notch Katsayıları (50 Hz):**
```cpp
const float NOTCH_B0 = 0.9875889381f;
const float NOTCH_B1 = -1.8785057900f;
const float NOTCH_B2 = 0.9875889381f;
const float NOTCH_A1 = -1.8785057900f;
const float NOTCH_A2 = 0.9751778762f;
```

#### 3. 2000 Hz Örnekleme Katsayıları (Satır 58-92)

**Preprocessor Koşulu:**
```cpp
#ifdef DATA_ACQUISITION_MODE
  // 2000 Hz katsayıları (KULLANILMIYOR - tarihsel amaçla saklandı)
#endif
```

**Durum:** Şu anda **kullanılmıyor**. Her iki mod da 1000 Hz kullanır (Sigma Project birleştirmesi).

**Neden saklandı?**
- Gelecekte 2000 Hz desteği geri eklenebilir
- Referans amaçlı (katsayı karşılaştırma)

### 🔧 Katsayı Üretim Detayları

**Python SciPy Kullanımı:**
```python
import scipy.signal as signal

# HPF: 20 Hz, 4th-order Butterworth
sos_hpf = signal.butter(4, 20, btype='high', fs=1000, output='sos')

# LPF: 450 Hz, 4th-order Butterworth
sos_lpf = signal.butter(4, 450, btype='low', fs=1000, output='sos')

# Notch: 50 Hz, Q=12.5
sos_notch = signal.iirnotch(50, Q=12.5, fs=1000)

# SOS (Second-Order Sections) → Biquad formatına dönüştür
# b = [b0, b1, b2], a = [1.0, a1, a2]
```

**Float Precision:**
- 10 ondalık basamak (`%.10f`)
- `f` suffix (float literal)
- Bit-exact tutarlılık için yüksek hassasiyet

### ⚠️ Önemli Uyarılar

1. **Elle Düzenleme Yapma:** Bu dosya otomatik oluşturulur. Elle yapılan değişiklikler bir sonraki üretimde kaybolur.

2. **Python/C++ Tutarlılığı:** C++ ve Python filtreleri **aynı katsayıları kullanmalı**. Doğrulama için:
   ```bash
   cd scripts_ai/validation
   python validate_filters.py
   # Çıkış: "✅ Filters match within tolerance"
   ```

3. **Powerline Frekansı:** 50 Hz (Avrupa/Asya) vs 60 Hz (Amerika) → Farklı katsayılar gerekir

---

## 4. model.h - TensorFlow Lite Model (Otomatik Oluşturulan)

### 🎯 Amaç
Python'da eğitilmiş TensorFlow Lite modelini C++ byte array olarak içerir. ESP32'de çalıştırılabilir formata dönüştürülmüş model.

### 🤖 Otomatik Üretim

**Eğitim ve Dönüştürme Scripti:**
```bash
cd scripts_ai/critical
python train_test_model.py

# Çıkışlar:
# - ../data/models/emg_model_TIMESTAMP.tflite (binary model)
# - ../data/models/emg_model_TIMESTAMP.h (C++ header)
```

**Script Konumu:** [train_test_model.py](../scripts_ai/critical/train_test_model.py)

**Model Kopyalama:**
```bash
cp scripts_ai/data/models/emg_model_TIMESTAMP.h include/model.h
```

### 📦 Dosya Yapısı

#### 1. Dosya Başlığı (Satır 1-11)
```cpp
// Auto-generated TensorFlow Lite model for ESP32-S3
// Wide & Deep MLP with TD4 Features (MAV, WL, ZC, SSC)
//
// Generated: 2025-11-26 00:33:11
// Model size: 195584 bytes
// Format: Float32 (no quantization for TFLite v2.1.1 compatibility)
// Input: 24 features (4 TD4 × 6 EMG sensors)
// Output: 11 gestures

#ifndef MODEL_H
#define MODEL_H
```

**Önemli Bilgiler:**
- **Format:** Float32 (Int8 yerine - TFLite v2.1.1 uyumluluğu için)
- **Boyut:** ~195 KB (Flash'da saklanır, RAM kullanmaz)
- **Giriş:** 24 TD4 özelliği
- **Çıkış:** 11 jest sınıfı

#### 2. Model Byte Array (Satır 13+)
```cpp
const unsigned char model_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  // ... ~195,000 byte
};
```

**Depolama:**
- `const` → Flash belleğinde saklanır (ROM)
- RAM kullanmaz (sadece TFLite tensor arena RAM kullanır: 30KB)

#### 3. Python Dönüşüm Kodu
```python
# train_test_model.py içinde
with open(f'models/emg_model_{timestamp}.h', 'w') as f:
    f.write(f'// Auto-generated TensorFlow Lite model\n')
    f.write(f'// Model size: {model_size} bytes\n')
    f.write(f'const unsigned char model_tflite[] = {{\n')

    # Hex formatında byte-by-byte yazma
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        f.write(f'  {hex_str},\n')

    f.write('};\n')
```

### 🏗️ Model Mimarisi

**Wide & Deep MLP:**
```
Input Layer: 24 features (TD4)
  ↓
Dense(256, relu) + Dropout(0.3)
  ↓
Dense(128, relu) + Dropout(0.2)
  ↓
Dense(64, relu) + Dropout(0.1)
  ↓
Output Layer: 11 classes (softmax)
```

**Eğitim Detayları (train_test_model.py):**
```python
model = Sequential([
    Dense(256, activation='relu', input_shape=(24,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(11, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### 🔄 C++ Kullanımı

**main.cpp içinde:**
```cpp
#include "model.h"

// TFLite model yükleme
const tflite::Model* model = tflite::GetModel(model_tflite);

// Schema versiyonu kontrolü (CRITICAL!)
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report("Model schema mismatch!");
  while(1) { /* HALT */ }
}

// Interpreter oluşturma
static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

// Inference
interpreter->Invoke();
```

### ⚠️ Önemli Uyarılar

1. **Schema Version Mismatch:**
   - TFLite kütüphanesi: v2.1.1 (ESP32 lib/)
   - Model: TFLite Converter ile aynı versiyon gerekli
   - **Çözüm:** Float32 model kullan (Int8 v2.2+ gerektirir)

2. **Model Boyutu:**
   - 195 KB Flash kullanır (ESP32-S3'te 16 MB var, sorun yok)
   - RAM'de tensor arena: 30 KB (ESP32-S3'te 512 KB var, sorun yok)

3. **Elle Düzenleme Yapma:**
   - Otomatik oluşturulan dosya
   - Elle değişiklik yapılırsa bir sonraki eğitimde kaybolur

4. **Yeniden Eğitim Gereksinimi:**
   - DSP filtreleri değiştirilirse **model yeniden eğitilmeli**
   - TD4 özellik hesaplama değişirse **model yeniden eğitilmeli**
   - Sigma Project güncellemesi sonrası tüm modeller **uyumsuz**

### 📊 Model Performansı

**Beklenen Doğruluk (Literatür):**
- TD4 özellikleri: %85-95
- Wide & Deep MLP: +%5-10 artış
- **Toplam:** %90-98 doğruluk (temiz verilerle)

**Inference Süresi:**
- ESP32-S3 @ 240 MHz: ~20 ms
- Float32 hesaplama (SIMD optimizasyonu yok)

---

## 5. README - PlatformIO Varsayılan Açıklama

### 🎯 Amaç
PlatformIO tarafından otomatik oluşturulan standart header dosyaları açıklaması. Proje-spesifik içerik yok.

### 📦 İçerik Özeti
```
- Header dosyaları nedir?
- #include direktifi nasıl kullanılır?
- Header dosyalarının avantajları
- GCC dokümantasyonu referansları
```

**Not:** Bu dosya **proje-spesifik değildir**. PlatformIO şablon dosyasıdır.

---

## 🔗 Dosyalar Arası Bağımlılıklar

```
main.cpp / data_acquisition.cpp
  ↓ includes
functions.h
  ↓ defines
  ├─ NUM_FEATURES = 24
  ├─ NUM_SENSORS = 6
  ├─ RAW_WINDOW_SIZE = 250
  ├─ SAMPLING_FREQ = 1000
  ├─ ADC_MAX_GLOBAL = 4095
  ├─ GPIO pin definitions
  └─ Function prototypes

filters.h
  ↓ defines
  ├─ Filter enable flags
  ├─ BiquadFilter struct
  ├─ SensorFilterState struct
  └─ Filter function prototypes
  ↓ includes
filter_coefficients_50hz.h
  ↓ defines
  ├─ HPF_STAGE1_B0, B1, B2, A1, A2
  ├─ HPF_STAGE2_B0, B1, B2, A1, A2
  ├─ LPF_STAGE1_B0, B1, B2, A1, A2
  ├─ LPF_STAGE2_B0, B1, B2, A1, A2
  └─ NOTCH_B0, B1, B2, A1, A2

model.h
  ↓ defines
  └─ model_tflite[] byte array
```

---

## 🛠️ Header Dosyaları ile Çalışma

### 1. Yeni Sabit Ekleme (functions.h)
```cpp
// functions.h içine ekle
#define NEW_CONSTANT 42

// functions.cpp içinde kullan
int value = NEW_CONSTANT;
```

### 2. Filtre Konfigürasyonu Değiştirme (filters.h)
```cpp
// filters.h içinde
#define ENABLE_MOVING_AVG 1  // 0 → 1 (aktif et)
#define MA_WINDOW_SIZE 10    // 5 → 10 (pencere büyüt)
```

### 3. Yeni Filtre Katsayıları Oluşturma
```bash
cd scripts_ai/filters
python generate_filter_coefficients.py 60  # 60Hz için
cp ../../include/filter_coefficients_50hz.h ../../include/filter_coefficients_60hz.h

# filters.cpp içinde:
# #include "filter_coefficients_60hz.h"
```

### 4. Yeni Model Yükleme
```bash
cd scripts_ai/critical
python train_test_model.py

# Eğitim tamamlandıktan sonra:
cp ../data/models/emg_model_<TIMESTAMP>.h ../../include/model.h

# Yeniden derle
cd ../..
pio run -e real_time_inference -t upload
```

---

## ⚠️ Yaygın Hatalar ve Çözümler

### 1. Multiple Definition Error
**Hata:**
```
multiple definition of `filter_states'
```

**Sebep:** `filter_states` hem header'da tanımlanmış hem de `.cpp` dosyasında

**Çözüm:** Header'da `extern` kullan, `.cpp` dosyasında tanımla
```cpp
// filters.h
extern SensorFilterState filter_states[NUM_SENSORS];

// filters.cpp
SensorFilterState filter_states[NUM_SENSORS];
```

### 2. Undefined Reference
**Hata:**
```
undefined reference to `filter_sample'
```

**Sebep:** Fonksiyon bildirilmiş ama tanımlanmamış

**Çözüm:**
```cpp
// filters.h - BİLDİRİM
float filter_sample(int sensor_index, float raw_sample);

// filters.cpp - TANIM
float filter_sample(int sensor_index, float raw_sample) {
  // ... implementasyon
}
```

### 3. Model Schema Mismatch
**Hata:**
```
Model schema mismatch! (version 3, expected 3)
```

**Sebep:** TFLite model SOFTMAX v2 kullanıyor, kütüphane v1 destekliyor

**Çözüm:** Float32 model kullan (Int8 yerine)
```python
# train_test_model.py içinde
converter.target_spec.supported_ops = []  # Int8 quantization KAPALI
```

### 4. ADC Pin Conflict
**Hata:**
```
Serial upload başarısız, COM11 bulunamıyor
```

**Sebep:** GPIO 1-2 UART0 ile çakışıyor

**Çözüm:** functions.h içinde GPIO 4-7, 15-16 kullan (zaten güncel)

---

## 📚 İlgili Dosyalar

**Kaynak Dosyalar (src/):**
- [main.cpp](../src/main.cpp) - Real-time inference
- [data_acquisition.cpp](../src/data_acquisition.cpp) - Veri toplama
- [functions.cpp](../src/functions.cpp) - TD4 özellik çıkarma
- [filters.cpp](../src/filters.cpp) - DSP filtre implementasyonu

**Python Pipeline:**
- [generate_filter_coefficients.py](../scripts_ai/filters/generate_filter_coefficients.py) - Katsayı üretimi
- [train_test_model.py](../scripts_ai/critical/train_test_model.py) - Model eğitimi ve dönüştürme
- [validate_filters.py](../scripts_ai/validation/validate_filters.py) - Python/C++ filtre tutarlılık kontrolü

**Dokümantasyon:**
- [src/README.md](../src/README.md) - Kaynak dosyalar detayları
- [CLAUDE.md](../CLAUDE.md) - Proje genel dokümantasyonu

---

## 🔍 Header Guard Detayları

**Neden Header Guard Gerekli?**

```cpp
// math_utils.h
#ifndef MATH_UTILS_H
#define MATH_UTILS_H

int add(int a, int b);

#endif
```

**Sorun Senaryosu (Header Guard Olmadan):**
```cpp
// file1.cpp
#include "math_utils.h"  // add() bildirimi
#include "geometry.h"    // geometry.h da math_utils.h include eder
// SONUÇ: add() iki kez bildirildi → DERLEME HATASI
```

**Çözüm (Header Guard ile):**
```cpp
// İlk include
#ifndef MATH_UTILS_H  // Tanımlı değil, devam et
#define MATH_UTILS_H  // Şimdi tanımla
int add(int a, int b);
#endif

// İkinci include
#ifndef MATH_UTILS_H  // Tanımlı, atla
// ... (bu kısım işlenmez)
#endif
```

**Modern Alternatif:**
```cpp
#pragma once  // Derleyici-spesifik, daha hızlı

int add(int a, int b);
```

**Not:** Bu projede `#ifndef` / `#define` / `#endif` yaklaşımı kullanılır (daha taşınabilir).

---

**Son Güncelleme:** 2025-11-30
**Proje:** ESP32-S3 Bionic Hand - Real-Time Gesture Classification
