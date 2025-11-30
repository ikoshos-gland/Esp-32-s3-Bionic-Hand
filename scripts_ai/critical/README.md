# 🎯 Critical Pipeline - End-to-End ML Workflow

Bu klasör, ESP32-S3 Bionic Hand projesi için **temel makine öğrenmesi pipeline'ını** içerir. Ham EMG verilerinden başlayıp, ESP32'de çalışan TensorFlow Lite modeline kadar tüm süreci yöneten 2 kritik script bulunur.

---

## 📋 İçindekiler

- [Klasör Amacı](#-klasör-amacı)
- [Pipeline Akışı](#-pipeline-akışı)
- [Scriptler](#-scriptler)
  - [1. feature_extraction.py](#1-feature_extractionpy)
  - [2. train_test_model.py](#2-train_test_modelpy)
- [Adım Adım Kullanım](#-adım-adım-kullanım)
- [Detaylı Parametre Açıklamaları](#-detaylı-parametre-açıklamaları)
- [Çıktı Dosyaları](#-çıktı-dosyaları)
- [Örnek Komutlar](#-örnek-komutlar)
- [Troubleshooting](#-troubleshooting)
- [Teknik Detaylar](#-teknik-detaylar)

---

## 🎯 Klasör Amacı

Bu klasördeki scriptler, **veri biliminin temelini** oluşturur:

```
Ham EMG Verileri (CSV) → TD4 Özellikler (NPZ) → TFLite Model + C++ Header
```

**Neden "Critical" (Kritik)?**
- Bu 2 script olmadan model eğitilemez
- Tüm pipeline bu scriptlere bağlı
- Herhangi bir değişiklik tüm sistemi etkiler
- ESP32'deki gerçek zamanlı sınıflandırma bu scriptlerin çıktısına dayanır

**Kimler Kullanmalı?**
- Model eğitmek isteyen geliştiriciler
- Yeni EMG verisi topladıktan sonra
- Farklı hiperparametreler denemek isteyenler
- Model performansını iyileştirmek isteyenler

---

## 🔄 Pipeline Akışı

### **Aşama 1: Veri Toplama** (bu klasörün dışında)
```bash
# ESP32'den 6 EMG sensörü verisi toplama
python data_acquisition/scripts/training_collection.py
```
**Çıktı:** `data/training_data_YYYYMMDD_HHMMSS.csv`

### **Aşama 2: Özellik Çıkarımı** (bu klasör - Script 1)
```bash
python critical/feature_extraction.py data/training_data_20251130.csv
```
**Çıktı:** `data/features/training_data_20251130_td4_features.npz`

### **Aşama 3: Model Eğitimi** (bu klasör - Script 2)
```bash
python critical/train_test_model.py
```
**Çıktılar:**
- `data/models/mlp_td4_YYYYMMDD_HHMMSS_float32.tflite` (TFLite modeli)
- `data/models/mlp_td4_YYYYMMDD_HHMMSS_float32.h` (ESP32 için C++ header)
- `data/models/mlp_td4_YYYYMMDD_HHMMSS.keras` (Keras modeli)
- `data/plots/training_history_YYYYMMDD_HHMMSS.png` (Eğitim grafikleri)
- `data/plots/confusion_matrix_YYYYMMDD_HHMMSS.png` (Karmaşıklık matrisi)

### **Aşama 4: ESP32'ye Yükleme** (bu klasörün dışında)
```bash
cp data/models/mlp_td4_*_float32.h src/model.h
pio run -e real_time_inference -t upload
```

---

## 📜 Scriptler

### 1. **feature_extraction.py**

#### 📝 Ne İşe Yarar?

Ham EMG sinyallerini (ADC değerleri) makine öğrenmesi için uygun sayısal özelliklere dönüştürür.

**Giriş:** CSV dosyası (6 EMG sensörü × N örnek)
```csv
Timestamp,EMG1,EMG2,EMG3,EMG4,EMG5,EMG6,Movement,Phase
0.000,2048,2100,1950,2200,2050,2150,Fist,HAREKET
0.001,2055,2105,1955,2205,2055,2155,Fist,HAREKET
...
```

**Çıkış:** NPZ dosyası (sıkıştırılmış NumPy array)
```python
{
  'features': array([[0.15, 0.32, 0.08, ...], ...]),  # Shape: (n_windows, 24)
  'labels': array(['Fist', 'Open', 'Rest', ...]),     # Shape: (n_windows,)
  'feature_names': ['mav_EMG1', 'wl_EMG1', ...],      # Shape: (24,)
  'sensor_columns': ['EMG1', 'EMG2', ..., 'EMG6']     # Shape: (6,)
}
```

#### 🔬 İşlem Adımları

**ADIM 0: DSP Filtreleme (İsteğe Bağlı, Varsayılan: Açık)**
```
Ham ADC → High-Pass (20 Hz) → Low-Pass (450 Hz) → Notch (50/60 Hz) → Temiz EMG
```
- **Amaç:** Gürültü azaltma, DC drift kaldırma, powerline interferans temizleme
- **C++ ile uyumlu:** ESP32'deki filtrelerle aynı (validate_filters.py ile doğrulanır)

**ADIM 1: HAZIRLIK Verilerini Temizleme**
```python
# "HAZIRLIK" fazındaki veriler kaldırılır (hareket öncesi)
df = df[~df['Phase'].str.upper().str.contains('HAZIRLIK')]
```
- **Neden?** HAZIRLIK sırasında kas kasılmamış, bu veri modeli yanıltır

**ADIM 2: Pencere Oluşturma (Sliding Window)**
```
Window size: 250ms (250 örnek @ 1000 Hz)
Overlap: 125ms (50% örtüşme)
Hop size: 125 örnek
```
- Her pencere 250 ardışık EMG örneği içerir
- %50 örtüşme daha fazla eğitim verisi sağlar

**ADIM 3: TD4 Özellik Hesaplama (Her Sensör İçin)**

Her 250 örneklik pencere için:

1. **DC Offset Kaldırma**
   ```python
   mean_val = np.mean(raw_signal)
   centered_signal = raw_signal - mean_val
   ```

2. **TD4 Özellikleri Hesapla**
   - **MAV (Mean Absolute Value):** Ortalama genlik
     ```python
     mav = np.mean(np.abs(centered_signal))
     ```
   - **WL (Waveform Length):** Sinyal karmaşıklığı
     ```python
     wl = np.sum(np.abs(np.diff(centered_signal)))
     ```
   - **ZC (Zero Crossings):** Frekans tahmini
     ```python
     zc = count_zero_crossings(centered_signal, threshold=15.0)
     ```
   - **SSC (Slope Sign Changes):** Frekans içeriği
     ```python
     ssc = count_slope_sign_changes(centered_signal, threshold=15.0)
     ```

3. **Global Normalizasyon** (ADC_MAX = 4095)
   ```python
   mav_normalized = mav / 4095.0
   wl_normalized = wl / (4095.0 * window_length)
   zc_normalized = zc / window_length
   ssc_normalized = ssc / window_length
   ```

**ADIM 4: Rest Sınıfını Dengeleme**
```python
# "Rest" sınıfı genellikle fazla, diğer sınıfların 1.2× medyanına düşürülür
target_count = int(np.median(non_rest_counts) * 1.2)
```

#### ⚙️ Parametreler

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `window_size_ms` | 250 | Analiz penceresi (ms) - **C++ ile eşleşmeli** |
| `overlap_ms` | 125 | Pencere örtüşmesi (ms) |
| `sampling_rate` | 1000 | Örnekleme frekansı (Hz) |
| `zc_threshold_adc` | 15.0 | Zero crossing eşiği (ADC birimi) |
| `ssc_threshold_adc` | 15.0 | Slope sign change eşiği (ADC birimi) |
| `adc_max` | 4095.0 | 12-bit ADC maksimum değeri |
| `apply_filters` | True | DSP filtrelerini uygula |
| `powerline_freq` | 50 | Powerline frekansı (50 Hz Avrupa, 60 Hz Amerika) |

#### 💾 Çıktı Dosyası

**Konum:** `scripts_ai/data/features/[DOSYA_ADI]_td4_features.npz`

**İçerik:**
```python
import numpy as np

data = np.load('training_data_20251130_td4_features.npz')
print(data['features'].shape)      # (n_windows, 24)
print(data['labels'].shape)        # (n_windows,)
print(data['feature_names'])       # ['mav_EMG1', 'wl_EMG1', 'zc_EMG1', ...]
print(np.unique(data['labels']))   # ['Rest', 'Fist', 'Open', ...]
```

---

### 2. **train_test_model.py**

#### 📝 Ne İşe Yarar?

TD4 özelliklerini kullanarak **Wide & Deep MLP** (Çok Katmanlı Algılayıcı) modelini eğitir ve ESP32 için TensorFlow Lite formatına dönüştürür.

#### 🧠 Model Mimarisi

```
Input (24 features)
    ↓
Dense(256) + ReLU + Dropout(0.3)    ← Wide layer
    ↓
Dense(128) + ReLU + Dropout(0.2)    ← Deep layer 1
    ↓
Dense(64) + ReLU + Dropout(0.1)     ← Deep layer 2
    ↓
Dense(11) + Softmax                  ← Output (11 gestures)
```

**Toplam Parametre:** ~86,000 (Float32: ~344 KB)

#### 🔬 İşlem Adımları

**ADIM 1: Veri Yükleme**
```python
# En yeni NPZ dosyasını otomatik bulur
npz_files = glob('data/features/*_td4_features.npz')
latest = max(npz_files, key=os.path.getctime)
```

**ADIM 2: Label Encoding (ÇOK ÖNEMLİ!)**
```python
# SABİT hareket sırası (C++ ile AYNI olmalı!)
GESTURE_NAMES_FIXED = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]

# Alfabetik sıralama KULLANILMAZ (eski bug!)
label_map = {label: idx for idx, label in enumerate(GESTURE_NAMES_FIXED)}
```
- **Neden önemli?** Model "Fist"i index 1 olarak öğrenir
- ESP32 de aynı indexi "Fist" olarak yorumlamalı
- Sıra değişirse tahminler yanlış eşleşir!

**ADIM 3: Veri Bölme (Train/Validation/Test)**
```python
# TEMPORAL LEAKAGE ÖNLEMİ!
X_train, X_temp = train_test_split(features, shuffle=False)  # shuffle=False!
X_val, X_test = train_test_split(X_temp, shuffle=False)

# Dağılım: %60 Train / %20 Validation / %20 Test
```
- **shuffle=False neden?** Ardışık pencereler aynı hareket → karıştırırsan train/test'e dağılır → yapay yüksek accuracy!

**ADIM 4: Model Eğitimi**
```python
# Callbacks
early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)

# Eğitim
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)
```

**ADIM 5: Model Değerlendirme**
```python
# Test seti üzerinde performans
test_results = model.evaluate(X_test, y_test)
# Metrics: loss, accuracy, precision, recall

# Confusion matrix ve classification report
cm = confusion_matrix(y_true, y_pred)
```

**ADIM 6: TFLite Dönüşümü (Float32)**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# NO quantization - Float32 for ESP32 compatibility
tflite_model = converter.convert()
```
- **Neden Float32?** ESP32'deki TFLite v2.1.1 ile uyumlu
- **Neden Int8 değil?** Eski TFLite versiyonlarında operator versiyonu hataları

**ADIM 7: C++ Header Oluşturma**
```python
# TFLite modeli byte array olarak C++ header'a dönüştürülür
c_array = ', '.join([f'0x{b:02x}' for b in tflite_data])

header = f"""
const unsigned char model_tflite[] = {{
  {c_array}
}};
const unsigned int model_tflite_len = {len(tflite_data)};
"""
```

#### ⚙️ Hiperparametreler (Dosya Başında Tanımlı)

```python
# Training parameters
EPOCHS = 150                      # Maksimum epoch sayısı
BATCH_SIZE = 32                   # Mini-batch boyutu
LEARNING_RATE = 0.001            # Adam optimizer learning rate
VALIDATION_SPLIT = 0.2           # %20 validation
TEST_SPLIT = 0.2                 # %20 test

# Model architecture
LAYER_1_UNITS = 256              # İlk katman nöron sayısı
LAYER_2_UNITS = 128              # İkinci katman
LAYER_3_UNITS = 64               # Üçüncü katman
DROPOUT_RATE_1 = 0.3             # İlk dropout oranı
DROPOUT_RATE_2 = 0.2             # İkinci dropout
DROPOUT_RATE_3 = 0.1             # Üçüncü dropout

# Early stopping
EARLY_STOPPING_PATIENCE = 15     # Kaç epoch iyileşme yoksa dur
```

**Bu parametreleri değiştirerek:**
- Daha büyük model: LAYER_*_UNITS artır (daha yüksek accuracy, daha yavaş)
- Daha küçük model: LAYER_*_UNITS azalt (daha hızlı, daha düşük accuracy)
- Overfitting varsa: DROPOUT_RATE artır
- Underfitting varsa: DROPOUT_RATE azalt veya EPOCHS artır

#### 💾 Çıktı Dosyaları

**1. Keras Modeli** (`.keras`)
- TensorFlow formatında tam model
- Python'da yeniden yüklenebilir
- **Boyut:** ~344 KB

**2. TFLite Modeli** (`.tflite`)
- TensorFlow Lite formatı (Float32)
- Mobile/embedded cihazlar için optimize
- **Boyut:** ~340 KB

**3. C++ Header** (`.h`)
- ESP32 için `model_tflite[]` byte array
- `src/model.h` olarak kopyalanmalı
- **Boyut:** ~1.2 MB (hex string formatı)

**4. Training History Plot** (`training_history_*.png`)
- Accuracy ve Loss grafikleri
- Train vs Validation karşılaştırması

**5. Confusion Matrix** (`confusion_matrix_*.png`)
- 11×11 matris (11 hareket)
- Hangi hareketlerin karıştığını gösterir

#### 📊 Çıktı Yorumlama

**Training History Grafiği:**
```
Accuracy: Train ↑, Validation ↑ → İyi!
Accuracy: Train ↑, Validation ~ → Overfitting
Loss: Train ↓, Validation ↓ → İyi!
Loss: Train ↓, Validation ↑ → Overfitting
```

**Confusion Matrix:**
```
           Rest  Fist  Open  ...
Rest       [245]   2     1   ...  ← Rest 245 kez doğru tahmin edildi
Fist         3  [198]   5   ...  ← Fist 3 kez Rest olarak yanlış tahmin
Open         1    4  [210]  ...
```
- Diyagonal (köşegen): Doğru tahminler
- Diyagonal dışı: Hatalı tahminler
- Hangi hareketler karışıyor? → Daha fazla veri topla

---

## 🚀 Adım Adım Kullanım

### **Senaryo 1: İlk Kez Model Eğitimi**

```bash
# ADIM 1: Proje klasörüne git
cd C:\Users\MERT\Documents\PlatformIO\Projects\functions_real_time\scripts_ai

# ADIM 2: Veri toplama (ESP32'den)
cd ../data_acquisition/scripts
python training_collection.py
# Çıktı: ../../data/training_data_20251130_143022.csv

# ADIM 3: Özellik çıkarımı
cd ../../scripts_ai
python critical/feature_extraction.py ../data/training_data_20251130_143022.csv
# Çıktı: data/features/training_data_20251130_143022_td4_features.npz

# ADIM 4: Model eğitimi
python critical/train_test_model.py
# Çıktılar:
#   data/models/mlp_td4_20251130_143522.keras
#   data/models/mlp_td4_20251130_143522_float32.tflite
#   data/models/mlp_td4_20251130_143522_float32.h
#   data/plots/training_history_20251130_143522.png
#   data/plots/confusion_matrix_20251130_143522.png

# ADIM 5: ESP32'ye yükleme
cp data/models/mlp_td4_20251130_143522_float32.h ../src/model.h
cd ..
pio run -e real_time_inference -t upload
```

### **Senaryo 2: Farklı DSP Filtresi ile Özellik Çıkarımı**

```bash
# 60 Hz powerline (Amerika) için
python critical/feature_extraction.py data/training_data.csv --powerline-freq 60

# DSP filtrelerini devre dışı bırak (önerilmez!)
python critical/feature_extraction.py data/training_data.csv --no-filters
```

### **Senaryo 3: Hiperparametre Değiştirme**

`critical/train_test_model.py` dosyasını aç ve başındaki sabitleri değiştir:

```python
# Daha büyük model (daha yüksek accuracy için)
LAYER_1_UNITS = 512  # 256 → 512
LAYER_2_UNITS = 256  # 128 → 256
LAYER_3_UNITS = 128  # 64 → 128
EPOCHS = 200         # 150 → 200

# Daha agresif regularization (overfitting varsa)
DROPOUT_RATE_1 = 0.5  # 0.3 → 0.5
DROPOUT_RATE_2 = 0.4  # 0.2 → 0.4
```

Sonra tekrar çalıştır:
```bash
python critical/train_test_model.py
```

---

## 🔧 Detaylı Parametre Açıklamaları

### **feature_extraction.py - TD4FeatureExtractor Class**

```python
extractor = TD4FeatureExtractor(
    window_size_ms=250,        # Analiz penceresi (ms)
    overlap_ms=125,            # Pencere örtüşmesi (ms)
    sampling_rate=1000,        # Örnekleme frekansı (Hz)
    zc_threshold_adc=15.0,     # Zero crossing eşiği (ADC birimi)
    ssc_threshold_adc=15.0,    # Slope sign change eşiği (ADC birimi)
    adc_max=4095.0             # 12-bit ADC maksimum
)
```

**window_size_ms (250)**
- Hangi süre zarfında EMG sinyali analiz edilecek?
- 250ms = **hızlı tepki** (gerçek zamanlı kontrol için ideal)
- 500ms = daha stabil özellikler, ama yavaş tepki
- **C++ ile EŞLEŞMELİ!** (`RAW_WINDOW_SIZE` in functions.h)

**overlap_ms (125)**
- Ardışık pencereler ne kadar örtüşecek?
- 125ms = %50 örtüşme → **2× daha fazla eğitim verisi**
- 0ms = örtüşme yok → daha az veri
- 200ms = %80 örtüşme → çok fazla benzer veri (overfitting riski)

**zc_threshold_adc / ssc_threshold_adc (15.0)**
- Gürültüyü filtrelemek için minimum değişim
- 15.0 ADC birimi = 12-bit ADC'nin %0.37'si
- Çok düşük → gürültü ZC/SSC olarak sayılır
- Çok yüksek → gerçek frekans bileşenleri kaçırılır

### **train_test_model.py - Model Architecture**

**Neden Wide & Deep?**
- **Wide (256 nöron):** Ham özellikleri doğrudan öğrenir (hızlı)
- **Deep (128→64):** Karmaşık kombinasyonları öğrenir (doğru)
- **Dropout:** Overfitting önleme (test accuracy'yi artırır)

**Dropout Oranları:**
```python
DROPOUT_RATE_1 = 0.3  # İlk katman: %30 nöron rastgele kapatılır
DROPOUT_RATE_2 = 0.2  # İkinci katman: %20
DROPOUT_RATE_3 = 0.1  # Üçüncü katman: %10
```
- İlk katmanlarda daha yüksek dropout → genel özellikleri öğrenir
- Son katmanlarda daha düşük dropout → spesifik kombinasyonları öğrenir

**Early Stopping:**
```python
EARLY_STOPPING_PATIENCE = 15  # 15 epoch iyileşme yoksa dur
```
- Validation loss 15 epoch boyunca düşmezse eğitim durur
- Overfitting'i önler (gereksiz epoch'larda zaman kaybetmez)

---

## 📦 Çıktı Dosyaları

### **NPZ Dosyası Yapısı** (feature_extraction.py çıktısı)

```python
import numpy as np

# NPZ dosyasını yükle
data = np.load('data/features/training_data_20251130_td4_features.npz')

# İçerik:
features = data['features']          # Shape: (n_windows, 24)
labels = data['labels']              # Shape: (n_windows,)
feature_names = data['feature_names'] # ['mav_EMG1', 'wl_EMG1', ...]
sensor_columns = data['sensor_columns'] # ['EMG1', 'EMG2', ..., 'EMG6']

# Örnek:
print(features[0])
# [0.15, 0.32, 0.08, 0.12,  # EMG1: MAV, WL, ZC, SSC
#  0.18, 0.28, 0.10, 0.15,  # EMG2: MAV, WL, ZC, SSC
#  ...]

print(labels[0])   # 'Fist'
```

### **TFLite Model Boyutu**

| Format | Boyut | Açıklama |
|--------|-------|----------|
| Keras (.keras) | ~344 KB | Tam model (Float32) |
| TFLite Float32 (.tflite) | ~340 KB | ESP32 için optimize |
| C++ Header (.h) | ~1.2 MB | Hex string formatı (kaynak kod) |

**ESP32'de kullanım:**
```cpp
#include "model.h"  // model_tflite[] array tanımlı

// TFLite Micro ile yükle
model = tflite::GetModel(model_tflite);
```

---

## 💡 Örnek Komutlar

### **Özellik Çıkarımı - Farklı Seçenekler**

```bash
# Standart kullanım (DSP filtreleri aktif, 50 Hz)
python critical/feature_extraction.py data/training_data.csv

# Amerika için (60 Hz powerline)
python critical/feature_extraction.py data/training_data.csv --powerline-freq 60

# DSP filtrelerini devre dışı bırak (debug için)
python critical/feature_extraction.py data/training_data.csv --no-filters

# Farklı sensör isimleri
python critical/feature_extraction.py data/training_data.csv \
    --sensor-columns Sensor1 Sensor2 Sensor3 Sensor4 Sensor5 Sensor6
```

### **Model Eğitimi - Batch İşleme**

```bash
# Birden fazla veri dosyasını birleştir
cd scripts_ai

# 1. Her dosyadan özellik çıkar
python critical/feature_extraction.py data/session1.csv
python critical/feature_extraction.py data/session2.csv
python critical/feature_extraction.py data/session3.csv

# 2. NPZ dosyalarını Python'da birleştir (custom script yazman gerekir)
python merge_features.py  # Örnek: 3 NPZ'yi birleştir

# 3. Birleştirilmiş veri ile model eğit
python critical/train_test_model.py
```

---

## 🐛 Troubleshooting

### **Problem 1: "No *_td4_features.npz files found"**

**Neden:** `train_test_model.py` çalışıyor ama NPZ dosyası yok.

**Çözüm:**
```bash
# Önce özellik çıkarımı yap
python critical/feature_extraction.py data/training_data.csv

# Sonra model eğit
python critical/train_test_model.py
```

---

### **Problem 2: "ValueError: Gesture names in data do not match GESTURE_NAMES_FIXED!"**

**Neden:** CSV dosyasındaki hareket isimleri `GESTURE_NAMES_FIXED` ile uyuşmuyor.

**Kontrol et:**
```python
# train_test_model.py içinde:
GESTURE_NAMES_FIXED = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]

# CSV'deki Movement sütunu bu isimlerle TAMAMEN aynı olmalı!
```

**Çözüm 1:** CSV'deki hareket isimlerini düzelt
**Çözüm 2:** `GESTURE_NAMES_FIXED` listesini CSV'ye göre güncelle (ÖNERİLMEZ - C++ de değiştirmen gerekir!)

---

### **Problem 3: Model accuracy çok düşük (~10-20%)**

**Olası nedenler:**

1. **Veri kalitesi kötü**
   ```bash
   # Veriyi kontrol et
   python validation/validate_noise_data.py data/training_data.csv
   ```
   - ADC overflow var mı?
   - Sensör variyansı çok düşük mü?

2. **DSP filtreleri tutarsız**
   ```bash
   # Python ve C++ filtrelerini karşılaştır
   python validation/validate_filters.py
   ```
   - Filtre katsayıları aynı mı?

3. **Hiperparametreler kötü**
   - DROPOUT_RATE çok yüksek → underfitting
   - LAYER_*_UNITS çok düşük → model kapasitesi yetersiz
   - EPOCHS çok az → eğitim tamamlanmamış

**Çözüm:**
```python
# train_test_model.py içinde:
LAYER_1_UNITS = 512  # 256'dan artır
LAYER_2_UNITS = 256
EPOCHS = 200         # 150'den artır
DROPOUT_RATE_1 = 0.2 # 0.3'ten azalt (overfitting yoksa)
```

---

### **Problem 4: Overfitting (Train accuracy yüksek, Test accuracy düşük)**

**Belirtiler:**
```
Train Accuracy: 98%
Test Accuracy: 65%  ← PROBLEM!
```

**Çözüm 1:** Dropout artır
```python
DROPOUT_RATE_1 = 0.5  # 0.3 → 0.5
DROPOUT_RATE_2 = 0.4  # 0.2 → 0.4
```

**Çözüm 2:** L2 regularization artır
```python
# train_test_model.py, build_wide_deep_mlp() fonksiyonunda:
model.add(layers.Dense(
    256, activation='relu',
    kernel_regularizer=regularizers.l2(0.01)  # 0.001 → 0.01
))
```

**Çözüm 3:** Daha fazla veri topla
```bash
# 2-3 kez daha veri toplama oturumu yap
python data_acquisition/scripts/training_collection.py
```

---

### **Problem 5: "Confusion matrix shows random predictions"**

**Neden:** Tüm hareketler birbirine karışıyor (model random tahmin yapıyor).

**Kontrol et:**
```bash
# 1. Veri kalitesi
python validation/validate_noise_data.py data/training_data.csv

# 2. Preprocessing tutarlılığı
python validation/validate_preprocessing.py data/training_data.csv

# 3. Label sırası tutarlılığı
# train_test_model.py'deki GESTURE_NAMES_FIXED ile
# src/main.cpp'deki gesture_names[] aynı sırada mı?
```

**Çözüm:** Yukarıdaki sorunları düzelt ve yeniden eğit.

---

### **Problem 6: ESP32'de tahminler yanlış (Python'da doğru)**

**Neden:** Label sırası tutarsızlığı!

**Kontrol et:**
```cpp
// src/main.cpp içinde:
const char* gesture_names[] = {
    "Rest", "Fist", "Open", "Point", "Victory", "OK",
    "ThumbUp", "ThumbDn", "Grasp", "Pinch", "WristFlex"
};
```

```python
# train_test_model.py içinde:
GESTURE_NAMES_FIXED = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]
```

**TAMAMEN AYNI SIRADA OLMALI!**

---

### **Problem 7: TFLite dönüşümü hatası**

```
ValueError: Cannot set tensor: Dimension mismatch
```

**Neden:** Model input/output boyutları uyumsuz.

**Kontrol et:**
```python
# Model input shape:
print(model.input_shape)  # (None, 24) olmalı

# Model output shape:
print(model.output_shape)  # (None, 11) olmalı

# NPZ features shape:
data = np.load('features.npz')
print(data['features'].shape)  # (n_samples, 24) olmalı
```

---

### **Problem 8: "Memory allocation failed" (ESP32'de)**

**Neden:** Model ESP32'nin SRAM'ine sığmıyor.

**Çözüm 1:** Tensor arena boyutunu artır
```cpp
// src/main.cpp içinde:
constexpr int kTensorArenaSize = 40 * 1024;  // 30KB → 40KB
```

**Çözüm 2:** Model boyutunu küçült
```python
# train_test_model.py içinde:
LAYER_1_UNITS = 128  # 256 → 128
LAYER_2_UNITS = 64   # 128 → 64
LAYER_3_UNITS = 32   # 64 → 32
```

---

## 🔬 Teknik Detaylar

### **TD4 Özellikleri Matematiksel Formüller**

Bir EMG sinyal penceresi \( x[n] \) için (n = 0, 1, ..., N-1):

**1. MAV (Mean Absolute Value)**
```
MAV = (1/N) × Σ|x[n]|
```
- Sinyalin ortalama genliği
- Kas kasılma gücü ile doğru orantılı

**2. WL (Waveform Length)**
```
WL = Σ|x[n+1] - x[n]|
```
- Sinyal karmaşıklığı
- Yüksek frekans bileşenleri varsa artar

**3. ZC (Zero Crossings)**
```
ZC = Σ[sgn(x[n] × x[n+1] < 0) AND |x[n] - x[n+1]| ≥ threshold]
```
- Sinyalin sıfırı kaç kez geçtiği
- Frekans tahmini (daha fazla ZC = daha yüksek frekans)

**4. SSC (Slope Sign Changes)**
```
SSC = Σ[sgn((x[n] - x[n-1]) × (x[n] - x[n+1])) AND |(x[n] - x[n-1]) × (x[n] - x[n+1])| ≥ threshold]
```
- Eğim işareti değişimleri (tepe ve vadiler)
- Frekans içeriği göstergesi

### **Normalizasyon Stratejisi**

**Global ADC Normalizasyon (Kullanılan Yöntem):**
```python
mav_normalized = mav / ADC_MAX  # ADC_MAX = 4095
```
- **Avantaj:** Sensörler arası genlik farkları korunur
- **Dezavantaj:** Sensör montajı değişirse normalize edilmiş değerler değişir

**Alternatif: Per-Window Normalizasyon**
```python
mav_normalized = (mav - mav_min) / (mav_max - mav_min)
```
- **Avantaj:** Her pencere [0,1] aralığına ölçeklenir
- **Dezavantaj:** Genlik bilgisi kaybolur (zayıf/güçlü kasılma ayırt edilemez)

### **Sliding Window Parametreleri**

```
Window Size: 250ms
Overlap: 125ms (50%)
Hop Size: 125ms

Örnek:
Window 1: samples [0:250]     → t = 0.000s - 0.250s
Window 2: samples [125:375]   → t = 0.125s - 0.375s
Window 3: samples [250:500]   → t = 0.250s - 0.500s
```

**Neden 50% örtüşme?**
- Daha fazla eğitim verisi (2× pencere sayısı)
- Hareket geçişlerini kaçırma riski azalır
- Temporal smoothness (zamansal pürüzsüzlük)

### **Wide & Deep Mimarisi (Cheng et al. 2016)**

**Wide Component (İlk katman):**
- Doğrusal kombinasyonlar öğrenir
- "Bu sensör aktif → Fist" gibi basit kurallar

**Deep Component (2.-3. katmanlar):**
- Non-linear kombinasyonlar öğrenir
- "EMG1 yüksek VE EMG3 düşük VE EMG5 orta → Point" gibi karmaşık desenler

**Neden Dropout?**
- Eğitim sırasında rastgele nöronlar devre dışı bırakılır
- Model tek bir özelliğe bağımlı kalmaz → generalization iyileşir

### **Float32 vs Int8 Quantization**

| Özellik | Float32 | Int8 |
|---------|---------|------|
| **Boyut** | 344 KB | ~86 KB (4× küçük) |
| **Hız** | Baseline | 2-3× hızlı |
| **Accuracy** | %100 | %95-98 (küçük düşüş) |
| **ESP32 Uyumluluğu** | ✅ TFLite v2.1.1+ | ⚠️ TFLite v2.4+ gerekli |

**Bu projede Float32 kullanılıyor çünkü:**
- ESP32'deki TFLite Micro v2.1.1 (eski versiyon)
- Int8 operator version hataları veriyor
- Float32 yeterince hızlı (20ms inference @ 240MHz)

---

## 📚 Referanslar

**TD4 Özellikleri:**
- Hudgins et al. (1993): "A New Strategy for Multifunction Myoelectric Control"
- Phinyomark et al. (2012): "Feature Reduction and Selection for EMG Signal Classification"

**Wide & Deep Architecture:**
- Cheng et al. (2016): "Wide & Deep Learning for Recommender Systems"

**EMG Sınıflandırma:**
- Atzori et al. (2014): "Electromyography data for non-invasive naturally-controlled robotic hand prostheses"
- Oskoei & Hu (2007): "Myoelectric Control Systems—A Survey"

**DSP Filtering:**
- De Luca (2002): "Surface Electromyography: Detection and Recording"

---

## 🎓 Öğrenme Kaynakları

### **Başlangıç Seviyesi**
1. [Keras Sequential Model Guide](https://keras.io/guides/sequential_model/)
2. [EMG Signal Processing Basics](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4105824/)

### **Orta Seviye**
1. [Understanding Dropout](https://jmlr.org/papers/v15/srivastava14a.html)
2. [TensorFlow Lite Micro Overview](https://www.tensorflow.org/lite/microcontrollers)

### **İleri Seviye**
1. TD4 features paper: [ResearchGate Link](https://www.researchgate.net/publication/224146154_A_new_strategy_for_multifunction_myoelectric_control)
2. Wide & Deep Learning: [Google Research Blog](https://research.google/pubs/pub45413/)

---

## 📞 İletişim & Destek

**Sorularınız için:**
- GitHub Issues: [functions_real_time/issues](https://github.com/your-repo/issues)
- Proje dokümantasyonu: [README.md](../../README.md)
- Script açıklamaları: [SCRIPT_EXPLANATIONS.md](../SCRIPT_EXPLANATIONS.md)

**Bu README'yi güncel tutun!**
- Yeni özellik eklediyseniz buraya ekleyin
- Hiperparametre değiştirdiyseniz belirtin
- Yeni troubleshooting bulguları paylaşın

---

**Son güncelleme:** 2025-11-30
**Versiyon:** 1.0
**Yazar:** ESP32 Bionic Hand Project
