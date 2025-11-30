# 📚 scripts_ai - Tüm Script'lerin Detaylı Açıklaması

Bu doküman, `scripts_ai` klasöründeki **her bir script'in ne işe yaradığını** detaylıca açıklar.

---

## 📂 Yeni Klasör Yapısı

```
scripts_ai/
├── README.md                          # Ana README (bu dosya değil, genel README)
├── SCRIPT_EXPLANATIONS.md             # Bu dosya (detaylı açıklamalar)
│
├── critical/                          # 🎯 END-TO-END PIPELINE (Kritik)
│   ├── README.md
│   ├── feature_extraction.py          # TD4 özellikleri çıkarır
│   └── train_test_model.py            # Model eğitir ve TFLite'a dönüştürür
│
├── filters/                           # 🔧 DSP Filtre Araçları
│   ├── README.md
│   ├── dsp_filters.py                 # Python DSP filtre kütüphanesi
│   └── generate_filter_coefficients.py # C++ katsayı üretici
│
├── validation/                        # ✅ Kalite Kontrol Araçları
│   ├── README.md
│   ├── validate_noise_data.py         # ADC overflow ve sensör kontrolü
│   ├── validate_filters.py            # Python-C++ filtre uyum testi
│   ├── validate_preprocessing.py      # Feature extraction doğrulama
│   └── validate_pipeline.py           # End-to-end veri akışı testi
│
└── visualization/                     # 📊 Görselleştirme Araçları
    ├── README.md
    └── visualize_signal_quality.py    # EMG sinyal kalitesi analizi
```

---

## 🎯 critical/ - Ana İşlem Hattı

### 1. feature_extraction.py

**Ne İşe Yarar:**  
Ham EMG verilerinden (CSV) makine öğrenimi için **TD4 özellikleri** çıkarır.

**Giriş:**
- CSV dosyası (EMG1-EMG6 sütunları, Movement, Phase)

**Çıkış:**
- NPZ dosyası (24 TD4 features × N windows)

**TD4 Özellikleri:**
1. **MAV (Mean Absolute Value):** Ortalama genlik
2. **WL (Waveform Length):** Sinyal karmaşıklığı
3. **ZC (Zero Crossings):** Sıfır geçiş sayısı
4. **SSC (Slope Sign Changes):** Eğim değişim sayısı

**İşlem Adımları:**
```
CSV → DSP Filtre (HPF+LPF+Notch) → HAZIRLIK filtrele → 
Sliding Window (250ms, 125ms overlap) → TD4 Features → 
Rest Balance → Min-Max Normalize → NPZ kaydet
```

**Kullanım:**
```bash
python scripts_ai/critical/feature_extraction.py data/training_data.csv
```

**Çıktı:**
```
data/features/td4_features_YYYYMMDD_HHMMSS.npz
```

**Önemli Notlar:**
- DSP filtreleri otomatik uygulanır (HPF 20Hz, LPF 450Hz, Notch 50Hz)
- HAZIRLIK fazındaki veriler otomatik filtrelenir
- Rest sınıfı balanced edilir (1.2× median)
- C++ firmware ile tam uyumlu

---

### 2. train_test_model.py

**Ne İşe Yarar:**  
TD4 özelliklerini kullanarak **Wide & Deep MLP** modelini eğitir ve ESP32 için **TFLite** formatına dönüştürür.

**Giriş:**
- NPZ dosyası (feature_extraction.py çıktısı)
- Otomatik olarak en son NPZ'yi bulur

**Çıkış:**
1. TFLite model (ESP32 için): `emg_model_YYYYMMDD_HHMMSS.tflite`
2. C++ header: `emg_model_YYYYMMDD_HHMMSS.h`
3. Training plots: `confusion_matrix.png`, `training_history.png`

**Model Mimarisi:**
```
Input(24) → Dense(256) → Dropout(0.3) →
Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.1) →
Dense(11) Softmax
```

**Eğitim Parametreleri (değiştirilebilir):**
- EPOCHS: 150
- BATCH_SIZE: 32
- LEARNING_RATE: 0.001
- VALIDATION_SPLIT: 0.2
- TEST_SPLIT: 0.2

**Kullanım:**
```bash
python scripts_ai/critical/train_test_model.py
```

**Çıktı Dizini:**
```
data/models/
├── emg_model_20251126_010530.tflite
├── emg_model_20251126_010530.h
├── confusion_matrix.png
└── training_history.png
```

**Önemli Notlar:**
- **FIXED gesture order** kullanır (alfabetik değil!)
- C++ ile aynı label sırası
- Float32 TFLite (maksimum uyumluluk)
- Early stopping ve learning rate scheduling

---

## 🔧 filters/ - DSP Filtre Araçları

### 1. dsp_filters.py

**Ne İşe Yarar:**  
Python tarafında EMG sinyallerini filtrelemek için **IIR filter bank** sağlar.

**Filtreler:**
1. **HPF (High-Pass):** 20 Hz, 4th-order Butterworth → DC offset kaldırır
2. **LPF (Low-Pass):** 450 Hz, 4th-order Butterworth → Yüksek frekans gürültüsü
3. **Notch:** 50/60 Hz, 2nd-order IIR → Powerline gürültüsü

**Kullanım (NumPy array):**
```python
from filters.dsp_filters import EMGFilterBank

filter_bank = EMGFilterBank(
    sampling_rate=2000,
    powerline_freq=50
)

filtered = filter_bank.filter_signal(raw_emg)
```

**Kullanım (Pandas DataFrame):**
```python
filtered_df = filter_bank.filter_dataframe(
    df,
    sensor_columns=['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6']
)
```

**Frekans Yanıtı:**
```python
filter_bank.plot_frequency_response(save_path='response.png')
```

**Önemli Notlar:**
- SOS (Second-Order Sections) format kullanır → Sayısal kararlılık
- Zero-phase filtering (`sosfiltfilt`)
- C++ ile aynı katsayılar

---

### 2. generate_filter_coefficients.py

**Ne İşe Yarar:**  
ESP32 firmware'i için **C++ formatta** filtre katsayıları üretir.

**Kullanım:**
```bash
# 50 Hz powerline (Türkiye, Avrupa)
python scripts_ai/filters/generate_filter_coefficients.py 50

# 60 Hz powerline (Amerika)
python scripts_ai/filters/generate_filter_coefficients.py 60
```

**Çıktı:**
1. Terminal: Frekans yanıtı özeti + C++ kodu
2. Dosya: `../src/filter_coefficients_50hz.h`

**İçerik:**
```cpp
// 1000 Hz SAMPLING RATE
const float HPF_STAGE1_B0 = 0.9565436241f;
const float HPF_STAGE1_B1 = -1.9130872481f;
...

// 2000 Hz SAMPLING RATE
#ifdef DATA_ACQUISITION_MODE
const float HPF_STAGE1_B0 = 0.9823456789f;
...
#endif
```

**Neden İki Set Katsayı?**
- 1000 Hz: Real-time inference (ESP32)
- 2000 Hz: Data acquisition (training için veri toplama)

**Önemli Notlar:**
- Katsayılar a0 ile normalize edilir
- Python ile tam uyumlu
- Powerline frekansı doğru seçilmeli (Türkiye için 50 Hz)

---

## ✅ validation/ - Kalite Kontrol Araçları

### 1. validate_noise_data.py

**Ne İşe Yarar:**  
Veri kalitesi sorunlarını tespit eder.

**Kontrol Edilen Şeyler:**
1. **ADC Overflow:** 12-bit aşımı (> 4095)
2. **Düşük Varyans:** Kopuk/arızalı sensörler
3. **Gesture Separability:** Sınıflar ayırt edilebilir mi?

**Kullanım:**
```bash
python scripts_ai/validation/validate_noise_data.py data/training_data.csv
```

**Örnek Çıktı:**
```
✅ ADC Range Check:
   EMG1: Max=4095 ✅
   EMG2: Max=65535 ⚠️ OVERFLOW!

✅ Sensor Variance:
   EMG1: σ²=1250.5 ✅
   EMG5: σ²=45.2 ⚠️ LOW!

✅ Gesture Separability: 2.34 ✅
```

**Ne Zaman Kullan:**
- Her veri toplama sonrası
- Model accuracy düşükse
- Sensör değişikliği sonrası

---

### 2. validate_filters.py

**Ne İşe Yarar:**  
Python ve C++ filtrelerin **aynı sonucu** verdiğini doğrular.

**Test Edilen Şeyler:**
1. WL normalization bug check
2. Frekans yanıtı (HPF, LPF, Notch)
3. Time-domain filtering
4. TD4 feature extraction

**Kullanım:**
```bash
python scripts_ai/validation/validate_filters.py
```

**Örnek Çıktı:**
```
✅ WL Normalization: Difference < 1e-6
✅ HPF cutoff: 20.1 Hz (expected 20 Hz)
✅ LPF cutoff: 449.8 Hz (expected 450 Hz)
✅ Notch rejection: -42.3 dB @ 50 Hz
✅ ALL TESTS PASSED
```

**Ne Zaman Kullan:**
- Filtre değişikliği sonrası
- C++ katsayıları update ettikten sonra
- Python-C++ uyumu şüpheli ise

---

### 3. validate_preprocessing.py

**Ne İşe Yarar:**  
Feature extraction pipeline'ının **doğruluğunu** test eder.

**Test Edilen Şeyler:**
1. DC offset removal (mean subtraction)
2. Global normalization (ADC_MAX = 4095)
3. TD4 feature calculation
4. Full pipeline (CSV → Features)

**Kullanım:**
```bash
# Synthetic data ile otomatik test
python scripts_ai/validation/validate_preprocessing.py

# Gerçek CSV ile test
python scripts_ai/validation/validate_preprocessing.py data/test_sample.csv
```

**Örnek Çıktı:**
```
✅ DC Offset Removal: Mean = 0.0000000
✅ Global Normalization: Range [-0.49, 0.51] ✅
✅ TD4 Features:
   MAV: 0.00305 ✅
   WL: 0.12450 ✅
   ZC: 15 ✅
   SSC: 18 ✅
✅ Python-C++ Difference: < 1e-5
```

**Ne Zaman Kullan:**
- Pipeline değişikliği sonrası
- Feature extraction kodunu değiştirdikten sonra
- C++ firmware güncellemesi sonrası

---

### 4. validate_pipeline.py

**Ne İşe Yarar:**  
End-to-end **veri akışını** simüle eder ve label encoding'i doğrular.

**Kontrol Edilen Şeyler:**
1. CSV format ve sütunlar
2. Movement distribution
3. Phase distribution (HAZIRLIK, MOVEMENT, REST)
4. Window estimation
5. Label encoding order (CRITICAL!)
6. EMG data range

**Kullanım:**
```bash
python scripts_ai/validation/validate_pipeline.py data/training_data.csv
```

**Örnek Çıktı:**
```
✅ CSV Structure: 1,234,567 rows, 9 columns
✅ Movements: 11 gestures found
✅ Estimated windows: ~10,000
✅ Label encoding: Matches C++ order
✅ EMG ranges: All within [0, 4095]

Pipeline is ready for training!
```

**Ne Zaman Kullan:**
- Model eğitimi öncesi (zorunlu!)
- Yeni veri seti ile
- Label encoding değişikliği sonrası

---

## 📊 visualization/ - Görselleştirme Araçları

### 1. visualize_signal_quality.py

**Ne İşe Yarar:**  
Her gesture için **EMG sinyal kalitesini** görselleştirir ve analiz eder.

**Çıktılar:**
1. **Signal Samples:** Her gesture için 6 EMG kanalının örnek sinyali
2. **Quality Metrics:** Variance, SNR, peak-to-peak
3. **Quality Summary:** Sensor comparison, separability heatmap

**Kullanım:**
```bash
python scripts_ai/visualization/visualize_signal_quality.py data/training_data.csv
```

**Çıktı Klasörü:**
```
plots/signal_quality/
├── signal_samples_Rest.png
├── signal_samples_Fist.png
├── signal_samples_Open.png
...
├── quality_summary.png
└── separability_heatmap.png
```

**Örnek Çıktı:**
```
📊 Sensor Quality:
   EMG1: σ²=1250.5, SNR=18.3 dB ✅
   EMG2: σ²=1180.3, SNR=17.8 dB ✅
   EMG3: σ²=45.2, SNR=8.2 dB ⚠️ LOW
   
✅ Plots saved to: plots/signal_quality/
```

**Ne Zaman Kullan:**
- Veri toplama sonrası
- Model accuracy düşükse
- Yeni sensör test ederken
- Görsel veri kalitesi analizi için

---

## 🔄 Tam Workflow (Start to Finish)

```
1. VERİ TOPLAMA
   └─→ data_acquisition/scripts/training_data_collection.py
       └─→ CSV: training_data_YYYYMMDD.csv

2. VERİ KALİTE KONTROL
   ├─→ validation/validate_noise_data.py ✅ ADC overflow, varyans
   ├─→ visualization/visualize_signal_quality.py 📊 Sinyal kalitesi
   └─→ validation/validate_pipeline.py ✅ Pipeline test

3. ÖZELLİK ÇIKARIM
   └─→ critical/feature_extraction.py
       └─→ NPZ: td4_features_YYYYMMDD_HHMMSS.npz

4. MODEL EĞİTİMİ
   └─→ critical/train_test_model.py
       ├─→ TFLite: emg_model_YYYYMMDD_HHMMSS.tflite
       ├─→ Header: emg_model_YYYYMMDD_HHMMSS.h
       └─→ Plots: confusion_matrix.png, training_history.png

5. ESP32 DEPLOYMENT
   ├─→ Copy .h to src/
   ├─→ Update firmware
   ├─→ Compile & upload
   └─→ Test!
```

---

## 🛠️ Yardımcı Araçlar (Support)

### DSP Filtre Geliştirme
```
filters/generate_filter_coefficients.py
   └─→ C++ header: filter_coefficients_50hz.h
   
validation/validate_filters.py
   └─→ Python-C++ uyum testi
```

### Pipeline Doğrulama
```
validation/validate_preprocessing.py
   └─→ Feature extraction doğrulama

validation/validate_pipeline.py
   └─→ End-to-end test
```

---

## ❓ Hangi Script Ne Zaman Kullanılır?

### 🆕 İlk Kurulum
1. `filters/generate_filter_coefficients.py` → C++ katsayılar
2. `validation/validate_filters.py` → Filtre uyum testi

### 📊 Veri Toplama Sonrası
1. `validation/validate_noise_data.py` → Veri kalitesi
2. `visualization/visualize_signal_quality.py` → Görsel analiz
3. `validation/validate_pipeline.py` → Pipeline test

### 🎓 Model Eğitimi
1. `critical/feature_extraction.py` → Özellik çıkarım
2. `critical/train_test_model.py` → Model eğit

### 🔧 Debugging
1. **Model accuracy düşük:**
   - `validation/validate_noise_data.py`
   - `visualization/visualize_signal_quality.py`
   
2. **Python-C++ farklı sonuç:**
   - `validation/validate_filters.py`
   - `validation/validate_preprocessing.py`
   
3. **Label encoding hatası:**
   - `validation/validate_pipeline.py`

---

## 📊 Her Script'in Önemi (Kritiklik Sırası)

### ⭐⭐⭐⭐⭐ CRITICAL (Zorunlu)
1. `critical/feature_extraction.py` - Feature çıkarımı
2. `critical/train_test_model.py` - Model eğitimi
3. `validation/validate_noise_data.py` - Veri kalitesi

### ⭐⭐⭐⭐ HIGH (Şiddetle Tavsiye)
4. `validation/validate_pipeline.py` - Pipeline test
5. `filters/dsp_filters.py` - DSP filtreleme
6. `validation/validate_filters.py` - Filtre uyum

### ⭐⭐⭐ MEDIUM (Tavsiye)
7. `validation/validate_preprocessing.py` - Preprocessing test
8. `filters/generate_filter_coefficients.py` - C++ katsayılar

### ⭐⭐ LOW (Opsiyonel)
9. `visualization/visualize_signal_quality.py` - Görsel analiz

---

## 💡 Pro Tips

### Hızlı Test Pipeline
```bash
# Full validation pass
python scripts_ai/validation/validate_noise_data.py data/latest.csv
python scripts_ai/validation/validate_pipeline.py data/latest.csv
python scripts_ai/validation/validate_filters.py
```

### Batch Processing
```bash
# Birden fazla CSV için feature extraction
for csv in data/*.csv; do
    python scripts_ai/critical/feature_extraction.py "$csv"
done
```

### Custom Hiperparametre Testi
```python
# train_test_model.py içinde değiştir:
EPOCHS = 100  # Hızlı test için
LEARNING_RATE = 0.0001  # Daha stabil eğitim
LAYER_1_UNITS = 128  # Daha küçük model
```

---

## 🐛 Genel Troubleshooting

### Import Hatası
```python
# ModuleNotFoundError: No module named 'filters'
# Çözüm: scripts_ai dizininden çalıştır
cd scripts_ai
python critical/feature_extraction.py data.csv
```

### Path Hatası
```python
# FileNotFoundError: data/features/...
# Çözüm: data klasörü oluştur
mkdir -p data/features data/models plots/signal_quality
```

### Memory Error (Büyük CSV)
```python
# MemoryError when loading CSV
# Çözüm: Chunk processing kullan (feature_extraction.py'de)
```

---

**Proje:** ESP32-S3 Bionic Hand  
**Yazar:** MERT  
**Tarih:** 2025-11-26  
**Versiyon:** 2.0 (Yeniden organize edilmiş)
