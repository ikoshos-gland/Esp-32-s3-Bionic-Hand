# 🚀 Hızlı Başlangıç Rehberi

Bu dosya, `scripts_ai` klasöründeki araçları **en hızlı şekilde** kullanmanız için hazırlanmıştır.

---

## ⚡ 3 Dakikada Başlangıç

### 1️⃣ Veri Topladıktan Sonra
```bash
# Veri kalitesini kontrol et
python scripts_ai/validation/validate_noise_data.py data/training_data_20251126.csv

# ✅ ise devam et, ⚠️ ise sensörleri kontrol et
```

---

### 2️⃣ Özellik Çıkar
```bash
# TD4 features üret
python scripts_ai/critical/feature_extraction.py data/training_data_20251126.csv

# Çıktı: data/features/td4_features_YYYYMMDD_HHMMSS.npz
```

---

### 3️⃣ Model Eğit
```bash
# Model eğit ve TFLite oluştur
python scripts_ai/critical/train_test_model.py

# Çıktı: 
#   data/models/emg_model_YYYYMMDD_HHMMSS.tflite
#   data/models/emg_model_YYYYMMDD_HHMMSS.h
```

---

### 4️⃣ ESP32'ye Yükle
```bash
# .h dosyasını kopyala
cp data/models/emg_model_*.h src/emg_model.h

# Firmware compile et
pio run -t upload
```

---

## 📋 Checkpoint Listesi

**Veri Toplama Sonrası:**
- [ ] `validate_noise_data.py` çalıştır
- [ ] ADC overflow yok ✅
- [ ] Sensör varyansı yeterli ✅
- [ ] (Opsiyonel) `visualize_signal_quality.py` ile görselleştir

**Model Eğitimi Öncesi:**
- [ ] `validate_pipeline.py` çalıştır
- [ ] 11 movement mevcut ✅
- [ ] Label encoding doğru ✅
- [ ] EMG range [0, 4095] ✅

**Deployment Öncesi:**
- [ ] Test accuracy > 85% ✅
- [ ] TFLite boyutu < 256 KB ✅
- [ ] C++ header oluşturuldu ✅

---

## 🔧 İlk Kurulum (Sadece Bir Kez)

### Python Bağımlılıkları
```bash
pip install numpy pandas scipy matplotlib seaborn tensorflow scikit-learn
```

### Dizin Yapısı
```bash
cd scripts_ai
mkdir -p data/features data/models plots/signal_quality
```

### DSP Filtre Katsayıları (Türkiye için)
```bash
python filters/generate_filter_coefficients.py 50

# Çıktıyı C++ projesine kopyala
# src/filter_coefficients_50hz.h
```

---

## 🆘 Hızlı Sorun Giderme

### Problem: "No module named 'filters'"
```bash
# scripts_ai dizininden çalıştır
cd scripts_ai
python critical/feature_extraction.py data.csv
```

### Problem: ADC Overflow Hata Veriyor
```cpp
// ESP32 firmware'e ekle (filters.cpp)
filtered_value = constrain(filtered_value, 0, 4095);
```

### Problem: Model Accuracy %30'un Altında
```bash
# Veri kalitesini kontrol et
python validation/validate_noise_data.py data/training_data.csv

# Sinyal kalitesini görselleştir
python visualization/visualize_signal_quality.py data/training_data.csv
```

### Problem: Python-C++ Farklı Sonuç Veriyor
```bash
# Filtre uyumunu test et
python validation/validate_filters.py

# Preprocessing'i doğrula
python validation/validate_preprocessing.py
```

---

## 📚 Daha Fazla Bilgi

- **Ana README:** `scripts_ai/README.md`
- **Detaylı Açıklamalar:** `scripts_ai/SCRIPT_EXPLANATIONS.md`
- **Klasör READMEs:**
  - `critical/README.md` - Pipeline detayları
  - `filters/README.md` - DSP filtre teknik dökümanı
  - `validation/README.md` - Validasyon araçları
  - `visualization/README.md` - Görselleştirme rehberi

---

## 🎯 En Sık Kullanılan Komutlar

```bash
# Veri kontrolü
python validation/validate_noise_data.py data/latest.csv

# Feature extraction
python critical/feature_extraction.py data/latest.csv

# Model training
python critical/train_test_model.py

# Full validation
python validation/validate_noise_data.py data/latest.csv
python validation/validate_pipeline.py data/latest.csv
python validation/validate_filters.py

# Görselleştirme
python visualization/visualize_signal_quality.py data/latest.csv
```

---

**İyi çalışmalar! 🚀**
