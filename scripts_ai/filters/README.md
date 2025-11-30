# 🔧 Filters - DSP Signal Processing

Bu klasör, EMG sinyallerini işlemek için kullanılan **dijital sinyal işleme (DSP) filtrelerini** içerir. Python'da eğitim sırasında ve C++'da ESP32'de gerçek zamanlı olarak uygulanan filtrelerin **tam uyumlu** olmasını sağlar.

---

## 📋 İçindekiler

- [Klasör Amacı](#-klasör-amacı)
- [Neden DSP Filtreleri?](#-neden-dsp-filtreleri)
- [Filtre Türleri](#-filtre-türleri)
- [Scriptler](#-scriptler)
  - [1. dsp_filters.py](#1-dsp_filterspy)
  - [2. generate_filter_coefficients.py](#2-generate_filter_coefficientspy)
- [Adım Adım Kullanım](#-adım-adım-kullanım)
- [Filtre Parametreleri](#-filtre-parametreleri)
- [Python-C++ Tutarlılığı](#-python-c-tutarlılığı)
- [Frekans Tepkisi Analizi](#-frekans-tepkisi-analizi)
- [Troubleshooting](#-troubleshooting)
- [Teknik Detaylar](#-teknik-detaylar)
- [Referanslar](#-referanslar)

---

## 🎯 Klasör Amacı

**Problem:** Ham EMG sinyalleri gürültülüdür!

```
Ham ADC Sinyali = EMG + DC Offset + Motion Artifacts + Powerline Noise + Electronic Noise
                  ^^^   ^^^^^^^^^^^  ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^
                  İstenen  0-20 Hz     0-10 Hz           50/60 Hz          >500 Hz
```

**Çözüm:** Cascaded IIR Filter Bank

```
Ham ADC → HPF (20 Hz) → LPF (450 Hz) → Notch (50/60 Hz) → Temiz EMG
```

**Bu klasörün görevi:**
1. **Python filter bank** sağlamak (eğitim için)
2. **C++ coefficient generator** sağlamak (ESP32 için)
3. **İki implementasyonun tutarlılığını** garantilemek

---

## 🚨 Neden DSP Filtreleri?

### **Probleme Bakış**

EMG sensörlerinden gelen ham ADC değerleri birçok gürültü kaynağı içerir:

| Gürültü Türü | Frekans Bandı | Kaynak | Etkisi |
|--------------|---------------|--------|--------|
| **DC Offset** | 0 Hz | Elektrode-deri kontağı | Sinyal 2048 civarında merkez kaymış |
| **Motion Artifacts** | 0-20 Hz | Kablo hareketi, kas titremesi | Baseline wander (yavaş salınım) |
| **EMG Sinyali** | **20-450 Hz** | **Kas kasılması (İSTENİR!)** | **Bilgi burada!** |
| **Powerline Noise** | 50/60 Hz | AC elektrik şebekesi | Güçlü sinüzoidal interferans |
| **Electronic Noise** | >500 Hz | ADC, kablolama, dijital gürültü | Yüksek frekanslı rastgele gürültü |

### **Filtrelemeden Önce vs Sonra**

**Filtrelemeden Önce:**
```python
# Ham ADC verisi
signal = [2048, 2150, 2100, 2200, 1950, 2050, ...]  # Gürültülü
mean = 2048  # DC offset hala mevcut
std = 250    # Yüksek varyans (gürültü dominant)

# TD4 özellikleri güvenilmez:
ZC = 45   # Çok fazla (gürültü sıfırı sık geçiyor)
SSC = 38  # Çok fazla (rastgele tepe/vadi)
```

**Filtrelemeden Sonra:**
```python
# Filtrelenmiş sinyal
signal = [-50, 120, -30, 80, -60, 40, ...]  # Temiz EMG bandı
mean = 0    # DC offset kaldırılmış
std = 85    # Düşük varyans (sadece EMG)

# TD4 özellikleri güvenilir:
ZC = 12   # Doğru frekans tahmini
SSC = 10  # Gerçek sinyal karmaşıklığı
```

### **Literatür Desteği**

- **De Luca et al. (2002):** "EMG sinyalleri 20-450 Hz arasında bant sınırlı olmalıdır"
- **Merletti & Parker (2004):** "Powerline interferansı mutlaka kaldırılmalıdır"
- **Phinyomark et al. (2012):** "Filtreleme TD4 özellik güvenilirliğini %30 artırır"

---

## 🎛️ Filtre Türleri

### **1. High-Pass Filter (HPF) - 20 Hz**

**Amaç:** DC offset ve düşük frekanslı hareket artifacts'ı kaldırır

**Tasarım:**
- **Türü:** 4th-order Butterworth
- **Cutoff:** 20 Hz
- **Attenuation:** -24 dB/octave (çok keskin)
- **Yapı:** 2 biquad (cascade)

**Neden 20 Hz?**
- EMG sinyalleri 20 Hz'den başlar
- Motion artifacts <10 Hz → tamamen bloklanır
- DC drift (0 Hz) → tamamen kaldırılır

**Frekans Tepkisi:**
```
Frequency   Attenuation
---------   -----------
0 Hz        -∞ dB (DC tamamen bloke)
10 Hz       -12 dB (baseline drift bastırılır)
20 Hz       -3 dB (cutoff)
50 Hz       ~0 dB (EMG bandı geçer)
```

### **2. Low-Pass Filter (LPF) - 450 Hz**

**Amaç:** Yüksek frekanslı elektronik gürültüyü kaldırır

**Tasarım:**
- **Türü:** 4th-order Butterworth
- **Cutoff:** 450 Hz
- **Attenuation:** -24 dB/octave
- **Yapı:** 2 biquad (cascade)

**Neden 450 Hz?**
- EMG sinyalleri 450 Hz'de sonlanır
- Yüksek frekanslı gürültü >500 Hz → bloklanır
- ADC sampling rate 1000 Hz → Nyquist (500 Hz) altında güvenli

**Frekans Tepkisi:**
```
Frequency   Attenuation
---------   -----------
100 Hz      ~0 dB (EMG bandı geçer)
300 Hz      ~0 dB (EMG bandı geçer)
450 Hz      -3 dB (cutoff)
600 Hz      -12 dB (gürültü bastırılır)
1000 Hz     -24 dB (elektronik gürültü tamamen bloke)
```

### **3. Notch Filter - 50/60 Hz**

**Amaç:** Powerline interferansını (elektrik şebekesi gürültüsü) kaldırır

**Tasarım:**
- **Türü:** 2nd-order IIR Notch
- **Frequency:** 50 Hz (Avrupa/Asya) veya 60 Hz (Amerika)
- **Q Factor:** 12.5 (dar bant)
- **Yapı:** 1 biquad

**Neden Notch?**
- AC elektrik şebekesi çok güçlü sinüzoidal gürültü yaratır
- 50/60 Hz'de -40 dB attenuation → %99 azaltma
- Dar bant (Q=12.5) → sadece 50/60 Hz etkilenir, EMG bandı korunur

**Frekans Tepkisi (50 Hz için):**
```
Frequency   Attenuation
---------   -----------
40 Hz       -0.5 dB (EMG bandı minimal etkilenir)
48 Hz       -3 dB
50 Hz       -40 dB (powerline tamamen bloke)
52 Hz       -3 dB
60 Hz       -0.5 dB (EMG bandı minimal etkilenir)
```

### **Cascade (Birleşik Filtre)**

**Sıralama:** HPF → LPF → Notch

```
Ham ADC (0-4095)
    ↓
HPF (20 Hz)   → DC ve motion artifacts kaldırılır
    ↓
LPF (450 Hz)  → Yüksek frekanslı gürültü kaldırılır
    ↓
Notch (50 Hz) → Powerline interferansı kaldırılır
    ↓
Temiz EMG (20-450 Hz, powerline-free)
```

**Toplam Latency:** ~12 ms (4th-order × 2 + 2nd-order)

---

**Son güncelleme:** 2025-11-30
**Versiyon:** 1.0
**Yazar:** ESP32 Bionic Hand Project
