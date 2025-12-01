import sys
import os
import time
import csv
import struct
import serial
import serial.tools.list_ports
import threading
import numpy as np
from collections import deque
from datetime import datetime

# PyQt6 ve Py QtGraph (Modern Arayüz Kütüphaneleri)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QComboBox, 
                             QProgressBar, QGroupBox, QGridLayout, QMessageBox, QTabWidget)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

import pyqtgraph as pg

# ==========================================
# KONFİGÜRASYON
# ==========================================
BAUD = 921600
WINDOW_SIZE = 1000  # Grafikte gösterilecek veri genişliği
FEATURE_WINDOW = 200  # Feature extraction için pencere boyutu
UI_REFRESH_RATE = 50 # ms (Ekranı 50ms'de bir güncelle = 20 FPS). Lag'ı engelleyen ayar budur.

# Feature extraction thresholds (matching C++ implementation)
ZC_THRESHOLD = 15.0
SSC_THRESHOLD = 15.0

# ==========================================
# DATA PROTOCOL - UPDATED 2025
# ==========================================
# ESP32 now sends FILTERED + BIPOLAR ENCODED data (matches real-time inference)
# 
# C++ Pipeline:
#   Raw ADC → HPF (20Hz) → LPF (450Hz) → Notch (50/60Hz) → Bipolar Encoding
# 
# Binary Protocol:
#   Header: 0xA5 0x5A (2 bytes)
#   Seq: uint16_t (2 bytes)
#   EMG1-6: uint16_t [0, 4095] - bipolar encoded filtered signals (12 bytes)
# 
# Decoding (automatic in this script):
#   Python receives uint16_t [0, 4095]
#   Decoding: decoded_float = uint16_value - 2047.5
#   Result: Bipolar signal [-2047.5, +2047.5] centered around 0
# 
# WHY THIS MATTERS:
#   - Training data now matches inference data characteristics
#   - Model learns on filtered signals (DC removed, noise reduced)
#   - Visualization shows bipolar signals centered around 0
# ==========================================

# Dosya Yolları (Otomatik Algılama)
# Bu dosyanın bulunduğu klasörden yukarı çıkarak data ve images klasörlerini bulur
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) # Gerekirse burayı kendi klasör yapına göre ayarla
if 'real_time_esp322' not in PROJECT_ROOT: # Basit bir fallback
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data') 
IMAGES_DIR = os.path.join(CURRENT_DIR, 'images_hand') # Updated: Now inside data_acquisition

# Protokol Ayarları
MOVEMENT_DURATION = 4  # Hareket Süresi (sn)
REST_DURATION = 3      # Dinlenme Süresi (sn)
REPETITIONS = 10       # Her hareket kaç kere tekrar edilecek
GESTURE_NAMES = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]
MOVEMENTS = list(range(1, 11)) # 1'den 10'a kadar olan hareket ID'leri

# ==========================================
# ARKA PLAN İŞÇİSİ (THREAD) - SERIAL OKUMA & KAYIT
# ==========================================
class SerialWorker(QThread):
    # GUI'ye sinyal göndermek için
    data_received = pyqtSignal(object) 
    finished_saving = pyqtSignal(str)
    
    def __init__(self, port, baud_rate):
        super().__init__()
        self.port = port
        self.baud = baud_rate
        self.running = False
        self.recording = False
        self.ser = None
        self.csv_file = None
        self.csv_writer = None
        self.start_time = None
        self.current_filename = None

        # Protokol Durumu (Etiketleme için)
        self.current_movement_id = 0
        self.current_movement_name = "Rest"
        self.current_phase = "HAZIRLIK"
        self.current_rep = 0

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=1)
            time.sleep(2) # ESP32 Reset Beklemesi
            self.ser.reset_input_buffer()
            self.ser.write(b'S') # Stream başlat komutu
            self.running = True
            
            buffer = bytearray()
            packet_size = 16 # Header(2) + Seq(2) + 6xADC(2)
            
            while self.running:
                if self.ser.in_waiting:
                    # Veriyi oku
                    buffer.extend(self.ser.read(self.ser.in_waiting))
                    
                    # Paketleri ayrıştır
                    while len(buffer) >= packet_size:
                        # Header Kontrolü (0xA5, 0x5A)
                        if buffer[0] == 0xA5 and buffer[1] == 0x5A:
                            packet_data = buffer[:packet_size]
                            buffer = buffer[packet_size:]
                            
                            # Binary'den sayıya çevir
                            unpacked = struct.unpack('<H HHHHHH', packet_data[2:])
                            seq = unpacked[0]
                            emg_vals = unpacked[1:] # (EMG1, EMG2, ..., EMG6)
                            
                            # ============================================================
                            # DATA INTERPRETATION - UPDATED 2025
                            # ============================================================
                            # emg_vals are BIPOLAR ENCODED filtered signals from C++
                            # - C++ applied: HPF → LPF → Notch → Bipolar Encoding
                            # - Values are uint16_t [0, 4095]
                            # - Decoding happens in feature_extraction.py: value - 2047.5
                            # - Result: Bipolar signal [-2047.5, +2047.5] (centered at 0)
                            # 
                            # IMPORTANT: We save the RAW uint16_t values to CSV
                            # Feature extraction will decode them later
                            # ============================================================
                            
                            current_ts = time.time()
                            
                            # 1. Veriyi GUI'ye gönder (Görselleştirme için)
                            self.data_received.emit(emg_vals)
                            
                            # 2. CSV'ye Yaz (Kayıt için - EN YÜKSEK HIZDA)
                            if self.recording and self.csv_writer:
                                if self.start_time is None:
                                    self.start_time = current_ts
                                
                                # Etiket Mantığı (Labeling)
                                label_name = self.current_movement_name
                                if self.current_phase == "DINLENME":
                                    label_name = "Rest"
                                
                                self.csv_writer.writerow([
                                    f"{current_ts - self.start_time:.4f}",
                                    seq,
                                    label_name,
                                    self.current_phase,
                                    self.current_rep,
                                    *emg_vals
                                ])
                        else:
                            # Header bozuksa 1 byte kaydırarak aramaya devam et
                            del buffer[0]
                else:
                    self.msleep(1) # CPU'yu boğmamak için minik bekleme
                    
        except Exception as e:
            print(f"Serial Error: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.write(b'E') # Stream durdur
                self.ser.close()

    def start_recording(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Timestamp', 'Seq', 'Movement', 'Phase', 'Rep', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6'])
        self.recording = True
        self.start_time = time.time()
        self.current_filename = filename
        print(f"\n{'='*60}")
        print(f"📝 KAYIT BAŞLATILDI")
        print(f"📂 Dosya: {filename}")
        print(f"{'='*60}\n")

    def stop_recording(self):
        self.recording = False
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

        # Terminale başarı mesajı yazdır
        if hasattr(self, 'current_filename') and self.current_filename is not None:
            print(f"\n{'='*60}")
            print(f"✅ KAYIT TAMAMLANDI")
            print(f"📂 Dosya: {self.current_filename}")
            print(f"📊 Dosya boyutu: {os.path.getsize(self.current_filename) / 1024:.2f} KB")
            print(f"{'='*60}\n")
        else:
            print("\n  Kayıt yapılmadı (dosya yok)")

        self.finished_saving.emit("Kayıt Tamamlandı")

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def update_protocol_state(self, mov_id, mov_name, phase, rep):
        """Protokol thread'inden gelen durumu güncelle"""
        self.current_movement_id = mov_id
        self.current_movement_name = mov_name
        self.current_phase = phase
        self.current_rep = rep

# ==========================================
# PROTOKOL YÖNETİCİSİ (Logic Thread)
# ==========================================
class ProtocolThread(QThread):
    status_update = pyqtSignal(dict) # GUI'yi güncellemek için
    protocol_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        # 1. Başlangıç Beklemesi (Kullanıcı hazırlansın)
        self.emit_state(0, "Rest", "HAZIRLANIYOR...", 0, 3, 0)
        time.sleep(3)
        
        # 2. Baseline Rest (İlk dinlenme verisi)
        self.emit_state(0, "Rest", "DINLENME (BASELINE)", 0, 5, 0)
        time.sleep(5)
        
        total_steps = len(MOVEMENTS) * REPETITIONS
        current_step = 0

        # Hareket Döngüsü
        for mov_id in MOVEMENTS:
            mov_name = GESTURE_NAMES[mov_id]
            
            for rep in range(1, REPETITIONS + 1):
                if not self.running: return
                
                current_step += 1
                progress_pct = int((current_step / total_steps) * 100)
                
                # --- HAREKET FAZI ---
                start_t = time.time()
                while time.time() - start_t < MOVEMENT_DURATION:
                    if not self.running: return
                    left = MOVEMENT_DURATION - (time.time() - start_t)
                    self.emit_state(mov_id, mov_name, "HAREKET", rep, left, progress_pct)
                    time.sleep(0.05) 
                
                # --- DINLENME FAZI ---
                start_t = time.time()
                while time.time() - start_t < REST_DURATION:
                    if not self.running: return
                    left = REST_DURATION - (time.time() - start_t)
                    self.emit_state(mov_id, mov_name, "DINLENME", rep, left, progress_pct)
                    time.sleep(0.05)
        
        self.protocol_finished.emit()

    def emit_state(self, mid, mname, phase, rep, time_left, prog):
        self.status_update.emit({
            "id": mid, "name": mname, "phase": phase, 
            "rep": rep, "time": time_left, "progress": prog
        })

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

# ==========================================
# FEATURE EXTRACTION FUNCTIONS
# ==========================================
def extract_features(signal_window):
    """
    Extract TD4 features from a signal window
    Returns: MAV, WL, ZC, SSC
    """
    signal = np.array(signal_window, dtype=np.float32)
    
    # Mean Absolute Value
    mav = np.mean(np.abs(signal))
    
    # Waveform Length
    wl = np.sum(np.abs(np.diff(signal)))
    
    # Zero Crossings
    zc = 0
    for i in range(len(signal) - 1):
        if signal[i] * signal[i+1] < 0 and abs(signal[i] - signal[i+1]) >= ZC_THRESHOLD:
            zc += 1
    
    # Slope Sign Changes
    ssc = 0
    for i in range(1, len(signal) - 1):
        if (signal[i] - signal[i-1]) * (signal[i] - signal[i+1]) >= SSC_THRESHOLD:
            ssc += 1
    
    return mav, wl, zc, ssc

# ==========================================
# ANA ARAYÜZ (GUI)
# ==========================================
class BionicHandGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32-S3 EMG | Advanced Data Collection & Feature Visualization")
        self.resize(1200, 800) # Daha küçük başlangıç boyutu
        self.center() # Ekranı ortala
        
        # Veri Tamponları - RAW (unfiltered) and FILTERED
        self.serial_worker = None
        self.protocol_thread = None
        
        # Store both raw (received as-is) and filtered (decoded bipolar) signals
        self.raw_buffers = [deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(6)]
        self.filtered_buffers = [deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(6)]
        
        # Feature buffers (store computed features over time)
        self.feature_values = {
            'MAV': [0] * 6,
            'WL': [0] * 6,
            'ZC': [0] * 6,
            'SSC': [0] * 6
        }
        
        # Arayüzü Kur
        self.init_ui()
        self.refresh_ports()

        # --- LAG ENGELLEYİCİ ZAMANLAYICI ---
        # Grafikleri her veri geldiğinde değil, belirli aralıklarla güncelliyoruz.
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots_loop)
        self.plot_timer.start(UI_REFRESH_RATE) 

    def center(self):
        """Pencereyi ekranın ortasına konumlandır"""
        qr = self.frameGeometry()
        cp = self.screen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def init_ui(self):
        # Professional Dark Theme (Monochrome/Minimalist)
        # Inspired by VS Code, Linear, and modern engineering tools
        self.setStyleSheet("""
            QMainWindow { 
                background-color: #121212;
                color: #E0E0E0;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            
            /* Typography */
            QLabel { 
                color: #E0E0E0; 
                font-size: 14px; 
            }
            
            /* Containers */
            QGroupBox { 
                border: 1px solid #333333; 
                border-radius: 6px; 
                margin-top: 24px; 
                font-weight: bold; 
                color: #FFFFFF;
                background-color: #1E1E1E;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                subcontrol-position: top left; 
                padding: 0px 8px;
                background-color: transparent;
                color: #AAAAAA;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            /* Buttons */
            QPushButton { 
                background-color: #2D2D2D;
                color: #FFFFFF; 
                border: 1px solid #3E3E3E;
                padding: 10px 16px; 
                border-radius: 4px; 
                font-weight: 600; 
                font-size: 13px; 
            }
            QPushButton:hover { 
                background-color: #3E3E3E;
                border-color: #555555;
            }
            QPushButton:pressed {
                background-color: #1A1A1A;
            }
            QPushButton:disabled { 
                background-color: #1A1A1A; 
                color: #555555; 
                border: 1px solid #2D2D2D; 
            }
            
            /* Primary Action Button (Connect/Start) */
            QPushButton#primary_btn {
                background-color: #E0E0E0;
                color: #121212;
                border: none;
            }
            QPushButton#primary_btn:hover {
                background-color: #FFFFFF;
            }
            
            /* Destructive Action Button (Stop) */
            QPushButton#stop_btn { 
                background-color: #2D2D2D;
                color: #FF5252;
                border: 1px solid #552222;
            }
            QPushButton#stop_btn:hover { 
                background-color: #3E2222;
                border-color: #FF5252;
            }
            
            /* Save Button */
            QPushButton#save_btn {
                background-color: #2D2D2D;
                color: #4CAF50;
                border: 1px solid #224422;
            }
            QPushButton#save_btn:hover {
                background-color: #223322;
                border-color: #4CAF50;
            }
            
            /* Inputs */
            QComboBox {
                background-color: #1E1E1E;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 4px;
                color: #E0E0E0;
            }
            QComboBox::drop-down {
                border: none;
            }
            
            /* Progress Bar */
            QProgressBar { 
                border: none;
                background-color: #2D2D2D; 
                height: 6px;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk { 
                background-color: #FFFFFF;
                border-radius: 3px;
            }
            
            /* Tabs */
            QTabWidget::pane {
                border: 1px solid #333333;
                background-color: #1E1E1E;
                border-radius: 6px;
            }
            QTabBar::tab {
                background: #121212;
                color: #888888;
                padding: 12px 24px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: 600;
                border: 1px solid transparent;
            }
            QTabBar::tab:selected {
                background: #1E1E1E;
                color: #FFFFFF;
                border-top: 2px solid #FFFFFF;
            }
            QTabBar::tab:hover {
                color: #CCCCCC;
            }
            
            /* Scroll Area */
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------------- Sol Panel (Kontrol & Rehber) ----------------
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setFixedWidth(380) # Sol tarafın genişliği
        left_widget.setLayout(left_panel)
        main_layout.addWidget(left_widget)

        # 1. Bağlantı Kutusu
        conn_group = QGroupBox("Cihaz Bağlantısı")
        conn_layout = QGridLayout()
        
        self.port_combo = QComboBox()
        self.port_combo.setFixedHeight(30)
        self.refresh_btn = QPushButton("⟳")
        self.refresh_btn.setFixedWidth(40)
        self.refresh_btn.clicked.connect(self.refresh_ports)
        
        self.connect_btn = QPushButton("BAĞLAN")
        self.connect_btn.setObjectName("primary_btn") # Use primary style
        self.connect_btn.clicked.connect(self.toggle_connection)
        
        conn_layout.addWidget(QLabel("Port:"), 0, 0)
        conn_layout.addWidget(self.port_combo, 0, 1)
        conn_layout.addWidget(self.refresh_btn, 0, 2)
        conn_layout.addWidget(self.connect_btn, 1, 0, 1, 3)
        conn_group.setLayout(conn_layout)
        left_panel.addWidget(conn_group)

        # 2. Eğitim Kontrolü
        proto_group = QGroupBox("Veri Toplama İşlemi")
        proto_layout = QVBoxLayout()

        self.start_btn = QPushButton("EĞİTİMİ BAŞLAT")
        self.start_btn.setObjectName("primary_btn") # Use primary style
        self.start_btn.clicked.connect(self.start_protocol)
        self.start_btn.setEnabled(False)

        self.stop_btn = QPushButton("DURDUR")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_protocol)
        self.stop_btn.setEnabled(False)

        self.save_btn = QPushButton("KAYDET VE KAPAT")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.clicked.connect(self.save_and_close)
        self.save_btn.setEnabled(False)
        # Removed inline style, handled by stylesheet

        proto_layout.addWidget(self.start_btn)
        proto_layout.addWidget(self.stop_btn)
        proto_layout.addWidget(self.save_btn)
        proto_group.setLayout(proto_layout)
        left_panel.addWidget(proto_group)

        # 3. Görsel Rehber (En Önemli Kısım)
        status_group = QGroupBox("Talimatlar")
        status_layout = QVBoxLayout()
        
        # Resim
        self.image_label = QLabel("HAZIR")
        self.image_label.setFixedSize(340, 340)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000000; border: 1px solid #333; border-radius: 4px;")
        
        # Hareket Adı
        self.lbl_action = QLabel("BEKLENİYOR")
        self.lbl_action.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_action.setFont(QFont("Segoe UI", 24, QFont.Weight.Bold))
        self.lbl_action.setStyleSheet("color: #888888; letter-spacing: 2px;")
        
        # Geri Sayım
        self.lbl_timer = QLabel("0.0")
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer.setFont(QFont("Segoe UI", 64, QFont.Weight.Light))
        self.lbl_timer.setStyleSheet("color: #FFFFFF;")
        
        # Detay Bilgi
        self.lbl_info = QLabel("Set: 0/0")
        self.lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_layout.addWidget(self.image_label, 0, Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.lbl_action)
        status_layout.addWidget(self.lbl_timer)
        status_layout.addWidget(self.lbl_info)
        status_group.setLayout(status_layout)
        left_panel.addWidget(status_group)
        
        # İlerleme Çubuğu
        self.prog_bar = QProgressBar()
        self.prog_bar.setValue(0)
        left_panel.addWidget(self.prog_bar)
        
        left_panel.addStretch() # Boşluğu alta it

        # ---------------- Sağ Panel (Grafikler & Features) ----------------
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel)
        
        # Tab Widget for different views
        self.tab_widget = QTabWidget()
        right_panel.addWidget(self.tab_widget)
        
        # Tab 1: Signal Comparison (Raw vs Filtered)
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        self.tab_widget.addTab(signal_tab, "📊 Signal Comparison")
        
        # Tab 2: Features
        features_tab = QWidget()
        features_layout = QVBoxLayout(features_tab)
        self.tab_widget.addTab(features_tab, "🔬 Feature Extraction")
        
        # ========== SIGNAL COMPARISON TAB ==========
        pg.setConfigOptions(antialias=True) # Better quality for comparison view
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#000000')
        signal_layout.addWidget(self.graphics_layout)
        
        self.raw_plots = []
        self.filtered_plots = []
        self.raw_curves = []
        self.filtered_curves = []
        
        # Enhanced color schemes
        raw_colors = ['#FF006E', '#FB5607', '#FFBE0B', '#8338EC', '#3A86FF', '#06FFA5']
        filtered_colors = ['#00E5FF', '#00E676', '#FFEA00', '#FF6E40', '#E040FB', '#76FF03']
        
        for i in range(6):
            row = i
            
            # Raw signal plot (left column)
            p_raw = self.graphics_layout.addPlot(row=row, col=0)
            p_raw.setYRange(-1000, 1000)
            p_raw.setXRange(0, WINDOW_SIZE)
            p_raw.showGrid(x=True, y=True, alpha=0.15)
            p_raw.addLine(y=0, pen=pg.mkPen('#555', width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
            p_raw.setTitle(f"CH{i+1} - RAW (Unfiltered)", color=raw_colors[i], size="10pt")
            p_raw.getAxis('left').setStyle(showValues=True)
            p_raw.getAxis('left').setWidth(40)
            p_raw.getAxis('left').setLabel('ADC', color='#888')
            p_raw.getAxis('bottom').setStyle(showValues=False)
            p_raw.setLabel('bottom', 'Samples', color='#888')
            
            curve_raw = p_raw.plot(pen=pg.mkPen(color=raw_colors[i], width=2))
            self.raw_plots.append(p_raw)
            self.raw_curves.append(curve_raw)
            
            # Filtered signal plot (right column)
            p_filt = self.graphics_layout.addPlot(row=row, col=1)
            p_filt.setYRange(-1000, 1000)
            p_filt.setXRange(0, WINDOW_SIZE)
            p_filt.showGrid(x=True, y=True, alpha=0.15)
            p_filt.addLine(y=0, pen=pg.mkPen('#555', width=1, style=pg.QtCore.Qt.PenStyle.DashLine))
            p_filt.setTitle(f"CH{i+1} - FILTERED (HPF→LPF→Notch)", color=filtered_colors[i], size="10pt")
            p_filt.getAxis('left').setStyle(showValues=True)
            p_filt.getAxis('left').setWidth(40)
            p_filt.getAxis('left').setLabel('Bipolar', color='#888')
            p_filt.getAxis('bottom').setStyle(showValues=False)
            p_filt.setLabel('bottom', 'Samples', color='#888')
            
            curve_filt = p_filt.plot(pen=pg.mkPen(color=filtered_colors[i], width=2))
            self.filtered_plots.append(p_filt)
            self.filtered_curves.append(curve_filt)
        
        # ========== FEATURES TAB ==========
        from PyQt6.QtWidgets import QScrollArea
        
        # Create scrollable area for features
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #121212;
            }
            QScrollBar:vertical {
                background: #1E1E1E;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #444444;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #666666;
            }
        """)
        features_layout.addWidget(scroll_area)
        
        # Container widget for the scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(20)
        scroll_area.setWidget(scroll_content)
        
        self.feature_labels = {}
        feature_names = ['MAV', 'WL', 'ZC', 'SSC']
        feature_colors = ['#00E5FF', '#00E676', '#FFEA00', '#FF6E40']
        feature_descriptions = ['Mean Absolute Value', 'Waveform Length', 'Zero Crossings', 'Slope Sign Changes']
        
        for ch in range(6):
            # Create a card for each channel
            channel_card = QGroupBox()
            channel_card.setStyleSheet(f"""
                QGroupBox {{
                    background-color: #1E1E1E;
                    border: 1px solid #333;
                    border-left: 4px solid {filtered_colors[ch]};
                    border-radius: 4px;
                    margin-top: 10px;
                    padding: 15px;
                }}
            """)
            
            card_layout = QVBoxLayout(channel_card)
            
            # Channel header
            ch_label = QLabel(f"CHANNEL {ch+1}")
            ch_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            ch_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
            ch_label.setStyleSheet(f"""
                color: {filtered_colors[ch]}; 
                background-color: transparent;
                letter-spacing: 1px;
            """)
            card_layout.addWidget(ch_label)
            
            # Feature grid for this channel
            feature_grid = QGridLayout()
            feature_grid.setSpacing(10)
            feature_grid.setContentsMargins(0, 10, 0, 0)
            
            for f_idx, (fname, fcolor, fdesc) in enumerate(zip(feature_names, feature_colors, feature_descriptions)):
                # Feature name
                name_lbl = QLabel(f"{fname}")
                name_lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
                name_lbl.setStyleSheet(f"color: #888; padding: 4px;")
                name_lbl.setFixedWidth(60)
                feature_grid.addWidget(name_lbl, f_idx, 0)
                
                # Feature value (RAW)
                val_lbl = QLabel("0.00")
                val_lbl.setFont(QFont("Consolas", 13))
                val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                val_lbl.setFixedWidth(100)
                val_lbl.setStyleSheet(f"""
                    color: #EEE; 
                    padding: 4px 8px; 
                    background-color: #121212; 
                    border: 1px solid #333;
                    border-radius: 3px;
                """)
                feature_grid.addWidget(val_lbl, f_idx, 1)

                # Feature value (NORMALIZED)
                norm_lbl = QLabel("N: 0.00")
                norm_lbl.setFont(QFont("Consolas", 11))
                norm_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                norm_lbl.setFixedWidth(100)
                norm_lbl.setStyleSheet(f"""
                    color: #888; 
                    padding: 4px 8px; 
                    background-color: #121212; 
                    border: 1px solid #333;
                    border-radius: 3px;
                """)
                feature_grid.addWidget(norm_lbl, f_idx, 2)
                
                # Description
                desc_lbl = QLabel(fdesc)
                desc_lbl.setFont(QFont("Segoe UI", 10))
                desc_lbl.setStyleSheet("color: #666; padding: 4px;")
                desc_lbl.setWordWrap(True)
                feature_grid.addWidget(desc_lbl, f_idx, 3)
                
                self.feature_labels[f"{fname}_CH{ch}"] = val_lbl
                self.feature_labels[f"{fname}_CH{ch}_NORM"] = norm_lbl
            
            card_layout.addLayout(feature_grid)
            scroll_layout.addWidget(channel_card)
        
        scroll_layout.addStretch()

    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports:
            # Sadece olası cihazları listele (filtreli)
            desc = p.description
            if "CP210" in desc or "CH340" in desc or "USB Serial" in desc or "COM" in desc:
                self.port_combo.addItem(p.device)

    def toggle_connection(self):
        if self.serial_worker is None: # BAĞLAN
            port = self.port_combo.currentText()
            if not port: return
            
            try:
                self.serial_worker = SerialWorker(port, BAUD)
                self.serial_worker.data_received.connect(self.update_data)
                self.serial_worker.start()
                
                self.connect_btn.setText("KES")
                self.connect_btn.setStyleSheet("background-color: #D32F2F;")
                self.start_btn.setEnabled(True)
                self.port_combo.setEnabled(False)
                self.refresh_btn.setEnabled(False)
                print("Bağlandı.")
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Bağlantı Hatası:\n{e}")
            
        else: # KES
            self.stop_protocol()
            self.serial_worker.stop()
            self.serial_worker = None

            self.connect_btn.setText("BAĞLAN")
            self.connect_btn.setStyleSheet("background-color: #007ACC;")
            self.start_btn.setEnabled(False)
            self.save_btn.setEnabled(False)  # Bağlantı kesildiğinde kaydet butonu pasif
            self.port_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            print("Bağlantı kesildi.")

    def start_protocol(self):
        if not self.serial_worker: return

        # Dosya ismi oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(DATA_DIR, f'training_data_{timestamp}.csv')

        # Kaydı Başlat
        self.serial_worker.start_recording(filename)

        # Protokolü Başlat
        self.protocol_thread = ProtocolThread()
        self.protocol_thread.status_update.connect(self.update_protocol_ui)
        self.protocol_thread.protocol_finished.connect(self.on_protocol_finished)
        self.protocol_thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(True)  # Kaydet butonunu aktif et
        self.connect_btn.setEnabled(False) # Eğitim sırasında port koparılmasın

    def stop_protocol(self):
        if self.protocol_thread:
            self.protocol_thread.stop()
            self.protocol_thread = None

        if self.serial_worker:
            self.serial_worker.stop_recording()
            self.serial_worker.update_protocol_state(0, "Rest", "DURDURULDU", 0)

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # save_btn aktif kalsın (kullanıcı hala kaydetmek isteyebilir)
        self.connect_btn.setEnabled(True)

        self.lbl_action.setText("DURDURULDU")
        self.lbl_action.setStyleSheet("color: #F44336;")
        self.image_label.clear()
        self.image_label.setText("STOP")

    def on_protocol_finished(self):
        self.stop_protocol()
        QMessageBox.information(self, "Bitti", "Eğitim tamamlandı!\nVeriler başarıyla kaydedildi.")

    def save_and_close(self):
        """Veriyi kaydet ve programı kapat"""
        # Protokolü durdur (hala çalışıyorsa)
        if self.protocol_thread:
            self.protocol_thread.stop()
            self.protocol_thread = None

        # Kayıt devam ediyorsa durdur
        if self.serial_worker and self.serial_worker.recording:
            print("\n🔄 Manuel kaydetme başlatıldı...")
            self.serial_worker.stop_recording()
            msg = "Veriler başarıyla kaydedildi.\nProgram kapanıyor..."
        else:
            print("\n⚠️  Kayıt zaten durdurulmuş. Program kapanıyor...")
            msg = "Program kapanıyor..."

        # Bilgi mesajı göster
        QMessageBox.information(self, "Bilgi", msg)

        # Programı kapat
        print("👋 Program kapatılıyor...\n")
        self.close()

    @pyqtSlot(dict)
    def update_protocol_ui(self, status):
        """UI Rehberini Güncelle"""
        phase = status['phase']
        name = status['name']
        rep = status['rep']
        time_left = status['time']
        prog = status['progress']
        mov_id = status['id']

        # Worker'a durumu ilet (CSV için)
        if self.serial_worker:
            self.serial_worker.update_protocol_state(mov_id, name, phase, rep)

        # Görsel Yükle
        img_id = mov_id if phase == "HAREKET" else 0
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpeg")
        
        if os.path.exists(img_path):
            pixmap = QPixmap(img_path)
            self.image_label.setPixmap(pixmap.scaled(340, 340, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            self.image_label.setText(f"Görsel Yok\n{img_id}.jpeg")

        # Metinler
        self.lbl_action.setText(name if phase == "HAREKET" else "DINLENME")
        self.lbl_info.setText(f"Tekrar: {rep}/{REPETITIONS}")
        self.lbl_timer.setText(f"{time_left:.1f}")
        self.prog_bar.setValue(prog)

        # Renk Durumları
        if phase == "HAREKET":
            self.lbl_timer.setStyleSheet("color: #00E676;") # Yeşil
            self.lbl_action.setStyleSheet("color: #00E676;")
            self.image_label.setStyleSheet("border: 4px solid #00E676; border-radius: 8px;")
        elif phase == "DINLENME":
            self.lbl_timer.setStyleSheet("color: #FF9100;") # Turuncu
            self.lbl_action.setStyleSheet("color: #FF9100;")
            self.image_label.setStyleSheet("border: 4px solid #FF9100; border-radius: 8px;")
        else:
            self.lbl_timer.setStyleSheet("color: #AAA;")
            self.image_label.setStyleSheet("border: 1px solid #444;")

    @pyqtSlot(object)
    def update_data(self, emg_vals):
        """
        Serial'den gelen veriyi SADECE hafızaya ekler.
        RAW: as received (uint16_t [0-4095])
        FILTERED: decoded bipolar signal (float [-2047.5, +2047.5])
        """
        for i, val in enumerate(emg_vals):
            # Store raw value
            self.raw_buffers[i].append(val)
            
            # Decode to bipolar (matching C++ encoding)
            decoded = val - 2047.5
            self.filtered_buffers[i].append(decoded)

    def update_plots_loop(self):
        """
        Timer ile belirli aralıklarla (örn. 50ms) çağrılır.
        Updates both signal plots and feature calculations.
        """
        if self.serial_worker and self.serial_worker.running:
            for i in range(6):
                # Update RAW signal plot
                if len(self.raw_buffers[i]) > 0:
                    raw_data = list(self.raw_buffers[i])
                    # Center raw data for visualization
                    mean_raw = sum(raw_data) / len(raw_data) if len(raw_data) > 0 else 2047.5
                    centered_raw = [val - mean_raw for val in raw_data]
                    self.raw_curves[i].setData(centered_raw)
                
                # Update FILTERED signal plot
                if len(self.filtered_buffers[i]) > 0:
                    filtered_data = list(self.filtered_buffers[i])
                    self.filtered_curves[i].setData(filtered_data)
                    
                    # Extract features from the latest window
                    if len(filtered_data) >= FEATURE_WINDOW:
                        window = filtered_data[-FEATURE_WINDOW:]
                        mav, wl, zc, ssc = extract_features(window)
                        
                        # Store feature values
                        self.feature_values['MAV'][i] = mav
                        self.feature_values['WL'][i] = wl
                        self.feature_values['ZC'][i] = zc
                        self.feature_values['SSC'][i] = ssc
                        
                        # Calculate NORMALIZED values (matching C++ & Training)
                        # MAV: / 4095.0
                        # WL: / (4095.0 * FEATURE_WINDOW)
                        # ZC: / FEATURE_WINDOW
                        # SSC: / FEATURE_WINDOW
                        mav_norm = mav / 4095.0
                        wl_norm = wl / (4095.0 * FEATURE_WINDOW)
                        zc_norm = zc / FEATURE_WINDOW
                        ssc_norm = ssc / FEATURE_WINDOW

                        # Update feature labels (RAW)
                        self.feature_labels[f"MAV_CH{i}"].setText(f"{mav:.2f}")
                        self.feature_labels[f"WL_CH{i}"].setText(f"{wl:.2f}")
                        self.feature_labels[f"ZC_CH{i}"].setText(f"{int(zc)}")
                        self.feature_labels[f"SSC_CH{i}"].setText(f"{int(ssc)}")

                        # Update feature labels (NORMALIZED)
                        self.feature_labels[f"MAV_CH{i}_NORM"].setText(f"N:{mav_norm:.3f}")
                        self.feature_labels[f"WL_CH{i}_NORM"].setText(f"N:{wl_norm:.3f}")
                        self.feature_labels[f"ZC_CH{i}_NORM"].setText(f"N:{zc_norm:.3f}")
                        self.feature_labels[f"SSC_CH{i}_NORM"].setText(f"N:{ssc_norm:.3f}")

    def closeEvent(self, event):
        self.stop_protocol()
        if self.serial_worker:
            self.serial_worker.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BionicHandGUI()
    window.show()
    sys.exit(app.exec())