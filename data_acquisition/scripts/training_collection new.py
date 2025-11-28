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
                             QProgressBar, QGroupBox, QGridLayout, QMessageBox)
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor

import pyqtgraph as pg

# ==========================================
# KONFİGÜRASYON
# ==========================================
BAUD = 921600
WINDOW_SIZE = 1000  # Grafikte gösterilecek veri genişliği
UI_REFRESH_RATE = 50 # ms (Ekranı 50ms'de bir güncelle = 20 FPS). Lag'ı engelleyen ayar budur.

# Dosya Yolları (Otomatik Algılama)
# Bu dosyanın bulunduğu klasörden yukarı çıkarak data ve images klasörlerini bulur
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR)) # Gerekirse burayı kendi klasör yapına göre ayarla
if 'real_time_esp322' not in PROJECT_ROOT: # Basit bir fallback
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data') 
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images_hand')

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
            print("\n⚠️  Kayıt yapılmadı (dosya yok)")

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
# ANA ARAYÜZ (GUI)
# ==========================================
class BionicHandGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32-S3 EMG | Gelişmiş Veri Toplama")
        self.resize(1280, 720)
        
        # Veri Tamponları
        self.serial_worker = None
        self.protocol_thread = None
        self.data_buffers = [deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE) for _ in range(6)]
        
        # Arayüzü Kur
        self.init_ui()
        self.refresh_ports()

        # --- LAG ENGELLEYİCİ ZAMANLAYICI ---
        # Grafikleri her veri geldiğinde değil, belirli aralıklarla güncelliyoruz.
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plots_loop)
        self.plot_timer.start(UI_REFRESH_RATE) 

    def init_ui(self):
        # Dark Mode Stil
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #EEE; }
            QLabel { color: #EEE; font-family: Segoe UI, Arial; font-size: 14px; }
            QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; font-weight: bold; color: #AAA; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 3px; }
            QPushButton { background-color: #007ACC; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }
            QPushButton:hover { background-color: #009BE5; }
            QPushButton:disabled { background-color: #333; color: #666; }
            QPushButton#stop_btn { background-color: #D32F2F; }
            QPushButton#stop_btn:hover { background-color: #F44336; }
            QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; color: white; background-color: #222; height: 20px; }
            QProgressBar::chunk { background-color: #007ACC; width: 10px; }
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
        self.save_btn.setStyleSheet("""
            QPushButton#save_btn { background-color: #FF6F00; }
            QPushButton#save_btn:hover { background-color: #FF8F00; }
            QPushButton#save_btn:disabled { background-color: #333; color: #666; }
        """)

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
        self.image_label.setStyleSheet("background-color: #222; border-radius: 8px; font-weight: bold;")
        
        # Hareket Adı
        self.lbl_action = QLabel("BEKLENİYOR")
        self.lbl_action.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_action.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        self.lbl_action.setStyleSheet("color: #AAA;")
        
        # Geri Sayım
        self.lbl_timer = QLabel("0.0")
        self.lbl_timer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_timer.setFont(QFont("Arial", 56, QFont.Weight.Bold))
        self.lbl_timer.setStyleSheet("color: #00E676;")
        
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

        # ---------------- Sağ Panel (Grafikler) ----------------
        right_panel = QVBoxLayout()
        main_layout.addLayout(right_panel)
        
        # Grafik Widget'ı
        pg.setConfigOptions(antialias=False) # Performans için antialias kapalı
        self.graphics_layout = pg.GraphicsLayoutWidget()
        self.graphics_layout.setBackground('#050505') # Çok koyu gri
        right_panel.addWidget(self.graphics_layout)
        
        self.plots = []
        self.curves = []
        # Kanal Renkleri (Neon tarzı)
        colors = ['#FF0055', '#00FF55', '#0055FF', '#FFFF00', '#00FFFF', '#FF00FF']
        
        for i in range(6):
            # 3 satır x 2 sütun düzeni
            row = i // 2
            col = i % 2

            p = self.graphics_layout.addPlot(row=row, col=col)
            # Y ekseni -1000 ile +1000 arasında (sıfır merkezli)
            p.setYRange(-1000, 1000)
            p.setXRange(0, WINDOW_SIZE)
            p.showGrid(x=True, y=True, alpha=0.3)

            # Sıfır çizgisi ekle (referans için)
            p.addLine(y=0, pen=pg.mkPen('#888', width=1, style=pg.QtCore.Qt.PenStyle.DashLine))

            # Başlık ve Eksenleri Gizle/Küçült
            p.setTitle(f"EMG CH {i+1}", color=colors[i], size="9pt")
            p.getAxis('left').setStyle(showValues=True) # Değerleri göster (ölçek için)
            p.getAxis('left').setWidth(35) # Eksen genişliği
            p.getAxis('bottom').setStyle(showValues=False)

            curve = p.plot(pen=pg.mkPen(color=colors[i], width=1.5))
            self.plots.append(p)
            self.curves.append(curve)

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
        Burada çizim yapılmaz, bu yüzden çok hızlıdır ve kasmaz.
        """
        for i, val in enumerate(emg_vals):
            self.data_buffers[i].append(val)

    def update_plots_loop(self):
        """
        Timer ile belirli aralıklarla (örn. 50ms) çağrılır.
        Ekranı güncelleyen fonksiyon budur.
        """
        if self.serial_worker and self.serial_worker.running:
            for i in range(6):
                # Veri varsa curve'ü güncelle
                if len(self.data_buffers[i]) > 0:
                    # DC offset removal: Ortalamayı çıkar (veri sıfır etrafında görünsün)
                    data = list(self.data_buffers[i])
                    if len(data) > 0:
                        mean = sum(data) / len(data)
                        centered_data = [val - mean for val in data]
                        self.curves[i].setData(centered_data)
                    else:
                        self.curves[i].setData(data)

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