"""
ESP32-S3 Biyonik El - 6 Kanallı EMG Veri Toplama
TD4 Feature Extraction uyumlu CSV formatı üretir.
"""
import serial
import struct
import time
import csv
from datetime import datetime
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from collections import deque
import threading
from scipy import signal
import serial.tools.list_ports

# ==========================================
# KONFİGÜRASYON
# ==========================================
# ESP32-S3 genelde yüksek baud rate ile sorunsuz çalışır
BAUD = 921600
# COM11 - ESP32-S3 USB Serial Device
PORT = 'COM11' 

# Dosya Yolları
# Ana proje root'una git, sonra scripts_ai/data/ altına kaydet
PROJECT_ROOT_MAIN = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # real_time_esp322/
DATA_DIR = os.path.join(PROJECT_ROOT_MAIN, 'scripts_ai', 'data')
IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'images_hand')

# Hareket Protokolü (Model eğitimiyle uyumlu)
# Modelde 0: Rest, 1-10: Hareketler
MOVEMENT_DURATION = 4  # saniye (Veri toplama süresi)
REST_DURATION = 3      # saniye (Ara dinlenme süresi - Bu da veri olarak kaydedilecek)
REPETITIONS = 10       # Her hareket için tekrar sayısı (Eğitim için bol veri iyidir)

# Hareket İsimleri (Eğitim koduyla eşleşmeli)
GESTURE_NAMES = [
    'Rest', 'Fist', 'Open', 'Point', 'Victory', 'OK',
    'ThumbUp', 'ThumbDn', 'Grasp', 'Pinch', 'WristFlex'
]
# Toplanacak aktif hareketler (1'den 10'a kadar)
MOVEMENTS = list(range(1, 11)) 

# Grafik Ayarları
WINDOW_SIZE = 500
PLOT_UPDATE_INTERVAL = 100 # ms (S3 hızlıdır, arayüzü hızlandırdık)

# ==========================================
# GLOBAL DEĞİŞKENLER
# ==========================================
emg_data = [deque(maxlen=WINDOW_SIZE) for _ in range(6)] # 6 Kanal deque listesi
time_data = deque(maxlen=WINDOW_SIZE)

packet_count = 0
start_time = None
current_phase = {
    "movement_id": 0,    # 0=Rest, 1..10=Hareket
    "movement_name": "Rest",
    "phase": "HAZIRLIK", # EKRANDA GÖZÜKEN DURUM
    "time_left": 0, 
    "rep": 0
}

csv_file = None
csv_writer = None
running = True

# ==========================================
# SINIFLAR VE FONKSİYONLAR
# ==========================================

class TrainingPlotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title('ESP32-S3 - Veri Toplama (TD4 Uyumlu)')

        # Grid: Sol (Resim), Sağ (6 EMG Grafiği)
        gs = GridSpec(3, 4, figure=self.fig)

        # Sol taraf: Görsel (1 sütun genişliğinde)
        self.ax_image = self.fig.add_subplot(gs[:, 0:1])
        self.ax_image.axis('off')

        # Sağ taraf: 6 Grafik (3 satır, 3 sütun genişliğinde)
        self.axes = []
        self.lines = []
        colors = ['#FF0000', '#00FF00', '#0000FF', '#FF00FF', '#00FFFF', '#FFA500']
        
        # 3x2 düzen (6 sensör)
        pos = [(0,1), (0,2), (0,3), (1,1), (1,2), (1,3)] # 6'lı grid yerine 3x3 alana yaydık ama alt alta dizeceğiz
        # Daha düzgün 3 satır 2 sütun (sağ taraf için)
        
        self.ax1 = self.fig.add_subplot(gs[0, 1:])
        self.ax2 = self.fig.add_subplot(gs[1, 1:])
        self.ax3 = self.fig.add_subplot(gs[2, 1:])
        
        # Yerleşim değişikliği: 6 grafiği alt alta koymak yerine
        # 3 satır yapıp her satıra 2 sensör çizdirelim (daha temiz görünür)
        # Ama kolaylık olsun diye 6 ayrı subplot oluşturuyorum:
        self.fig.clf()
        gs = GridSpec(3, 3, figure=self.fig)
        self.ax_image = self.fig.add_subplot(gs[:, 0]) # Sol sütun resim
        self.ax_image.axis('off')
        
        # Grafikler
        self.axes = [
            self.fig.add_subplot(gs[0, 1]), self.fig.add_subplot(gs[0, 2]),
            self.fig.add_subplot(gs[1, 1]), self.fig.add_subplot(gs[1, 2]),
            self.fig.add_subplot(gs[2, 1]), self.fig.add_subplot(gs[2, 2])
        ]

        channel_names = [f'EMG {i+1}' for i in range(6)]
        
        self.value_texts = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], color=colors[i], linewidth=1)
            self.lines.append(line)
            ax.set_ylim(0, 4096) # 12-bit ADC
            ax.set_xlim(0, WINDOW_SIZE)
            ax.grid(True, alpha=0.3)
            ax.set_title(channel_names[i], fontsize=9, pad=2)
            
            # Anlık değer texti
            txt = ax.text(0.95, 0.9, '', transform=ax.transAxes, ha='right', fontweight='bold')
            self.value_texts.append(txt)

        self.last_img_id = -1
        self.load_image(0)

    def load_image(self, movement_id):
        """Görseli güncelle"""
        try:
            # Rest sırasında (ID 0) veya ID 99 durumunda 0.jpg göster
            img_id = movement_id if movement_id < len(GESTURE_NAMES) else 0
            img_path = os.path.join(IMAGES_DIR, f'{img_id}.jpg')
            
            if os.path.exists(img_path):
                img = Image.open(img_path)
                self.ax_image.clear()
                self.ax_image.axis('off')
                self.ax_image.imshow(img)
            else:
                self.ax_image.text(0.5, 0.5, f'NO IMAGE\n{img_id}.jpg', ha='center')

            # Başlık güncelle
            ph = current_phase["phase"]
            name = current_phase["movement_name"]
            rep = current_phase["rep"]
            t = current_phase["time_left"]
            
            color = 'green' if ph == "HAREKET" else ('orange' if ph == "DINLENME" else 'blue')
            
            self.ax_image.set_title(
                f"{ph}\n{name}\nTekrar: {rep}/{REPETITIONS}\nKalan: {t:.1f}s",
                fontsize=16, fontweight='bold', color=color, pad=20
            )

        except Exception as e:
            print(f"Görsel hatası: {e}")

    def update_plot(self, frame):
        """Animasyon döngüsü"""
        if not running: return
        
        # Grafikleri çiz
        x = range(len(emg_data[0]))
        for i in range(6):
            if len(emg_data[i]) > 0:
                # Görselleştirme için basit filtre (Notch filtresi yavaşlatabilir, burada raw gösteriyoruz)
                # İstersen notch_filter_50hz fonksiyonunu açabilirsin.
                data_list = list(emg_data[i])
                self.lines[i].set_data(x, data_list)
                self.value_texts[i].set_text(str(data_list[-1]))

        # Görsel güncelleme kontrolü
        mov_id = current_phase["movement_id"]
        # Eğer faz "DINLENME" ise, görsel olarak 0 (Rest) gösterelim
        display_id = mov_id if current_phase["phase"] == "HAREKET" else 0
        
        # Sadece durum değiştiyse veya her 5 frame'de bir text güncelle
        self.load_image(display_id)

def find_esp32_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "CP210" in p.description or "CH340" in p.description or "USB Serial" in p.description:
            return p.device
    return ports[0].device if ports else None

def parse_packet(data):
    """16-byte binary paket: Header(2) + Seq(2) + 6xADC(2)"""
    if len(data) != 16 or data[0] != 0xA5 or data[1] != 0x5A:
        return None
    
    unpacked = struct.unpack('<H HHHHHH', data[2:]) # Little endian
    return {
        'seq': unpacked[0],
        'emg': unpacked[1:] # Tuple of 6
    }

def read_serial_thread(ser):
    """Arka plan veri okuma"""
    global packet_count, start_time
    buffer = bytearray()
    
    while running:
        try:
            if ser.in_waiting:
                buffer.extend(ser.read(ser.in_waiting))
                
                while len(buffer) >= 16:
                    # Header ara
                    idx = buffer.find(b'\xA5\x5A')
                    if idx == -1:
                        buffer = bytearray() # Header yoksa çöpü at
                        break
                    
                    if idx > 0:
                        buffer = buffer[idx:] # Hizala
                        
                    if len(buffer) < 16:
                        break # Yeterli veri yok
                        
                    packet_data = buffer[:16]
                    buffer = buffer[16:]
                    
                    parsed = parse_packet(packet_data)
                    if parsed:
                        packet_count += 1
                        ts = time.time()
                        if start_time is None: start_time = ts
                        
                        # Datayı kuyruklara ekle
                        for i in range(6):
                            emg_data[i].append(parsed['emg'][i])
                        
                        # CSV Kayıt (ÖNEMLİ: Feature Extraction ile uyum)
                        if csv_writer:
                            # Model eğitimi için Label belirleme:
                            # HAREKET fazındaysak -> Hareketin ID'si (1-10)
                            # DINLENME fazındaysak -> 0 (Rest)
                            
                            label_name = current_phase["movement_name"]
                            if current_phase["phase"] == "DINLENME":
                                label_name = "Rest"
                            
                            csv_writer.writerow([
                                f"{ts - start_time:.4f}", # Timestamp
                                parsed['seq'],
                                label_name,       # Movement Name (feature_extraction buna bakar)
                                current_phase["phase"],
                                current_phase["rep"],
                                *parsed['emg']    # EMG1..EMG6 unpacking
                            ])
            else:
                time.sleep(0.001) # CPU rahatlat
        except Exception as e:
            print(f"Serial Read Error: {e}")
            break

def protocol_thread():
    """Eğitim zamanlaması"""
    global running
    print("Eğitim protokolü başladı...")
    time.sleep(2) # Başlangıç gecikmesi

    # Başlangıçta uzun bir Rest verisi al (Baseline için)
    print("Initial REST data collection...")
    current_phase.update({
        "movement_id": 0,
        "movement_name": "Rest",
        "phase": "DINLENME",
        "time_left": 5,
        "rep": 0
    })
    time.sleep(5)

    for movement_id in MOVEMENTS:
        mov_name = GESTURE_NAMES[movement_id]
        
        for rep in range(1, REPETITIONS + 1):
            if not running: break
            
            # 1. HAREKET FAZI (Veri = Hareket Sınıfı)
            current_phase.update({
                "movement_id": movement_id,
                "movement_name": mov_name,
                "phase": "HAREKET",
                "rep": rep,
                "time_left": MOVEMENT_DURATION
            })
            
            # Geri sayım
            t_start = time.time()
            while time.time() - t_start < MOVEMENT_DURATION:
                if not running: break
                current_phase["time_left"] = MOVEMENT_DURATION - (time.time() - t_start)
                time.sleep(0.1)
                
            # 2. DINLENME FAZI (Veri = Rest Sınıfı)
            current_phase.update({
                "movement_id": movement_id, # Görselde kafa karışmasın diye ID kalabilir
                "movement_name": mov_name,  # Ama CSV'ye "Rest" yazılacak (read_serial'da handle ediliyor)
                "phase": "DINLENME",
                "rep": rep,
                "time_left": REST_DURATION
            })
            
            t_start = time.time()
            while time.time() - t_start < REST_DURATION:
                if not running: break
                current_phase["time_left"] = REST_DURATION - (time.time() - t_start)
                time.sleep(0.1)

    print("Protokol tamamlandı.")
    running = False

def main():
    global csv_file, csv_writer, running, PORT

    # Port bulma
    if PORT is None:
        PORT = find_esp32_port()
        if PORT is None:
            print("ESP32 bulunamadı! Lütfen USB'yi takın.")
            return
    
    print(f"Port: {PORT}, Baud: {BAUD}")
    
    # Dosya hazırlığı
    os.makedirs(DATA_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f'training_data_S3_{timestamp}.csv')
    
    try:
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        # Header (feature_extraction.py bu sütunları bekler)
        csv_writer.writerow(['Timestamp', 'Seq', 'Movement', 'Phase', 'Rep', 'EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6'])
        
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # ESP32 Reset Beklemesi
        ser.reset_input_buffer()
        ser.write(b'S') # Stream başlat
        print("Veri akışı başlatıldı.")
        
        # Threadleri başlat
        t_serial = threading.Thread(target=read_serial_thread, args=(ser,))
        t_proto = threading.Thread(target=protocol_thread)
        
        t_serial.start()
        t_proto.start()
        
        # Arayüz
        plotter = TrainingPlotter()
        ani = animation.FuncAnimation(plotter.fig, plotter.update_plot, interval=PLOT_UPDATE_INTERVAL)
        plt.show()
        
        running = False # Pencere kapanınca bitir
        t_proto.join()
        t_serial.join()
        
        ser.write(b'E')
        ser.close()
        
    except Exception as e:
        print(f"HATA: {e}")
    finally:
        if csv_file: csv_file.close()
        print(f"Veri kaydedildi: {filename}")

if __name__ == '__main__':
    main()