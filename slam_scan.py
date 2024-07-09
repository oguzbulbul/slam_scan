import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from rplidar import RPLidar

# EKF SLAM değişkenleri
durum_boyutu = 3  # [x, y, theta] - Durum boyutu
kontrol_boyutu = 2  # [v, omega] - Kontrol boyutu
olcum_boyutu = 2  # [menzil, yön] - Ölçüm boyutu

# Başlangıç durumu [x, y, theta]
durum = np.zeros(durum_boyutu)
# Başlangıç kovaryansı
kovaryans = np.eye(durum_boyutu) * 0.1  # Başlangıç kovaryans matrisi

# Süreç gürültüsü kovaryansı
surec_gurultusu = np.diag([0.1, 0.1, np.deg2rad(5)])  # Kontrol gürültüsü

# Ölçüm gürültüsü kovaryansı
olcum_gurultusu = np.diag([0.1, np.deg2rad(1)])  # Ölçüm gürültüsü

# Durum geçiş modeli
def hareket_modeli(durum, kontrol, zaman_adimi):
    theta = durum[2]
    v = kontrol[0]
    omega = kontrol[1]

    if omega == 0:
        dx = v * zaman_adimi * np.cos(theta)
        dy = v * zaman_adimi * np.sin(theta)
        dtheta = 0
    else:
        dx = -(v / omega) * np.sin(theta) + (v / omega) * np.sin(theta + omega * zaman_adimi)
        dy = (v / omega) * np.cos(theta) - (v / omega) * np.cos(theta + omega * zaman_adimi)
        dtheta = omega * zaman_adimi

    return durum + np.array([dx, dy, dtheta])

# Hareket modelinin Jacobian'ı
def jacobian_hareket_modeli(durum, kontrol, zaman_adimi):
    theta = durum[2]
    v = kontrol[0]
    omega = kontrol[1]

    if omega == 0:
        J = np.array([
            [1, 0, -v * zaman_adimi * np.sin(theta)],
            [0, 1, v * zaman_adimi * np.cos(theta)],
            [0, 0, 1]
        ])
    else:
        J = np.array([
            [1, 0, (v / omega) * (np.cos(theta) - np.cos(theta + omega * zaman_adimi))],
            [0, 1, (v / omega) * (np.sin(theta + omega * zaman_adimi) - np.sin(theta))],
            [0, 0, 1]
        ])

    return J

# Ölçüm modeli
def olcum_modeli(durum, hedef):
    dx = hedef[0] - durum[0]
    dy = hedef[1] - durum[1]
    menzil = np.sqrt(dx**2 + dy**2)
    yon = math.atan2(dy, dx) - durum[2]

    return np.array([menzil, yon])

# Ölçüm modelinin Jacobian'ı
def jacobian_olcum_modeli(durum, hedef):
    dx = hedef[0] - durum[0]
    dy = hedef[1] - durum[1]
    menzil = np.sqrt(dx**2 + dy**2)

    H = np.array([
        [-dx / menzil, -dy / menzil, 0],
        [dy / (menzil**2), -dx / (menzil**2), -1]
    ])

    return H

# EKF SLAM algoritması
def ekf_slam(durum, kovaryans, kontrol, olcumler, hedefler, zaman_adimi):
    # Tahmin adımı
    durum_tahmin = hareket_modeli(durum, kontrol, zaman_adimi)  # Durum tahmini
    F = jacobian_hareket_modeli(durum, kontrol, zaman_adimi)  # Hareket modelinin Jacobian'ı
    kovaryans_tahmin = F @ kovaryans @ F.T + surec_gurultusu  # Kovaryans güncelleme

    # Güncelleme adımı
    for i, hedef in enumerate(hedefler):
        olcum_tahmin = olcum_modeli(durum_tahmin, hedef)  # Tahmin edilen ölçüm
        H = jacobian_olcum_modeli(durum_tahmin, hedef)  # Ölçüm modelinin Jacobian'ı

        S = H @ kovaryans_tahmin @ H.T + olcum_gurultusu  # Yenilik kovaryansı
        K = kovaryans_tahmin @ H.T @ np.linalg.inv(S)  # Kalman kazancı

        yenilik = olcumler[i] - olcum_tahmin  # Yeni ölçüm
        yenilik[1] = ((yenilik[1] + np.pi) % (2 * np.pi)) - np.pi  # Açıyı normalize et

        durum_tahmin = durum_tahmin + K @ yenilik
        kovaryans_tahmin = (np.eye(len(kovaryans_tahmin)) - K @ H) @ kovaryans_tahmin

    return durum_tahmin, kovaryans_tahmin

def main():
    lidar_port = '/dev/ttyUSB0'
    baud_rate = 256_000
    max_scans = 1000
    kontrol = np.array([1.0, 0.1])  # [v, omega]
    zaman_adimi = 0.1  # Zaman adımı

    try:
        lidar = RPLidar(lidar_port, baudrate=baud_rate)
        
        for i, scan in enumerate(lidar.iter_scans()):
            olcumler = []
            hedefler = []

            for (_, angle, distance) in scan:
                if distance > 0:  # Filter out invalid measurements
                    rad_angle = np.deg2rad(angle)
                    olcumler.append([distance, rad_angle])
                    hedefler.append([distance * np.cos(rad_angle), distance * np.sin(rad_angle)])

            if len(olcumler) < 2:
                continue

            olcumler = np.array(olcumler)
            hedefler = np.array(hedefler)

            # EKF SLAM çalıştır
            global durum, kovaryans
            durum, kovaryans = ekf_slam(durum, kovaryans, kontrol, olcumler, hedefler, zaman_adimi)

            # Sonuçları sakla
            trajektori.append(durum.copy())

            # Görselleştirme
            plt.figure()
            plt.scatter(hedefler[:, 0], hedefler[:, 1], s=50, c='b', marker='.', label='Hedefler')
            plt.scatter(durum[0], durum[1], s=50, c='r', marker='x', label='Robot')
            plt.xlabel('X Konumu')
            plt.ylabel('Y Konumu')
            plt.legend()
            plt.title('EKF SLAM')
            plt.show()

            if i >= max_scans:
                break

    except Exception as e:
        print(f"Failed to initialize or read from the lidar: {e}")
    finally:
        try:
            lidar.stop()
            lidar.stop_motor()
            lidar.disconnect()
            print("Lidar disconnected successfully.")
        except Exception as e:
            print(f"Failed to properly disconnect the lidar: {e}")

if __name__ == "__main__":
    trajektori = []
    main()
