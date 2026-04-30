import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random


# ==========================================
# YARDIMCI FONKSİYONLAR (HİÇBİRİ DEĞİŞTİRİLMEDİ)
# ==========================================
def print_stats(name, image):
    """Verilen görüntünün istatistiksel değerlerini terminale yazar."""
    print(f"--- {name} İstatistikleri ---")
    print(f"Min Değer: {np.min(image)}")
    print(f"Max Değer: {np.max(image)}")
    print(f"Mean (Ort): {np.mean(image):.2f}")
    print(f"Median (Med): {np.median(image)}\n")


def show_side_by_side_app1(img_original, img_processed, title_processed, step_num, cmap=None):
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"ADIM {step_num}: LÜTFEN KODUN DEVAM ETMESİ İÇİN BU PENCEREYİ KAPATIN (X)", fontsize=12,
                 fontweight='bold', color='red')
    plt.subplot(1, 2, 1)
    plt.imshow(img_original)
    plt.title('Orijinal (RGB)')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    if cmap:
        plt.imshow(img_processed, cmap=cmap)
    else:
        plt.imshow(img_processed)
    plt.title(title_processed)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_three_images(img1, img2, img3, t1, t2, t3, main_title):
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"{main_title}\nLÜTFEN DEVAM ETMEK İÇİN PENCEREYİ KAPATIN (X)", color='red', fontweight='bold')
    plt.subplot(1, 3, 1), plt.imshow(img1, cmap='gray' if len(img1.shape) == 2 else None), plt.title(t1), plt.axis(
        'off')
    plt.subplot(1, 3, 2), plt.imshow(img2, cmap='gray' if len(img2.shape) == 2 else None), plt.title(t2), plt.axis(
        'off')
    plt.subplot(1, 3, 3), plt.imshow(img3, cmap='gray' if len(img3.shape) == 2 else None), plt.title(t3), plt.axis(
        'off')
    plt.tight_layout()
    plt.show()


def add_salt_pepper_noise(image, prob=0.05):
    noisy = np.copy(image)
    probs = np.random.random(noisy.shape[:2])
    noisy[probs < (prob / 2)] = 0
    noisy[probs > 1 - (prob / 2)] = 255
    return noisy


def show_side_by_side_app3(img1, img2, title1, title2, main_title, cmap1=None, cmap2=None):
    plt.figure(figsize=(10, 5))
    plt.suptitle(f"{main_title}\nLÜTFEN KODUN DEVAM ETMESİ İÇİN BU PENCEREYİ KAPATIN (X)", color='red',
                 fontweight='bold')
    plt.subplot(1, 2, 1), plt.imshow(img1, cmap=cmap1), plt.title(title1), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(img2, cmap=cmap2), plt.title(title2), plt.axis('off')
    plt.tight_layout()
    plt.show()


def show_channels(img_rgb, r, g, b):
    plt.figure(figsize=(12, 4))
    plt.suptitle("ADIM 4: RGB Kanallarının Ayrılması\nLÜTFEN KODUN BİTMESİ İÇİN BU PENCEREYİ KAPATIN (X)", color='red',
                 fontweight='bold')
    plt.subplot(1, 4, 1), plt.imshow(img_rgb), plt.title('Orijinal (RGB)'), plt.axis('off')
    plt.subplot(1, 4, 2), plt.imshow(r, cmap='Reds'), plt.title('Kırmızı (R) Kanalı'), plt.axis('off')
    plt.subplot(1, 4, 3), plt.imshow(g, cmap='Greens'), plt.title('Yeşil (G) Kanalı'), plt.axis('off')
    plt.subplot(1, 4, 4), plt.imshow(b, cmap='Blues'), plt.title('Mavi (B) Kanalı'), plt.axis('off')
    plt.tight_layout()
    plt.show()


# ==========================================
# ANA ÇALIŞTIRMA BLOĞU
# ==========================================
img_bgr = cv.imread('lena.jpg')

if img_bgr is None:
    print("Hata: 'lena.jpg' bulunamadı!")
else:
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)

    # ------------------------------------------
    # UYGULAMA 1: RENK UZAYLARI VE İSTATİSTİK
    # ------------------------------------------
    print("===== UYGULAMA 1 BAŞLADI =====")
    img_ycrcb = cv.cvtColor(img_bgr, cv.COLOR_BGR2YCrCb)
    img_hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    img_negative = 255 - img_rgb

    # Tüm matris değerlerini (Min, Max vb.) yazdırıyoruz
    print_stats("Orijinal (RGB)", img_rgb)
    print_stats("BGR", img_bgr)
    show_side_by_side_app1(img_rgb, img_bgr, 'BGR Formatı', step_num=1)

    print_stats("YCrCb", img_ycrcb)
    show_side_by_side_app1(img_rgb, img_ycrcb, 'YCrCb Formatı', step_num=2)

    print_stats("HSV", img_hsv)
    show_side_by_side_app1(img_rgb, img_hsv, 'HSV Formatı', step_num=3)

    print_stats("Negatif", img_negative)
    show_side_by_side_app1(img_rgb, img_negative, 'Negatif Format\n(g(x,y) = 255 - f(x,y))', step_num=4)

    # Histogram (Adım 5)
    plt.figure(figsize=(8, 5))
    plt.suptitle("ADIM 5: FİNAL - UYGULAMA 1 BİTTİ (PENCEREYİ KAPATABİLİRSİNİZ)", fontsize=12, fontweight='bold',
                 color='red')
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        hist = cv.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Görüntü Histogramı (RGB)')
    plt.xlabel('Piksel Değeri (0-255)')
    plt.ylabel('Piksel Sayısı')
    plt.show()
    print("===== UYGULAMA 1 TAMAMLANDI =====\n")

    # ------------------------------------------
    # UYGULAMA 2: UZAMSAL FİLTRELEME
    # ------------------------------------------
    print("===== UYGULAMA 2 BAŞLADI =====")
    mean_3x3 = cv.blur(img_rgb, (3, 3), borderType=cv.BORDER_CONSTANT)
    mean_5x5 = cv.blur(img_rgb, (5, 5), borderType=cv.BORDER_CONSTANT)
    show_three_images(img_rgb, mean_3x3, mean_5x5, "Orijinal", "Mean 3x3", "Mean 5x5",
                      "ADIM 2.1: Mean (Ortalama) Filtresi")

    gauss_3x3 = cv.GaussianBlur(img_rgb, (3, 3), 0, borderType=cv.BORDER_CONSTANT)
    gauss_5x5 = cv.GaussianBlur(img_rgb, (5, 5), 0, borderType=cv.BORDER_CONSTANT)
    show_three_images(img_rgb, gauss_3x3, gauss_5x5, "Orijinal", "Gaussian 3x3", "Gaussian 5x5",
                      "ADIM 2.2: Gaussian Filtresi")

    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    laplacian_raw = cv.Laplacian(gray, cv.CV_64F, ksize=3, borderType=cv.BORDER_CONSTANT)
    laplacian = cv.convertScaleAbs(laplacian_raw)
    show_three_images(img_rgb, gray, laplacian, "Orijinal (RGB)", "Grayscale", "Laplacian Kenar Tespiti",
                      "ADIM 2.3: Laplacian Filtresi")

    noisy_img = add_salt_pepper_noise(img_rgb)
    median_3x3 = cv.medianBlur(noisy_img, 3)
    median_5x5 = cv.medianBlur(noisy_img, 5)
    show_three_images(noisy_img, median_3x3, median_5x5, "Gürültülü Orijinal", "Median 3x3", "Median 5x5",
                      "ADIM 2.4: Median Filtresi (Gürültü Temizleme)")
    print("===== UYGULAMA 2 TAMAMLANDI =====\n")

    # ------------------------------------------
    # UYGULAMA 3: OTSU EŞİKLEME METODU
    # ------------------------------------------
    print("===== UYGULAMA 3 BAŞLADI =====")
    gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    ret, thresh_otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    print(f"Otsu Algoritmasının Otomatik Bulduğu Eşik Değeri: {ret}")

    show_side_by_side_app3(gray, thresh_otsu,
                           "Orijinal (Grayscale)",
                           f"Otsu Binary (Eşik: {ret})",
                           "ADIM 3: Otsu Metodu ile İkili (Binary) Görüntüye Geçiş",
                           cmap1='gray', cmap2='gray')
    print("===== UYGULAMA 3 TAMAMLANDI =====\n")

    # ------------------------------------------
    # UYGULAMA 4: RGB VE GRAYSCALE İLİŞKİSİ
    # ------------------------------------------
    print("===== UYGULAMA 4 BAŞLADI =====")
    r, g, b = cv.split(img_rgb)
    show_channels(img_rgb, r, g, b)

    print("\n--- Grayscale (Gri Seviye) Dönüşüm Matematiği ---")
    print("Soru: Grayscale bir görüntü, RGB kanallarının hangi oranda bileşimidir?")
    print("Cevap: Bilgisayarlı görüde ve OpenCV'de (cv.COLOR_RGB2GRAY) standart formül şudur:")
    print("Y (Gri) = (0.299 * R) + (0.587 * G) + (0.114 * B)")
    print("Nedeni: İnsan gözü biyolojik olarak Yeşil (G) renge çok daha duyarlıdır, ")
    print("Mavi (B) renge ise en az duyarlıdır. Bu yüzden en yüksek ağırlık katsayısı yeşile verilmiştir.")
    print("=========================================\n")
    print("TÜM LABORATUVAR GÖREVLERİ BAŞARIYLA TAMAMLANDI!")