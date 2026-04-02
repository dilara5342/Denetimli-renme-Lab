import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. VERİ HAZIRLIĞI
df = pd.read_excel("CarSales.xlsx")  # Dosya adını kontrol et!
df = df.dropna(subset=['Price_in_thousands', 'Horsepower'])

X = df['Horsepower'].values
y = df['Price_in_thousands'].values

# NORMALİZASYON (Gradyan inişinin çalışması için ŞART)
X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# 2. GRADYAN İNİŞİ PARAMETRELERİ
w = 0.0
b = 0.0
learning_rate = 0.01  # Adım büyüklüğü
iterations = 5000  # Döngü sayısı

m = len(y_norm)  # Veri sayısı

# 3. ALGORİTMA (FROM SCRATCH)
for i in range(iterations):
    y_pred = w * X_norm + b  # Tahmin yap

    # Türevleri hesapla (Gradyanlar)
    dw = (1 / m) * np.sum((y_pred - y_norm) * X_norm)
    db = (1 / m) * np.sum(y_pred - y_norm)

    # Güncelleme yap
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Her 500 adımda bir hatayı yazdır (İşleyişi görmek için)
    if i % 500 == 0:
        mse = np.mean((y_pred - y_norm) ** 2)
        print(f"Adım {i}: Hata (MSE) = {mse}")

print(f"\nFinal değerleri: w = {w}, b = {b}")

# 4. GÖRSELLEŞTİRME
plt.scatter(X_norm, y_norm, color='blue', alpha=0.5, label='Gerçek Veri')
plt.plot(X_norm, w * X_norm + b, color='red', label='Gradyan İnişi Çizgisi')
plt.xlabel('Horsepower (Normalize)')
plt.ylabel('Fiyat (Normalize)')
plt.legend()
plt.show()

# Her adımda hatayı bir listeye kaydet (Döngünün içine ekle)
mse_list.append(mse)

plt.plot(range(iterations), mse_list)
plt.title('Hata Dususu (Cost History)')
plt.xlabel('Iterasyon')
plt.ylabel('MSE')
plt.show()

