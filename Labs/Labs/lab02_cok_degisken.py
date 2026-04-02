import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. VERİ HAZIRLIĞI (2 Değişken Seçiyoruz)
df = pd.read_excel("CarSales.xlsx")
df = df.dropna(subset=['Price_in_thousands', 'Horsepower', 'Engine_size'])

X1 = df['Horsepower'].values
X2 = df['Engine_size'].values
y = df['Price_in_thousands'].values

# NORMALİZASYON (0-1 arasına sıkıştırma)
X1_n = (X1 - X1.min()) / (X1.max() - X1.min())
X2_n = (X2 - X2.min()) / (X2.max() - X2.min())
y_n = (y - y.min()) / (y.max() - y.min())

# 2. PARAMETRELER (2 adet w, 1 adet b)
w1, w2, b = 0.0, 0.0, 0.0
learning_rate = 0.01
iterations = 5000
m = len(y_n)

# 3. ÇOKLU GRADYAN İNİŞİ DÖNGÜSÜ
for i in range(iterations):
    # Tahmin (2 değişkenli)
    y_pred = w1 * X1_n + w2 * X2_n + b

    # Türevlerin (Gradyanların) hesaplanması
    dw1 = (1 / m) * np.sum((y_pred - y_n) * X1_n)
    dw2 = (1 / m) * np.sum((y_pred - y_n) * X2_n)
    db = (1 / m) * np.sum(y_pred - y_n)

    # Güncelleme
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b = b - learning_rate * db

    if i % 1000 == 0:
        mse = np.mean((y_pred - y_n) ** 2)
        print(f"Adım {i}: MSE = {mse}")

print(f"\nFinal değerleri: w1={w1:.4f}, w2={w2:.4f}, b={b:.4f}")

# 4. GÖRSELLEŞTİRME (Gerçek vs Tahmin Grafiği)
plt.figure(figsize=(8, 6))
plt.scatter(y_n, y_pred, color='green', alpha=0.5)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Mükemmel tahmin çizgisi
plt.xlabel('Gerçek Fiyat (Normalize)')
plt.ylabel('Tahmin Edilen Fiyat (Normalize)')
plt.title('2 Değişkenli Gradyan İnişi Başarısı')
plt.show()