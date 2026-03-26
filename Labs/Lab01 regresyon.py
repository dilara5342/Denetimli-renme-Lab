import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Veriyi Oku
df = pd.read_excel("CarSales.xlsx")

# 2. Veriyi Temizle
df_temiz = df.dropna(subset=['Price_in_thousands', 'Horsepower', 'Engine_size', 'Curb_weight'])

# 3. Temizlendiğini Kontrol Et
print(df_temiz.isnull().sum())


#--- A. BASİT DOĞRUSAL REGRESYON ---

#1. Girdi (X) ve Çıktı (y) belirleme
X_simple = df_temiz[['Horsepower']] # Çift parantez önemli, sklearn tablo formatı ister
y = df_temiz['Price_in_thousands']

# 2. Modeli Kur ve Eğit (Öğrenme aşaması)
model_simple = LinearRegression()
model_simple.fit(X_simple, y)

# 3. Tahmin Yap ve Tablo İçin Değerleri Hesapla
y_tahmin_simple = model_simple.predict(X_simple)
print("A Şıkkı MSE Değeri:", mean_squared_error(y, y_tahmin_simple))
print("A Şıkkı R-Squared Değeri:", r2_score(y, y_tahmin_simple))

# 4. Rapor İçin Grafiği Çizdir
plt.scatter(X_simple, y, color='blue', alpha=0.5) # Gerçek veriler
plt.plot(X_simple, y_tahmin_simple, color='red', linewidth=2) # Modelimizin çizgisi
plt.title('Basit Regresyon: Beygir Gücü ve Fiyat')
plt.xlabel('Beygir Gücü (Horsepower)')
plt.ylabel('Fiyat (Bin Dolar)')
plt.show()

# --- B. ÇOKLU DOĞRUSAL REGRESYON ---

# 1. Girdileri Çoğaltıyoruz (Artık 3 farklı değişkene bakıyoruz)
X_multi = df_temiz[['Horsepower', 'Engine_size', 'Curb_weight']]

# 2. Yeni Modeli Kur ve Eğit
model_multi = LinearRegression()
model_multi.fit(X_multi, y)

# 3. Yeni Tahminler ve Hata Hesaplama
y_tahmin_multi = model_multi.predict(X_multi)
print("B Şıkkı MSE Değeri:", mean_squared_error(y, y_tahmin_multi))
print("B Şıkkı R-Squared Değeri:", r2_score(y, y_tahmin_multi))

# 4. Grafiği Çizdir
plt.figure() # Bir öncekine karışmasın diye yeni ve temiz bir grafik penceresi açar
plt.scatter(y, y_tahmin_multi, color='green', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, linestyle='--')
plt.title('Coklu Regresyon: Gercek Fiyat vs Tahmin Edilen')
plt.xlabel('Gercek Fiyat (Bin Dolar)')
plt.ylabel('Tahmin Edilen Fiyat (Bin Dolar)')
plt.show()

from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# --- C. POLİNOMİYAL REGRESYON (Derece=3) ---

# 1. Beygir gücü verimizi (X_simple) kavis çizebilecek şekilde 3. derece polinoma dönüştürüyoruz
poly_donusturucu = PolynomialFeatures(degree=3)
X_poly = poly_donusturucu.fit_transform(X_simple)

# 2. Yeni kavisli verimizle modelimizi eğitiyoruz
model_poly = LinearRegression()
model_poly.fit(X_poly, y)

# 3. Tahmin ve Performans (Word'deki tablonun 3. satırı için)
y_tahmin_poly = model_poly.predict(X_poly)
print("C Şıkkı MSE Değeri:", mean_squared_error(y, y_tahmin_poly))
print("C Şıkkı R-Squared Değeri:", r2_score(y, y_tahmin_poly))

# 4. Grafiği Çizdirme
# Eğriyi pürüzsüz çizebilmek için noktaları küçükten büyüğe sıralamamız gerekiyor
X_sirali = np.sort(X_simple, axis=0)
X_poly_sirali = poly_donusturucu.transform(X_sirali)
y_tahmin_sirali = model_poly.predict(X_poly_sirali)

plt.figure()
plt.scatter(X_simple, y, color='blue', alpha=0.5, label='Gerçek Arabalar')
plt.plot(X_sirali, y_tahmin_sirali, color='purple', linewidth=2, label='Polinom Eğrisi (Degree=3)')
plt.title('Polinomiyal Regresyon: Beygir Gucu vs Fiyat')
plt.xlabel('Beygir Gucu (Horsepower)')
plt.ylabel('Fiyat (Bin Dolar)')
plt.legend()
plt.show()

from sklearn.linear_model import Ridge

# --- D. RIDGE REGRESYON ---

# 1. Modeli kuruyoruz. 'alpha' parametresi modelin ne kadar dizginleneceğini belirler.
# Hocan test etmeni istediği için buraya alpha=10 diyoruz. (Bunu değiştirip deneyebilirsin de)
model_ridge = Ridge(alpha=10.0)

# B şıkkındaki 3 değişkenli verimizi (X_multi) kullanıyoruz
model_ridge.fit(X_multi, y)

# 2. Tahmin ve Performans (Tablonun son satırı için)
y_tahmin_ridge = model_ridge.predict(X_multi)
print("D Şıkkı MSE Değeri:", mean_squared_error(y, y_tahmin_ridge))
print("D Şıkkı R-Squared Değeri:", r2_score(y, y_tahmin_ridge))

# 3. Grafiği Çizdirme (Gerçek vs Tahmin)
plt.figure()
plt.scatter(y, y_tahmin_ridge, color='orange', alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, linestyle='--')
plt.title('Ridge Regresyon (alpha=10): Gercek vs Tahmin')
plt.xlabel('Gercek Fiyat (Bin Dolar)')
plt.ylabel('Tahmin Edilen Fiyat (Bin Dolar)')
plt.show()