import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# ==========================================
# GÖREV 1: VERİ ÖN İŞLEME (Sessiz Çalışır)
# ==========================================
df = pd.read_csv('Social_Network_Ads.csv')
df = df.drop(['User ID', 'Gender'], axis=1)

X = df.drop('Purchased', axis=1).values
y = df['Purchased'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ==========================================
# GÖREV 2: LOJİSTİK REGRESYON MODELİ
# ==========================================
print("\n--- GÖREV 2: LOJİSTİK REGRESYON SONUÇLARI ---")
classifier_log = LogisticRegression(random_state=0)
classifier_log.fit(X_train, y_train)

y_pred_log = classifier_log.predict(X_test)

acc_log = accuracy_score(y_test, y_pred_log)
cm_log = confusion_matrix(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log = recall_score(y_test, y_pred_log)
f1_log = f1_score(y_test, y_pred_log)

print(f"Accuracy (Doğruluk)  : {acc_log:.2f}")
print(f"Precision (Kesinlik) : {prec_log:.2f}")
print(f"Recall (Duyarlılık)  : {rec_log:.2f}")
print(f"F1-Score             : {f1_log:.2f}")
print(f"Confusion Matrix:\n{cm_log}")


from sklearn.linear_model import LinearRegression

# ==========================================
# GÖREV 3: LİNEER REGRESYON MODELİ
# ==========================================
print("\n--- GÖREV 3: LINEAR REGRESYON MODELİ ---")

# 1. Modeli Kuruyoruz ve Eğitiyoruz
regressor_lin = LinearRegression()
regressor_lin.fit(X_train, y_train)

# 2. Test verisi üzerinde tahmin yaptırıyoruz (Çıktılar sürekli olacak)
y_pred_lin_cont = regressor_lin.predict(X_test)

# ---> RAPOR İÇİN KANIT 1: Sürekli Çıktıları Yazdırıyoruz
print("1) Threshold Uygulanmadan Önceki Sürekli (Continuous) Çıktılar (İlk 5 adet):")
print(y_pred_lin_cont[:5])

# 3. 0.5 Threshold Uygulaması (Küsuratları 0 veya 1'e çevirme)
y_pred_lin = [1 if i >= 0.5 else 0 for i in y_pred_lin_cont]

# ---> RAPOR İÇİN KANIT 2: Çevrilmiş Çıktıları Yazdırıyoruz
print("\n2) 0.5 Threshold Uygulandıktan Sonraki Sınıflandırılmış Çıktılar (İlk 5 adet):")
print(y_pred_lin[:5])

# 4. Hoca'nın İstediği Tüm Başarı Notlarını (Metrikleri) Hesaplıyoruz
acc_lin = accuracy_score(y_test, y_pred_lin)
cm_lin = confusion_matrix(y_test, y_pred_lin)
prec_lin = precision_score(y_test, y_pred_lin)
rec_lin = recall_score(y_test, y_pred_lin)
f1_lin = f1_score(y_test, y_pred_lin)

# Sonuçları yazdırıyoruz
print(f"\n3) Model Metrikleri:")
print(f"Accuracy (Doğruluk)  : {acc_lin:.2f}")
print(f"Precision (Kesinlik) : {prec_lin:.2f}")
print(f"Recall (Duyarlılık)  : {rec_lin:.2f}")
print(f"F1-Score             : {f1_lin:.2f}")
print(f"Confusion Matrix:\n{cm_lin}")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# ==========================================
# GÖREV 5: BONUS - GÖRSELLEŞTİRME
# ==========================================
print("\n--- GÖREV 5: KARAR SINIRI (DECISION BOUNDARY) GÖRSELLEŞTİRME ---")
print("Grafik penceresi açılıyor... Lütfen bekleyin ve açılan penceredeki grafikleri kaydedin.")

# 1 satır, 2 sütunluk yan yana bir tablo oluşturuyoruz
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Arka planı boyamak için çok sık bir ızgara (meshgrid) oluşturuyoruz
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 0.5, stop = X_set[:, 0].max() + 0.5, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 0.5, stop = X_set[:, 1].max() + 0.5, step = 0.01))

renkler = ['red', 'green']

# --- 1. LOJİSTİK REGRESYON ÇİZİMİ ---
axes[0].contourf(X1, X2, classifier_log.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
axes[0].set_xlim(X1.min(), X1.max())
axes[0].set_ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    axes[0].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = renkler[i], label = j, edgecolors='black')
axes[0].set_title('Logistic Regression (Karar Sınırı)')
axes[0].set_xlabel('Yaş (Ölçeklenmiş)')
axes[0].set_ylabel('Maaş (Ölçeklenmiş)')
axes[0].legend()

# --- 2. LINEAR REGRESYON ÇİZİMİ ---
# Sürekli çıktıları grafiğe dökebilmek için 0.5 threshold'u burada da uyguluyoruz
lin_pred_grid = regressor_lin.predict(np.array([X1.ravel(), X2.ravel()]).T)
lin_pred_grid_class = np.where(lin_pred_grid >= 0.5, 1, 0).reshape(X1.shape)

axes[1].contourf(X1, X2, lin_pred_grid_class,
             alpha = 0.5, cmap = ListedColormap(('red', 'green')))
axes[1].set_xlim(X1.min(), X1.max())
axes[1].set_ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    axes[1].scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = renkler[i], label = j, edgecolors='black')
axes[1].set_title('Linear Regression (0.5 Threshold Karar Sınırı)')
axes[1].set_xlabel('Yaş (Ölçeklenmiş)')
axes[1].set_ylabel('Maaş (Ölçeklenmiş)')
axes[1].legend()

plt.tight_layout()
plt.show()