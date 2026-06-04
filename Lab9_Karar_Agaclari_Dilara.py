import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import warnings

# Uyarıları kapatmak için (çıktı temiz görünsün diye)
warnings.filterwarnings('ignore')

print("=" * 50)
print("PRATİK UYGULAMA 1: AZ VERİ VS ÇOK VERİ KIYASLAMASI")
print("=" * 50)

# ==========================================
# 1. AZ VERİLİ ORİJİNAL SET (Deneme.py'deki 10 satır)
# ==========================================
veri_az = {
    'Yas': [34, 42, np.nan, 23, 55, 40, 28, 48, np.nan, 62],
    'Kredi_Skoru': [650, 580, 710, 620, np.nan, 690, 510, 730, 640, 600],
    'Sehir': ['İstanbul', 'Ankara', 'İstanbul', 'İzmir', 'Ankara', 'İstanbul', 'İzmir', 'Ankara', 'İstanbul', 'İzmir'],
    'Cinsiyet': ['Kadın', 'Erkek', 'Erkek', 'Kadın', np.nan, 'Erkek', 'Kadın', 'Erkek', 'Kadın', 'Erkek'],
    'Terk_Etti_Mi': [0, 1, 0, 0, 1, 0, 1, 1, 0, 1]
}
df_az = pd.DataFrame(veri_az)

# ==========================================
# 2. YAPAY ZEKA İLE ÇOĞALTILMIŞ SET (50 Satır)
# ==========================================
np.random.seed(42)
yaslar = np.random.randint(20, 65, 50).astype(float)
yaslar[np.random.choice(50, 5, replace=False)] = np.nan

skorlar = np.random.randint(500, 800, 50).astype(float)
skorlar[np.random.choice(50, 4, replace=False)] = np.nan

sehirler = np.random.choice(['İstanbul', 'Ankara', 'İzmir'], 50)
cinsiyetler = np.random.choice(['Kadın', 'Erkek'], 50)
cinsiyetler[np.random.choice(50, 3, replace=False)] = np.nan

terk = [1 if (s < 600 or y > 50) else 0 for s, y in zip(skorlar, yaslar)]
veri_cok = pd.DataFrame(
    {'Yas': yaslar, 'Kredi_Skoru': skorlar, 'Sehir': sehirler, 'Cinsiyet': cinsiyetler, 'Terk_Etti_Mi': terk})


# ==========================================
# 3. İKİ VERİ SETİNİ KIYASLAYAN FONKSİYON
# ==========================================
def agaci_egit_ve_raporla(df, veri_adi):
    print(f"\n---> {veri_adi.upper()} <---")

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    df[['Yas', 'Kredi_Skoru']] = num_imputer.fit_transform(df[['Yas', 'Kredi_Skoru']])
    df[['Sehir', 'Cinsiyet']] = cat_imputer.fit_transform(df[['Sehir', 'Cinsiyet']])

    df_encoded = pd.get_dummies(df, columns=['Sehir', 'Cinsiyet'], drop_first=True)

    X_pratik1 = df_encoded.drop('Terk_Etti_Mi', axis=1)
    y_pratik1 = df_encoded['Terk_Etti_Mi']

    X_train_p1, X_test_p1, y_train_p1, y_test_p1 = train_test_split(X_pratik1, y_pratik1, test_size=0.3,
                                                                    random_state=42)

    model = DecisionTreeClassifier(random_state=42, max_depth=3)
    model.fit(X_train_p1, y_train_p1)

    acc = accuracy_score(y_test_p1, model.predict(X_test_p1))
    print(f"Toplam Veri Sayısı: {len(df)}")
    print(f"Test Doğruluk Oranı (Accuracy): % {acc * 100:.2f}")


# Pratik 1 Kodlarını Çalıştır
agaci_egit_ve_raporla(df_az, "Az Kayıtlı (Orijinal) Veri Seti")
agaci_egit_ve_raporla(veri_cok, "Yapay Zeka İle Çoğaltılmış (50 Satır) Veri Seti")

print("\n\n" + "=" * 50)
print("PRATİK UYGULAMA 2: HEART ATTACK VERİ SETİ (OVERFITTING & BUDAMA)")
print("=" * 50)

# ==========================================
# ADIM 1: Veri Yükleme ve Eksik Veri Doldurma
# ==========================================
df_heart = pd.read_csv('Heart attack.csv')

sayisal = df_heart.select_dtypes(include=['float64', 'int64']).columns
df_heart[sayisal] = df_heart[sayisal].fillna(df_heart[sayisal].mean())

kategorik = df_heart.select_dtypes(include=['object']).columns
for col in kategorik:
    df_heart[col] = df_heart[col].fillna(df_heart[col].mode()[0])

# RAPOR İÇİN GÜNCELLENEN KISIM: Tüm sütunlardaki eksik verilerin sayısını liste halinde basar
print("Eksik veri kaldı mı? (Tüm sütunlar 0 olmalı):\n")
print(df_heart.isnull().sum())

# ==========================================
# ADIM 2: Önişleme (Encoding) ve X-y Ayrımı
# ==========================================
df_encoded_heart = pd.get_dummies(df_heart, drop_first=True)

X = df_encoded_heart.drop('stroke', axis=1)
y = df_encoded_heart['stroke']

print(f"\nX (Öznitelikler) Boyutu: {X.shape}")
print(f"y (Hedef) Boyutu       : {y.shape}\n")

# ==========================================
# ADIM 3: Train / Test Ayrımı
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ==========================================
# ADIM 4: Kısıtlamasız Model (Overfitting Gözlemi)
# ==========================================
model_sinirsiz = DecisionTreeClassifier(random_state=42)
model_sinirsiz.fit(X_train, y_train)

print("=== SINIRSIZ MODEL (OVERFITTING) SONUÇLARI ===")
print(f"Eğitim Doğruluğu: % {accuracy_score(y_train, model_sinirsiz.predict(X_train)) * 100:.2f}")
print(f"Test Doğruluğu  : % {accuracy_score(y_test, model_sinirsiz.predict(X_test)) * 100:.2f}\n")

# ==========================================
# ADIM 5: Budanmış (Pruned) Model (max_depth=4)
# ==========================================
model_budanmis = DecisionTreeClassifier(max_depth=4, random_state=42)
model_budanmis.fit(X_train, y_train)

print("=== BUDANMIŞ MODEL (MAX DERİNLİK = 4) SONUÇLARI ===")
print(f"Eğitim Doğruluğu: % {accuracy_score(y_train, model_budanmis.predict(X_train)) * 100:.2f}")
print(f"Test Doğruluğu  : % {accuracy_score(y_test, model_budanmis.predict(X_test)) * 100:.2f}")

# ==========================================
# ADIM 6: Ağaç Görselleştirme (HD KALİTE)
# ==========================================
plt.figure(figsize=(40, 20))
plot_tree(model_budanmis,
          feature_names=X.columns,
          class_names=['Saglikli', 'Inme/Felc'],
          filled=True,
          rounded=True,
          fontsize=12)

plt.title('Kalp Krizi/İnme Riski Karar Ağacı (Budanmış)', fontsize=20)
plt.savefig('karar_agaci_HD.png', dpi=300, bbox_inches='tight')
print("\n[BİLGİ] 'karar_agaci_HD.png' dosyası projenin bulunduğu klasöre yüksek kalitede kaydedildi!")
plt.show()