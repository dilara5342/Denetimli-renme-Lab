

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import logging
import os

# TensorFlow uyarılarını gizleme (Daha temiz bir terminal çıktısı için)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ==========================================
# YARDIMCI FONKSİYON: PERFORMANS METRİKLERİ YAZDIRMA (GÖREV 6 İÇİN)
# ==========================================
def print_metrics(model, X_test, y_test, title):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- {title} ---")
    print(f"Accuracy         : {acc:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print(f"Confusion Matrix :\n{cm}")

print("===== LAB 4: YAPAY SİNİR AĞLARI BAŞLIYOR =====")

# ==========================================
# GÖREV 1: VERİ SETİNİN HAZIRLANMASI
# ==========================================
wine = load_wine()
X = wine.data
y = wine.target

# Veriyi bölme ve Standartlaştırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# GÖREV 2: HAM VERİ İLE EĞİTİM
# ==========================================
model_raw = Sequential([Dense(16, input_shape=(13,), activation='relu'), Dense(3, activation='softmax')])
model_raw.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_raw = model_raw.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=0)
print_metrics(model_raw, X_test, y_test, "GÖREV 2: HAM VERİ (Ölçeklendirilmemiş)")

# ==========================================
# GÖREV 3: STANDARDIZE VERİ İLE EĞİTİM (EN İYİ MODEL)
# ==========================================
model_scaled = Sequential([Dense(16, input_shape=(13,), activation='relu'), Dense(3, activation='softmax')])
model_scaled.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_scaled = model_scaled.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=0)
print_metrics(model_scaled, X_test_scaled, y_test, "GÖREV 3: STANDARDIZE VERİ (ReLU + LR=0.001)")

# --- GRAFİK 1: Ham vs Standart Veri Karşılaştırması ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_raw.history['loss'], label='Eğitim Loss')
plt.plot(history_raw.history['val_loss'], label='Test Loss')
plt.title('Ham Veri (Kararsız)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_scaled.history['loss'], label='Eğitim Loss')
plt.plot(history_scaled.history['val_loss'], label='Test Loss')
plt.title('Standart Veri (Kararlı)')
plt.legend()
plt.suptitle("Görev 2 ve 3: Ham vs Standart Veri Loss Karşılaştırması\n(KODUN DEVAM ETMESİ İÇİN BU PENCEREYİ KAPATIN)")
plt.show()

# ==========================================
# GÖREV 4: AKTİVASYON FONKSİYONLARI (SIGMOID)
# ==========================================
model_sig = Sequential([Dense(16, input_shape=(13,), activation='sigmoid'), Dense(3, activation='softmax')])
model_sig.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_sig = model_sig.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=0)
print_metrics(model_sig, X_test_scaled, y_test, "GÖREV 4: SIGMOID AKTİVASYONU")

# --- GRAFİK 2: Sigmoid vs ReLU Karşılaştırması ---
plt.figure(figsize=(8, 5))
plt.plot(history_sig.history['val_loss'], label='Sigmoid (Yavaş Düşer)', linestyle='--')
plt.plot(history_scaled.history['val_loss'], label='ReLU (Hızlı Düşer - Görev 3 Modeli)')
plt.title("Görev 4: Sigmoid vs ReLU Test Loss Karşılaştırması\n(KODUN DEVAM ETMESİ İÇİN BU PENCEREYİ KAPATIN)")
plt.legend()
plt.show()

# ==========================================
# GÖREV 5: LEARNING RATE (ÖĞRENME ORANI) ANALİZİ
# ==========================================
# LR = 1.0 (Aşırı Yüksek)
model_lr_high = Sequential([Dense(16, input_shape=(13,), activation='relu'), Dense(3, activation='softmax')])
model_lr_high.compile(optimizer=Adam(learning_rate=1.0), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_lr_high = model_lr_high.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=0)
print_metrics(model_lr_high, X_test_scaled, y_test, "GÖREV 5: LR = 1.0 (Overshooting)")

# LR = 1e-7 (Aşırı Düşük)
model_lr_low = Sequential([Dense(16, input_shape=(13,), activation='relu'), Dense(3, activation='softmax')])
model_lr_low.compile(optimizer=Adam(learning_rate=1e-7), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_lr_low = model_lr_low.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_test_scaled, y_test), verbose=0)
print_metrics(model_lr_low, X_test_scaled, y_test, "GÖREV 5: LR = 1e-7 (Aşırı Yavaş)")

# --- GRAFİK 3: Learning Rate Karşılaştırması ---
plt.figure(figsize=(8, 5))
plt.plot(history_lr_high.history['loss'], label='LR = 1.0 (Overshooting / Ezberleme)')
plt.plot(history_scaled.history['loss'], label='LR = 0.001 (İdeal - Görev 3 Modeli)')
plt.plot(history_lr_low.history['loss'], label='LR = 1e-7 (Çok Yavaş)')
plt.title("Görev 5: Farklı Learning Rate (LR) Değerlerinin Karşılaştırması\n(KODUN BİTMESİ İÇİN BU PENCEREYİ KAPATIN)")
plt.legend()
plt.show()

print("\n===== BÜTÜN GÖREVLER, METRİKLER VE GRAFİKLER BAŞARIYLA TAMAMLANDI! =====")
