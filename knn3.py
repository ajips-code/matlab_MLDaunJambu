import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Membaca dataset dari file Excel
dataset = pd.read_excel('features-fix3.xlsx')

# Mengubah kolom kelas menjadi numerik (0 untuk 'daunjambuair' dan 1 untuk 'daunjambubiji')
dataset['Class'] = dataset['Class'].map({'daunjambuair': 0, 'daunjambubiji': 1})

# Memisahkan fitur dan kelas
fitur = dataset.iloc[:, 1:-1]
kelas = dataset['Class']

# Konversi X dan y menjadi array numpy
fitur = np.array(fitur)
kelas = np.array(kelas)

# Memisahkan data latih dan data uji, data latih 80% dan data uji 20%
X_train, X_test, y_train, y_test = train_test_split(fitur, kelas, test_size=0.2, random_state=42)

# Skala fitur menggunakan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Buat model K-NN
classifier = KNeighborsClassifier(n_neighbors=3)

# Latih model K-NN menggunakan data latih
classifier.fit(X_train_scaled, y_train)

# Evaluasi menggunakan validasi silang
scores = cross_val_score(classifier, X_test_scaled, y_test, cv=5)

# Tampilkan akurasi untuk setiap lipatan validasi silang
for fold, score in enumerate(scores, 1):
    print(f"Akurasi Fold-{fold}: {score}")

# Pencarian grid untuk menemukan hyperparameter terbaik
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(fitur, kelas)

# Evaluasi model menggunakan data uji
accuracy = classifier.score(X_test_scaled, y_test)
print("Akurasi: ", accuracy)

# Menampilkan parameter terbaik dan skor validasi silang terbaik
print("Parameter Terbaik:", grid_search.best_params_)
print("Akurasi Terbaik:", grid_search.best_score_)

# Path ke folder testing
testing_folder = "testing/"

# Inisialisasi list untuk menyimpan fitur dan kelas data tes
X_test_images = []
y_test_images = []

# Looping untuk setiap file dalam folder testing
for file_name in os.listdir(testing_folder):
    # Mengambil path file
    file_path = os.path.join(testing_folder, file_name)

    # Menambahkan fitur ke X_test_images
    X_test_images.append(file_path)

    # Menambahkan kelas ke y_test_images
    if "daunjambuair" in file_name:
        y_test_images.append(0)
    elif "daunjambubiji" in file_name:
        y_test_images.append(1)

# Konversi X_test_images dan y_test_images menjadi array numpy
X_test_images = np.array(X_test_images)
y_test_images = np.array(y_test_images)

# Skala fitur menggunakan StandardScaler
X_test_images_scaled = scaler.transform(X_test_images)

# Klasifikasi data tes menggunakan model yang sudah dilatih
predicted_labels = classifier.predict(X_test_images_scaled)

# Menampilkan hasil klasifikasi dan akurasi pada data tes
for i, label in enumerate(predicted_labels):
    if label == 0:
        print(f"Gambar {i+1}: daunjambuair")
    else:
        print(f"Gambar {i+1}: daunjambubiji")

accuracy_test_images = np.mean(predicted_labels == y_test_images)
print("Akurasi pada data tes (gambar):", accuracy_test_images)
