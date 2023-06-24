import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops, regionprops_table
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV

# Membaca dataset dari file Excel
file = 'features-fix3.xlsx'
dataset = pd.read_excel(file, header=None)

fitur = dataset.iloc[:, :-2].values
kelas = dataset.iloc[:, -2].values

# Data testing
testing_folder = "testing/"
tes_fitur = []
tes_kelas = []

# Proses ekstraksi fitur pada data testing
for i in range(1, 16):
    # Mengambil path file
    file_path = testing_folder + f"daunjambuair{i}.jpg"

    # Membaca citra
    src = cv2.imread(file_path, 1)
    
    # Preprocessing
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask.copy(), None, iterations=10)
    mask = cv2.erode(mask.copy(), None, iterations=10)
    cropped = cv2.bitwise_and(src, src, mask=mask)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # GLCM
    glcm = graycomatrix(gray, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    glcm_props = [property for name in ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy'] for property in graycoprops(glcm, name)[0]]
    
    # Shape
    label_img = label(mask)
    props = regionprops(label_img)
    eccentricity = getattr(props[0], 'eccentricity')
    area = getattr(props[0], 'area')
    perimeter = getattr(props[0], 'perimeter')
    metric = (4 * np.pi * area) / (perimeter * perimeter)
    
    # Menyimpan fitur dan kelas
    tes_fitur.append(glcm_props + [metric, eccentricity])
    tes_kelas.append('daunjambuair')
    
for i in range(1, 16):
   # Mengambil path file
    file_path = testing_folder + f"daunjambubiji{i}.jpg"

    # Membaca citra
    src = cv2.imread(file_path, 1)
    
    # Preprocessing
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.dilate(mask.copy(), None, iterations=10)
    mask = cv2.erode(mask.copy(), None, iterations=10)
    cropped = cv2.bitwise_and(src, src, mask=mask)
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # GLCM
    glcm = graycomatrix(gray, distances=[5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    glcm_props = [property for name in ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy'] for property in graycoprops(glcm, name)[0]]
    
    # Shape
    label_img = label(mask)
    props = regionprops(label_img)
    eccentricity = getattr(props[0], 'eccentricity')
    area = getattr(props[0], 'area')
    perimeter = getattr(props[0], 'perimeter')
    metric = (4 * np.pi * area) / (perimeter * perimeter)
    
    # Menyimpan fitur dan kelas
    tes_fitur.append(glcm_props + [metric, eccentricity])
    tes_kelas.append('daunjambubiji')

# Hapus kolom string (path file gambar)
# Hapus kolom string yang tidak dibutuhkan
fitur_numeric = fitur[:, :-2]  # Hapus 2 kolom terakhir yang berisi path file dan kelas
tes_fitur_numeric = np.array(tes_fitur)[:, :-2]  # Hapus 2 kolom terakhir yang berisi path file dan kelas

# Skala fitur menggunakan StandardScaler
scaler = StandardScaler()
fitur_scaled = scaler.fit_transform(fitur_numeric)
tes_fitur_scaled = scaler.transform(tes_fitur_numeric)

# Buat model K-NN
classifier = KNeighborsClassifier(n_neighbors=3)

# Latih model K-NN menggunakan data latih
classifier.fit(fitur_scaled, kelas)

# Evaluasi model menggunakan data testing
kelas_pred = classifier.predict(tes_fitur_scaled)
accuracy = accuracy_score(tes_kelas, kelas_pred)
print("Accuracy:", accuracy)

# # Evaluasi menggunakan validasi silang
# scores = cross_val_score(classifier, fitur_scaled, kelas, cv=5)

# # Tampilkan akurasi untuk setiap lipatan validasi silang
# for fold, score in enumerate(scores, 1):
#     print(f"Akurasi Fold-{fold}: {score}")

# # Pencarian grid untuk menemukan hyperparameter terbaik
# param_grid = {
#     'n_neighbors': [3, 5, 7, 9, 11],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean', 'manhattan']
# }

# grid_search = GridSearchCV(classifier, param_grid, cv=5)
# grid_search.fit(fitur_scaled, kelas)

# # Evaluasi model menggunakan data tes
# kelas_pred = classifier.predict(tes_fitur)
# print("Prediksi Kelas:", kelas_pred)

# # Menampilkan hasil klasifikasi dan akurasi pada data tes
# for i, label in enumerate(kelas_pred):
#     if label == 'daunjambuair':
#         print(f"Gambar {i+1}: daunjambuair")
#     else:
#         print(f"Gambar {i+1}: daunjambubiji")

# accuracy_test = np.mean(kelas_pred == tes_kelas)
# print("Akurasi pada data tes:", accuracy_test)

# # Menampilkan parameter terbaik dan skor validasi silang terbaik
# print("Parameter Terbaik:", grid_search.best_params_)
# print("Akurasi Terbaik:", grid_search.best_score_)
