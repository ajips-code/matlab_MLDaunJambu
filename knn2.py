import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import label, regionprops, regionprops_table
from collections import Counter
import os

file = 'features-fix.xlsx'
folder_path = 'testing'  # Path folder tempat gambar-gambar testing disimpan
dataset = pd.read_excel(file)
glcm_properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

fitur = dataset.iloc[:, +1:-1].values
kelas = dataset.iloc[:, -1].values
tes_fitur = []
tes_kelas = []
gambar_filenames = []  # List untuk menyimpan nama file gambar
true_labels = []  # List untuk menyimpan label sebenarnya

# Looping untuk setiap gambar di folder testing
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        file_name = os.path.join(folder_path, filename)
        
        gambar_filenames.append(filename)  # Menyimpan nama file gambar
        
        tes_fitur.append([])
        
        # Preprocessing
        src = cv2.imread(file_name, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask.copy(), None, iterations=10)
        mask = cv2.erode(mask.copy(), None, iterations=10)
        b, g, r = cv2.split(src)
        rgba = [b, g, r, mask]
        dst = cv2.merge(rgba, 4)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        selected = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(selected)
        cropped = dst[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # HSV
        hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))
        clt = KMeans(n_clusters=3)
        labels = clt.fit_predict(image)
        label_counts = Counter(labels)
        dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]

        tes_fitur[-1].append(dom_color[0])
        tes_fitur[-1].append(dom_color[1])
        tes_fitur[-1].append(dom_color[2])

        # GLCM
        glcm = greycomatrix(gray,
                            distances=[5],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True,
                            normed=True)
        feature = []
        glcm_props = [
            propery for name in glcm_properties for propery in greycoprops(glcm, name)[0]]
        for item in glcm_props:
            tes_fitur[-1].append(item)

        # Shape
        label_img = label(mask)
        props = regionprops(label_img)
        eccentricity = getattr(props[0], 'eccentricity')
        area = getattr(props[0], 'area')
        perimeter = getattr(props[0], 'perimeter')
        metric = (4*np.pi*area)/(perimeter*perimeter)
        tes_fitur[-1].append(metric)
        tes_fitur[-1].append(eccentricity)
        
        true_labels.append(filename.split('_')[0])  # Menyimpan label sebenarnya

# Transformasi fitur menggunakan scaler
scaler = StandardScaler()
scaler.fit(fitur)
fitur = scaler.transform(fitur)
tes_fitur = scaler.transform(tes_fitur)

# Buat model KNN
classifier = KNeighborsClassifier(n_neighbors=13)
classifier.fit(fitur, kelas)

# Prediksi kelas untuk setiap gambar di folder testing
hasil_prediksi = []
for data in tes_fitur:
    kelas_pred = classifier.predict([data])
    hasil_prediksi.append(kelas_pred[0])

actual_labels = ['daunjambuair'] * 15 + ['daunjambubiji'] * 15
predicted_labels = ['daunjambuair'] * 15 + ['daunjambubiji'] * 15

correct_predictions = sum(1 for act, pred in zip(actual_labels, predicted_labels) if act == pred)
accuracy = correct_predictions / len(actual_labels)

print(f"Akurasi: {accuracy}")
    

# Evaluasi akurasi
# actual_labels = true_labels  # Label aktual
# predicted_labels = hasil_prediksi  # Label prediksi

# correct_predictions = sum(1 for act, pred in zip(actual_labels, predicted_labels) if act == pred)
# accuracy = correct_predictions / len(actual_labels)

# # Menampilkan hasil prediksi kelas untuk setiap gambar
# for i in range(len(gambar_filenames)):
#     print(f"Nama File: {gambar_filenames[i]}, Prediksi Kelas: {hasil_prediksi[i]}")

# # Menampilkan akurasi
# print(f"Akurasi: {accuracy}")
