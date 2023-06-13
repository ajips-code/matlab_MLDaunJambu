import cv2
import numpy as np
import xlsxwriter
import warnings
import pandas as pd
from collections import Counter
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Membuat file Excel untuk menyimpan fitur
workbook = xlsxwriter.Workbook('features-fix2.xlsx')
worksheet = workbook.add_worksheet()

# Jenis dataset yang akan diproses
jenis = ['daunjambuair', 'daunjambubiji']

# Properti HSV yang akan diekstrak
hsv_properties = ['hue', 'saturation', 'value']

# Properti GLCM yang akan diekstrak
glcm_properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
angles = ['0', '45', '90', '135']

# Properti bentuk yang akan diekstrak
shape_properties = ['metric', 'eccentricity']

# Menulis judul kolom di file Excel
worksheet.write(0, 0, 'File')
kolom = 1

# ---------------------------
# Menulis header di file Excel
for i in hsv_properties:
    worksheet.write(0, kolom, i)
    kolom += 1
for i in glcm_properties:
    for j in angles:
        worksheet.write(0, kolom, i + " " + j)
        kolom += 1
for i in shape_properties:
    worksheet.write(0, kolom, i)
    kolom += 1
worksheet.write(0, kolom, 'Class')
kolom += 1
baris = 1

# Looping untuk setiap dataset
for i in jenis:
    jum_per_data = 75 if i == 'daunjambuair' else 45  # Mengatur jumlah gambar yang diinginkan
    for j in range(1, jum_per_data + 1):
        kolom = 0
        file_name = "dataset/" + i + str(j) + ".jpg"
        print(file_name)
        worksheet.write(baris, kolom, file_name)
        kolom += 1
        # Preprocessing
        src = cv2.imread(file_name, 1)
        tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.dilate(mask.copy(), None, iterations=10)
        mask = cv2.erode(mask.copy(), None, iterations=10)
        b, g, r = cv2.split(src)
        rgba = [b, g, r, mask]
        dst = cv2.merge(rgba, 4)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        selected = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(selected)
        cropped = dst[y:y+h, x:x+w, :3]  # Mengekstrak hanya saluran BGR
        mask = mask[y:y+h, x:x+w]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # HSV
        hsv_image = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))

        # clt = KMeans(n_clusters=3)
        clt = KMeans(n_clusters=3, n_init=10)
        labels = clt.fit_predict(image)
        label_counts = Counter(labels)
        dom_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
        worksheet.write(baris, kolom, dom_color[0])
        kolom += 1
        worksheet.write(baris, kolom, dom_color[1])
        kolom += 1
        worksheet.write(baris, kolom, dom_color[2])
        kolom += 1

        dom_color_hsv = np.full(cropped.shape, dom_color, dtype='uint8')
        dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
        output_image = np.hstack((cropped, dom_color_bgr))

        # GLCM
        glcm = greycomatrix(gray,
                            distances=[5],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True,
                            normed=True)
        glcm_props = [
            property for name in glcm_properties for property in greycoprops(glcm, name)[0]]
        # Add this line to ignore the warning
        warnings.filterwarnings('ignore', category=UserWarning, module='skimage.feature')
        for item in glcm_props:
            worksheet.write(baris, kolom, item)
            kolom += 1

        # Shape
        label_img = label(mask)
        props = regionprops(label_img)
        eccentricity = getattr(props[0], 'eccentricity')
        area = getattr(props[0], 'area')
        perimeter = getattr(props[0], 'perimeter')
        metric = (4 * np.pi * area) / (perimeter * perimeter)
        worksheet.write(baris, kolom, metric)
        kolom += 1
        worksheet.write(baris, kolom, eccentricity)
        kolom += 1

        worksheet.write(baris, kolom, i)
        kolom += 1
        baris += 1

# Menutup file Excel
workbook.close()

# --------------------------------------
# Membaca file Excel dan melatih model K-NN

# Membaca dataset dari file Excel
dataset = pd.read_excel('features-fix2.xlsx')

# Mengubah kolom kelas menjadi numerik (0 untuk 'daunjambuair' dan 1 untuk 'daunjambubiji')
dataset['Class'] = dataset['Class'].map({'daunjambuair': 0, 'daunjambubiji': 1})

# Memisahkan fitur dan kelas
fitur = dataset.iloc[:, 1:-1]
kelas = dataset['Class']

# Skala fitur menggunakan StandardScaler
scaler = StandardScaler()
fitur = scaler.fit_transform(fitur)

# Buat model K-NN
classifier = KNeighborsClassifier(n_neighbors=3)

# Evaluasi menggunakan validasi silang
scores = cross_val_score(classifier, fitur, kelas, cv=5)

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

# Menampilkan parameter terbaik dan skor validasi silang terbaik
print("Parameter Terbaik:", grid_search.best_params_)
print("Akurasi Terbaik:", grid_search.best_score_)
