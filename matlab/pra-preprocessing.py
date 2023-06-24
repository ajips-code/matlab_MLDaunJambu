import cv2
import numpy as np
import xlsxwriter
import warnings
import pandas as pd
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
from cryptography.utils import CryptographyDeprecationWarning
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
from scapy.all import *

# Fungsi untuk menghilangkan noise dengan filter median atau filter gaussian
def denoise_image(image, median_ksize, gaussian_ksize):
    denoised = cv2.medianBlur(image, median_ksize)
    denoised = cv2.GaussianBlur(image, (gaussian_ksize, gaussian_ksize), 0)
    return denoised

# Fungsi untuk meningkatkan kontras dan kecerahan gambar
def enhance_image(image, alpha, beta):
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return enhanced

# Fungsi untuk segmentasi daun dari latar belakang
def segment_leaf(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    mask = cv2.drawContours(np.zeros_like(mask), [largest_contour], 0, 255, thickness=cv2.FILLED)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

# Membuat file Excel untuk menyimpan fitur
workbook = xlsxwriter.Workbook('features-fix3.xlsx', {'nan_inf_to_errors': True})
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
    jum_per_data = 164 if i == 'daunjambuair' else 106  # Mengatur jumlah gambar yang diinginkan
    for j in range(1, jum_per_data + 1):
        kolom = 0
        file_name = "dataset/" + i + str(j) + ".jpg"
        print(file_name)
        worksheet.write(baris, kolom, file_name)
        kolom += 1
        # Preprocessing
        src = cv2.imread(file_name, 1)

        # Menghilangkan noise dengan filter median atau filter gaussian
        denoised = denoise_image(src, median_ksize=5, gaussian_ksize=5)

        # Meningkatkan kontras dan kecerahan gambar
        enhanced = enhance_image(denoised, alpha=1.5, beta=30)

        # Segmentasi daun dari latar belakang
        segmented = segment_leaf(enhanced, threshold=0)

        # Ekstraksi fitur menggunakan metode yang telah ditentukan
        # HSV
        hsv_image = cv2.cvtColor(segmented, cv2.COLOR_BGR2HSV)
        image = hsv_image.reshape((hsv_image.shape[0] * hsv_image.shape[1], 3))

        # KMeans clustering
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

        dom_color_hsv = np.full(segmented.shape, dom_color, dtype='uint8')
        dom_color_bgr = cv2.cvtColor(dom_color_hsv, cv2.COLOR_HSV2BGR)
        output_image = np.hstack((segmented, dom_color_bgr))

        # GLCM
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray,
                            distances=[5],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256,
                            symmetric=True,
                            normed=True)
        glcm_props = [property for name in glcm_properties for property in graycoprops(glcm, name)[0]]
        warnings.filterwarnings('ignore', category=UserWarning, module='skimage.feature')
        for item in glcm_props:
            worksheet.write(baris, kolom, item)
            kolom += 1

        # Shape
        mask = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
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

        cv2.imwrite("preprocessed/" + i + str(j) + ".png", segmented)
        print("prosessing selesai")

workbook.close()
