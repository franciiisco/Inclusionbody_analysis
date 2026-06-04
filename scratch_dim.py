import cv2
import os
img_path = "/home/francisco/Descargas/Imágenes209/MEI_R2_t14.tif"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is not None:
    print(f"Dimensions: {img.shape}")
else:
    print("Could not read image")
