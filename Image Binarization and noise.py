import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

image = cv2.imread('panther.jpeg', 0)

mu = int(input())
s = int(input())

# NOISE
noise = np.random.normal(mu, pow(s, 0.5), image.shape)
#  noisy image
nimage4 = image+noise
nimage4= np.clip(nimage4,0,255) #.astype('int32')


def num_of_pixels(image):
    pixel_values, pixel_count = np.unique(image, return_counts=True)
    Pixel_val = np.arange(0, 256)
    Pixel_count = np.zeros(256, dtype=int)
    Pixel_count[pixel_values.astype(int)] = pixel_count
    return Pixel_val, Pixel_count

a_, b_ = num_of_pixels(nimage4)
total_pixels = b_.sum()

l = []
max_between_class_variance = 0
max_index = None

for i in range(len(b_)):
    b = b_[:i]
    f = b_[i:]

    if i == 0 or i == 255:
        w_mean_b = 0 if i == 0 else np.sum(np.arange(i) * b) / np.sum(b)
        w_mean_f = 0 if i == 255 else np.sum(np.arange(i, len(b_)) * f) / np.sum(f)
    else:
        w_mean_b = np.sum(np.arange(i) * b) / np.sum(b)             ## weighted mean
        w_mean_f = np.sum(np.arange(i, len(b_)) * f) / np.sum(f)

    weight_b = np.sum(b) / total_pixels                             ## weights
    weight_f = np.sum(f) / total_pixels

    between_class_variance = weight_b * weight_f * (w_mean_b - w_mean_f)**2  ## between class variance

    if between_class_variance > max_between_class_variance:
        max_between_class_variance = between_class_variance
        max_index = i
    l.append(between_class_variance)

threshold_value = max_index

##binarization

binary_image = np.where(nimage4 > threshold_value, 255, 0).astype(np.uint8)
plt.figure(figsize = (18,9))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(nimage4, cmap='gray')
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title("Binarized Image")
plt.axis("off")
# cv2_imshow(image)
# cv2_imshow(noisyimage)
# cv2_imshow(binary_image)
print(threshold_value)
