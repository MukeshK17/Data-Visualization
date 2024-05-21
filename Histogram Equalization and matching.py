import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Load the grayscale image
imported_image = Image.open("grey_3.png")
image = ImageOps.grayscale(imported_image)
g_a_p = np.array(image)       # g_a_p is grayscale array of pixels.

def num_of_pixels(g_a_p):
    pixel_values, pixel_count = np.unique(g_a_p, return_counts=True)
    Pixel_val = np.arange(0, 256)
    Pixel_count = [0] * 256

    for i in range(len(pixel_values)):
        Pixel_count[int(pixel_values[i])] = pixel_count[i]

    return Pixel_val, Pixel_count

Pixel_val, no_of_pixels = num_of_pixels(g_a_p)

#original image
plt.axis("off")
plt.imshow(g_a_p, cmap='gray')
plt.title('Original Image')
plt.show()

#CDF
y = no_of_pixels / np.sum(no_of_pixels)
CDF = np.cumsum(y)
CDF_ = CDF * 255
CDF_ = CDF_.round()

# Mapping of original pixel values to their equalized values using CDF
equalized_image = CDF_[g_a_p]

## Unique intensities with their pixel values
equalized_pixel_val, equalized_no_of_pixels = num_of_pixels(equalized_image)

plt.tight_layout()
plt.show()
#equalized image
plt.axis("off")
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.show()
#histogram of the original image
plt.bar(Pixel_val, no_of_pixels)
plt.title('Original Histogram')
plt.show()
#histogram of the equalized image
plt.bar(equalized_pixel_val, equalized_no_of_pixels)
plt.title('Equalized Histogram')
plt.show()

## Image matching

def num_of_pixels(image):
    pixel_values, pixel_count = np.unique(image, return_counts=True)
    Pixel_val = np.arange(0, 256)
    Pixel_count = np.zeros(256, dtype=int)
    Pixel_count[pixel_values.astype(int)] = pixel_count
    return Pixel_val, Pixel_count

def mcdf(no_of_pixels):
    # Calculate CDF
    y = no_of_pixels / np.sum(no_of_pixels)
    CDF = np.cumsum(y)
    CDF_ = CDF * 255
    return CDF_

# Load images using cv2
g_a_p1 = cv2.imread("imageA.png", cv2.IMREAD_GRAYSCALE)
g_a_p2 = cv2.imread("imageB.png", cv2.IMREAD_GRAYSCALE)

Pixel_val1, no_of_pixels1 = num_of_pixels(g_a_p1)
Pixel_val2, no_of_pixels2 = num_of_pixels(g_a_p2)

# cdf
histogram1 = mcdf(no_of_pixels1)
histogram2 = mcdf(no_of_pixels2)

# Perform histogram matching
def find_nearest_match(source_cdf, target_cdf, pixel_value):
    min_difference = float('inf')
    matched_pixel = 0

    for i in range(256):
        difference = abs(source_cdf[pixel_value] - target_cdf[i])
        if difference < min_difference:
            min_difference = difference
            matched_pixel = i

    return matched_pixel

def histogram_match(source_cdf, target_cdf, image):
    matched_image = np.array([[find_nearest_match(source_cdf, target_cdf, pixel) for pixel in row] for row in image ])
    return matched_image

# Perform histogram matching
matched_image = histogram_match(histogram1,histogram2, g_a_p1)
from matplotlib import pyplot as plt

# Convert images to RGB before displaying
g_a_p1_rgb = cv2.cvtColor(g_a_p1, cv2.COLOR_BGR2RGB)
g_a_p2_rgb = cv2.cvtColor(g_a_p2, cv2.COLOR_BGR2RGB)
matched_image_rgb = cv2.cvtColor(matched_image.astype(np.uint8), cv2.COLOR_BGR2RGB)

# Display of results
plt.figure(figsize=(12, 4))

plt.subplot(1,3,1)
plt.imshow(g_a_p1_rgb)
plt.title('Source Image')
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(g_a_p2_rgb)
plt.title('Target Image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(matched_image_rgb)
plt.title('Matched Image')
plt.axis('off')

plt.show()

