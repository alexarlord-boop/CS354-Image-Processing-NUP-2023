import cv2  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# STEP 1: Import the image
lena_image = cv2.imread("lena.png")

# STEP 2: Convert to grayscale
gray_image = cv2.cvtColor(lena_image, cv2.COLOR_BGR2GRAY)

# STEP 3: Extract histogram using PDF and CDF
hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])  # getting histogram out of 1D image array
pdf = hist / np.sum(hist)  # getting probability density function values
cdf = np.cumsum(pdf)  # getting cumulative density function values


# STEP 4: Perform Histogram Equalization (HE)


equalized_image = np.interp(gray_image.flatten(), bins[:-1], cdf * 255)  # new intensity values
equalized_image = equalized_image.reshape(gray_image.shape).astype(
    np.uint8)  # appropriate range of 0-255 for image display

# display the results
# original greyscale image
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# original greyscale image histogram
plt.subplot(2, 2, 2)
plt.hist(gray_image.flatten(), 256, [0, 256], color='r')
plt.title("Original Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# equalized image
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title("Equalized Image")
plt.axis('off')

# equalized image histogram
plt.subplot(2, 2, 4)
plt.hist(equalized_image.flatten(), 256, [0, 256], color='r')
plt.title("Equalized Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
