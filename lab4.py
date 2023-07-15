import cv2
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Import the image
image = cv2.imread("lena.png")

# STEP 2: Add Gaussian noise to the image
mean = 0
sigma = 0.1
noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# STEP 3 & 4
b, g, r = cv2.split(image)
equalized_b = cv2.equalizeHist(b)
equalized_g = cv2.equalizeHist(g)
equalized_r = cv2.equalizeHist(r)

# Display the enhanced RGB components and histograms
# plt.subplot(2, 3, 1)
# plt.imshow(cv2.cvtColor(cv2.merge([equalized_b, g, r]), cv2.COLOR_BGR2RGB))
# plt.title("Enhanced Blue Component")
# plt.axis('off')
#
# plt.subplot(2, 3, 2)
# plt.imshow(cv2.cvtColor(cv2.merge([b, equalized_g, r]), cv2.COLOR_BGR2RGB))
# plt.title("Enhanced Green Component")
# plt.axis('off')
#
# plt.subplot(2, 3, 3)
# plt.imshow(cv2.cvtColor(cv2.merge([b, g, equalized_r]), cv2.COLOR_BGR2RGB))
# plt.title("Enhanced Red Component")
# plt.axis('off')
#
#
# plt.subplot(2, 3, 4)
# plt.hist(equalized_b.flatten(), 256, [0, 256], color='b')
# plt.title("Equalized Histogram - Blue")
# plt.xlabel("Intensity")
# plt.ylabel("Frequency")
#
# plt.subplot(2, 3, 5)
# plt.hist(equalized_g.flatten(), 256, [0, 256], color='g')
# plt.title("Equalized Histogram - Green")
# plt.xlabel("Intensity")
# plt.ylabel("Frequency")
#
# plt.subplot(2, 3, 6)
# plt.hist(equalized_r.flatten(), 256, [0, 256], color='r')
# plt.title("Equalized Histogram - Red")
# plt.xlabel("Intensity")
# plt.ylabel("Frequency")
#
# plt.tight_layout()
# plt.show()

# STEP 5: Merge the enhanced RGB components and compare with the noisy image
enhanced_rgb = cv2.merge([equalized_b, equalized_g, equalized_r])
comparison_rgb = np.concatenate((enhanced_rgb, noisy_image), axis=1)
#
# plt.imshow(cv2.cvtColor(comparison_rgb, cv2.COLOR_BGR2RGB))
# plt.title("Enhanced Image (RGB) vs Noisy Image")
# plt.axis('off')
# plt.show()

# STEP 6: Split the image into cylindrical color model (HSI) components and equalize the intensity
hsi_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsi_image[:, :, 2] = cv2.equalizeHist(hsi_image[:, :, 2])

# Split the equalized HSI image into components
h, s, i = cv2.split(cv2.cvtColor(hsi_image, cv2.COLOR_HSV2BGR))
hist_i = cv2.equalizeHist(i)

# Display the enhanced HSI intensity component and histogram
# plt.subplot(1, 3, 1)
# plt.imshow(i, cmap='gray')
# plt.title("Enhanced Intensity Component (HSI)")
# plt.axis('off')
#
# plt.subplot(1, 3, 2)
# plt.hist(hist_i.flatten(), 256, [0, 256], color='b')
# plt.title("Equalized Histogram - Intensity (HSI)")
# plt.xlabel("Intensity")
# plt.ylabel("Frequency")
#
# plt.tight_layout()
# plt.show()

# STEP 7: Compare the enhanced RGB image with the enhanced HSI image
comparison_hsi = np.concatenate((enhanced_rgb, cv2.cvtColor(hsi_image, cv2.COLOR_BGR2RGB)), axis=1)
#
# plt.imshow(cv2.cvtColor(comparison_hsi, cv2.COLOR_BGR2RGB))
# plt.title("Enhanced Image (RGB) vs Enhanced Image (HSI)")
# plt.axis('off')
# plt.show()

# STEP 8: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# STEP 9: Pseudo-color the grayscale image using a lookup table
cmap = plt.get_cmap('jet')
pseudo_colored_image = cmap(gray_image)

# Display the pseudo-colored image
# plt.imshow(pseudo_colored_image)
# plt.title("Pseudo-colored Image")
# plt.axis('off')
# plt.show()
# Define the number of intensity slices
num_slices = 6

# Create a figure with subplots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

# Loop over the intensity slices
for i in range(num_slices):
    # Calculate the lower and upper thresholds for the slice
    lower_thresh = i * 255 // num_slices
    upper_thresh = (i + 1) * 255 // num_slices

    # Create a mask based on the threshold range
    mask = (lower_thresh <= gray_image) & (gray_image < upper_thresh)

    # Create a pseudo-colored image using the Jet color scheme
    pseudo_colored_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    pseudo_colored_image = cv2.bitwise_and(pseudo_colored_image, pseudo_colored_image, mask=mask.astype(np.uint8))

    # Plot the intensity-sliced pseudo-colored image
    row = i // 3
    col = i % 3
    axs[row, col].imshow(cv2.cvtColor(pseudo_colored_image, cv2.COLOR_BGR2RGB))
    axs[row, col].set_title(f"Slice {i+1}")
    axs[row, col].axis('off')

plt.tight_layout()
plt.show()
