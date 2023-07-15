import cv2
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Import the image & Convert to grayscale
original_image = cv2.imread("lena.png")
lena_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# STEP 2: Add Gaussian noise to the image
mean = 0
sigma = 0.1
gaussian_noise = np.random.normal(mean, sigma, lena_image.shape)
noisy_image = lena_image + gaussian_noise

# STEP 3: Apply 3x3 and 7x7 Mean masks
mean_mask_3x3 = cv2.blur(noisy_image, (3, 3))
mean_mask_7x7 = cv2.blur(noisy_image, (7, 7))

# STEP 4: Add salt-and-pepper noise to the original image
salt_pepper_noise = np.random.choice([0, 255], size=lena_image.shape)
noisy_image_2 = np.copy(lena_image)
noisy_image_2[salt_pepper_noise == 0] = 0
noisy_image_2[salt_pepper_noise == 255] = 255

# STEP 5: Apply 3x3 and 7x7 Median masks to the seasoned image
median_mask_3x3 = cv2.medianBlur(noisy_image_2, 3)
median_mask_7x7 = cv2.medianBlur(noisy_image_2, 7)

# Display the original image, salt-and-pepper noisy image, the mean and the median filtered images
# plt.subplot(2, 3, 1)
# plt.imshow(lena_image, cmap='gray')
# plt.title("Original Image")
# plt.axis('off')
#
# plt.subplot(2, 3, 2)
# plt.imshow(mean_mask_3x3, cmap='gray')
# plt.title("3x3 Mean Mask")
# plt.axis('off')
#
# plt.subplot(2, 3, 3)
# plt.imshow(mean_mask_7x7, cmap='gray')
# plt.title("7x7 Mean Mask")
# plt.axis('off')
#
# plt.subplot(2, 3, 4)
# plt.imshow(noisy_image_2, cmap='gray')
# plt.title("Salt-and-Pepper Noisy")
# plt.axis('off')
#
# plt.subplot(2, 3, 5)
# plt.imshow(median_mask_3x3, cmap='gray')
# plt.title("3x3 Median Mask")
# plt.axis('off')
#
# plt.subplot(2, 3, 6)
# plt.imshow(median_mask_7x7, cmap='gray')
# plt.title("7x7 Median Mask")
# plt.axis('off')
#
# plt.tight_layout()
# plt.show()

# STEP 7: Apply Fourier Transformation (FT) to the original image
f = np.fft.fft2(lena_image)
f_shift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(f_shift))

# Display the magnitude and phase of the Fourier spectrum
plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.angle(f_shift), cmap='gray')
plt.title("Phase Spectrum")
plt.axis('off')

plt.tight_layout()
plt.show()

# STEP 8: Apply a low-pass Fourier mask to the seasoned image
rows, cols = lena_image.shape
crow, ccol = rows // 2, cols // 2
d = 50  # Radius of the circular mask
mask = np.zeros((rows, cols), np.uint8)
cv2.circle(mask, (ccol, crow), d, 1, -1)
f_shift_filtered = f_shift * mask
f_filtered = np.fft.ifftshift(f_shift_filtered)
seasoned_filtered = np.abs(np.fft.ifft2(f_filtered))

# STEP 9: Subtract the smoothed image from the original to obtain an unsharp mask
unsharp_mask = lena_image - seasoned_filtered

# STEP 10: Add the unsharp mask back to the original image for sharpening
sharpened_image = lena_image + unsharp_mask

# Display the original, unsharp mask, and sharpened images
plt.subplot(1, 3, 1)
plt.imshow(lena_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(unsharp_mask, cmap='gray')
plt.title("Unsharp Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sharpened_image, cmap='gray')
plt.title("Sharpened Image")
plt.axis('off')

plt.tight_layout()
plt.show()
