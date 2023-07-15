import cv2
import numpy as np
import matplotlib.pyplot as plt

# STEP 1: Import another image of Lena
image = cv2.imread("lena.png")

# STEP 2: Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
