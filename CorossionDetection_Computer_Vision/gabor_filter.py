#!/usr/bin/env python
##############################################
# Gabor filter
##############################################
"""
For image processing and computer vision, Gabor filters are generally
used in texture analysis, edge detection, feature extraction, etc.

In image processing, a Gabor filter, named after Dennis Gabor, is a linear filter used for texture analysis, which essentially means that it analyzes whether there is any specific frequency content in the image in specific directions in a localized region around the point or region of analysis.

ksize - Size of the filter returned.
sigma - Standard deviation of the gaussian envelope.
theta - Orientation of the normal to the parallel stripes of a Gabor function.
lambda - Wavelength of the sinusoidal factor.
gamma - Spatial aspect ratio.
psi - Phase offset.
ktype - Type of filter coefficients. It can be CV_32F or CV_64F.
indicates the type and range of values that each pixel in the Gabor kernel can hold.
Basically float32 or float64

"""
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

ksize = 5  # Use size that makes sense to the image and feature size. Large may not be good.
# On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 3  # Large sigma on small features will fully miss the features.
theta = 1 * np.pi / 4  # /4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1 * np.pi / 4  # 1/4 works best for angled.
gamma = 0.4  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
# Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0  # Phase offset. I leave it to 0.

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

plt.imshow(kernel)

img = cv2.imread('Hydrangeas.jpg')
# img = cv2.imread('BSE_Image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

kernel_resized = cv2.resize(kernel, (400, 400))  # Resize image
cv2.imshow('Kernel', kernel_resized)
cv2.imshow('Original Img.', img)
cv2.imshow('Filtered', fimg)
cv2.waitKey(10000)
cv2.destroyAllWindows()