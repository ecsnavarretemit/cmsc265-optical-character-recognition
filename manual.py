#!/usr/bin/env python

# manual.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np

# reference: http://stackoverflow.com/questions/23506105/extracting-text-opencv#answer-23672571
image = os.path.join(os.getcwd(), "assets/img/manual.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

output = cv_image.copy()

###############################
# Contours ::start
###############################

# Blurring (Plain)
processed = cv2.blur(cv_image, (5, 5))

# Blurring (Gaussian)
# processed = cv2.GaussianBlur(cv_image, (3, 3), 0)

# Filtering
# filter_kernel = np.ones((5, 5), np.float32) / 25
# processed = cv2.filter2D(cv_image, -1, filter_kernel)

# thresholding
_, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

transformed = cv2.dilate(thresh, kernel, iterations=13)
# transformed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

_, contours, _ = cv2.findContours(transformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# for each contour found, draw a rectangle around it on original image
for contour in contours:
  if cv2.contourArea(contour) <= 50:
    continue

  # get rectangle bounding contour
  [x, y, w, h] = cv2.boundingRect(contour)

  # discard areas that are too large
  if h > 150 or w > 150:
    continue

  # discard areas that are too small
  if h < 10 or w < 10 or h == 28:
    continue

  # draw rectangle around contour on original image
  cv2.rectangle(output, (x, y), (x + w, y + h), (65, 203, 62), 2)

###############################
# Contours ::end
###############################

cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


