#!/usr/bin/env python

# document.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
import numpy as np

# reference: http://stackoverflow.com/questions/23506105/extracting-text-opencv#answer-23672571
image = os.path.join(os.getcwd(), "assets/img/japbend.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

output = cv_image.copy()

# Filtering
blur = cv2.GaussianBlur(cv_image, (5, 5), 0)
processed = cv2.bilateralFilter(blur, 15, 75, 75)

filter_kernel = np.ones((5, 5), np.float32) / 25
processed = cv2.filter2D(processed, -1, filter_kernel)

# thresholding
thresh = cv2.adaptiveThreshold(processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# dilate
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# get contours (_, contours, hierarchy)
im_contours, contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.imshow('Contours', im_contours)
cv2.waitKey(0)

# for each contour found, draw a rectangle around it on original image
for contour in contours:
  # get rectangle bounding contour
  [x, y, w, h] = cv2.boundingRect(contour)

  # # discard areas that are too large
  # if h > 300 and w > 300:
  #   print('large')
  #   continue

  # # discard areas that are too small
  # if h < 10 or w < 10:
  #   print('small')
  #   continue

  # draw rectangle around contour on original image
  cv2.rectangle(output, (x, y), (x + w, y + h), (65, 203, 62), 2)

cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


