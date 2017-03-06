#!/usr/bin/env python

# manual.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
from app import recognize_characters

# function to filter out contours on an image
def filter_contours(contour):
  # get rectangle bounding contour
  [_, _, w, h] = cv2.boundingRect(contour)

  # remove any large contours present
  if h > 150 or w > 150:
    return False

  return True

# read the image
image = os.path.join(os.getcwd(), "assets/img/manual.jpg")
cv_image = cv2.imread(image)

# convert to grayscale
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# apply some smoothing and thresholding to convert to binary image (inverted)
processed = cv2.blur(gray_image, (5, 5))
_, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# apply some morphological operation (dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
transformed = cv2.dilate(thresh, kernel, iterations=13)

# box out all possible characters present in the image
recognize_characters(transformed, cv_image, filter_fn=filter_contours)

cv2.imshow('Output', cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


