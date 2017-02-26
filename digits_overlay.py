#!/usr/bin/env python

# digits_overlay.py
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

  # remove any small contours present
  if h < 40 or w < 40:
    return False

  return True

image = os.path.join(os.getcwd(), "assets/img/digits-overlay.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

output = cv_image.copy()

# threshold
_, thresh = cv2.threshold(cv_image, 150, 255, cv2.THRESH_BINARY_INV)

# dilate
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=5)

# box out all possible characters present in the image
recognize_characters(dilated, output, filter_fn=filter_contours)

cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


