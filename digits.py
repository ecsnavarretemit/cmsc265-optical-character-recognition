#!/usr/bin/env python

# digits.py
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
  if h > 300 and w > 300:
    return False

  # remove any small contours present
  if h < 40 or w < 40:
    return False

  return True

image = os.path.join(os.getcwd(), "assets/img/digits.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

output = cv_image.copy()

# threshold
blur = cv2.GaussianBlur(cv_image, (5, 5), 0)
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# dilate
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=13)

# box out all possible characters present in the image
recognize_characters(dilated, output, filter_fn=filter_contours)

cv2.imshow('Output', cv2.pyrDown(cv2.pyrDown(output)))
cv2.waitKey(0)
cv2.destroyAllWindows()


