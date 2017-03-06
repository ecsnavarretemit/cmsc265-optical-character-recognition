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

# read the image
image = os.path.join(os.getcwd(), "assets/img/digits-overlay.jpg")
cv_image = cv2.imread(image)

# convert to grayscale
gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

# apply some thresholding to convert to binary image (inverted)
_, thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

# apply some morphological operation (dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=5)

# box out all possible characters present in the image
num_characters = recognize_characters(dilated, cv_image, random_shape_color=True, filter_fn=filter_contours)

print(f"Number of Boxes/Connected Components: {num_characters}")

cv2.imshow('Output', cv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


