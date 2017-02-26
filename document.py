#!/usr/bin/env python

# document.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import os
import cv2
from app import recognize_characters

image = os.path.join(os.getcwd(), "assets/img/document.jpg")
cv_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

output = cv_image.copy()

# Blurring (Plain)
processed = cv2.blur(cv_image, (3, 3))

# thresholding
_, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# dilate
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
dilated = cv2.dilate(thresh, kernel, iterations=1)

# box out all possible characters present in the image
recognize_characters(dilated, output)

cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()


