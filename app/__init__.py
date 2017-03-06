# __init__.py
#
# Copyright(c) Exequiel Ceasar Navarrete <esnavarrete1@up.edu.ph>
# Licensed under MIT
# Version 1.0.0-alpha1

import cv2
import random

def recognize_characters(src, dst, **kwargs):
  shape_color = kwargs.get('shape_color', (65, 203, 62))
  shape_thickness = kwargs.get('shape_thickness', 2)
  random_shape_color = kwargs.get('random_shape_color', False)
  filter_fn = kwargs.get('filter_fn', None)

  _, contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  # filter the contours first before processing
  if filter_fn is not None and callable(filter_fn):
    contours = filter(filter_fn, contours)

  # set default color
  new_shape_color = shape_color

  counter = 0

  # for each contour found, draw a rectangle around it on original image
  for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # generate random rgb colors for the box
    if random_shape_color is True:
      r = random.randint(0, 255)
      g = random.randint(0, 255)
      b = random.randint(0, 255)

      new_shape_color = (r, g, b)

    # draw rectangle around contour on original image
    cv2.rectangle(dst, (x, y), (x + w, y + h), new_shape_color, shape_thickness)

    # increment the value so that we can count how many contours are detected
    counter += 1

  return counter


