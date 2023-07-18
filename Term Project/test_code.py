# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:49:33 2021

@author: 이병화
"""

import cv2

img = cv2.imread('img.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()