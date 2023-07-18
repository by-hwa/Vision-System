# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 01:01:52 2021

@author: 이병화
"""

# Opencv Example

import cv2
import numpy as np

# image load
I = cv2.imread('lane.jpg')
#cv2.imshow('lane image', I)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

print(I.shape)

Gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY) # R, G, B 색깔에 대한 채널

# Binarization using cv2
Binary = (Gray > 150 ) * 255
BinaryI = cv2.convertScaleAbs(Binary) # int 32 -> uint8

# array, for
height,width = BinaryI.shape
Output = np.zeros((height,width), np.uint8) # numpy array
for i in range(0, height, 1):
    for j in range(0, width, 1):
        if Gray[i, j] > 150:
            Output[i, j] = 255
        else:
            Output[i, j] = 0

cv2.imshow('lane image', Output)
cv2.waitKey(0)
cv2.destroyAllWindows()