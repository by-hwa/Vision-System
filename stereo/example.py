# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 13:12:26 2021

@author: 이병화
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('./selected/left/tsukuba_daylight_L_00100.png',0)
imgR = cv2.imread('./selected/right/tsukuba_daylight_R_00100.png',0)

cv2.imshow('Left Image', imgL)
cv2.imshow('Right Image', imgR)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.colorbar()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()