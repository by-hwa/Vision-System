# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 00:29:00 2021

@author: 이병화
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import math

img = cv2.imread('lane_image.jpg')

x_sobel_mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
y_sobel_mask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
xgra = np.array([[1,-1]])
ygra = np.array([[-1],[1]])

lx = cv2.filter2D(img, -1, x_sobel_mask)
ly = cv2.filter2D(img, -1, y_sobel_mask)
sdx = cv2.filter2D(img, -1, xgra)
sdy = cv2.filter2D(img, -1, ygra)
dx = cv2.filter2D(img, cv2.CV_64F, xgra)
dy = cv2.filter2D(img, cv2.CV_64F, ygra)


fx = pow(dx,2)
fy = pow(dy,2)

imgmag = np.sqrt(fx+fy)
imgtan = np.rad2deg(np.arctan(sdy/sdx))

imgmag = imgmag.astype('uint8')
imgtan = imgtan.astype('uint8')

# x-detection 표시
plt.subplot(2,2,1)
plt.imshow(lx,cmap='jet')
plt.title('x-detection')
plt.axis("off")

# y-detection 표시
plt.subplot(2,2,3)
plt.imshow(ly)
plt.title('y-detection')
plt.axis("off")

# Magnitue 표시
plt.subplot(2,2,2)
plt.imshow(imgmag)
plt.title('Magnitude')
plt.axis("off")

#Angle 표시
plt.subplot(2,2,4)
plt.imshow(imgtan, cmap='jet')
plt.title('Angle')
plt.axis("off")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()