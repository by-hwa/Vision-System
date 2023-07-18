# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 01:51:20 2021

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
dx = cv2.filter2D(img, -1, xgra)
dy = cv2.filter2D(img, -1, ygra)


fx = pow(lx,2) 
fy = pow(ly,2)

imgmag = np.sqrt(fx+fy)
imgtan = np.arctan(ly/lx)