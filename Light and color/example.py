# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 13:25:00 2021

@author: 이병화
"""

import cv2
import matplotlib.pyplot as plt

imgRGB = cv2.imread('cube.png')
cv2.imshow('cube',imgRGB)

b,g,r=cv2.split(imgRGB)

cv2.imshow('b',b)
cv2.imshow('g',g)
cv2.imshow('r',r)

imgHSV = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(imgHSV)

cv2.imshow('h',h)
cv2.imshow('s',s)
cv2.imshow('v',v)

cv2.waitKey(0)
cv2.destroyAllWindows()