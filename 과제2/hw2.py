# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 21:55:42 2021

@author: 이병화
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('book1.jpg', 0) # queryImage, 0-> gray
img2 = cv2.imread('book2.jpg', 0) # trainImage
cv2.imshow("Query image",img1)
cv2.imshow("Reference image",img2)

# create SIFT feature extractor object
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# brute force matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.2*n.distance:
        good.append([m])
        
img4=None
img5=None

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
img4 = cv2.drawKeypoints(img1, kp1,img4,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img5 = cv2.drawKeypoints(img1, kp1,img5)

cv2.imshow("11",img4)
cv2.imshow("111",img5)
cv2.imshow("matching",img3), plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()