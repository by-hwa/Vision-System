# -*- coding: utf-8 -*-
"""
Created on Fri May 28 16:05:58 2021

@author: 이병화
"""

import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

img = cv2.imread('Lena.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1,4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0), 2)

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()