# -*- coding: utf-8 -*-
"""
Created on Tue May 11 02:17:47 2021

@author: 이병화
"""

import sys
import numpy as np
import cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cv2.waitKey(33) < 0:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)
    
capture.release()
cv2.destoryAllWindows()