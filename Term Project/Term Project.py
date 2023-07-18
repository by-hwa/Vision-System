# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:22:21 2021

@author: 이병화
"""

import numpy as np
import cv2
import glob
import sys
import os
import cv2.aruco


'''
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob('./chess/*.jpg')
#images = glob.glob('./calibration/*.png')


for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
    
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        
        img = cv2.drawChessboardCorners(img, (7,6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
        
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

img = cv2.imread('./chess/KakaoTalk_20210512_224651717_15.jpg')
#img = cv2.imread('./calibration/00010.png')
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png', dst)

tot_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error
print("total error: ", tot_error/len(objpoints))
'''

# linedetection and following

# edge검출 ROI처리
# 카메라 사용
#cap1 = cv2.VideoCapture(0)

#if not cap1.isOpened():
    #print('Camera open failed!')
    #sys.exit()

#mtx = np.load('mtx.npy')
#dist = np.load('dist.npy')
    
#img = cv2.imread('./Inteld435i/00006_image.png')
#h,  w = img.shape[:2]
#newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

#x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
#cap1 = cv2.imwrite('calibresult.png', dst)


cap1 = cv2.imread('./Inteld435i/00070_image.png')
#cap1 = cv2.imread('./image/image_018.jpg')

caphsv = cv2.cvtColor(cap1, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(caphsv)

edges = cv2.Canny(s,100,200)

roi_h = edges.shape[0]
roi_w = edges.shape[1]

region = np.array([
        [[60, roi_h], [285, 240],[330, 230], [roi_w, 430]]
    ], dtype = np.int32)

mask = np.zeros_like(s)
cv2.fillPoly(mask, region, 255)
region = np.array([
        [[140, roi_h], [290, 230],[350, 230], [600, 480]]
    ], dtype = np.int32)
cv2.fillPoly(mask,region, 0)

roimg = cv2.bitwise_and(edges, mask)

cv2.imshow('Canny',edges)
cv2.imshow('ROI',roimg)

#hought transform 차선검출
lines = cv2.HoughLines(roimg, 1, np.pi/180, 180)
#lines = cv2.HoughLinesP(roimg, 1,np.pi/180,50,100,10)
cap2 = cap1

if lines is not None:
    for line in lines :
        #x1,y1,x2,y2 = line[0]
        r, theta = line[0]
        tx, ty = np.cos(theta), np.sin(theta)
        x0, y0 = tx*r, ty*r
        x1, y1 = int(x0 + 1000*(-ty)), int(y0 + 1000 * tx)
        x2, y2 = int(x0 - 1000*(-ty)), int(y0 - 1000 * tx)
        print(theta*(180/3.14))
        cap2 = cv2.line(cap2, (x1,y1), (x2,y2), (0,0,255), 1)
    
    
#cv2.circle(cap1,(310,210),5,(0,0,255),-1)
cv2.imshow('roimg',cap2)


#object detection and localization

#hog = cv2.HOGDescriptor((48,96),(16,16),(8,8),(8,8),9)


#hog=cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#detected, _ = hog.detectMultiScale(cap1)

#for (x,y,w,h) in detected: 
#    cv2.rectangle(cap1, (x,y),(x+w,y+h), (50,200,50),3)
#cv2.imshow('HOG', cap2)



net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap3 = cv2.resize(cap1, None, fx=0.4, fy=0.4)
height, width, channels = cap3.shape

blob = cv2.dnn.blobFromImage(cap3, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        #color = colors[i]
        color = colors[10]
        x = round(x*2.5)
        y = round(y*2.5)
        w = round(w*2.5)
        h = round(h*2.5)
        cv2.rectangle(cap2, (x, y), ((x + w), (y + h)), color, 2)
        
        
cv2.imshow("Image", cap2)
# Markerdetection

#독자기능

pts1 = np.float32([[285, 240],[380, 260],[60, 480], [640, 450]])
pts2 = np.float32([[160,0],[480,0],[160,480],[480,480]])

M = cv2.getPerspectiveTransform(pts1,pts2)

dst = cv2.warpPerspective(cap1,M,(640,480))

cv2.imshow("birdeye",dst)

cv2.waitKey()
cv2.destroyAllWindows()
