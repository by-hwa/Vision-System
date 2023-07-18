# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:07:16 2021

@author: 이병화
"""

import numpy as np
import cv2
import glob
import sys
import os
import cv2.aruco
import math
import argparse
import imutils

#hsv 이미지처리
def hsv(img):
    caphsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h,s,v= cv2.split(caphsv)

    lower_yel = (15,110,85)
    upper_yel = (30,255,255)
    

    yelimg = cv2.inRange(caphsv,lower_yel,upper_yel)

    lower_red = (160,100,85)
    upper_red = (190,255,255)

    redimg = cv2.inRange(caphsv,lower_red,upper_red)

    mask = yelimg + redimg

    hsvimg = cv2.bitwise_and(img,img,mask = mask)
    return hsvimg


#hough transfrom
def hough(img, rho, theta, threshold, min_line_len, max_line_gap,img1): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
   #drawlines(img1, lines)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img1, (x1, y1), (x2, y2), color=[0,0,255], thickness=2)

    return img1

#calibration
img = cv2.imread('./Inteld435i/00005_image.png')

mtx = np.array([[602.3005,0,314.6993],[0,610.4634,235.3530],[0,0,1]])
dist = np.array([[0.1194,-0.2331,0,0,0]])

h, w = img.shape[:2]

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

cap1 = cv2.undistort(img, mtx, dist, None, newcameramtx)

#cv2.imshow('origin',img)
cv2.imshow('undistort',cap1)

#ROI
cap2 = cap1[320:480, 0:640].copy()
cv2.imshow('ROI',cap2)

#Bird eye view
pts1 = np.float32([[200, 0],[470, 0],[60, 160], [640, 130]])
pts2 = np.float32([[60,0],[580,0],[60,160],[580,160]])

M = cv2.getPerspectiveTransform(pts1,pts2)

cap2 = cv2.warpPerspective(cap2,M,(640,160))

cv2.imshow("birdeye",cap2)

#HSV 이미지 처리

hsvi = hsv(cap2)

edges = cv2.Canny(hsvi,50,200)



cv2.imshow('edges',edges)

#hough transform

a = 0

lines = cv2.HoughLinesP(edges, 1, 1*np.pi/180, 40, np.array([]), 5, 20)
if lines is not None:
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(cap2, (x1, y1), (x2, y2), color=[0,0,255], thickness=2)
            b = math.atan2((y2-y1),(x2-x1))*(180/3.14)
            a=a+abs(b)
    c = a/len(lines)
    print("각도 :",c)     

cv2.imshow('lines',cap2)



#objectdetection

cap3 = cap1.copy()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap4 = cv2.resize(cap3, None, fx=0.4, fy=0.4)
height, width, channels = cap3.shape

blob = cv2.dnn.blobFromImage(cap4, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
        cv2.rectangle(cap3, (x, y), ((x + w), (y + h)), color, 2)
        print((602.325*1700)/h,"mm")
        
cv2.imshow("Image", cap3)

# marker detection

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


# load the input image from disk and resize it
print("[INFO] loading image...")
image = cap1
#image = imutils.resize(image, width=600)
# loop over the types of ArUco dictionaries

for (arucoName, arucoDict) in ARUCO_DICT.items():
	# load the ArUCo dictionary, grab the ArUCo parameters, and
	# attempt to detect the markers for the current dictionary
	arucoDict = cv2.aruco.Dictionary_get(arucoDict)
	arucoParams = cv2.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv2.aruco.detectMarkers(
		image, arucoDict, parameters=arucoParams)
	# if at least one ArUco marker was detected display the ArUco
	# name to our terminal
    
	if len(corners) > 0:
		print("[INFO] detected {} markers for '{}'".format(
			len(corners), arucoName))


cap6 = cap1.copy()

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
(corners, ids, rejected) = cv2.aruco.detectMarkers(cap6, arucoDict,
	parameters=arucoParams)
rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.02, mtx, dist)
arc = cv2.aruco.drawDetectedMarkers(cap6, corners, ids, (0,255,0))
if rvec is not None:
    for i in range(len(rvec)):
        arc =cv2.aruco.drawAxis(cap6,mtx, dist, rvec[i], tvec[i], 0.01)

#corner = corners.reshape((4, 2))
#(a1,a2,a3,a4)=corners

#im1 = np.float32(a1,a2,a3,a4)
#im2 = np.float32()

cv2.imshow('mark',arc)


'''
# verify *at least* one ArUco marker was detected
if len(corners) > 0:
	# flatten the ArUco IDs list
	ids = ids.flatten()
	# loop over the detected ArUCo corners
	for (markerCorner, markerID) in zip(corners, ids):
		# extract the marker corners (which are always returned in
		# top-left, top-right, bottom-right, and bottom-left order)
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		# convert each of the (x, y)-coordinate pairs to integers
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
        # draw the bounding box of the ArUCo detection
		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
		# compute and draw the center (x, y)-coordinates of the ArUco
		# marker
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
		# draw the ArUco marker ID on the image
		cv2.putText(image, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		print("[INFO] ArUco marker ID: {}".format(markerID))
		# show the output image
		cv2.imshow("Image1", image)
		cv2.waitKey(0)
'''


# 기능 통합

cap5 = cap1.copy()

hsv1 = hsv(cap5)

roi_h = cap5.shape[0]
roi_w = cap5.shape[1]

region = np.array([
        [[60, roi_h], [285, 240],[330, 230], [roi_w, 430]]
    ], dtype = np.int32)

mask = np.zeros_like(cap5)

cv2.fillPoly(mask, region, 255)

region = np.array([
        [[140, roi_h], [290, 230],[350, 230], [600, 480]]
    ], dtype = np.int32)
cv2.fillPoly(mask,region, 0)

cap5 = cv2.bitwise_and(hsv1, mask)

edgesa = cv2.Canny(cap5,100,200)

cap7 = cap1.copy()

lin = hough(edgesa, 1, 1*np.pi/180, 20,10,20,cap7)

edgesa = hough(edgesa, 1, 1*np.pi/180, 20,10,20,cap3)


al = cv2.aruco.drawDetectedMarkers(edgesa, corners, ids, (0,255,0))
if rvec is not None:
    for i in range(len(rvec)):
        al =cv2.aruco.drawAxis(edgesa,mtx, dist, rvec[i], tvec[i], 0.01)
        
arrow = cv2.imread('arrow.png')
arrow = cv2.resize(arrow, dsize =(0,0), fx=0.1,fy=0.1,interpolation=cv2.INTER_LINEAR)

ih, iw = arrow.shape[:2]
#rotation
if lines is not None:
    M = cv2.getRotationMatrix2D((iw/2,ih/2),(180-c),1)
    arrot = cv2.warpAffine(arrow,M,(iw,iw))

    mask = np.full_like(arrot,255)

    al = cv2.seamlessClone(arrot,al,mask,(70,380),cv2.NORMAL_CLONE)

cv2.imshow('d',cap2)

cap8 = cap2.copy()

hsv = cv2.cvtColor(cap8,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

cv2.imshow('dd',s)

vv = cv2.Canny(s,100,200)

cv2.imshow('vv',vv)

vc = vv[0:160, 100:550].copy()

cv2.imshow('vc',vc)

vc = np.sum(vc,0)

if sum(vc)>0:
    print('멈춰!')

cv2.imshow('line',lin)
cv2.imshow('all',al)

cv2.waitKey()
cv2.destroyAllWindows()