# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:32:15 2020

@author: Alina Shcherbinina
"""

import numpy as np
import cv2
import time

cam=cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mask", cv2.WINDOW_KEEPRATIO)

colorLower=np.array([0,100,100],dtype="uint8")
colorUpper=np.array([255,255,255],dtype="uint8")

prev_time = time.time()
curr_time = time.time()

prev_x = 0
prev_y = 0 
curr_x = 0
curr_y = 0
radii = 1

while cam.isOpened():
    ret, frame = cam.read()
    curr_time = time.time()
    blurred=cv2.GaussianBlur(frame,(11,11),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv, colorLower, colorUpper)
    mask=cv2.erode(mask, None, iterations=2)
    mask=cv2.dilate(mask, None, iterations=2)
    
    # cv2.imshow('mask', mask)

    cnts, _ = cv2.findContours(
        mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        (curr_x, curr_y), radii = cv2.minEnclosingCircle(c)
        if radii > 20:
            cv2.circle(frame, (int(curr_x), int(curr_y)),
                       int(radii), (0, 255, 255), 2)
            cv2.circle(frame, (int(curr_x), int(curr_y)), 5, (0,0,255), -1)
            
    time_diff = curr_time - prev_time
    prev_time = curr_time
    
    dist = ((prev_x - curr_x) ** 2 + (prev_y - curr_y) ** 2) ** 0.5
    prev_x = curr_x
    prev_y = curr_y
    
    pxl_rer_m = (74 / 1000) / radii
    dist = dist * pxl_rer_m
    velocity = dist/(time_diff + 10 ** -16)
    
    cv2.putText(frame, f"velocity = {velocity:1f}m/s", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))

    cv2.imshow("Mask",mask)
    cv2.imshow("Camera",frame)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()