# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 00:30:24 2017

@author: Marco
"""
from collections import deque
import numpy as np
import cv2
import goal_detector

def main():
    i=0
    videopath = "C:\Users\Marco\Desktop\Studium\projects\Kickercam\match1.h264"
    camera = cv2.VideoCapture(videopath)
    print(np.shape(camera))
    (grabbed,frame) = camera.read()

    while i<12 and grabbed:
        print(np.sum(frame))
        i+=1
        if(i%1==0):
            print("iteration",i)

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            goal_rects= goal_detector.detect_goals(hsv)
            print(goal_rects)
            if goal_rects==None:
                (grabbed,frame) = camera.read()
                continue;
            for rect in goal_rects:
                cv2.rectangle(
                        hsv,
                        tuple(rect[:2]),
                        (rect[0]+rect[2], rect[1]+rect[3]),
                        (0, 0, 255), 6)
            
            cv2.imshow('image', hsv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        (grabbed,frame) = camera.read()
        
        #print(np.shape(frame))
        
main()