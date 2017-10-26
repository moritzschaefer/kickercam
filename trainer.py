# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:40:23 2017

@author: Marco
"""

import numpy as np
import cv2
from .goal_detector import GoalDetector

class Trainer:
    def __init__(self,videopath):
        self.videopath = videopath
        
    def saveTrainingdata(self, folder):
        
        gd = GoalDetector()
        camera = cv2.VideoCapture(self.videopath)
        (grabbed, frame) = camera.read()
        while grabbed:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            obs = gd.step(hsv)
            resized = cv2.resize(hsv, (960, 540))
            cv2.imshow('image', resized)
            if obs[0] or obs[1]:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
    
            (grabbed, frame) = camera.read()
