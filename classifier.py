# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:45:50 2017

@author: Marco
"""

import numpy as np
import cv2
from sklearn import svm

class Classifier:
    def __init__(self):
        pass

    def extractFeatures(self, obs_cord, hsv_img, diff_img):
        x = obs_cord[0]
        y = obs_cord[1]
        x_width = obs_cord[2]
        y_width = obs_cord[3]

        obstacle = hsv_img[x:x+x_width, y:y+y_width, :].copy()
        means = np.median(obstacle, axis=(0, 1))
        moments = np.asarray(cv2.moments(diff_img, binaryImage=True).values())
        features = np.concatenate(([x_width, y_width], means, moments))
        return features
    
    def train(self, data, label):
        assert(np.shape(data)[0] == np.shape(label)[0])
        self.clf = svm.SVC(class_weight={1: 2, 0: 1})
        
        self.clf.fit(data, label)
    
    def predict(self, data):
        return self.clf.predict(data)

    def test(self,data,label):
        print(self.clf.score(data,label))