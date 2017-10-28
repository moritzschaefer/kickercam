# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:45:50 2017

@author: Marco
"""

import numpy as np
import cv2
from sklearn import svm

i=0

class Classifier:
    def __init__(self):
        pass

    def extract_features(self, obs_cord, hsv_img, goal_rect, diff_img):
        global i
        goal_img = hsv_img[
            goal_rect[1]:goal_rect[1] + goal_rect[3],
            goal_rect[0]:goal_rect[0] + goal_rect[2]]
        x, y, width, height = obs_cord

        obstacle = goal_img[y:y + height, x:x + width, :]
        cv2.imwrite('balls/{}.jpg'.format(i), cv2.cvtColor(obstacle, cv2.COLOR_HSV2BGR))
        means = np.median(obstacle, axis=(0, 1))
        moments = np.asarray(cv2.moments(diff_img, binaryImage=True).values())
        i+=1
        features = np.concatenate(([width, height], means, moments))
        return features
    
    def train(self, data, label):
        assert(np.shape(data)[0] == np.shape(label)[0])
        self.clf = svm.SVC(class_weight={1: 2, 0: 1})
        
        self.clf.fit(data, label)
    
    def predict(self, data):
        return self.clf.predict(data)

    def test(self,data,label):
        print(self.clf.score(data,label))
