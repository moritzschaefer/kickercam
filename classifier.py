# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:45:50 2017

@author: Marco
"""

import numpy as np
import cv2


class Classifier:
    def __init__(self):
        pass

    def extractFeatures(self, obs_cord, hsv_img, diff_img):
        x = obs_cord[0]
        y = obs_cord[1]
        x_width = obs_cord[2]
        y_width = obs_cord[3]

        obstacle = hsv_img[x:x+x_width, y+y_width, :].copy
        means = np.median(obstacle, axis=(0, 1))
        moments = np.asarray(cv2.moments(diff_img, binaryImage=True))
        features = np.concatenate(([x_width, y_width], means, moments))
        return features
    

        