# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:40:23 2017

@author: Marco
"""

import numpy as np
import cv2
from goal_detector import GoalDetector
from classifier import Classifier


class Trainer:
    def __init__(self, videopath):
        self.videopath = videopath

    def save_training_data(self, folder):
        gd = GoalDetector()
        classifier = Classifier()
        camera = cv2.VideoCapture(self.videopath)
        imagecount = 0
        (grabbed, frame) = camera.read()
        features = []
        while grabbed:  # and imagecount <= 100:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            obs, diff = gd.step(hsv)
            obs = gd.relative_to_absolute_position(obs)
            if obs[0]:
                for obstacle, diff_img in zip(obs[0], diff[0]):
                    features.append(classifier.extract_features(
                        obstacle, hsv, gd.goal_rects[0], diff_img))
                    cv2.imwrite(folder + str(imagecount) +
                                ".jpg", frame[400:800, 0:400, :])
                    imagecount += 1

            if obs[1]:
                for obstacle, diff_img in zip(obs[1], diff[1]):
                    features.append(classifier.extract_features(
                        obstacle, hsv, gd.goal_rects[1], diff_img))
                    cv2.imwrite(folder + str(imagecount) +
                                ".jpg", frame[400:800, 1200:, :])
                    imagecount += 1
            (grabbed, frame) = camera.read()
        features = np.asarray(features)
        labels = np.zeros(len(features))
        np.savetxt(folder + "Features.txt", features)
        np.savetxt(folder + "Labels.txt", labels)


def main():
    tr = Trainer("./match2.h264")
    tr.save_training_data("./trainingdata/")


if __name__ == "__main__":
    main()
