# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 12:40:23 2017

@author: Marco
"""

import os
import numpy as np
import cv2
from goal_detector import GoalDetector
from classifier import Classifier
from sklearn.model_selection import train_test_split


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
            for i in [0, 1]:
                if obs[i]:
                    for obstacle, diff_img in zip(obs[i], diff[i]):
                        features.append(classifier.extract_features(
                            obstacle, hsv, gd.goal_rects[i]))
                        x, y, w, h = cv2.boundingRect(obstacle)
                        x += gd.goal_rects[i][0]
                        if i == 1:
                            x -= 1200
                        y += gd.goal_rects[i][1] - 400
                        if i == 0:
                            cutout = frame[400:800, 0:400, :].copy()
                        else:
                            cutout = frame[400:800, 1200:, :].copy()
                        cv2.rectangle(cutout, (x, y),
                                      (x + w, y + h), (255, 0, 0), 2)
                        cv2.imwrite(
                            '{}/{:06d}.jpg'.format(folder, imagecount), cutout)
                        imagecount += 1
            (grabbed, frame) = camera.read()
        features = np.asarray(features)
        labels = np.zeros(len(features))
        np.savetxt(folder + "Features.txt", features)
        if not os.path.isfile('{}/Labels.txt'.format(folder)):
            np.savetxt('{}/Labels.txt'.format(folder), labels)

    def train_model(self, folder):
        features = np.loadtxt(folder + "Features.txt")
        labels = np.loadtxt(folder + "Labels.txt")
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.33, random_state=42)

        classifier = Classifier()
        classifier.train(X_train, y_train, X_test, y_test)
        classifier.dump(folder)


def main():
    tr = Trainer("./match2.h264")
    if not os.path.isfile('./trainingdata/Features.txt'):
        tr.save_training_data("./trainingdata/")
    tr.train_model("./trainingdata/")


if __name__ == "__main__":
    main()
