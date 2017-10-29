# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 20:45:50 2017

@author: Marco
"""

import pickle
import numpy as np
import cv2
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn import preprocessing

i = 0


class Classifier:
    def __init__(self, folder=None):
        if folder:
            self.load(folder)

    def dump(self, folder):
        with open("{}/model.pkl".format(folder), 'wb') as f:
            pickle.dump(self.clf, f)
        with open("{}/scaler.pkl".format(folder), 'wb') as f:
            pickle.dump(self.scaler, f)

    def load(self, folder):
        with open("{}/model.pkl".format(folder), "rb") as f:
            self.clf = pickle.load(f)
        with open("{}/scaler.pkl".format(folder), "rb") as f:
            self.scaler = pickle.load(f)

    def extract_features(self, contour, hsv_img, goal_rect):
        global i
        goal_img = hsv_img[
            goal_rect[1]:goal_rect[1] + goal_rect[3],
            goal_rect[0]:goal_rect[0] + goal_rect[2]]
        x, y, width, height = cv2.boundingRect(contour)

        obstacle = goal_img[y:y + height, x:x + width, :].copy()
        thresholded = (obstacle[:, :, 2] > 40).astype('uint8')
        eroded_obstacle = cv2.erode(
            thresholded,
            np.ones((7, 7), np.uint8))
        obstacle = cv2.bitwise_and(obstacle, obstacle, mask=~eroded_obstacle)
        # moment features
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        bgr_obstacle = cv2.cvtColor(obstacle, cv2.COLOR_HSV2BGR)

        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        try:
            ellipse = cv2.fitEllipse(contour)
        except:
            ellipse = ((3, 3), (1, 1), 0)

        cv2.imwrite('balls/{}.jpg'.format(i), bgr_obstacle)

        obstacle_values = obstacle[obstacle[:, :, 2] > 0]
        hsv_medians = np.median(obstacle_values, axis=0)
        hsv_means = np.mean(obstacle_values, axis=0)
        hsv_mins = np.min(obstacle_values, axis=0)
        hsv_maxes = np.max(obstacle_values, axis=0)

        i += 1

        features = [
            width, height, width / height, cx, cy, area, perimeter, radius,
            ellipse[0][1] - ellipse[0][0], ellipse[1][1] - ellipse[1][0],
            ellipse[1][1] - ellipse[1][0] / ellipse[0][1] - ellipse[0][0]]
        # return hsv_medians
        return np.concatenate((features, hsv_medians,
                               hsv_means, hsv_mins, hsv_maxes))

    def train(self, X_train, y_train, X_test, y_test):
        # assert(np.shape(data)[0] == np.shape(label)[0])
        self.scaler = preprocessing.StandardScaler().fit(X_train)

        # self.clf.fit(data, label)

        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000], 'class_weight': ['balanced']},
                            {'kernel': ['linear'], 'C': [
                                1, 10, 100, 1000], 'class_weight': ['balanced']},
                            ]

        scores = ['precision', 'recall', 'f1']
        score = 'f1'

        # for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        self.clf = GridSearchCV(svm.SVC(), tuned_parameters, cv=4,
                                scoring='%s_macro' % score, verbose=0)
        self.clf.fit(self.scaler.transform(X_train), y_train)

        print("Best parameters set found on development set:")
        print()
        print(self.clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = self.clf.cv_results_['mean_test_score']
        stds = self.clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, self.clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, self.clf.predict(
            self.scaler.transform(X_test))
        print(classification_report(y_true, y_pred))
        print()

    def predict(self, obstacle, hsv, goal_rect):
        features = self.extract_features(obstacle, hsv, goal_rect).reshape(1, -1)

        return self.clf.predict(self.scaler.transform(features))
