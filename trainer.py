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
    def __init__(self,videopath):
        self.videopath = videopath
        
    def saveTrainingdata(self, folder):
        
        gd = GoalDetector()
        classifier = Classifier()
        camera = cv2.VideoCapture(self.videopath)
        imagecount = 0
        (grabbed, frame) = camera.read()
        features = []
        while grabbed:  # and imagecount <= 100:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            obs, diff = gd.step(hsv)
            obs = gd.relativeToTotalPosition(obs)
            if obs[0]:
                for obstacle, diff_img in zip(obs[0], diff[0]):
                    features.append(classifier.extractFeatures(obstacle, hsv, diff_img))
                    cv2.imwrite( folder + str(imagecount) + ".jpg", frame[400:800, 0:400, :])
                    imagecount += 1
                
            if obs[1]:
                for obstacle, diff_img in zip(obs[1], diff[1]):
                    features.append(classifier.extractFeatures(obstacle, hsv, diff_img))
                    cv2.imwrite(folder + str(imagecount) + ".jpg", frame[400:800, 1200:, :])
                    imagecount += 1
            (grabbed, frame) = camera.read()
        features = np.asarray(features)
        labels = np.zeros(len(features))
        np.savetxt(folder + "Features.txt", features)
        np.savetxt(folder + "Labels.txt", labels)
        
    def trainModel(self, folder):
        features = np.loadtxt(folder + "Features.txt")
        labels = np.loadtxt(folder + "Labels.txt")
        classifier = Classifier()
        trainingdata = features[0:2500, :]
        testdata = features[2500:, :]
        traininglabels = labels[0:2500]
        testlabels = labels[2500:]
        
        classifier.train(trainingdata, traininglabels)
            
        resultlabels = classifier.predict(testdata)
        print(np.sum(resultlabels))

        resultlabels = classifier.predict(trainingdata) 
        print(np.sum(traininglabels))
        print(np.sum(resultlabels))
        
        classifier.test(trainingdata,traininglabels)
        classifier.test(testdata,testlabels)
                
def main():
    tr = Trainer("./match2.h264")    
    #tr.saveTrainingdata("./trainingdata/")
    tr.trainModel("./trainingdata/")
    
if __name__ == "__main__":
    main()
