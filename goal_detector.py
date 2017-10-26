#!/usr/bin/env python
# TODO refactor: do the whole pipeline twice (once for each goal). right it's
# each pipeline step always iteration over both goals
import numpy as np
import cv2

GOAL_HEIGHT = 235
GOAL_WIDTH = 42
ALLOWED_DIFF = 40
SEARCH_WIDTH = 250
SEARCH_HEIGHT = 500
WIDTH = 1600
HEIGHT = 1200
GOAL_UPDATE_INTERVAL = 50
MIN_OBSTACLE_LENGTH = 10
DEBUG = True


class GoalDetector:
    def __init__(self):
        self.frame_count = 0
        self.goal_rects = []
        self.goal_images = []

    def detect_goals(self, hsv_img):
        '''
        Detect the two goals on the field and save their position and
        current appearing (hsV value)
        :returns: A tuple (rects, values) containing the positions and image
        Values (only brightness channel)  of the two rects
        '''
        search_y = (HEIGHT-SEARCH_HEIGHT)//2
        rects = []
        goal_img = []  # image values

        for search_x in [0, WIDTH-SEARCH_WIDTH]:
            possible_area = hsv_img[
                    search_y:search_y+SEARCH_HEIGHT,
                    search_x:search_x+SEARCH_WIDTH, :]
            dark_areas = (possible_area[:, :, 1] + possible_area[:, :, 2]  < 10).astype(np.uint8)
            new_image, contours, _ = cv2.findContours(
                    dark_areas,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)

            # TODO improve: use the best rect min((rect_w-w)^2+(rect_h-h)^2)
            minh = [0] * len(contours)
            for i in range(len(contours)):   
                rect = cv2.boundingRect(contours[i])  # x,y,w,h
                minh[i]=abs(rect[2]-GOAL_WIDTH)+  abs(rect[3]-GOAL_HEIGHT)                           
            if(min(minh) <= ALLOWED_DIFF):
                rect = cv2.boundingRect(contours[np.argmin(minh)])
                rects.append([
                            rect[0]+search_x,
                            rect[1]+search_y,
                            rect[2],
                            rect[3]])
                goal_img.append(
                            possible_area[
                                rect[1]:rect[1]+rect[3],
                                rect[0]:rect[0]+rect[2], 2])          
            else:
                print('Didnot find rectangle for search_x {}'.format(search_x))
                raise ValueError("No Goals found in Image")    
        if(np.sum(cv2.erode(np.uint8(goal_img[0] > 35), np.ones((9, 9), np.uint8))) > 15 or 
            np.sum(cv2.erode(np.uint8(goal_img[1] > 35), np.ones((9, 9), np.uint8))) > 15):
            raise ValueError("Probably something in the goal")
        return rects, goal_img

    def goal_obstacles(self, hsv_img):
        '''
        Find images harnessing diff image and return them

        TODO goal_obstacles should return two lists, one for the one
        goal, one for the other

        :returns: A list of rects with relative positions
        '''
        goals_obstacles = [[], []]
        goals_diff = [[], []]
        for rect, goal_img, obstacles, diff_img in \
                zip(self.goal_rects, self.goal_images, goals_obstacles, goals_diff):
            current_goal_img = hsv_img[
                    rect[1]:rect[1]+rect[3],
                    rect[0]:rect[0]+rect[2],
                    2]
            diff = ((goal_img - current_goal_img) > 15).astype(np.uint8)
            diff = cv2.erode(diff, np.ones((9, 9), np.uint8))
            # TODO maybe cut borders?
            _, contours, _ = cv2.findContours(
                    diff,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                # not important
                rect = cv2.boundingRect(contours[i])  # x,y,w,h
                if rect[2] > MIN_OBSTACLE_LENGTH and \
                        rect[3] > MIN_OBSTACLE_LENGTH:
                    obstacles.append(rect)
                    diff_img.append(diff)
                    cv2.imshow('obstacle', diff * 255)

        return goals_obstacles, goals_diff

    def step(self, hsv_img):
        '''
        Process one frame
        :hsv_img: The frame image to be processed
        '''
        # Update goal information once in a while
        if self.frame_count % GOAL_UPDATE_INTERVAL == 0:
            # TODO  check if there is no obstacle in the goal at the moment (it needs to be black for 95% of the pixels (or so)
            try:
                self.goal_rects, self.goal_images = self.detect_goals(hsv_img)
            except ValueError as e:
                print(e)
            else:
                if DEBUG:
                    hsv2 = hsv_img
                    hsv2[self.goal_rects[0][1]:
                         self.goal_rects[0][1] + self.goal_rects[0][3],
                         self.goal_rects[0][0]:
                         self.goal_rects[0][0] + self.goal_rects[0][2], :] = \
                             [255,0,0]
                    hsv2[self.goal_rects[1][1]:self.goal_rects[1][1]
                         + self.goal_rects[1][3],
                         self.goal_rects[1][0]:self.goal_rects[1][0]
                         + self.goal_rects[1][2],:] = [255,0,0]
                    hsv2 = cv2.resize(hsv2, (960, 540))
                    cv2.imshow("goal", hsv2[:, :, 2])
                    cv2.waitKey(40)
        # Get obstacles harnessing goal diff images
        goals_obstacles, goal_diff = self.goal_obstacles(hsv_img)

        for goal_rect, obstacles in zip(self.goal_rects, goals_obstacles):
            for obstacle in obstacles:
                y = goal_rect[1]+obstacle[1]
                x = goal_rect[0]+obstacle[0]
                cv2.rectangle(
                        hsv_img,
                        (x-5, y-5),
                        (x+obstacle[2]+5, y+obstacle[3]+5),
                        (255, 0, 0),
                        6)
        self.frame_count += 1
        return goals_obstacles, goal_diff

    def relativeToTotalPosition(self,goals_obstacles):
        '''
        Tranforms the relative position to global position and add the eroded border
        
        '''
        for goal_rect, obstacles in zip(self.goal_rects, goals_obstacles):
            for obstacle in obstacles:                
                obstacle = (goal_rect[0]+obstacle[0]-5,
                            goal_rect[1]+obstacle[1]-5,
                            obstacle[2]+5, obstacle[3]+5)
        return goals_obstacles
def main():
    '''
    detect and print the detected goals
    '''

    gd = GoalDetector()
    videopath = './match1.h264'
    camera = cv2.VideoCapture(videopath)
    print(np.shape(camera))

    (grabbed, frame) = camera.read()
    for i in range(500):
        camera.read()
    while grabbed:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        obs, _ = gd.step(hsv)
        resized = cv2.resize(hsv, (960, 540))
        cv2.imshow('image', resized)
        if obs[0] or obs[1]:
            cv2.waitKey(0)
        else:
            cv2.waitKey(1)

        (grabbed, frame) = camera.read()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
