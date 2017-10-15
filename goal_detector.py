#!/usr/bin/env python
# TODO refactor: do the whole pipeline twice (once for each goal). right it's
# each pipeline step always iteration over both goals
import numpy as np
import cv2

GOAL_HEIGHT = 235
GOAL_WIDTH = 42
ALLOWED_DIFF = 20
SEARCH_WIDTH = 250
SEARCH_HEIGHT = 500
WIDTH = 1600
HEIGHT = 1200
GOAL_UPDATE_INTERVAL = 50
MIN_OBSTACLE_LENGTH = 10


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
                    search_x:search_x+SEARCH_WIDTH, 2]
            dark_areas = (possible_area < 10).astype(np.uint8)
            new_image, contours, _ = cv2.findContours(
                    dark_areas,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)

            # TODO improve: use the best rect min((rect_w-w)^2+(rect_h-h)^2)
            for i in range(len(contours)):
                # not important
                rect = cv2.boundingRect(contours[i])  # x,y,w,h
                if abs(rect[2]-GOAL_WIDTH) <= ALLOWED_DIFF and \
                        abs(rect[3]-GOAL_HEIGHT) <= ALLOWED_DIFF:
                    rects.append([
                        rect[0]+search_x,
                        rect[1]+search_y,
                        rect[2],
                        rect[3]])
                    goal_img.append(
                            possible_area[
                                rect[1]:rect[1]+rect[3],
                                rect[0]:rect[0]+rect[2]])
                    break
            else:
                print('Didnot find rectangle for search_x {}'.format(search_x))
        return rects, goal_img

    def goal_obstacles(self, hsv_img):
        '''
        Find images harnessing diff image and return them

        TODO goal_obstacles should return two lists, one for the one
        goal, one for the other

        :returns: A list of rects with relative positions
        '''
        goals_obstacles = [[], []]
        for rect, goal_img, obstacles in \
                zip(self.goal_rects, self.goal_images, goals_obstacles):
            current_goal_img = hsv_img[
                    rect[1]:rect[1]+rect[3],
                    rect[0]:rect[0]+rect[2],
                    2]
            diff = ((goal_img - current_goal_img) > 15).astype(np.uint8)
            diff = cv2.erode(diff, np.ones((5,5), np.uint8))
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
                    cv2.imshow('obstacle', diff * 255)

        return goals_obstacles

    def step(self, hsv_img):
        '''
        Process one frame
        :hsv_img: The frame image to be processed
        '''
        # Update goal information once in a while
        if self.frame_count % GOAL_UPDATE_INTERVAL == 0:
            # TODO  check if there is no obstacle in the goal at the moment (it needs to be black for 95% of the pixels (or so)
            self.goal_rects, self.goal_images = self.detect_goals(hsv_img)

        # Get obstacles harnessing goal diff images
        goals_obstacles = self.goal_obstacles(hsv_img)

        for goal_rect, obstacles in zip(self.goal_rects, goals_obstacles):
            for obstacle in obstacles:
                y = goal_rect[1]+obstacle[1]
                x = goal_rect[0]+obstacle[0]
                cv2.rectangle(
                        hsv_img,
                        (x, y),
                        (x+obstacle[0], y+obstacle[1]),
                        (0, 0, 255),
                        6)

        self.frame_count += 1
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
    for i in range(100):
        camera.read()
    while grabbed:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        obs = gd.step(hsv)

        cv2.imshow('image', hsv)
        if obs[0] or obs[1]:
            cv2.waitKey(0)
        else:
            cv2.waitKey(40)

        (grabbed, frame) = camera.read()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
