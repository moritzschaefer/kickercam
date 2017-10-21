#!/usr/bin/env python
import numpy as np
import cv2

GOAL_HEIGHT = 235
GOAL_WIDTH = 42
ALLOWED_DIFF = 10
SEARCH_WIDTH = 200
SEARCH_HEIGHT = 400
WIDTH = 1600
HEIGHT = 1200


def detect_goals(hsv_img):
    '''
    return a tuple with the rects for the two goals
    '''
    search_y = (HEIGHT-SEARCH_HEIGHT)//2
    rects = []

    for search_x in [0, WIDTH-SEARCH_WIDTH]:
        possible_area = hsv_img[
                search_y:search_y+SEARCH_HEIGHT,
                search_x:search_x+SEARCH_WIDTH, 2]
        dark_areas = (possible_area < 10).astype(np.uint8)
        new_image, contours, _ = cv2.findContours(
                dark_areas,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(contours)):
            # not important
            rect = cv2.boundingRect(contours[i])  # x,y,w,h
            #print(rect)
            if abs(rect[2]-GOAL_WIDTH) <= ALLOWED_DIFF and \
                    abs(rect[3]-GOAL_HEIGHT) <= ALLOWED_DIFF:
                continue
        else:
            return rects
            raise ValueError('No rectangle found')  # TODO what now?
        rects.append([rect[0]+search_x, rect[1]+search_y, rect[2], rect[3]])
    return rects


def main():
    '''
    detect and print the detected goals
    '''
    img = cv2.imread('image-010.jpg', cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    goal_rects = detect_goals(hsv)
    for rect in goal_rects:
        cv2.rectangle(
                img,
                tuple(rect[:2]),
                (rect[0]+rect[2], rect[1]+rect[3]),
                (0, 0, 255), 6)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
