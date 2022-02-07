import cv2 as cv
import numpy as np

import json

from aux_scripts import translate

def read_obstacles_from_json(filename):
    input_file = open(filename)
    data = json.load(input_file)
    obstacles = data["obstacles"]

    return obstacles

def update_box(obstacle, box_start, box_end, interest_point):
    # find intersection between current box and obstacle - If none exists, the box is unchanged
    # If an intersection exists, we need to update the box according to the intersection polygon
    # and position of interest_point
    for vertex in obstacle:
        row, col = translate.coordinates_to_pixels(vertex[0], vertex[1])

        # vertex is inside box
        if box_start[0] <= row <= box_end[0] and box_start[1] <= col <= box_end[1]:
            # check position w.r.t interest_point
            if row > interest_point[0]:
                box_end = (row, box_end[1])
            else:
                box_start = (row, box_start[1])

            if col > interest_point[1]:
                box_end = (box_end[0], col)
            else:
                box_start = (box_start[0], col)

    return box_start, box_end

def draw_obstacle(img, obstacle, im_height, im_width, axis_range):
    curr_obstacle = []
    for vertex in obstacle:
        row, col = translate.coordinates_to_pixels(vertex[0], vertex[1])

        curr_obstacle.append([col, row])

    curr_obstacle = np.array(curr_obstacle, np.int32)
    curr_obstacle = curr_obstacle.reshape((-1, 1, 2))

    # color obstacle in white
    cv.fillPoly(img, [curr_obstacle], (255, 255, 255))

    return img