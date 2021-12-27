import cv2 as cv
import numpy as np

import json

from aux_scripts import translate

def read_obstacles_from_json(filename):
    input_file = open(filename)
    data = json.load(input_file)
    obstacles = data["obstacles"]

    return obstacles


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