import numpy as np
import cv2 as cv

import json

import params

from aux_scripts import translate


def max_abs_value(obstacle):
    max_val = 0
    for vertex in obstacle:
        curr_max_val = np.max(np.abs(vertex))
        if curr_max_val > max_val:
            max_val = curr_max_val

    return max_val


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


if __name__ == "__main__":
    im_height = params.im_height
    im_width = params.im_width
    axis_range = params.axis_range

    for i in range(9):
        for j in range(18):
            in_filename = "input_json_obstacles\\"+str(i)+"_0\\"+str(j)+".json"
            out_filename = "input_png_obstacles\\stage 1\\"+str(i)+"_0\\"+str(j)+".png"

            obstacles = read_obstacles_from_json(in_filename)

            # Initialize a black background image
            img = np.zeros((im_height, im_width, 3), np.uint8)

            for obstacle in obstacles:
                # eliminate obstacles outside of range x in [-axis_range, axis_range] and y in [-axis_range, axis_range]
                if max_abs_value(obstacle) <= axis_range:
                    img = draw_obstacle(img, obstacle, im_height, im_width, axis_range)

            cv.imwrite(out_filename, img)
