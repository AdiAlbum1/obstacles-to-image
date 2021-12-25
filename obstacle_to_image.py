import numpy as np
import cv2 as cv

import json

import params


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
        curr_vertex = vertex

        # translate vertex
        curr_vertex[0] += axis_range
        curr_vertex[1] += axis_range

        # scale vertex
        curr_vertex[0] = round(curr_vertex[0] * im_width / (2 * axis_range))
        curr_vertex[1] = round(curr_vertex[1] * im_height / (2 * axis_range))

        curr_obstacle.append(curr_vertex)

    curr_obstacle = np.array(curr_obstacle, np.int32)
    curr_obstacle = curr_obstacle.reshape((-1, 1, 2))

    # color obstacle in white
    cv.fillPoly(img, [curr_obstacle], (255, 255, 255))

    return img


if __name__ == "__main__":
    filename = "input_obstacles\\0_0\\7.json"

    im_height = params.im_height
    im_width = params.im_width
    axis_range = params.axis_range

    obstacles = read_obstacles_from_json(filename)

    # Initialize a black background image
    img = np.zeros((im_height, im_width, 3), np.uint8)

    for obstacle in obstacles:
        # eliminate obstacles outside of range x in [-axis_range, axis_range] and y in [-axis_range, axis_range]
        if max_abs_value(obstacle) <= axis_range:
            img = draw_obstacle(img, obstacle, im_height, im_width, axis_range)

    cv.imwrite("img.png", img)
    cv.imshow("obstacle map", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
