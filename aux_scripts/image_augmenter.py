import random

import cv2 as cv
import numpy as np

from aux_scripts import obstacle_drawer
import params


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def translate_along_y_axis(image):
    y_translation = random.randint(int(-params.im_height / 2), int(params.im_height / 2))

    M = np.float32([
        [1, 0, 0],
        [0, 1, y_translation]
    ])

    result = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    pixel_row = int((params.im_height / 2) + y_translation)

    return result, pixel_row

def randomly_generate_obstacles_avoiding_passageway(forbidden_row, forbidden_col):
    num_obstacles = random.randint(0, params.max_obstacles)

    # create obstacle background - a black image
    all_img = np.zeros((params.im_height, params.im_width, 1), np.uint8)

    # draw obstacles
    for i in range(num_obstacles):
        # generate random obstacle
        obstacle_index = random.randint(0, params.max_num_additional_obstacles)
        obstacle_json_filename = "input_json_obstacles/obstacles/" + str(obstacle_index) + ".json"
        obstacles = obstacle_drawer.read_obstacles_from_json(obstacle_json_filename)

        curr_img = np.zeros((params.im_height, params.im_width, 1), np.uint8)

        for obstacle in obstacles:
            curr_img = obstacle_drawer.draw_obstacle(curr_img, obstacle, params.im_height, params.im_width,
                                                     params.axis_range)

        # randomly rotate obstacle
        angle = random.uniform(0, 360)
        curr_img = rotate_image(curr_img, angle)

        # randomly translate shape
        x_position = random.randint(int(-params.im_width / 2), int(params.im_width / 2))
        y_position = random.randint(int(-params.im_height / 2), int(params.im_height / 2))

        M = np.float32([
            [1, 0, x_position],
            [0, 1, y_position]
        ])

        obstacle_row = int((params.im_height/2) + y_position)
        obstacle_col = int((params.im_width/2) + x_position)

        curr_img = cv.warpAffine(curr_img, M, (curr_img.shape[1], curr_img.shape[0]))

        # if (obstacle_row, obstacle_col) is too near (forbidden_row, forbidden_col) ignore it
        if forbidden_row - 15 < obstacle_row < forbidden_row + 15 and forbidden_col - 15 < obstacle_col < forbidden_col + 15:
            continue

        all_img = cv.bitwise_or(all_img, curr_img)

    return all_img