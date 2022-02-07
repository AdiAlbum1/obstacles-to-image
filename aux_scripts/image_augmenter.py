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

def randomly_generate_obstacles_avoiding_passageway(forbidden_row, forbidden_col, box_start, box_end, interest_point):
    num_obstacles = round(random.expovariate(params.lambda_scale) + random.randint(0,3))
    # create obstacle background - a black image
    all_img = np.zeros((params.im_height, params.im_width, 1), np.uint8)

    # draw obstacles
    for i in range(num_obstacles):
        # generate random obstacle
        obstacle_index = random.randint(0, params.max_num_additional_obstacles)
        obstacle_json_filename = "input_json_obstacles/obstacles/" + str(obstacle_index) + ".json"
        obstacles = obstacle_drawer.read_obstacles_from_json(obstacle_json_filename)

        curr_img = np.zeros((params.im_height, params.im_width, 1), np.uint8)

        obstacle = obstacles[0]

        # randomly rotate obstacle
        angle = random.uniform(0, 360)
        mat = cv.getRotationMatrix2D((0,0), angle, 1)
        rotated_obstacle = []
        for vector in obstacle:
            rotated_vector = np.matmul(mat, [vector[0], vector[1], 1])
            rotated_obstacle.append(rotated_vector)

        # randomly translate shape
        x_position = random.randint(-params.axis_range, params.axis_range)
        y_position = random.randint(-params.axis_range, params.axis_range)

        translated_obstacle = []
        for vector in rotated_obstacle:
            translated_vector = (vector[0] + x_position, vector[1] + y_position)
            translated_obstacle.append(translated_vector)

        curr_img = obstacle_drawer.draw_obstacle(curr_img, translated_obstacle, params.im_height, params.im_width,
                                                 params.axis_range)

        obstacle_row = int((params.im_height/2) + y_position)
        obstacle_col = int((params.im_width/2) + x_position)

        # if (obstacle_row, obstacle_col) is too near (forbidden_row, forbidden_col) ignore it
        if forbidden_row - 15 < obstacle_row < forbidden_row + 15 and forbidden_col - 15 < obstacle_col < forbidden_col + 15:
            continue

        box_start, box_end = obstacle_drawer.update_box(translated_obstacle, box_start, box_end, interest_point)

        all_img = cv.bitwise_or(all_img, curr_img)

    return all_img, (box_start, box_end)