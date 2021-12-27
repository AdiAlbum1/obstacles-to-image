import numpy as np
import cv2 as cv

import params

from aux_scripts import obstacle_drawer


def max_abs_value(obstacle):
    max_val = 0
    for vertex in obstacle:
        curr_max_val = np.max(np.abs(vertex))
        if curr_max_val > max_val:
            max_val = curr_max_val

    return max_val


if __name__ == "__main__":
    im_height = params.im_height
    im_width = params.im_width
    axis_range = params.axis_range

    for i in range(9):
        for j in range(18):
            in_filename = "input_json_obstacles\\"+str(i)+"_0\\"+str(j)+".json"
            out_filename = "input_png_obstacles\\stage 1\\"+str(i)+"_0\\"+str(j)+".png"

            obstacles = obstacle_drawer.read_obstacles_from_json(in_filename)

            # Initialize a black background image
            img = np.zeros((im_height, im_width, 3), np.uint8)

            for obstacle in obstacles:
                # eliminate obstacles outside of range x in [-axis_range, axis_range] and y in [-axis_range, axis_range]
                if max_abs_value(obstacle) <= axis_range:
                    img = obstacle_drawer.draw_obstacle(img, obstacle, im_height, im_width, axis_range)

            cv.imwrite(out_filename, img)