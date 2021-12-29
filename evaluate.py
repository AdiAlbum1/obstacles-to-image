import torch
import numpy as np
import cv2 as cv

from net import Net
import params

from aux_scripts import translate
from generate_base_obstacle_images import generate_base_obstacle_image

if __name__ == "__main__":
    in_filename = "evaluate\\test_scene_(6,-4).json"
    in_filename_2 = "evaluate\\test_scene_(4,2).json"

    img = generate_base_obstacle_image(in_filename)
    img_2 = generate_base_obstacle_image(in_filename_2)

    ground_truth_coords = (6, -4)
    ground_truth_coords_2 = (4, 2)

    row, col = translate.coordinates_to_pixels(ground_truth_coords[0], ground_truth_coords[1])
    row_2, col_2 = translate.coordinates_to_pixels(ground_truth_coords_2[0], ground_truth_coords_2[1])

    normalized_row, normalized_col = row / params.im_height, col / params.im_width
    normalized_row_2, normalized_col_2 = row_2 / params.im_height, col_2 / params.im_width

    print(normalized_row, normalized_col)
    cv.imshow("obstacle", img)
    cv.waitKey(0)

    print(normalized_row_2, normalized_col_2)
    cv.imshow("obstacle_2", img_2)
    cv.waitKey(0)

    model_state_dict = torch.load("test_model.pt")
    model = Net()
    model.load_state_dict(model_state_dict)
    model = model.eval()

    # normalize image to [0,1]
    img = img / 255.0
    img_2 = img_2 / 255.0

    # generate a batch of size 1
    img = np.reshape(img, (1, params.im_height, params.im_width))
    img_2 = np.reshape(img_2, (1, params.im_height, params.im_width))

    batch = torch.from_numpy(np.array([img, img_2]))

    results = model(batch.float())

    print("Network results 1:" + str(results))
    print("Ground truth 1:" + str((normalized_row, normalized_col)))
    print("Ground truth 2:" + str((normalized_row_2, normalized_col_2)))


