import numpy as np
import torch

import params
from generate_base_obstacle_images import generate_obstacle_image
from net import Net
from aux_scripts import translate

def find_k_narrow_passageways(input_scene_img):
    # normalize image to [0,1] range
    input_scene_img = input_scene_img / 255.0

    # load model
    model_state_dict = torch.load(params.network_path)
    model = Net()
    model.load_state_dict(model_state_dict)
    model = model.eval()


def find_narrow_passageway(input_path):
    orig_obstacle_image = generate_obstacle_image(input_path)

    # normalize image to [0,1] range
    obstacle_image = orig_obstacle_image / 255.0

    # load model
    model_state_dict = torch.load(params.network_path)
    model = Net()
    model.load_state_dict(model_state_dict)
    model = model.eval()

    # generate a batch of size 1
    obstacle_image_batch = obstacle_image.reshape((1, 1, params.im_height, params.im_width))
    obstacle_image_batch = torch.from_numpy(obstacle_image_batch)

    # run model
    results = model(obstacle_image_batch.float())

    # decipher model results to obstacle coordinates
    results_x, results_y = results.cpu().data.numpy()[0]
    results_pix_row, results_pix_col = translate.net_output_to_pixels(results_x, results_y)
    results_coord_x, results_coord_y = translate.pixels_to_coordinates(results_pix_row, results_pix_col)

    return orig_obstacle_image, (results_coord_x, results_coord_y)
