import numpy as np
import torch

import params
from generate_base_obstacle_images import generate_obstacle_image
from aux_scripts import translate

# from models.current_best.net import Net
from net import Net


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
    pw_start_row, pw_start_col, pw_end_row, pw_end_col = results.cpu().data.numpy()[0]

    pw_start_row, pw_start_col = translate.net_output_to_pixels(pw_start_row, pw_start_col)
    pw_end_row, pw_end_col = translate.net_output_to_pixels(pw_end_row, pw_end_col)

    return orig_obstacle_image, (pw_start_row, pw_start_col, pw_end_row, pw_end_col)
