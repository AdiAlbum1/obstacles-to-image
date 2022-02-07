import params

from aux_scripts import  obstacle_drawer
from generate_base_obstacle_images import max_abs_value

def generate_passageway_width(input_path):
    obstacles = obstacle_drawer.read_obstacles_from_json(input_path)

    max_x = params.axis_range
    min_x = -params.axis_range

    min_x_val = None
    max_x_val = None

    for obstacle in obstacles:
        # eliminate obstacles outside of range x in [-axis_range, axis_range] and y in [-axis_range, axis_range]
        if max_abs_value(obstacle) <= params.axis_range:
            x_vals = [point[0] for point in obstacle]
            # left obstacle
            if min_x in x_vals:
                min_x_val = max(x_vals)

            # right obstacle
            if max_x in x_vals:
                max_x_val = min(x_vals)

    if min_x_val is None:
        min_x_val = min_x
    if max_x_val is None:
        max_x_val = max_x

    return max_x_val - min_x_val
