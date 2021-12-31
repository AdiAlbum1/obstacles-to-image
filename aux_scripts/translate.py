import params

def coordinates_to_pixels(x, y):
    row = round(params.im_height * ((-y / (2 * params.axis_range)) + 0.5))
    col = round(params.im_width * ((x / (2 * params.axis_range)) + 0.5))

    return row, col

def pixels_to_coordinates(row, col):
    x = params.axis_range * ((2 * col / params.im_width) - 1)
    y = params.axis_range * (1 - (2*row / params.im_height))

    return x, y

def net_output_to_pixels(out_x, out_y):
    row_pixels = round(params.im_height * out_x)
    col_pixels = round(params.im_width * out_y)

    return row_pixels, col_pixels

def translate_pixel_value_for_rotation(row, col, num_rows, num_cols, angle):
    assert angle in [0, 90, 180, 270]
    if angle == 0:
        new_row = row
        new_col = col
    elif angle == 90:
        new_row = num_cols - col
        new_col = row
    elif angle == 180:
        new_row = num_rows - row
        new_col = num_cols - col
    else:
        new_row = col
        new_col = num_rows - row
    return new_row, new_col