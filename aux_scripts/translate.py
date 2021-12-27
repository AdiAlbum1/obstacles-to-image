import params

def coordinates_to_pixels(x, y):
    row = y
    col = x

    # translate
    row += params.axis_range
    col += params.axis_range

    # scale

    row = round(row * params.im_height / (2 * params.axis_range))
    col = round(col * params.im_width / (2 * params.axis_range))

    return (row, col)

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