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