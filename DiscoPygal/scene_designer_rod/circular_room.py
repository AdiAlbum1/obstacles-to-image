import math


def rotate_polygon(input_polygon, rotation_angle):
    result_polygon = []

    for v in input_polygon:
        # x = v.x* math.cos(rotation_angle)+v.y*math.sin(rotation_angle)
        x = v[0] * math.cos(rotation_angle) - v[1] * math.sin(rotation_angle)
        y = v[0] * math.sin(rotation_angle) + v[1] * math.cos(rotation_angle)
        result_polygon.append([x, y])

    return result_polygon


def get_circular_room(center, inner_radius, number_of_blocks, block_height):
    if number_of_blocks < 3:
        raise Exception("invalid number of blocks")
    result = []
    base_polygon = []

    radians_per_block = 2 * math.pi / number_of_blocks

    # first block
    x = inner_radius * math.tan(0.5 * radians_per_block)
    y = inner_radius
    base_polygon.append([x, y])
    # x2 = x1
    # y2 = y1 + block_height
    base_polygon.append([x, y + block_height])
    # x3 = - x1
    # y3 = y2
    base_polygon.append([-x, +y + block_height])
    # x4 = x3
    # y4 = inner_radius
    base_polygon.append([-x, +y])

    result.append(base_polygon)

    # the remaining blocks
    for i in range(2, number_of_blocks + 1):  # the upper bound of the range is not executed
        result.append(rotate_polygon(base_polygon, (i - 1) * radians_per_block))

    # shift by center
    for q in result:
        for p in q:
            p[0] = p[0] + center[0]
            p[1] = p[1] + center[1]

    return result
