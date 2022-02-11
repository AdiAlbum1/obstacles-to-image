import cv2 as cv
import numpy as np

import json

import shapely.geometry

from aux_scripts import translate
from shapely.geometry import Polygon, Point

def read_obstacles_from_json(filename):
    input_file = open(filename)
    data = json.load(input_file)
    obstacles = data["obstacles"]

    return obstacles

def update_passageway(obstacle, passageway):
    # find intersection between current box and obstacle - If none exists, the box is unchanged
    # If an intersection exists, we need to update the box according to the intersection polygon
    # and position of interest_point

    # obstacle to pixels polygon
    obstacle_in_pixels = [translate.coordinates_to_pixels(vertex[0], vertex[1]) for vertex in obstacle]
    obstacle_poly = Polygon(obstacle_in_pixels)
    if not obstacle_poly.is_simple:
        return None


    # passageway to polygon
    passageway_poly = passageway.to_polygon()

    # obtain intersection polygon
    intersection = passageway_poly.intersection(obstacle_poly)
    if intersection.is_empty:
        return passageway

    # poi to shapely point
    poi_point = Point(passageway.get_poi())

    # if poi is inside intersrction polygon - this is an invalid obstacle
    if intersection.contains(poi_point):
        return None

    # if intersection isn't a polygon (say, a line or a multipolygon, ignore the obstacle
    if not isinstance(intersection, Polygon):
        return None

    intersection_vertices = list(intersection.exterior.coords)
    poi = passageway.get_poi()

    # if passageway is vertical, verify the intersection polygon is entirely either to the left or to the right of poi
    # and update passageway accordingly
    if not passageway.get_is_vertical():
        intersection_min_col = min([vertex[1] for vertex in intersection_vertices])
        intersection_max_col = max([vertex[1] for vertex in intersection_vertices])

        if intersection_max_col < poi[1]:
            # intersection is entirely to the left of poi
            new_passageway_start = (passageway.get_start()[0], intersection_max_col)
            passageway.update_start(new_passageway_start)
        elif intersection_min_col > poi[1]:
            # intersection is entirely to the right of poi
            new_passageway_end = (passageway.get_end()[0], intersection_min_col)
            passageway.update_end(new_passageway_end)
        else:
            # illegal
            return None
    # if passageway is horizonatal, verify the interesction polygon is entirely either above or below the poi
    else:
        intersection_min_row = min([vertex[0] for vertex in intersection_vertices])
        intersection_max_row = max([vertex[0] for vertex in intersection_vertices])

        if intersection_max_row < poi[0]:
            # intersection is entirely above of poi
            new_passageway_start = (intersection_max_row, passageway.get_start()[1])
            passageway.update_start(new_passageway_start)
        elif intersection_min_row > poi[0]:
            # intersection is entirely to the right of poi
            new_passageway_end = (intersection_min_row, passageway.get_end()[1])
            passageway.update_end(new_passageway_end)
        else:
            # illegal
            return None

    return passageway

def draw_obstacle(img, obstacle, im_height, im_width, axis_range):
    curr_obstacle = []
    for vertex in obstacle:
        row, col = translate.coordinates_to_pixels(vertex[0], vertex[1])

        curr_obstacle.append([col, row])

    curr_obstacle = np.array(curr_obstacle, np.int32)
    curr_obstacle = curr_obstacle.reshape((-1, 1, 2))

    # color obstacle in white
    cv.fillPoly(img, [curr_obstacle], (255, 255, 255))

    return img