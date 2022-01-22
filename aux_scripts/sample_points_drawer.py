import cv2 as cv
from aux_scripts.translate import coordinates_to_pixels

def draw_sample_points(scene_img, sample_points, near_passageway=True):
    scene_with_sample_points = scene_img.copy()
    for sample_point_x, sample_point_y in sample_points:
        row, col = coordinates_to_pixels(sample_point_x, sample_point_y)
        if near_passageway:
            scene_with_sample_points = cv.circle(scene_with_sample_points, (col, row), radius=1, color=(255, 0, 0),
                                                 thickness=-1)
        else:
            scene_with_sample_points = cv.circle(scene_with_sample_points, (col, row), radius=1, color=(0, 0, 255),
                                                 thickness=-1)
    return scene_with_sample_points