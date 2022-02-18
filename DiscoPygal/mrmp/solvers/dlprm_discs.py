import sys
import os.path
import cv2 as cv
import sklearn.neighbors
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from DiscoPygal.bindings import *
from DiscoPygal.geometry_utils import collision_detection
import networkx as nx
import random
import time
import math
import conversions
import sum_distances
import bounding_box
import inference
from aux_scripts import translate, sample_points_drawer


# Number of nearest neighbors to search for in the k-d tree
K = 15

# generate_path() is our main PRM function
# it constructs a PRM (probabilistic roadmap)
# and searches in it for a path from start (robots) to target (destinations)
def generate_path_disc(scene, robots, obstacles, disc_obstacles, destinations, argument, writer, isRunning):
    ###################
    # Preperations
    ###################
    t0 = time.perf_counter()
    path = []
    try:
        num_landmarks = int(argument)
    except Exception as e:
        print("argument is not an integer", file=writer)
        return path
    print("num_landmarks=", num_landmarks, file=writer)
    num_robots = len(robots)
    print("num_robots=", num_robots, file=writer)
    # for technical reasons related to the way the python bindings for this project were generated, we need
    # the condition "(dim / num_robots) >= 2" to hold
    if num_robots == 0:
        print("unsupported number of robots:", num_robots, file=writer)
        return path
    # compute the free C-space of a single robot by expanding the obstacles by the disc robot radius
    # and maintaining a representation of the complement of the expanded obstacles
    sources = [robot['center'] for robot in robots]
    radii = [robot['radius'] for robot in robots]
    collision_detectors = [collision_detection.Collision_detector(obstacles, disc_obstacles, radius) for radius in radii]
    min_x, max_x, min_y, max_y = bounding_box.calc_bbox(obstacles, sources, destinations, max(radii))

    # turn the start position of the robots (the array robots) into a d-dim point, d = 2 * num_robots
    sources = conversions.to_point_d(sources)
    # turn the target position of the robots (the array destinations) into a d-dim point, d = 2 * num_robots
    destinations = conversions.to_point_d(destinations)
    # we use the networkx Python package to define and manipulate graphs
    # G is an undirected graph, which will represent the PRM
    G = nx.Graph()
    points = [sources, destinations]
    # we also add these two configurations as nodes to the PRM G
    G.add_nodes_from([sources, destinations])
    print('Sampling landmarks', file=writer)

    scene_img, net_output = inference.find_narrow_passageway(scene)
    pw_start_row, pw_start_col, pw_end_row, pw_end_col = net_output
    pw_start = pw_start_row, pw_start_col
    pw_end = pw_end_row, pw_end_col

    # # ~~~ UNCOMMENT FOR BOUNDING BOX VISUALIZATION ~~~
    # color = (255, 0, 0)
    # img = cv.cvtColor(scene_img, cv.COLOR_GRAY2BGR)
    # img = cv.rectangle(img, pw_start[::-1], pw_end[::-1], color)
    # cv.imshow("scene_img", img)
    # cv.imwrite("evaluate\\bounding_box.png", img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    pw_start_coords = translate.pixels_to_coordinates(pw_start[0], pw_start[1])
    pw_end_coords = translate.pixels_to_coordinates(pw_end[0], pw_end[1])

    n_NP = num_landmarks // 2
    # n_NP = 0

    narrow_passageway_landmarks = []
    remaining_landmarks = []

    ######################
    # Sampling landmarks
    ######################
    for i in range(num_landmarks):
        if not isRunning[0]:
            print("Aborted", file=writer)
            return path, G

        if i < n_NP:
            p = sample_valid_landmark_in_narrow_passageway(min_x, max_x, min_y, max_y, collision_detectors, num_robots, radii,
                                                             pw_start_coords, pw_end_coords, narrow_passageway_landmarks)
        else:
            p = sample_valid_landmark(min_x, max_x, min_y, max_y, collision_detectors, num_robots, radii, remaining_landmarks)
        G.add_node(p)
        points.append(p)
        if i % 500 == 0:
            print(i, "landmarks sampled", file=writer)
    print(num_landmarks, "landmarks sampled", file=writer)

    # # ~~~ UNCOMMENT FOR SAMPLING POINT VISUALIZATION ~~~
    # scene_with_landmarks = cv.cvtColor(scene_img, cv.COLOR_GRAY2RGB)
    # scene_with_landmarks = sample_points_drawer.draw_sample_points(scene_with_landmarks, narrow_passageway_landmarks, True)
    # scene_with_landmarks = sample_points_drawer.draw_sample_points(scene_with_landmarks, remaining_landmarks, False)
    #
    # cv.imshow("scene", scene_img)
    # cv.imshow("scene with landmarks", scene_with_landmarks)
    # cv.waitKey(0)
    #
    # cv.imwrite("evaluate\\scene\\scene.png", scene_img)
    # cv.imwrite("evaluate\\scene\\our_prm_samples.png", scene_with_landmarks)
    #

    ### !!!
    # Distnace functions
    ### !!!
    distance = sum_distances.sum_distances(num_robots)
    custom_dist = sum_distances.numpy_sum_distance_for_n(num_robots)

    _points = np.array([point_d_to_arr(p) for p in points])


    ########################
    # Constract the roadmap
    ########################
    # User defined metric cannot be used with the kd_tree algorithm
    kdt = sklearn.neighbors.NearestNeighbors(n_neighbors=K, metric=custom_dist, algorithm='auto')
    # kdt = sklearn.neighbors.NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    kdt.fit(_points)
    print('Connecting landmarks', file=writer)
    for i in range(len(points)):
        if not isRunning[0]:
            print("Aborted", file=writer)
            return path, G

        p = points[i]
        k_neighbors = kdt.kneighbors([_points[i]], return_distance=False)

        if edge_valid(collision_detectors, p, destinations, num_robots, radii):
            d = distance.transformed_distance(p, destinations).to_double()
            G.add_edge(p, destinations, weight=d)
        for j in k_neighbors[0]:
            neighbor = points[j]
            if not G.has_edge(p, neighbor):
                # check if we can add an edge to the graph
                if edge_valid(collision_detectors, p, neighbor, num_robots, radii):
                    d = distance.transformed_distance(p, neighbor).to_double()
                    G.add_edge(p, neighbor, weight=d)
        if i % 500 == 0:
            print('Connected', i, 'landmarks to their nearest neighbors', file=writer)


    ########################
    # Finding a valid path
    ########################
    if nx.has_path(G, sources, destinations):
        temp = nx.dijkstra_path(G, sources, destinations, weight='weight')
        lengths = [0 for _ in range(num_robots)]
        if len(temp) > 1:
            for i in range(len(temp) - 1):
                p = temp[i]
                q = temp[i + 1]
                for j in range(num_robots):
                    dx = p[2 * j].to_double() - q[2 * j].to_double()
                    dy = p[2 * j + 1].to_double() - q[2 * j + 1].to_double()
                    lengths[j] += math.sqrt((dx * dx + dy * dy))
        print("A path of length", sum(lengths), "was found", file=writer)
        for i in range(num_robots):
            print('Length traveled by robot', i, ":", lengths[i], file=writer)
        for p in temp:
            path.append(conversions.to_point_2_list(p, num_robots))
    else:
        print("No path was found", file=writer)
    t1 = time.perf_counter()
    print("Time taken:", t1 - t0, "seconds", file=writer)
    return path, G


# throughout the code, wherever we need to return a number of type double to CGAL,
# we convert it using FT() (which stands for field number type)
def point_d_to_arr(p: Point_d):
    return [p[i].to_double() for i in range(p.dimension())]

# find one free landmark (milestone) within the bounding box
def sample_valid_landmark(min_x, max_x, min_y, max_y, collision_detectors, num_robots, radii, remaining_landmarks):
    while True:
        points = []
        # for each robot check that its configuration (point) is in the free space
        for i in range(num_robots):
            rand_x = FT(random.uniform(min_x, max_x))
            rand_y = FT(random.uniform(min_y, max_y))
            p = Point_2(rand_x, rand_y)
            if collision_detectors[i].is_point_valid(p):
                remaining_landmarks.append((rand_x.to_double(), rand_y.to_double()))
                points.append(p)
            else:
                break
        # verify that the robots do not collide with one another at the sampled configuration
        if len(points) == num_robots and not collision_detection.check_intersection_static(points, radii):
            return conversions.to_point_d(points)

def sample_valid_landmark_in_narrow_passageway(min_x, max_x, min_y, max_y, collision_detectors, num_robots, radii,
                                                 pw_start_coords, pw_end_coords, narrow_passageway_landmarks):

    np_std = 1
    while True:
        points = []

        # randomly select a robot which we'll select near the narrow passageway
        j = random.randint(0, num_robots-1)
        for i in range(num_robots):
            if i == j:
                rand_x = FT(random.uniform(pw_start_coords[0], pw_end_coords[0]))
                rand_y = FT(random.uniform(pw_start_coords[1], pw_end_coords[1]))
            else:
                rand_x = FT(random.uniform(min_x, max_x))
                rand_y = FT(random.uniform(min_y, max_y))

            p = Point_2(rand_x, rand_y)
            if collision_detectors[i].is_point_valid(p):
                narrow_passageway_landmarks.append((rand_x.to_double(), rand_y.to_double()))
                points.append(p)
            else:
                break
            # verify that the robots do not collide with one another at the sampled configuration
        if len(points) == num_robots and not collision_detection.check_intersection_static(points, radii):
            return conversions.to_point_d(points)


# check whether the edge pq is collision free
# the collision detection module sits on top of CGAL arrangements
def edge_valid(collision_detectors, p: Point_d, q: Point_d, num_robots, radii):
    p = conversions.to_point_2_list(p, num_robots)
    q = conversions.to_point_2_list(q, num_robots)
    edges = []
    # for each robot check that its path (line segment) is in the free space
    for i in range(num_robots):
        edge = Segment_2(p[i], q[i])
        if not collision_detectors[i].is_edge_valid(edge):
            return False
        edges.append(edge)
    # verify that the robots do not collide with one another along the C-space edge
    if collision_detection.check_intersection_against_robots(edges, radii):
        return False
    return True
