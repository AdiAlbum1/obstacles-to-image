from bindings import *
import random
import math
import conversions
import networkx as nx
import sklearn.neighbors
import numpy as np
import time

from geometry_utils.collision_detection import Collision_detector

def calc_bbox(obstacles, origin, destination, length):
    X = []
    Y = []
    X.append(origin.x())
    X.append(destination.x())
    Y.append(origin.y())
    Y.append(destination.y())
    for poly in obstacles:
        for point in poly.vertices():
            X.append(point.x())
            Y.append(point.y())
    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)
    # min_x = min(X)-length
    # max_x = max(X)+length
    # min_y = min(Y)-length
    # max_y = max(Y)+length

    return min_x, max_x, min_y, max_y

def point_d_to_arr(p: Point_d):
    return [p[i].to_double() for i in range(p.dimension())]


def generate_path(length, obstacles, origin, destination, argument, writer, isRunning):
    t0 = time.perf_counter()
    path = []
    try:
        num_landmarks = int(argument)
    except Exception as e:
        print("argument is not an integer", file=writer)
        return path
    
    # the radius by which the rod will be expanded
    epsilon = FT(0.1)

    polygons = [conversions.tuples_list_to_polygon_2(p) for p in obstacles]
    s = Point_2(FT(Gmpq(origin[0])), FT(Gmpq(origin[1])))
    clockwise = Point_2(FT(Gmpq(destination[0])), FT(Gmpq(destination[1])))
    bbox = calc_bbox(polygons, s, clockwise, length)
    x_range = (bbox[0].to_double(), bbox[1].to_double())
    y_range = (bbox[2].to_double(), bbox[3].to_double())
    z_range = (0, 2 * math.pi)

    begin = Point_d(3, [FT(Gmpq(origin[0])), FT(Gmpq(origin[1])), FT(Gmpq(origin[2]))])
    end = Point_d(3, [FT(Gmpq(destination[0])), FT(Gmpq(destination[1])), FT(Gmpq(destination[2]))])
    G = nx.DiGraph()
    G.add_nodes_from([begin, end])
    points = [begin, end]

    cd = Collision_detector(polygons, [], epsilon)

    # The number of nearest neighbors each vertex will try to connect to
    K = 25

    i = 0
    while i < num_landmarks:
        # sample new landmark
        rand_x = FT(random.uniform(x_range[0], x_range[1]))
        rand_y = FT(random.uniform(y_range[0], y_range[1]))
        rand_z = FT(random.uniform(z_range[0], z_range[1]))

        if cd.is_rod_position_valid(rand_x, rand_y, rand_z, length):
            p = Point_d(3, [rand_x, rand_y, rand_z])
            G.add_node(p)
            points.append(p)
            i += 1
            if i % 500 == 0:
                print(i, "landmarks sampled", file=writer)
    print(num_landmarks, "landmarks sampled", file=writer)

    # distance used for nearest neighbor search
    def custom_dist(p, q):
        sd = math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
        zd = abs(p[2] - q[2])
        if zd > math.pi:
            zd = math.pi*2 - zd
        sd += length.to_double() * zd
        return sd
    
    # distance used to weigh the edges
    def edge_weight(p, q):
        return math.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

    # sklearn works with numpy array of points represented as numpy arrays
    _points = np.array([point_d_to_arr(p) for p in points])

    # User defined metric cannot be used with the kd_tree algorithm
    nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=K, metric=custom_dist, algorithm='auto')
    # nearest_neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=K, algorithm='kd_tree')
    nearest_neighbors.fit(_points)
    print('Connecting landmarks', file=writer)
    for i in range(len(points)):
        if not isRunning[0]:
            print("Aborted", file=writer)
            return path, G

        p = points[i]
        # Obtain the K nearest neighbors
        k_neighbors = nearest_neighbors.kneighbors([_points[i]], return_distance=False)

        for j in k_neighbors[0]:
            neighbor = points[j]
            for clockwise in (True, False):
                # check if we can add an edge to the graph
                if cd.is_rod_motion_valid(p, neighbor, clockwise, length):
                    weight = edge_weight(point_d_to_arr(p), point_d_to_arr(neighbor))
                    G.add_edge(p, neighbor, weight=weight, clockwise=clockwise)
                    break
        if i % 100 == 0:
            print('Connected', i, 'landmarks to their nearest neighbors', file=writer)
        i += 1

    if nx.has_path(G, begin, end):
        shortest_path = nx.shortest_path(G, begin, end)
        print("path found", file=writer)
        print("distance:", nx.shortest_path_length(G, begin, end))

        if len(shortest_path) == 0:
            return path
        first = shortest_path[0]
        path.append((first[0], first[1], first[2], True))
        for i in range(1, len(shortest_path)):
            last = shortest_path[i-1]
            next = shortest_path[i]
            # determine correct direction
            clockwise = G.get_edge_data(last, next)["clockwise"]
            path.append((next[0], next[1], next[2], clockwise))
    else:
        print("no path was found", file=writer)
    t1 = time.perf_counter()
    print("Time taken:", t1 - t0, "seconds", file=writer)
    return path
    # print(path)
