import sys
import os.path
from bindings import *
import math
sys.path.insert(0, os.path.dirname(__file__))

# For the following functions the parameter n is the number of robots
# a point is a configuration of the robots and hence the dimension equals 2*num_robots

# Return a function that calculates the sum of distances of the robots for
# two configurations represented by a numpy array
def numpy_sum_distance_for_n(n):
    def distance(p, q):
        sum_of_distances = 0
        for i in range(n):
            dx = p[2 * i] - q[2 * i]
            dy = p[2 * i + 1] - q[2 * i + 1]
            sum_of_distances += math.sqrt(dx * dx + dy * dy)
        return sum_of_distances

    return distance

# The following 5 functions are needed for using CGAL's Kd_trees

def transformed_distance_for_n(n):
    # The following function returns the transformed distance between two points
    # (for Euclidean distance the transformed distance is the squared distance)
    def transformed_distance(p: Point_d, q: Point_d):
        sum_of_distances = 0
        for i in range(n):
            # when we use special number types of CGAL, we occasionally need to convert them to double, to_double()
            # for Python functions such as math.sqrt(), which expect a double (called `float' in Python)
            dx = p[2 * i].to_double() - q[2 * i].to_double()
            dy = p[2 * i + 1].to_double() - q[2 * i + 1].to_double()
            sum_of_distances += math.sqrt(dx * dx + dy * dy)
        return FT(sum_of_distances)

    return transformed_distance


def min_distance_to_rectangle_for_n(n):
    # The following function returns the transformed distance between the query
    # point q and the point on the boundary of the rectangle r closest to q.
    def min_distance_to_rectangle(q, r):
        distance = 0.0
        for i in range(n):
            temp = FT(0)
            h = q[2 * i]
            if h < r.min_coord(2 * i):
                temp += (r.min_coord(2 * i) - h) * (r.min_coord(2 * i) - h)
            if h > r.max_coord(2 * i):
                temp += (h - r.max_coord(2 * i)) * (h - r.max_coord(2 * i))
            h = q[2 * i + 1]
            if h < r.min_coord(2 * i + 1):
                temp += (r.min_coord(2 * i + 1) - h) * (r.min_coord(2 * i + 1) - h)
            if h > r.max_coord(2 * i + 1):
                temp += (h - r.max_coord(2 * i + 1)) * (h - r.max_coord(2 * i + 1))
            distance += math.sqrt(temp.to_double())
        return FT(distance)

    return min_distance_to_rectangle


# The following function returns the transformed distance between the query
# point q and the point on the boundary of the rectangle r furthest to q.
def max_distance_to_rectangle(q, r):
    return FT(Gmpq(1))  # not used


# The following function returns the transformed distance for a value d
# Fo example, if d is a value computed using the Euclidean distance, the transformed distance should be d*d
def transformed_distance_for_value(d):
    return d


# The following function returns the inverse of the transformed distance for a value d
# Fo example, if d is a squared distance value then its inverse should be sqrt(d)
def inverse_of_transformed_distance_for_value(d):
    return d

# Returns a Distance_python object that can be used as one of the parameters of the
# function K_neighbor_search_python() to pass a user defined metric
def sum_distances(num_robots):
    # preparing the distance object, which describes our distance metric for the nearest neighbor search
    distance = SS.Distance_python(transformed_distance_for_n(num_robots), min_distance_to_rectangle_for_n(num_robots),
                              max_distance_to_rectangle, transformed_distance_for_value,
                              inverse_of_transformed_distance_for_value)
    return distance
