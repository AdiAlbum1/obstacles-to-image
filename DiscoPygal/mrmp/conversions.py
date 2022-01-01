from bindings import *


def point_2_to_xy(p):
    return p.x().to_double(), p.y().to_double()


def xy_to_point_2(x, y):
    return Point_2(x, y)


def coords_list_to_polygon_2(lst):
    lst0 = []
    for i in range(len(lst) // 2):
        lst0.append(Point_2(lst[2 * i], lst[2 * i + 1]))
    p = Polygon_2(lst0)
    if p.is_clockwise_oriented():
      p.reverse_orientation()
    return p


def tuples_list_to_polygon_2(lst):
    lst0 = []
    for tuple in lst:
        lst0.append(Point_2(tuple[0], tuple[1]))
    p = Polygon_2(lst0)
    if p.is_clockwise_oriented():
      p.reverse_orientation()
    return p


def polygon_2_to_tuples_list(polygon):
    lst = [(p.x().to_double(), p.y().to_double()) for p in polygon.vertices()]
    return lst

# converting a d-dimensional point into an array of d/2 two-dimensional points
# some time we need the first representation (e.g., NN search) and sometime the second
def to_point_2_list(p: Point_d, num_robots) -> list:
    points = []
    for i in range(num_robots):
        points.append(Point_2(p[2 * i], p[2 * i + 1]))
    return points

# converting an array of d/2 two-dimensional points into a d-dimensional point
def to_point_d(points, min_dim=0) -> Point_d:
    res = []
    for p in points:
        res.append(p.x())
        res.append(p.y())
    for _ in range(len(points), min_dim):
        res.append(FT(0))
        res.append(FT(0))
    res = Point_d(len(res), res)
    return res
