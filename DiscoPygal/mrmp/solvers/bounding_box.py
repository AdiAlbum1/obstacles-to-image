from bindings import *


# calculating a bounding box for the scene so as to sample uniformly inside this box
def calc_bbox(obstacles, sources, destinations, radius):
    X = []
    Y = []
    expand = False
    for poly in obstacles:
        for point in poly.vertices():
            X.append(point.x())
            Y.append(point.y())

    # For scenes without obstacles, take bounding box of expanded robots+destinations
    if not obstacles:
        expand = True
    else:
        min_x = min(X)
        max_x = max(X)
        min_y = min(Y)
        max_y = max(Y)

        for point in sources + destinations:
            x, y = point.x(), point.y()
            if x < min_x or x > max_x or y < min_y or y > max_y:
                expand = True
                break

    for point in sources + destinations:
        x, y = point.x(), point.y()
        X.append(x)
        Y.append(y)

    min_x = min(X)
    max_x = max(X)
    min_y = min(Y)
    max_y = max(Y)

    if expand:
        offset = CGALPY.Ker.FT(2) * radius
        min_x = min_x - offset
        max_x = max_x + offset
        min_y = min_y - offset
        max_y = max_y + offset

    return min_x.to_double(), max_x.to_double(), min_y.to_double(), max_y.to_double()
