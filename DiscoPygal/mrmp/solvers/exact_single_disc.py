import time
import math

import networkx as nx

from bindings import *
from geometry_utils import collision_detection
from geometry_utils import display_arrangement

# Epsilon for approximated offset
EPS = 0.0001

def generate_path_disc(robots, obstacles, disc_obstacles, destinations, argument, writer, isRunning):
    """
    Exact motion planner for a single disc robot among polygonal obstacles
    The idea is to use a vertical decomposition of the (expanded) obstacles arrangement,
    and building a connectivity graph for the trapezoids. 
    From there a path is just connecting midpoints of each traveled edge in the graph.

    NOTE: The argument is ignored in this method (and can be empty)
    """

    #################
    # Preprocessing
    #################
    t0 = time.perf_counter()
    path = []
    G = nx.Graph()
    
    num_robots = len(robots)
    # this solvers assumes only the case for a signle robot
    if num_robots != 1:
        print("unsupported number of robots:", num_robots, file=writer)
        return path, G
    
    source = robots[0]['center']
    target = destinations[0]
    radius = robots[0]['radius']

    #############################################################
    # Generate the vertical decomposition and connectivity graph
    #############################################################

    arr = construct_cspace(obstacles, radius)
    arr = vertical_decomposition(arr)
    conn_graph, edge_dict = connectivity_graph(arr)
    collision_detector = collision_detection.Collision_detector(obstacles, [], radius)

    #########################################
    # Find indices of first and last faces
    #########################################
    pl = Aos2.Arr_trapezoid_ric_point_location(arr)
    
    source_idx = find_face_index(arr, source, pl)
    target_idx = find_face_index(arr, target, pl)

    if source_idx < 0:
        print("Could not find origin face.", file=writer)
        return path, G
    if target_idx < 0:
        print("Could not find target face.", file=writer)
        return path, G
    
    #####################################
    # Generate a path from start to end
    #####################################
    if source_idx == target_idx:
        # If we are very close, then linear interpolation is enough
        path = [[source], [target]]
        return path, G

    try:
        g_path = nx.shortest_path(conn_graph, source_idx, target_idx)
    except nx.exception.NetworkXNoPath:
        print("Could not find a path from source to dest", file=writer)
        return path, arr

    # Convert g_path to a linear path for the robot
    path = find_valid_path(g_path, source, target, edge_dict, collision_detector)
    
    t1 = time.perf_counter()
    print("Time taken:", t1 - t0, "seconds", file=writer)
    return path, arr


def calc_bbox(obstacles, sources, destinations, radius):
    """
    Get the scene data and construct a bounding box
    (Also expant the bounding box by twice some radius)
    """
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


def construct_cspace(obstacles, radius):
    """
    Get the (CGAL Polygonal) obstacles and the radius of the robot,
    and construct the expanded CSPACE arrangement (also with bounding box walls)
    """
    traits = Arr_face_overlay_traits(lambda x, y: x + y)
    
    # Compute an arrangement for each single Minkowski sum
    arrangements = []
    for polygon in obstacles:
        ms = MN2.approximated_offset_2(polygon, radius, EPS)
        arr = Arrangement_2()
        Aos2.insert(arr, [curve for curve in ms.outer_boundary().curves()])
        for hole in ms.holes():
            Aos2.insert(arr, [curve for curve in hole.curves()])
        ubf = arr.unbounded_face()
        ubf.set_data(0)
        invalid_face = next(next(ubf.inner_ccbs())).twin().face()
        invalid_face.set_data(1)
        for ccb in invalid_face.inner_ccbs():
            valid_face = next(ccb).twin().face()
            valid_face.set_data(0)
        arrangements.append(arr)

    # Overlay the arrangmemnt
    initial = Arrangement_2()
    ubf = initial.unbounded_face()
    ubf.set_data(0)
    arrangements.insert(0, initial)
    arr = initial
    for i in range(len(arrangements)-1):
        arr = Arrangement_2()
        Aos2.overlay(arrangements[i], arrangements[i+1], arr, traits)
        arrangements[i+1] = arr

    # Compute the bounding box of the arrangement and add it as edges
    min_x, max_x, min_y, max_y = calc_bbox(obstacles, [], [], radius)
    bb_arr = Aos2.Arrangement_2()
    Aos2.insert(bb_arr, [
        Curve_2(Point_2(min_x, min_y), Point_2(max_x, min_y)),
        Curve_2(Point_2(max_x, min_y), Point_2(max_x, max_y)),
        Curve_2(Point_2(max_x, max_y), Point_2(min_x, max_y)),
        Curve_2(Point_2(min_x, max_y), Point_2(min_x, min_y)),
    ])
    for face in bb_arr.faces():
        if face.is_unbounded():
            face.set_data(1)
        else:
            face.set_data(0)

    cspace = Aos2.Arrangement_2()
    Aos2.overlay(arr, bb_arr, cspace, traits)
    
    return cspace


def to_ker_point_2(point: Point_2):
    """
    Convert TPoint() to Ker.Point_2()
    """
    x, y = point.x(), point.y()
    # assert (not x.is_extended())
    # assert (not y.is_extended())
    return Ker.Point_2(x.a0(), y.a0())


def vertical_decomposition(arr):
    """
    Take an arrangement and add edges to it that represent the vertical decomposition
    """
    lst = []
    Aos2.decompose(arr, lst)
    vertical_walls = []
    for pair in lst:
        v, objects = pair
        for object in objects:
            v_point = to_ker_point_2(v.point())
            if type(object) == Vertex:
                v_other = to_ker_point_2(object.point())
                wall = Curve_2(Segment_2(v_point, v_other))
                vertical_walls.append(wall)
            if type(object) == Halfedge:
                line = Ker.Line_2(to_ker_point_2(object.source().point()), to_ker_point_2(object.target().point()))
                y_at_x = line.y_at_x(v_point.x())
                wall = Curve_2(Segment_2(v_point, Point_2(v_point.x(), y_at_x)))
                vertical_walls.append(wall)
    
    # Create an arrangement of vertical walls and overlay it
    walls_arr = Aos2.Arrangement_2()
    Aos2.insert(walls_arr, vertical_walls)
    for face in walls_arr.faces():
        face.set_data(0)
    
    traits = Arr_face_overlay_traits(lambda x, y: x + y)
    res = Aos2.Arrangement_2()
    Aos2.overlay(arr, walls_arr, res, traits)
    return res


def connectivity_graph(arr):
    """
    Get the connectivity graph from a vertical decomposition arrangement
    """
    conn_graph = nx.Graph()

    # Attach a face to each index
    face_dict = {}
    # For every edge in the graph, also add the common edge
    edge_dict = {}

    idx = 0
    for face in arr.faces():
        if face.is_unbounded() or face.data() > 0:
            face.set_data(-1)
        else:
            face.set_data(idx)
            face_dict[idx] = face
            conn_graph.add_node(idx)
            idx += 1
    
    unvisited_faces = list(face_dict.keys())

    queue = [unvisited_faces.pop(0)]
    while len(queue) > 0:
        face_idx = queue.pop(0)
        face = face_dict[face_idx]
        for edge in face.outer_ccb():
            adj_face_idx = edge.twin().face().data()
            if adj_face_idx in unvisited_faces:
                conn_graph.add_edge(face_idx, adj_face_idx)
                unvisited_faces.remove(adj_face_idx)
                queue.append(adj_face_idx)
                # Add to edge dict (both direction)
                edge_dict[(face_idx, adj_face_idx)] = edge
                edge_dict[(adj_face_idx, face_idx)] = edge

    return conn_graph, edge_dict


def find_face_index(arr, p, pl):
    """
    Get a point and find the index of the face in the arrangement that has it
    """
    p = TPoint(p.x(), p.y())
    obj = pl.locate(p)
    if obj.is_face():
        face = Face()
        obj.get_face(face)
        return face.data()
    if obj.is_halfedge():
        edge = Halfedge()
        obj.get_halfedge(edge)
        return edge.face().data()
    return -1

def find_valid_path(g_path, source, target, edge_dict, collision_detector):
    """
    Convert a graph path to a valid motion planning path.
    We do that by connecting midpoints of edges we pass - and since the arrangement
    has circle curves, we might need split some edges in half (for exact motion).
    """
    path = []
    path.append([source])
    for i in range(len(g_path)-1):
        # For every traveled edge in the connectivity graph, add its midpoint
        edge = edge_dict[(g_path[i], g_path[i+1])]
        midpoint = Ker.midpoint(to_ker_point_2(edge.source().point()), to_ker_point_2(edge.target().point()))
        # Check if we can traverse from previous point to current point
        prev_point = path[-1][0]
        if not collision_detector.is_edge_valid(Segment_2(prev_point, midpoint)):
            # If we have an intersection, split the path to two parts and slide along the normal
            mid_midpoint = Ker.midpoint(prev_point, midpoint)
            
            # Compute (unit) normal of the path
            normal = (prev_point.y().to_double() - midpoint.y().to_double(), midpoint.y().to_double() - prev_point.x().to_double())
            norm = math.sqrt(normal[0] * normal[0] + normal[1] * normal[1])
            normal = (normal[0] / norm, normal[1] / norm)

            alpha = EPS
            flag = True
            new_point = None
            while flag:
                # Try moing along and against the normal, until at least one of them works
                pos_point = Point_2(mid_midpoint.x().to_double() + alpha * normal[0], mid_midpoint.y().to_double() + alpha * normal[1])
                neg_point = Point_2(mid_midpoint.x().to_double() - alpha * normal[0], mid_midpoint.y().to_double() - alpha * normal[1])

                if collision_detector.is_edge_valid(Segment_2(prev_point, pos_point)) and collision_detector.is_edge_valid(Segment_2(pos_point, midpoint)):
                    new_point = pos_point
                    flag = False
                elif collision_detector.is_edge_valid(Segment_2(prev_point, neg_point)) and collision_detector.is_edge_valid(Segment_2(neg_point, midpoint)):
                    new_point = neg_point
                    flag = False
                else:
                    alpha += EPS
            path.append([new_point])

        path.append([midpoint])
    path.append([target])
    return path