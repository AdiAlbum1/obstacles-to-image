import time
import math

import networkx as nx

from bindings import *
from geometry_utils import collision_detection
from geometry_utils import display_arrangement

# Epsilon for approximated offset
EPS = 0.0001

class LinearSegment(object):
    """
    Class that represents a linear segment, which is defined uniquely by
    its endpoints.

    You may add any helper methods for this class.
    """
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    

class CircleSegment(object):
    """
    Class that represents a circular segment, which is defined uniquely by
    its radius, center, start-end angles and orientation (clockwise or counter-clockwise).

    You may add any helper methods for this class.
    """
    def __init__(self, r, x, y, start_angle, end_angle, clockwise):
        self.r = r
        self.x = x
        self.y = y
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.clockwise = clockwise


class ArrangementFace(object):
    """
    Class that represents an arrangement face, which is made of a sequence of segments
    (either linear or circle, and each face might have both kinds)
    Also each face has an index, for convinience.

    You may add any helper methods for this class.
    """
    def __init__(self, edges, index):
        self.edges = edges
        self.index = index


def generate_path_from_connectivity_graph(conn_graph, start, end, collision_detector):
    """
    #################
    # Excersice 4.1
    #################
    Here is the function that you need to write (you can only modify the code in this region).
    
    The inputs are a connectivity graph and start and end points. The goal is to return 
    a list of points, that form a valid motion path for a robot among the obstacels - and minimize the
    vertical clearence measure.

    Start and end are tuples of the form (x, y) where x, y are floats.

    You also get a collision detector object which you might use for planning a path.

    The connectivity graph, `conn_graph`, is a networkx.Graph object. The vertices are of 
    type ArrangementFace and the "common_edge" property is of type LinearSegment or CircleSegment
    (see the above classes).

    To get the "common_edge" property for some two faces, you can use:
    (where f1, f2 are vertices in the graph, i.e. of type ArrangementFace)

    >>> ... = conn_graph.get_edge_data(f1, f2)["common_edge"]
    
    The output should be a list of list of points, of the form:
    [
        [(x1, y1)],
        [(x2, y2)],
        ...
    ]
    If there is no path, return an empty list.

    You may add helper functions to the classes above, to this function of global methods which will
    be *directly* under this method. 

    You might also take a look at the function `generate_path_disc`, which is called by the mrmp software.
    You can also look at the exact planner which uses the CGAL implementation, but remember you goal here is 
    to minimize the vertical clearence.

    A (very) incorrect example solution is appended below, to explain further the syntax.
    """
    path = []
    for edge in conn_graph.edges: # just randomly traverse all the edges of the graph
        common_edge = conn_graph.get_edge_data(edge[0], edge[1])["common_edge"]
        if type(common_edge) is LinearSegment:
            point = ((common_edge.x1 + common_edge.x2)/2, (common_edge.y1 + common_edge.y2)/2) # midpoint
        elif type(common_edge) is CircleSegment:
            point = (common_edge.x, common_edge.y)
        path.append([point])
    return path



##########################################
# PUT ANY HELPER METHODS IN THIS SECTION
##########################################







##########################################
##########################################
##########################################



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

    start = (source.x().to_double(), source.y().to_double())
    end = (target.x().to_double(), target.y().to_double())

    #############################################################
    # Generate the vertical decomposition and connectivity graph
    #############################################################

    arr = construct_cspace(obstacles, radius)
    arr = vertical_decomposition(arr)
    cgal_conn_graph, edge_dict, face_dict = connectivity_graph(arr)
    conn_graph = convert_connectivity_graph_from_cgal(cgal_conn_graph, edge_dict, face_dict)
    collision_detector = collision_detection.Collision_detector(obstacles, [], radius)

    #####################################
    # Generate a path from start to end
    #####################################
    
    path = generate_path_from_connectivity_graph(conn_graph, start, end, collision_detector)

    if len(path) == 0:
        print("Could not find a path from source to dest", file=writer)

    # Convert each point back to CGAL format
    path = [[Point_2(point[0][0], point[0][1])] for point in path]
    
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

    return conn_graph, edge_dict, face_dict


def circle_segment_to_args(curve):
    s, t = curve.source(), curve.target()
    x1, y1 = s.x().to_double(), s.y().to_double()
    x2, y2 = t.x().to_double(), t.y().to_double()
    circle: Circle_2 = curve.supporting_circle()
    ori = curve.orientation()
    clockwise = False
    if ori == CGALPY.Ker.CLOCKWISE:
        clockwise = True
    center = circle.center()
    x = center.x().to_double()
    y = center.y().to_double()
    r = math.sqrt(circle.squared_radius().to_double())
    tmp = (x1-x)/r
    if tmp > 1:
        tmp = 0.999
    if tmp < -1:
        tmp = -0.999
    start_angle = math.acos(tmp)
    if y1 < y:
        start_angle = -start_angle
    tmp = (x2-x)/r
    if tmp > 1:
        tmp = 0.999
    if tmp < -1:
        tmp = -0.999
    end_angle = math.acos(tmp)
    if y2 < y:
        end_angle = -end_angle
    if start_angle < 0:
        start_angle += 2*math.pi
    if end_angle < 0:
        end_angle += 2*math.pi
    return r, x, y, start_angle, end_angle, clockwise


def linear_segment_to_args(curve):
    s, t = curve.source(), curve.target()
    x1, y1 = s.x().to_double(), s.y().to_double()
    x2, y2 = t.x().to_double(), t.y().to_double()
    return x1, y1, x2, y2


def cgal_edge_to_python_edge(edge):
    if edge.curve().is_circular():
        return CircleSegment(*circle_segment_to_args(edge.curve()))
    return LinearSegment(*linear_segment_to_args(edge.curve()))


def cgal_face_to_python_face(face):
    edges = []
    for edge in face.outer_ccb():
        edges.append(cgal_edge_to_python_edge(edge))        
    return ArrangementFace(edges, face.data())
    

def convert_connectivity_graph_from_cgal(conn_graph, edge_dict, face_dict):
    """
    Convert a CGAL connectivity graph to Python native one
    """
    new_graph = nx.Graph()
    for edge in conn_graph.edges:
        f1 = cgal_face_to_python_face(face_dict[edge[0]])
        f2 = cgal_face_to_python_face(face_dict[edge[1]])
        common_edge = cgal_edge_to_python_edge(edge_dict[(f1.index, f2.index)])
        new_graph.add_edge(f1, f2, common_edge=common_edge)
    return new_graph