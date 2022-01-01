from PyQt5.QtCore import Qt
from conversions import point_2_to_xy, polygon_2_to_tuples_list
from gui_mrmp import GUI_mrmp
import read_input
import ms_polygon_segment
import linear_path_intersection_test
from SceneBase import SceneBase, colors
from bindings import *

offset = -Vector_2(FT(0), FT(0))


class ScenePolygon(SceneBase):
    def __init__(self, gui: GUI_mrmp, writer):
        super(ScenePolygon, self).__init__(gui, writer)

    def draw_scene(self):
        super(ScenePolygon, self).draw_scene()
        for i in range(len(self.robots)):
            self.gui_robots.append(
                self.gui.add_polygon([point_2_to_xy(p) for p in self.robots[i]], fill_color=colors[i % len(colors)]))
        for i in range(len(self.destinations)):
            self.gui_destinations.append(
                self.gui.add_polygon([point_2_to_xy(self.destinations[i] + Vector_2(self.robots[i][0], p))
                                      for p in self.robots[i]], line_color=colors[i % len(colors)],
                                     fill_color=Qt.transparent))

    def load_scene(self, filename):
        self.robots = []
        self.obstacles = []
        self.path = []
        self.gui_robots = []
        self.gui_obstacles = []
        self.destinations = []
        self.gui_destinations = []
        try:
            self.robots, self.destinations, self.obstacles = read_input.read_scene(filename, mode='polygon')
        except Exception as e:
            print('load_scene:', e, file=self.writer)
        self.number_of_robots = len(self.robots)
        if self.gui:
            self.gui.empty_queue()
            self.draw_scene()
        print("Loaded scene from", filename, file=self.writer)

    def is_path_valid(self):
        robot_intersection_check = True
        if self.path is None:
            return False
        if len(self.path) == 0:
            return False
        robot_polygons = []
        path_polygons = []
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                source = []
                target = []
                for j in range(self.number_of_robots):
                    robot_polygons.append(Polygon_2(self.robots[j]))
                    source.append(self.path[i][j])
                    target.append(self.path[i + 1][j])
                    if source[j] != target[j]:
                        s = Segment_2(source[j], target[j])
                        pwh = ms_polygon_segment.minkowski_sum_polygon_segment(robot_polygons[j], s)
                    else:
                        pwh = ms_polygon_segment.minkowski_sum_polygon_point(robot_polygons[j], source[j])
                    path_polygons.append(pwh)
                # check that the robots don't intersect each other while performing the current step
                for j in range(self.number_of_robots):
                    for k in range(self.number_of_robots):
                        if j != k:
                            if linear_path_intersection_test.do_intersect(robot_polygons[j], robot_polygons[k], source,
                                                                          target):
                                robot_intersection_check = False

        obstacle_polygons = []
        for obs in self.obstacles:
            p = Polygon_2(obs)
            obstacle_polygons.append(p)

        path_set = Polygon_set_2()
        path_set.join_polygons_with_holes(path_polygons)
        obstacles_set = Polygon_set_2()
        obstacles_set.join_polygons(obstacle_polygons)

        lst = []
        path_set.polygons_with_holes(lst)
        for pwh in lst:
            p = pwh.outer_boundary()
            lst = polygon_2_to_tuples_list(p)
            if self.gui:
                self.gui.add_polygon(lst, fill_color=Qt.lightGray).polygon.setZValue(-3)
                for p in pwh.holes():
                    lst = polygon_2_to_tuples_list(p)
                    self.gui.add_polygon(lst, fill_color=Qt.white).polygon.setZValue(-2)

        # check that the origin matches the first point in the path
        origin_check = True
        for i in range(self.number_of_robots):
            if self.robots[i][0] - offset != self.path[0][i]:
                origin_check = False
        # check that the destination matches the last point in the path
        destination_check = True
        for i in range(self.number_of_robots):
            if self.destinations[i] != self.path[-1][i]:
                destination_check = False
        # check that there are no collisions with obstacles
        obstacle_intersection_check = True if not path_set.do_intersect(obstacles_set) else False
        res = (origin_check and destination_check and obstacle_intersection_check and robot_intersection_check)
        print("Valid path: ", res, file=self.writer)
        if not origin_check:
            print("Origin mismatch", file=self.writer)
        if not destination_check:
            print("Destination mismatch", file=self.writer)
        if not obstacle_intersection_check:
            print("Movement along path intersects with obstacles", file=self.writer)
        if not robot_intersection_check:
            print("The robots intersect each other", file=self.writer)
        return res
