from bindings import *

from gui.gui import QtCore
from geometry_utils.collision_detection import Collision_detector
import read_input
from conversions import point_2_to_xy, tuples_list_to_polygon_2, polygon_2_to_tuples_list, path_point_to_xyzd

FT, Gmpq, Point_2, Vector_2 = Ker.FT, Ker.Gmpq, Ker.Point_2, Ker.Vector_2
Polygon_set_2 = BSO2.Polygon_set_2

class Polygons_scene():
    def __init__(self, gui, writer):
        self.writer = writer
        self.gui = gui
        self.origin = None
        self.length = None
        self.obstacles = []
        self.gui_robot = None
        self.gui_obstacles = []
        self.destination = None
        self.gui_destination = None
        self.path = []

    def draw_scene(self):
        self.gui.clear_scene()
        if self.origin is not None:
            x1 = FT(Gmpq(self.origin[0])).to_double()
            y1 = FT(Gmpq(self.origin[1])).to_double()
            a = FT(Gmpq(self.origin[2])).to_double()
            l = self.length.to_double()
            self.gui_robot = self.gui.add_segment_angle(
                x1, y1, l, a, QtCore.Qt.red)
        for obstacle in self.obstacles:
            self.gui_obstacles = []
            self.gui_obstacles.append(
                self.gui.add_polygon(obstacle, QtCore.Qt.darkGray))
        if self.destination is not None:
            # self.gui_destination = gui.add_disc(4, int(self.destination[0]), int(self.destination[1]), Qt.green)
            da = FT(Gmpq(self.destination[2])).to_double()
            self.gui_destination = self.gui.add_segment_angle(int(self.destination[0]), int(self.destination[1]), l, da,
                                                              QtCore.Qt.green)

    def load_scene(self, filename):
        self.length = None
        self.origin = None
        self.obstacles = []
        self.path = []
        self.destination = None
        try:
            self.length, self.origin, destination, self.obstacles = read_input.read_scene(
                filename)
            self.set_destination(destination)
        except Exception as e:
            print('load_scene:', e, file=self.writer)
        self.gui.empty_queue()
        self.draw_scene()
        print("Loaded scene from", filename, file=self.writer)

    def save_scene(self, filename):
        def polygon_to_string(polygon):
            s = " "
            temp = [str(i) for i in polygon]
            line = str(len(polygon)) + " " + s.join(temp)
            line = line.replace("(", "")
            line = line.replace(")", "")
            line = line.replace(",", "")
            return line

        file = open(filename, 'w')
        # print(self.destination)
        line = str(self.destination[0]) + " " + str(self.destination[1])
        file.write(line + '\n')
        line = polygon_to_string(self.origin)
        file.write(line)
        if len(self.obstacles) > 0:
            file.write('\n')
        for i in range(len(self.obstacles)):
            line = polygon_to_string(self.obstacles[i])
            if i != len(self.obstacles) - 1:
                line = line + '\n'
            file.write(line)
        file.close()

    def set_destination(self, destination):
        dx1 = destination[0]
        dy1 = destination[1]
        da = destination[2]
        self.destination = (dx1, dy1, da)
        if self.gui_destination is not None:
            self.gui.scene.removeItem(self.gui_destination.line)
            # self.gui.add_disc(4, int(dx1), int(dy1), Qt.green)
        l = self.length.to_double()
        da = FT(Gmpq(self.destination[2])).to_double()
        self.gui_destination = self.gui.add_segment(int(self.destination[0]), int(self.destination[1]), l, da,
                                                    QtCore.Qt.green)

    def set_up_animation(self):
        self.draw_scene()
        if len(self.path) == 0:
            self.gui.queue_animation(self.gui.pause_animation())
            return
        if len(self.path) == 1:
            start = path_point_to_xyzd(self.path[0])
            anim = self.gui.segment_angle_animation(
                self.gui_robot, start[0], start[1], start[2], *start)
            self.gui.queue_animation(anim)
        else:
            for i in range(len(self.path) - 1):
                start = path_point_to_xyzd(self.path[i])
                end = path_point_to_xyzd(self.path[i + 1])
                anim = self.gui.segment_angle_animation(self.gui_robot, start[0], start[1], start[2],
                                                        *end, duration=10000/len(self.path) - 1)
                self.gui.queue_animation(anim)

    def is_path_valid(self):
        if self.path is None:
            return False
        if len(self.path) == 0:
            return False

        obstacle_polygons = []
        for obs in self.obstacles:
            p = tuples_list_to_polygon_2(obs)
            obstacle_polygons.append(p)
        
        epsilon = FT(0.1)

        cd = Collision_detector(obstacle_polygons, [], epsilon)

        x1 = FT(Gmpq(self.origin[0]))
        y1 = FT(Gmpq(self.origin[1]))
        a = FT(Gmpq(self.origin[2]))

        dx1 = FT(Gmpq(self.destination[0]))
        dy1 = FT(Gmpq(self.destination[1]))
        da = FT(Gmpq(self.destination[2]))

        # check that the origin matches the first point in the path
        check0 = (x1 == self.path[0][0] and y1 == self.path[0][1] and a == self.path[0][2])
        # check that the destination matches the last point in the path
        check1 = (dx1 == self.path[-1][0] and dy1 ==
                          self.path[-1][1] and da == self.path[-1][2])
        # check that there are no collisions
        check2 = all(cd.is_rod_motion_valid(self.path[i], self.path[i+1], self.path[i+1][3], self.length) for i in range(len(self.path) - 1))
        res = (check0 and check1 and check2)
        print("Valid path: ", res, file=self.writer)
        if not check0:
            print("Origin mismatch:", file=self.writer)
            print(self.path[0][0], self.path[0][1],
                  self.path[0][2], file=self.writer)
            print("Does not match origin:", file=self.writer)
            print(x1, y1, a, file=self.writer)
        if not check1:
            print("Destination mismatch:", file=self.writer)
            print(self.path[-1][0], self.path[-1][1],
                  self.path[-1][2], file=self.writer)
            print("Does not match destination:", file=self.writer)
            print(dx1, dy1, da, file=self.writer)
        if not check2:
            print("Movement along path intersects with obstacles", file=self.writer)
        return res
