from PyQt5.QtCore import Qt
from conversions import point_2_to_xy
from DiscoPygal.bindings import *
from gui_mrmp import GUI_mrmp
import read_input
from DiscoPygal.geometry_utils import collision_detection
from DiscoPygal.geometry_utils.collision_detection import Collision_detector
from SceneBase import SceneBase, colors

class SceneDisc(SceneBase):
    def __init__(self, gui: GUI_mrmp, writer):
        super(SceneDisc, self).__init__(gui, writer)

    def draw_scene(self):
        super(SceneDisc, self).draw_scene()
        for i in range(len(self.robots)):
            self.gui_robots.append(self.gui.add_disc(self.robots[i]['radius'].to_double(), *point_2_to_xy(self.robots[i]['center']),
                                                     fill_color=colors[i % len(colors)]))
        for i in range(len(self.destinations)):
            self.gui_destinations.append(
                self.gui.add_disc(self.robots[i]['radius'].to_double(), *point_2_to_xy(self.destinations[i]),
                                  line_color=colors[i % len(colors)], fill_color=Qt.transparent))

    def load_scene(self, filename):
        super(SceneDisc, self).load_scene(filename)
        try:
            self.robots, self.destinations, self.obstacles, self.disc_obstacles = read_input.read_scene(filename)
        except Exception as e:
            print('load_scene:', e, file=self.writer)
        self.number_of_robots = len(self.robots)
        self.collision_detectors = [Collision_detector(self.obstacles, self.disc_obstacles, robot['radius']) for robot in self.robots]
        if self.gui:
            self.gui.empty_queue()
            self.draw_scene()
        print("Loaded scene from", filename, file=self.writer)

    def is_path_valid(self):
        super(SceneDisc, self).is_path_valid()
        current = [robot["center"] for robot in self.robots]
        if not self.path:
            print('Empty path', file=self.writer)
            return True
        for step in self.path:
            edges = []
            for i in range(len(self.robots)):
                edges.append(Segment_2(current[i], step[i]))
            if self.is_step_valid(edges):
                current = step
            else:
                print('Invalid motion', file=self.writer)
                return False
        print('Valid motion', file=self.writer)
        for i in range(len(self.robots)):
            if current[i] != self.destinations[i]:
                print('Robot', i, 'at', current[i], 'did not reach its destination at', self.destinations[i],
                      file=self.writer)
        return True

    def is_step_valid(self, edges):
        # check intersection with robots
        if collision_detection.check_intersection_against_robots(edges, [robot['radius'] for robot in self.robots]):
            print('Intersection with robots', file=self.writer)
            return False
        # check intersection with obstacles
        for i, edge in enumerate(edges):
            if not self.collision_detectors[i].is_edge_valid(edge):
                print("Intersection with obstacles", file=self.writer)
                return False
        return True
