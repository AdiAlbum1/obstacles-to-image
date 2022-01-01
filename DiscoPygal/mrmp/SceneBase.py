import random

import networkx
from PyQt5.QtCore import Qt, QPointF
from conversions import point_2_to_xy
from gui_mrmp import GUI_mrmp

colors = [Qt.red, Qt.blue, Qt.magenta, Qt.yellow, Qt.green]

class SceneBase:
    def __init__(self, gui: GUI_mrmp, writer):
        self.writer = writer
        self.gui = gui
        self.number_of_robots = None
        self.robots = []
        self.obstacles = []
        self.disc_obstacles = []
        self.gui_robots = []
        self.gui_obstacles = []
        self.gui_disc_obstacles = []
        self.destinations = []
        self.gui_destinations = []
        self.path = []
        self.graph = networkx.Graph()

    def draw_scene(self):
        self.gui.clear_scene()
        self.gui_robots = []
        self.gui_obstacles = []
        self.gui_destinations = []
        for obstacle in self.obstacles:
            self.gui_obstacles.append(self.gui.add_polygon([point_2_to_xy(p) for p in obstacle.vertices()], Qt.darkGray))
        for obstacle in self.disc_obstacles:
            self.gui_disc_obstacles.append(self.gui.add_disc((obstacle['radius'].to_double()),
                                                             *point_2_to_xy(obstacle['center']), Qt.darkGray))

    def display_graph(self, i=0, percent=100.0):
        if not 0 <= i <= 3:
            raise Exception('invalid index:', i)
        self.draw_scene()
        for edge in self.graph.edges:
            if edge[0] != edge[1] and percent/100 > random.random():
                x1, y1 = edge[0][2*i].to_double(), edge[0][2*i+1].to_double()
                x2, y2 = edge[1][2*i].to_double(), edge[1][2*i+1].to_double()
                if not (x1, y1) == (x2, y2):
                    self.gui.add_segment(x1, y1, x2, y2, colors[i % len(colors)])

    def load_scene(self, filename):
        self.robots = []
        self.obstacles = []
        self.disc_obstacles = []
        self.path = []
        self.gui_robots = []
        self.gui_obstacles = []
        self.gui_disc_obstacles = []
        self.destinations = []
        self.gui_destinations = []

    def set_up_animation(self):
        self.draw_scene()
        animations = []
        if len(self.path) == 0:
            self.gui.queue_animation(self.gui.pause_animation())
            return
        if len(self.path) == 1:
            for i in range(self.number_of_robots):
                start = point_2_to_xy(self.path[0][i])
                animations.append(self.gui.linear_translation_animation(self.gui_robots[i], *start, *start))
            anim = self.gui.parallel_animation(*animations)
            self.gui.queue_animation(anim)
        else:
            for i in range(len(self.path) - 1):
                animations = []
                for j in range(self.number_of_robots):
                    start = point_2_to_xy(self.path[i][j])
                    end = point_2_to_xy(self.path[i + 1][j])
                    if start != end:
                        s = self.gui.add_segment(*start, *end, colors[j % len(colors)], opacity=0.5)
                        s.line.setZValue(2)
                    animations.append(self.gui.linear_translation_animation(self.gui_robots[j], *start, *end,
                                                                            duration=10000/len(self.path) - 1))
                anim = self.gui.parallel_animation(*animations)
                self.gui.queue_animation(anim)

    def draw_path(self):
        self.draw_scene()
        if len(self.path) > 1:
            for i in range(len(self.path) - 1):
                for j in range(self.number_of_robots):
                    start = point_2_to_xy(self.path[i][j])
                    end = point_2_to_xy(self.path[i + 1][j])
                    if start != end:
                        s = self.gui.add_segment(*start, *end, colors[j % len(colors)], opacity=0.5)
                        s.line.setZValue(2)

    def is_path_valid(self):
        pass
