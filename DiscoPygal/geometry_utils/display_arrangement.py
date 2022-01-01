"""
This module can be used to display an arrangement on a new window
"""

import sys
import math

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog

from DiscoPygal.gui.gui import GUI
from DiscoPygal.bindings import *

def segment_to_qt_args(curve):
    s, t = curve.source(), curve.target()
    x1, y1 = s.x().to_double(), s.y().to_double()
    x2, y2 = t.x().to_double(), t.y().to_double()
    return x1, y1, x2, y2


def circle_segment_to_qt_args(curve):
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


def circular_edge_points(edge, n=100):
    """
    Add polygonal approximation of a circular segment to a Qt Face
    """
    r, x, y, start_angle, end_angle, clockwise = circle_segment_to_qt_args(edge.curve())

    if start_angle > end_angle:
        end_angle += 2 * math.pi

    if clockwise:
        return []

    points = []
    for i in range(0, n+1):
        angle = start_angle + (end_angle - start_angle) * i / n
        points.append((r * math.cos(angle) + x, r * math.sin(angle) + y))
    return points


class GUIArrangement(GUI):
    def setupUi(self):
        #########################
        # Setup the Qt UI layout
        #########################
        # Resize the main window and set the stylesheet (CSS)
        MainWindow = self.mainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("QMainWindow { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }\n"
                                 "QLabel { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }")

        # Add a graphics widget to the main window
        self.graphicsView = QtWidgets.QGraphicsView(self.mainWindow)
        self.graphicsView.setEnabled(True)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 780, 580))
        self.graphicsView.setObjectName("graphicsView")

        # Zoom out a bit
        self.zoom = 20


    def show_arrangement(self, arr):
        for face in arr.faces():
            if face.is_unbounded():
                continue
            points = []
            for i, edge in enumerate(face.outer_ccb()):
                if i == 0:
                    points.append((edge.source().point().x().to_double(), edge.source().point().y().to_double()))
                if edge.curve().is_circular():
                    points += circular_edge_points(edge)
                points.append((edge.target().point().x().to_double(), edge.target().point().y().to_double()))
            
            color = QtCore.Qt.green
            if face.data() < 0:
                color = QtCore.Qt.red

            self.add_polygon(points, color, QtCore.Qt.transparent)
                

        for edge in arr.edges():
            # Check that the edge has a least one incident free space
            if edge.face().data() < 0 and edge.twin().face().data() < 0:
                continue

            if edge.curve().is_circular():
                self.add_circle_segment(*circle_segment_to_qt_args(edge.curve()))
            else:
                self.add_segment(*segment_to_qt_args(edge.curve()))




def display_arrangement(arr, init_qt=False):
    """
    Get an arrangement and open a window that displays it.
    Useful for debugging.

    Set init_qt=True if there is no other Qt window running
    """
    if init_qt:
        app = QtWidgets.QApplication(sys.argv)
    gui = GUIArrangement()
    gui.set_program_name("Display Arrangement")
    gui.show_arrangement(arr)
    gui.mainWindow.show()
    if init_qt:
        sys.exit(app.exec_())


if __name__ == "__main__":
    polygon = Polygon_2([Point_2(0, 0), Point_2(0, 1), Point_2(1, 0)])
    ms = MN2.approximated_offset_2(polygon, FT(0.4), 0.01)
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
    display_arrangement(arr, True)