import os

import networkx
from importlib import util
import webbrowser

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
# from DiscoPygal.bindings import Arrangement_2
from DiscoPygal.geometry_utils import display_arrangement
from DiscoPygal.gui.Worker import Worker
from DiscoPygal.gui.logger import Writer

from gui_mrmp import GUI_mrmp
from SceneDisc import SceneDisc
from ScenePolygon import ScenePolygon
import read_input

write = None
worker = None
mode = None


def display_help():
    webbrowser.open_new('file://' + os.path.realpath(r".\readme.md"))


def set_up_scene():
    current_mode = gui.comboBoxes['mode'].currentText()
    if current_mode == 'Disc':
        disc_mode()
    elif current_mode == 'Polygon':
        polygon_mode()
    scene_file = gui.get_field('scene')
    ps.load_scene(scene_file)
    gui.zoom = 30
    gui.redraw()


def disc_mode():
    global mode
    global ps
    mode = 'disc'
    ps = SceneDisc(gui, writer)
    gui.clear_scene()
    gui.redraw()


def polygon_mode():
    global mode
    global ps
    mode = 'polygon'
    ps = ScenePolygon(gui, writer)
    gui.clear_scene()
    gui.redraw()


def stop():
    if worker is not None:
        worker.stop()
    else:
        gui.stop_queue()


def disable():
    key: str
    for key in gui.pushButtons:
        if 'search' not in key:
            gui.set_button_text(key, 'Abort')
            gui.set_logic(key, stop)


def enable():
    gui.set_button_text('help', "Help")
    gui.set_logic('help', display_help)
    gui.set_label('mode', "Mode")
    gui.set_label('scene', "Scene File (.json):")
    gui.set_logic('scene', set_up_scene)
    gui.set_button_text('scene', "Load Scene")
    gui.set_label('planner', "Planner File (.py):")
    gui.set_logic('planner', generate_path)
    gui.set_label('argument', 'Solver argument:')
    gui.set_button_text('planner', "Generate Path")
    gui.set_label('solution', "Solution File (.txt):")
    gui.set_logic('solution', load_path)
    gui.set_button_text('solution', "Load Solution")
    gui.set_label('export', "Export Solution to:")
    gui.set_logic('export', export_path)
    gui.set_button_text('export', "Export Solution")
    gui.set_logic('animate', animate_path)
    gui.set_button_text('animate', "Animate Movement Along Path")
    gui.set_logic('verify', is_path_valid)
    gui.set_button_text('verify', "Check Path Validity")
    gui.set_label('index', "Robot index:")
    gui.set_button_text('graph', "Draw robot graph")
    gui.set_logic('graph', display_graph)
    gui.set_label('graphPercent', "Percent of Edges to Display")
    gui.set_button_text('clearGraph', "Clear")
    gui.set_logic('clearGraph', clear_graph)
    gui.set_button_text('searchScene', "..")
    gui.set_logic('searchScene', set_input_file)
    gui.set_button_text('searchPlanner', "..")
    gui.set_logic('searchPlanner', set_planner_file)
    gui.set_button_text('searchSolution', "..")
    gui.set_logic('searchSolution', set_solution_file)


def generate_path():
    global worker
    ps.path = []
    gui.empty_queue()
    path_name = gui.get_field('planner')
    argument = gui.get_field('argument')
    try:
        spec = util.spec_from_file_location(path_name, path_name)
        gp = util.module_from_spec(spec)
        spec.loader.exec_module(gp)
        if mode == 'disc':
            worker = Worker(gp.generate_path_disc, ps.robots, ps.obstacles, ps.disc_obstacles,
                            ps.destinations,
                            argument,
                            writer)
        if mode == 'polygon':
            worker = Worker(gp.generate_path_polygon, ps.robots, ps.obstacles, ps.destinations,
                            argument,
                            writer)
        worker.signals.finished.connect(done)
        print('Started generating path via:', path_name, file=writer)
        disable()
        threadpool.start(worker)
    except Exception as e:
        print('generate path:', e, file=writer)


def display_graph():
    try:
        if type(ps.graph) == networkx.Graph:
            i = gui.get_field('index')
            i = int(i)
            percent = float(gui.get_field('graphPercent'))
            percent = max(0.0, percent)
            percent = min(100.0, percent)
            ps.display_graph(i, percent)
        elif type(ps.graph) == Arrangement_2:
            display_arrangement.display_arrangement(ps.graph)
    except Exception as e:
        print('Error displaying robot graph:', e, file=writer)


def clear_graph():
    ps.draw_scene()


def done(res):
    global worker
    worker = None
    if res:
        res = res[0]
        try:
            G: networkx.Graph
            ps.path = res[0]
            if type(res[1]) != networkx.Graph:
                ps.graph = res[1]
            else:
                G = res[1]
                ps.graph = G
                print('Number of nodes:', G.number_of_nodes(), file=writer)
                print('Number of edges:', G.number_of_edges(), file=writer)
        except Exception as e:
            print(e, file=writer)
    enable()


def load_path():
    ps.path = []
    try:
        gui.empty_queue()
        path_name = gui.get_field('solution')
        read_input.load_path(ps.path, path_name, ps.number_of_robots)
        print('Loaded solution from', path_name, file=writer)
    except Exception as e:
        print('load_path:', e, file=writer)
        ps.path = []
    # ps.set_up_animation()


def export_path():
    path_name = gui.get_field('export')
    try:
        with open(path_name, 'w') as f:
            for i in range(len(ps.path)):
                p = ps.path[i]
                for robot in p:
                    x = robot.x().exact()
                    y = robot.y().exact()
                    f.write(str(x.numerator()) + "/" + str(x.denominator()) + " " + str(y.numerator()) + "/" + str(
                        y.denominator()) + " ")
                if i != len(ps.path) - 1:
                    f.write('\n')
        print("Path saved to", path_name)
    except Exception as e:
        print('export_path:', e, file=writer)


def is_path_valid():
    ps.draw_path()
    ps.is_path_valid()


def animate_path():
    ps.set_up_animation()
    gui.play_queue()
    disable()


def get_file():
    dlg = QFileDialog()
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setDirectory('')
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        return filenames[0]


def set_input_file():
    file_path = get_file()
    if file_path:
        gui.set_field('scene', file_path)


def set_planner_file():
    file_path = get_file()
    if file_path:
        gui.set_field('planner', file_path)


def set_solution_file():
    file_path = get_file()
    if file_path:
        gui.set_field('solution', file_path)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    gui = GUI_mrmp()
    writer = Writer(gui.textEdit)
    gui.set_program_name("Multi-robot Motion Planning")
    i = 1
    for field in gui.lineEdits:
        if i >= len(sys.argv):
            break
        gui.set_field(field, sys.argv[i])
        i += 1
    enable()
    gui.set_field('index', str(0))
    gui.set_field('graphPercent', str(100))
    gui.mainWindow.signal_drop.connect(
        lambda path: gui.set_field('scene', path))
    threadpool = QtCore.QThreadPool()
    gui.set_animation_finished_action(enable)
    gui.comboBoxes['mode'].addItems(["Disc", "Polygon"])
    mode = 'disc'
    ps = ScenePolygon(gui, writer)
    gui.mainWindow.show()
    sys.exit(app.exec_())
