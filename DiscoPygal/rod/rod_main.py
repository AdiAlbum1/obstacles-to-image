import webbrowser
import os
from importlib import util
import networkx
from PyQt5 import QtCore

from PyQt5.QtWidgets import QFileDialog

from gui.Worker import Worker
from gui.logger import Writer
from gui_rod import GUI_rod, QtWidgets
import read_input
from Polygons_scene import Polygons_scene
worker = None
writer = None

def display_help():
    webbrowser.open_new('file://' + os.path.realpath(r".\readme.md"))

def set_up_scene():
    gui.clear_scene()
    scene_file = gui.get_field('scene')
    ps.load_scene(scene_file)


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
    gui.set_label('mode', "Mode:")
    gui.set_label('scene', "Scene File (.json):")
    gui.set_logic('scene', set_up_scene)
    gui.set_button_text('scene', "Load Scene")
    gui.set_label('planner', "Planner File (.py):")
    gui.set_logic('planner', generate_path)
    gui.set_button_text('planner', "Generate Path")
    gui.set_button_text('searchScene', "..")
    gui.set_logic('searchScene', set_input_file)
    gui.set_button_text('searchPlanner', "..")
    gui.set_logic('searchPlanner', set_planner_file)
    gui.set_label('argument', "Solver argument")
    gui.set_label('solution', "Solution File (.txt):")
    gui.set_logic('solution', load_path)
    gui.set_button_text('solution', "Load Solution")
    gui.set_button_text('searchSolution', "..")
    gui.set_logic('searchSolution', set_solution_file)
    gui.set_label('export', "Export solution:")
    gui.set_logic('export', export_path)
    gui.set_button_text('export', "Export")
    gui.set_button_text('animate', "Animate Movement Along Path")
    gui.set_logic('animate', animate_path)
    gui.set_button_text('verify', "Check Path Validity")
    gui.set_logic('verify', is_path_valid)


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
        worker = Worker(gp.generate_path, ps.length, ps.obstacles, ps.origin,
                        ps.destination, argument, writer)
        worker.signals.finished.connect(done)
        print('Started generating path via:', path_name, file=writer)
        disable()
        threadpool.start(worker)
    except Exception as e:
        print('generate path:', e, file=writer)


def done(res):
    global worker
    worker = None
    if res:
        ps.path = res[0]
    enable()


def load_path():
    ps.path = []
    try:
        gui.empty_queue()
        path_name = gui.get_field('solution')
        read_input.load_path(ps.path, path_name)
        print("Loaded path from:", path_name, file=writer)
    except Exception as e:
        print('load_path:', e, file=writer)
        ps.path = []
    # ps.set_up_animation()

def export_path():
    path_name = gui.get_field('export')
    try:
        read_input.save_path(ps.path, path_name)
        print("path saved to:", path_name, file=writer)
    except Exception as e:
        print('save_path:', e, file=writer)

def is_path_valid():
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
    gui = GUI_rod()
    writer = Writer(gui.textEdit)
    gui.set_program_name("Robot Motion Planning - Rod")
    i = 1
    for field in gui.lineEdits:
        if i >= len(sys.argv):
            break
        gui.set_field(field, sys.argv[i])
        i += 1
    gui.comboBoxes['mode'].addItems(["Rod"])
    enable()
    gui.mainWindow.signal_drop.connect(lambda path: gui.set_field('scene', path))
    threadpool = QtCore.QThreadPool()
    gui.set_animation_finished_action(enable)
    ps = Polygons_scene(gui, writer)
    gui.mainWindow.show()
    sys.exit(app.exec_())
