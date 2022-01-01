import os
import webbrowser

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from gui_scene_designer_rod import *
from gui import RPolygon, RSegment, RDisc
import sys
import json
import math
import circular_room


def setup_logic(gui):
    """
    Connect and setup the logic of the UI
    """
    gui.set_logic('mode_polygonal_obstacles', lambda: set_drawing_mode(gui, MODE_POLYGON_OBSTACLE))
    gui.set_logic('mode_rod_start', lambda: set_drawing_mode(gui, MODE_ROD_START))
    gui.set_logic('mode_rod_end', lambda: set_drawing_mode(gui, MODE_ROD_END))
    set_drawing_mode(gui, MODE_POLYGON_OBSTACLE) # This is the default mode
    gui.set_animation_finished_action(lambda: None)

    # Set resolution
    gui.set_field('resolution', str(gui.resolution))
    gui.set_logic('set_resolution', lambda: set_resolution(gui))

    # Set rod defaults
    gui.set_field('rod_start_angle', str(gui.rod_start_angle))
    gui.set_field('rod_end_angle', str(gui.rod_end_angle))
    gui.set_field('rod_length', str(gui.rod_length))
    gui.set_logic('update_rod', lambda: update_rod(gui))

    gui.set_logic('help', display_help)
    gui.set_logic('clear', lambda: clear(gui))
    gui.set_logic('select_polygon', lambda: select_button(gui))
    gui.set_logic('save', lambda: export_scene(gui))
    gui.set_logic('load', lambda: load_scene(gui))
    gui.set_logic('file_browser', lambda: browse_input_file(gui))

    gui.scene.right_click_signal.connect(lambda x, y: right_click(gui, x, y))
    gui.scene.left_click_signal.connect(lambda x, y: left_click(gui, x, y))
    gui.mainWindow.signal_ctrl_z.connect(lambda: undo(gui))
    gui.mainWindow.signal_delete.connect(lambda: remove_selected(gui))
    gui.mainWindow.signal_esc.connect(lambda: clear_current_polyline(gui))
    gui.mainWindow.signal_drop.connect(lambda file_path: drop_input_file(gui, file_path))



def display_help():
    webbrowser.open_new('file://' + os.path.realpath(r".\\readme.md"))


def clear(gui):
    for obstacle in gui.gui_polygon_obstacles:
        gui.scene.removeItem(obstacle.polygon)
    gui.gui_polygon_obstacles.clear()
    gui.polygon_obstacles.clear()

    gui.rod_start = ()
    if gui.gui_rod_start is not None:
        gui.scene.removeItem(gui.gui_rod_start.line)
    gui.gui_rod_start = None

    gui.rod_end = ()
    if gui.gui_rod_end is not None:
        gui.scene.removeItem(gui.gui_rod_end.line)
    gui.gui_rod_end = None

    clear_current_polyline(gui)


def redraw_grid(gui, size):
    """
    Draw a grid in out gui with a given size
    """
    # Clear any old grid lines
    for segment in gui.grid:
        gui.scene.removeItem(segment.line)
    gui.grid.clear()

    length = size * gui.resolution
    
    # Draw the grid lines (except axes)
    color = QtCore.Qt.lightGray
    for i in range(-size, size):
        if i == 0:
            continue # We will draw the main axes sperately
        i = i * gui.resolution
        gui.grid.append(gui.add_segment(-length, i, length, i, line_color=color))
        gui.grid.append(gui.add_segment(i, -length, i, length, line_color=color))
    
    # Draw the axes
    color = QtCore.Qt.black
    gui.grid.append(gui.add_segment(-length, 0, length, 0, line_color=color))
    gui.grid.append(gui.add_segment(0, -length, 0, length, line_color=color))

    # Push all segments to the back
    for segment in gui.grid:
        segment.line.setZValue(-1)

def set_drawing_mode(gui, mode):
    """
    Set the drawing mode to the selected one
    (And update the labels accordingly)
    """
    gui.drawing_mode = mode
    gui.set_label('current_mode', MODE_CURRENT_STR.format(MODE_NAMES[mode])) 
    clear_current_polyline(gui)


def clear_current_polyline(gui):
    """
    Discard the polygline we currently draw
    """
    gui.polyline = []
    for vertex in gui.gui_current_polygon_vertices:
        gui.scene.removeItem(vertex.disc)
    gui.gui_current_polygon_vertices.clear()
    for edge in gui.gui_current_polygon_edges:
        gui.scene.removeItem(edge.line)
    gui.gui_current_polygon_edges.clear()


def submit_polygon_obstacle(gui):
    gui.polygon_obstacles.append(gui.polyline)
    gui.gui_polygon_obstacles.append(
        gui.add_polygon(gui.polyline, fill_color=QtCore.Qt.transparent, line_color=QtCore.Qt.blue)
    )
    clear_current_polyline(gui)


def submit_rod_start(gui, x, y):
    gui.rod_start = (x, y)
    if gui.gui_rod_start is not None:
        # Delete previous rod
        gui.scene.removeItem(gui.gui_rod_start.line)
    gui.gui_rod_start = gui.add_segment_angle(x, y, gui.rod_length, math.radians(gui.rod_start_angle), line_color=QtCore.Qt.red)


def submit_rod_end(gui, x, y):
    gui.rod_end = (x, y)
    if gui.gui_rod_end is not None:
        # Delete previous rod
        gui.scene.removeItem(gui.gui_rod_end.line)
    gui.gui_rod_end = gui.add_segment_angle(x, y, gui.rod_length, math.radians(gui.rod_end_angle), line_color=QtCore.Qt.green)


def update_rod(gui):
    gui.rod_length = float(gui.get_field('rod_length'))
    gui.rod_start_angle = float(gui.get_field('rod_start_angle'))
    gui.rod_end_angle = float(gui.get_field('rod_end_angle'))
    if gui.gui_rod_start is not None:
        submit_rod_start(gui, *gui.rod_start)
    if gui.gui_rod_end is not None:
        submit_rod_end(gui, *gui.rod_end)


def right_click(gui, x: float, y: float):
    """
    Handles the drawing by user (which is done by right clicking)
    """
    print('x', x, 'y', y)
    x = gui.resolution * round(x / gui.resolution)
    y = gui.resolution * round(y / gui.resolution)
    print(x, y)
    if gui.drawing_mode == MODE_POLYGON_OBSTACLE:
        if [x, y] in gui.polyline:
            if len(gui.polyline) >= 3:
                submit_polygon_obstacle(gui)
            else:
                clear_current_polyline(gui)
        else:
            gui.polyline.append([x,y])
            gui.gui_current_polygon_vertices.append(gui.add_disc(POINT_RADIUS, x, y, fill_color=QtCore.Qt.red))
            if len(gui.polyline) > 1:
                gui.gui_current_polygon_edges.append(
                    gui.add_segment(*gui.polyline[-2], *gui.polyline[-1], line_color=QtCore.Qt.red))
    elif gui.drawing_mode == MODE_ROD_START:
        submit_rod_start(gui, x, y)
    elif gui.drawing_mode == MODE_ROD_END:
        submit_rod_end(gui, x, y)


def left_click(gui, x: float, y: float):
    x = gui.resolution * round(x / gui.resolution)
    y = gui.resolution * round(y / gui.resolution)
    polygons = get_selected(gui, x, y)
    new_selection = {
        'index': None,
        'type': None
    }
    if polygons:
        new_selection['index'] = polygons[-1]
        new_selection['type'] = MODE_POLYGON_OBSTACLE
    select(gui, new_selection)


def get_selected(gui, x, y):
    print(x, y)
    gui.writerEdit.setPlainText(str((x, y)))
    polygon_indices = []
    disc_indices = []
    source_indices = []
    for i, polygon in enumerate(gui.polygon_obstacles):
        if [x, y] in polygon:
            polygon_indices.append(i)
    print('Polygon indices sharing the selected vertex:', *polygon_indices)
    gui.writerEdit.append('Polygon indices sharing the selected vertex:' + str(polygon_indices))
    return polygon_indices


def select(gui, new_selection):
    index = gui.selected['index']
    print("previously selected:", index)
    if index is not None:
        if gui.selected['type'] == MODE_POLYGON_OBSTACLE and 0 <= index < len(gui.polygon_obstacles):
            gui.scene.removeItem(gui.gui_polygon_obstacles[index].polygon)
            gui.gui_polygon_obstacles[index] = gui.add_polygon(gui.polygon_obstacles[index], fill_color=QtCore.Qt.transparent,
                                                           line_color=QtCore.Qt.blue)

    gui.selected['index'] = new_selection['index']
    gui.selected['type'] = new_selection['type']
    index = gui.selected['index']
    print("new selected:", index)
    if index is not None:
        if gui.selected['type'] == MODE_POLYGON_OBSTACLE and 0 <= index < len(gui.polygon_obstacles):
            gui.scene.removeItem(gui.gui_polygon_obstacles[index].polygon)
            gui.gui_polygon_obstacles[index] = gui.add_polygon(gui.polygon_obstacles[index], fill_color=QtCore.Qt.transparent,
                                                           line_color=QtCore.Qt.darkBlue)


def undo(gui):
    index = len(gui.polyline) - 1
    if 0 <= index < len(gui.polyline):
        gui.polyline.pop()
        gui.scene.removeItem(gui.gui_current_polygon_vertices.pop().disc)
        if len(gui.gui_current_polygon_edges) > 0:
            gui.scene.removeItem(gui.gui_current_polygon_edges.pop().line)


def remove_selected(gui):
    index = gui.selected['index']
    type = gui.selected['type']
    if type == MODE_POLYGON_OBSTACLE:
        if index is not None and 0 <= index < len(gui.polygon_obstacles):
            gui.polygon_obstacles.pop(index)
            gui.scene.removeItem(gui.gui_polygon_obstacles.pop(index).polygon)
    gui.selected['index'] = None


def select_button(gui):
    try:
        select(gui, {
            'index': int(gui.get_field('polygon_index')),
            'type': MODE_POLYGON_OBSTACLE
        })
    except Exception as e:
        print('select_button:', e)


def set_resolution(gui):
    try:
        gui.resolution = float(gui.get_field('resolution'))
        print('Resolution set to:', gui.resolution)
        gui.writerEdit.setPlainText('Resolution set to: ' + str(gui.resolution))
    except Exception as e:
        print('Failed to set resolution:', e)
    redraw_grid(gui, GRID_SIZE)


def export_scene(gui):
    filename = gui.get_field('output_path')
    if len(gui.rod_start) == 0 or len(gui.rod_end) == 0:
        print("No rod start, end. Cannot save scene.")
        gui.writerEdit.setPlainText("No rod start, end. Cannot save scene.")
        return
    d = {
        'obstacles': gui.polygon_obstacles,
        'origin': [gui.rod_start[0], gui.rod_start[1], math.radians(gui.rod_start_angle)],
        'destination': [gui.rod_end[0], gui.rod_end[1], math.radians(gui.rod_end_angle)],
        'length': gui.rod_length,
    }
    try:
        with open(filename, 'w') as fp:
            fp.write(json.dumps(d, indent=4, sort_keys=True))
            gui.writerEdit.setPlainText("Scene saved to: " + filename)
    except Exception as e:
        print("Failed to write to file", filename + ":", e)
        gui.writerEdit.setPlainText("Failed to write to file " + filename + ": " + str(e))


def load_scene(gui):
    filename = gui.get_field('input_path')
    try:
        with open(filename, 'r') as fp:
            d = json.load(fp)
        clear(gui)
        for polygon in d['obstacles']:
            gui.polyline = polygon
            submit_polygon_obstacle(gui)
        start_x, start_y, start_angle = d['origin']
        end_x, end_y, end_angle = d['destination']
        
        submit_rod_start(gui, start_x, start_y)
        submit_rod_end(gui, end_x, end_y)

        gui.set_field('rod_length', str(d['length']))
        gui.set_field('rod_start_angle', str(math.degrees(start_angle)))
        gui.set_field('rod_end_angle', str(math.degrees(end_angle)))
        update_rod(gui)
        
    except Exception as e:
        print("Failed to load from file " + filename + ": " + str(e))
        gui.writerEdit.setPlainText("Failed to load from file " + filename + ": " + str(e))


def get_file():
    dlg = QFileDialog()
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setDirectory('')
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        return filenames[0]


def browse_input_file(gui):
    file_path = get_file()
    if file_path:
        gui.set_field('input_path', file_path)


def drop_input_file(gui, file_path):
    gui.set_field('input_path', file_path)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI_scene_designer_rod()
    gui.set_program_name("Robot Motion Planning - Rod - Scene Designer")

    setup_logic(gui)

    # Draw the grid with origin
    redraw_grid(gui, GRID_SIZE)
    gui.add_disc(POINT_RADIUS, 0, 0)

    gui.mainWindow.show()
    sys.exit(app.exec_())