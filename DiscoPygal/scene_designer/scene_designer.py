import os
import webbrowser

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from gui_scene_designer import GUI_scene_designer
from gui import RPolygon, RSegment, RDisc
import sys
import json
import math
import circular_room

colors = [QtCore.Qt.yellow, QtCore.Qt.green, QtCore.Qt.red, QtCore.Qt.magenta,
          QtCore.Qt.darkYellow, QtCore.Qt.darkGreen, QtCore.Qt.darkRed, QtCore.Qt.darkMagenta]
current_color_index = 0

POINT_RADIUS = 0.1
GRID_SIZE = 200
resolution = 1.0
radius = 1.0
polygon_obstacles = []
gui_polygon_obstacles = []
disc_obstacles = []
gui_disc_obstacles = []
gui_disc_obstacles_centers = []
targets = []
gui_targets = []
sources = []
radii = []
gui_sources = []
polyline = []
gui_current_polygon_vertices = []
gui_current_polygon_edges = []
grid = []
selected = {
    'index': None,
    'type': None
}


def display_help():
    webbrowser.open_new('file://' + os.path.realpath(r".\readme.md"))


def add_circular_room():
    try:
        center = polyline[-1]
        inner_radius = float(gui.get_field('innerRadius'))
        number_of_blocks = int(gui.get_field('blockNumber'))
        block_height = float(gui.get_field('blockHeight'))
        result = circular_room.get_circular_room(center, inner_radius, number_of_blocks, block_height)
        polygon_obstacles.extend(result)
        for polygon in result:
            gui_polygon_obstacles.append(
                gui.add_polygon(polygon, fill_color=QtCore.Qt.transparent, line_color=QtCore.Qt.blue))
    except Exception as e:
        print('add_circular_room:', e)
    clear_current_polyline()


def add_target(target: [float, float], radius: float):
    targets.append(target)
    gui_targets.append(
        gui.add_disc(radius, *target, fill_color=QtCore.Qt.transparent, line_color=colors[current_color_index]))


def add_source(center: [float, float], radius: float):
    sources.append(center)
    gui_sources.append(
        gui.add_disc(radius, *center, fill_color=colors[current_color_index], line_color=colors[current_color_index]))


def add_disc_obstacle(center: [float, float], radius: float):
    disc_obstacles.append({
        'center': center,
        'radius': radius
    })
    gui_disc_obstacles.append(
        gui.add_disc(radius, *center, fill_color=QtCore.Qt.transparent, line_color=QtCore.Qt.blue)
    )
    gui_disc_obstacles_centers.append(
        gui.add_disc(POINT_RADIUS, *center, fill_color=QtCore.Qt.blue, line_color=QtCore.Qt.blue)
    )


def clear_current_polyline():
    global polyline
    polyline = []
    vertex: RDisc
    for vertex in gui_current_polygon_vertices:
        gui.scene.removeItem(vertex.disc)
    gui_current_polygon_vertices.clear()
    edge: RSegment
    for edge in gui_current_polygon_edges:
        gui.scene.removeItem(edge.line)
    gui_current_polygon_edges.clear()


def submit_disc_obstacle():
    print("1")
    add_disc_obstacle(polyline[0], radius)
    clear_current_polyline()


def submit_source_target():
    global current_color_index
    radii.append(radius)
    add_source(polyline[0], radius)
    add_target(polyline[1], radius)
    current_color_index = (current_color_index + 1) % len(colors)
    clear_current_polyline()


def submit_polygon_obstacle():
    polygon_obstacles.append(polyline)
    gui_polygon_obstacles.append(gui.add_polygon(polyline, fill_color=QtCore.Qt.transparent, line_color=QtCore.Qt.blue))
    clear_current_polyline()


def cancel_current_polygon():
    polyline.clear()
    for vertex in gui_current_polygon_vertices:
        gui.scene.removeItem(vertex.disc)
    gui_current_polygon_vertices.clear()
    edge: RSegment
    for edge in gui_current_polygon_edges:
        gui.scene.removeItem(edge.line)
    gui_current_polygon_edges.clear()


def pop_vertex():
    index = len(polyline) - 1
    if 0 <= index < len(polyline):
        polyline.pop()
        gui.scene.removeItem(gui_current_polygon_vertices.pop().disc)
        if gui_current_polygon_edges:
            gui.scene.removeItem(gui_current_polygon_edges.pop().line)


def remove_selected():
    index = selected['index']
    type = selected['type']
    if type == 'polygon_obstacle':
        if index is not None and 0 <= index < len(polygon_obstacles):
            polygon_obstacles.pop(index)
            gui.scene.removeItem(gui_polygon_obstacles.pop(index).polygon)
    if type == 'disc_obstacle':
        if index is not None and 0 <= index < len(disc_obstacles):
            disc_obstacles.pop(index)
            gui.scene.removeItem(gui_disc_obstacles.pop(index).disc)
            gui.scene.removeItem(gui_disc_obstacles_centers.pop(index).disc)
    if type == 'source':
        if index is not None and 0 <= index < len(sources):
            sources.pop(index)
            targets.pop(index)
            radii.pop(index)
            gui.scene.removeItem(gui_sources.pop(index).disc)
            gui.scene.removeItem(gui_targets.pop(index).disc)
    selected['index'] = None


def select(new_selection):
    index = selected['index']
    print("previously selected:", index)
    if index is not None:
        if selected['type'] == 'polygon_obstacle' and 0 <= index < len(polygon_obstacles):
            gui.scene.removeItem(gui_polygon_obstacles[index].polygon)
            gui_polygon_obstacles[index] = gui.add_polygon(polygon_obstacles[index], fill_color=QtCore.Qt.transparent,
                                                           line_color=QtCore.Qt.blue)
        if selected['type'] == 'disc_obstacle' and 0 <= index < len(disc_obstacles):
            gui.scene.removeItem(gui_disc_obstacles[index].disc)
            gui.scene.removeItem(gui_disc_obstacles_centers[index].disc)
            gui_disc_obstacles[index] = gui.add_disc(disc_obstacles[index]['radius'], *disc_obstacles[index]['center'],
                                                     fill_color=QtCore.Qt.transparent,
                                                     line_color=QtCore.Qt.blue)
            gui_disc_obstacles_centers[index] = gui.add_disc(POINT_RADIUS, *disc_obstacles[index]['center'],
                                                     fill_color=QtCore.Qt.transparent,
                                                     line_color=QtCore.Qt.blue)
    selected['index'] = new_selection['index']
    selected['type'] = new_selection['type']
    index = selected['index']
    print("new selected:", index)
    if index is not None:
        if selected['type'] == 'polygon_obstacle' and 0 <= index < len(polygon_obstacles):
            gui.scene.removeItem(gui_polygon_obstacles[index].polygon)
            gui_polygon_obstacles[index] = gui.add_polygon(polygon_obstacles[index], fill_color=QtCore.Qt.transparent,
                                                           line_color=QtCore.Qt.darkBlue)
        if selected['type'] == 'disc_obstacle' and 0 <= index < len(disc_obstacles):
            gui.scene.removeItem(gui_disc_obstacles[index].disc)
            gui.scene.removeItem(gui_disc_obstacles_centers[index].disc)
            gui_disc_obstacles[index] = gui.add_disc(disc_obstacles[index]['radius'], *disc_obstacles[index]['center'],
                                                     fill_color=QtCore.Qt.transparent,
                                                     line_color=QtCore.Qt.darkBlue)
            gui_disc_obstacles_centers[index] = gui.add_disc(POINT_RADIUS, *disc_obstacles[index]['center'],
                                                     fill_color=QtCore.Qt.transparent,
                                                     line_color=QtCore.Qt.darkBlue)


def right_click(x: float, y: float):
    print('x', x)
    print('y', y)
    x = resolution * round(x / resolution)
    y = resolution * round(y / resolution)
    print(x, y)
    if [x, y] in polyline:
        if len(polyline) == 1:
            submit_disc_obstacle()
        elif len(polyline) == 2:
            submit_source_target()
        else:
            submit_polygon_obstacle()
        return
    polyline.append([x, y])
    gui_current_polygon_vertices.append(gui.add_disc(POINT_RADIUS, x, y, fill_color=QtCore.Qt.red))
    if len(polyline) > 1:
        gui_current_polygon_edges.append(
            gui.add_segment(*polyline[-2], *polyline[-1], line_color=QtCore.Qt.red))


def left_click(x: float, y: float):
    x = resolution * round(x / resolution)
    y = resolution * round(y / resolution)
    polygons, discs, sources = get_selected(x, y)
    new_selection = {
        'index': None,
        'type': None
    }
    if polygons:
        new_selection['index'] = polygons[-1]
        new_selection['type'] = 'polygon_obstacle'
    if discs:
        new_selection['index'] = discs[-1]
        new_selection['type'] = 'disc_obstacle'
    if sources:
        new_selection['index'] = sources[-1]
        new_selection['type'] = 'source'
    select(new_selection)


def redraw_grid(size):
    for segment in grid:
        gui.scene.removeItem(segment.line)
    grid.clear()
    color = QtCore.Qt.lightGray
    # size = int(size/RESOLUTION)
    length = size * resolution
    for i in range(-size, size):
        if i == 0:
            continue
        i = i * resolution
        grid.append(gui.add_segment(-length, i, length, i, line_color=color))
        grid.append(gui.add_segment(i, -length, i, length, line_color=color))
    color = QtCore.Qt.black
    grid.append(gui.add_segment(-length, 0, length, 0, line_color=color))
    grid.append(gui.add_segment(0, -length, 0, length, line_color=color))
    for rline in grid:
        rline.line.setZValue(-1)


def redraw_robots():
    global current_color_index
    for robot in gui_sources:
        gui.scene.removeItem(robot.disc)
    for target in gui_targets:
        gui.scene.removeItem(target.disc)
    gui_sources.clear()
    gui_targets.clear()
    for i, robot in enumerate(sources):
        gui_sources.append(gui.add_disc(radii[i], *robot, fill_color=colors[current_color_index],
                                        line_color=colors[current_color_index]))
        gui_targets.append(gui.add_disc(radii[i], *(targets[i]), fill_color=QtCore.Qt.transparent,
                                        line_color=colors[current_color_index]))
        current_color_index = (current_color_index + 1) % len(colors)


def set_up():
    redraw_grid(GRID_SIZE)
    gui.add_disc(POINT_RADIUS, 0, 0)


def export_scene():
    filename = gui.get_field('saveLocation')
    d = {'obstacles': polygon_obstacles,
         'disc_obstacles': disc_obstacles,
         'radii': radii,
         'targets': targets,
         'sources': sources}
    try:
        with open(filename, 'w') as f:
            f.write(json.dumps(d, indent=4, sort_keys=True))
            gui.textEdit.setPlainText("Scene saved to: " + filename)
    except Exception as e:
        print('Failed to write to file', filename + ':', e)
        gui.textEdit.setPlainText('Failed to write to file ' + filename + ':' + str(e))


def load_scene():
    global polyline
    global radius
    global radii
    global current_color_index
    filename = gui.get_field('scene')
    try:
        with open(filename, 'r') as f:
            d = json.load(f)
            clear()

            radius = 1.0
            if 'obstacles' in d:
                for polygon in d['obstacles']:
                    polyline = polygon
                    submit_polygon_obstacle()
            if 'disc_obstacles' in d:
                for disc_obstacle in d['disc_obstacles']:
                    radius = disc_obstacle['radius']
                    polyline = [disc_obstacle['center']]
                    submit_disc_obstacle()
            if 'radii' in d:
                radii = d['radii']
                if 'sources' in d and 'targets' in d and len(d['sources']) == len(d['targets']):
                    for i, disc in enumerate(d['sources']):
                        radius = radii[i]
                        add_source(disc, radius)
                        add_target(d['targets'][i], radius)
                        current_color_index = (current_color_index + 1) % len(colors)
        gui.textEdit.setPlainText("Scene loaded from: " + filename)
    except Exception as e:
        print('Failed to load from file', filename + ':', e)
        gui.textEdit.setPlainText('Failed to load file ' + filename + ': ' + str(e))


def clear():
    gui_polygon: RPolygon
    for gui_polygon in gui_polygon_obstacles:
        gui.scene.removeItem(gui_polygon.polygon)
    gui_polygon_obstacles.clear()
    polygon_obstacles.clear()

    for gui_disc_obstacle in gui_disc_obstacles:
        gui.scene.removeItem(gui_disc_obstacle.disc)
    gui_disc_obstacles.clear()

    for gui_disc_obstacles_center in gui_disc_obstacles_centers:
        gui.scene.removeItem(gui_disc_obstacles_center.disc)
    gui_disc_obstacles_centers.clear()

    for gui_target in gui_targets:
        gui.scene.removeItem(gui_target.disc)
    gui_targets.clear()
    targets.clear()

    for gui_disc in gui_sources:
        gui.scene.removeItem(gui_disc.disc)
    gui_sources.clear()
    sources.clear()

    radii.clear()


def set_resolution():
    global resolution
    try:
        resolution = float(gui.get_field('resolution'))
        print('Resolution set to:', resolution)
        gui.textEdit.setPlainText('Resolution set to: ' + str(resolution))
    except Exception as e:
        print('Failed to set resolution:', e)
    redraw_grid(GRID_SIZE)


def set_radius():
    global radius
    try:
        radius = float(gui.get_field('radius'))
        print('Radius set to:', radius)
        gui.textEdit.setPlainText('Radius set to: ' + str(radius))
    except Exception as e:
        print('Failed to set resolution:', e)


def undo():
    global selected
    if polyline:
        pop_vertex()


def get_selected(x, y):
    print(x, y)
    gui.textEdit.setPlainText(str((x, y)))
    polygon_indices = []
    disc_indices = []
    source_indices = []
    for i, polygon in enumerate(polygon_obstacles):
        if [x, y] in polygon:
            polygon_indices.append(i)
    for i, disc in enumerate(disc_obstacles):
        if [x, y] == disc['center']:
            disc_indices.append(i)
    for i, source in enumerate(sources):
        if [x, y] == source:
            source_indices.append(i)
    print('Polygon indices sharing the selected vertex:', *polygon_indices)
    gui.textEdit.append('Polygon indices sharing the selected vertex:' + str(polygon_indices))
    print('Disc indices sharing the selected vertex:', *disc_indices)
    gui.textEdit.append('Disc indices sharing the selected vertex:' + str(disc_indices))
    print('Source indices sharing the selected vertex:', *source_indices)
    gui.textEdit.append('Disc indices sharing the selected vertex:' + str(source_indices))
    return polygon_indices, disc_indices, source_indices


def select_button():
    try:
        select(int(gui.get_field('index')))
    except Exception as e:
        print('select_button:', e)


def get_file():
    dlg = QFileDialog()
    dlg.setFileMode(QFileDialog.AnyFile)
    dlg.setDirectory('')
    if dlg.exec_():
        filenames = dlg.selectedFiles()
        return filenames[0]


def browse_input_file():
    file_path = get_file()
    if file_path:
        gui.set_field('scene', file_path)
        load_scene()


def drop_input_file(file_path):
    gui.set_field('scene', file_path)
    load_scene()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        resolution = float(sys.argv[1])
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI_scene_designer()
    gui.scene.right_click_signal.connect(right_click)
    gui.scene.left_click_signal.connect(left_click)
    gui.mainWindow.signal_ctrl_z.connect(undo)
    gui.mainWindow.signal_delete.connect(remove_selected)
    gui.mainWindow.signal_esc.connect(cancel_current_polygon)
    gui.mainWindow.signal_drop.connect(drop_input_file)
    gui.set_animation_finished_action(lambda: None)
    gui.set_label('scene', 'Input Path:')
    gui.set_logic('load', load_scene)
    gui.set_button_text('load', 'Load Scene')
    gui.set_label('saveLocation', 'Output Path:')
    gui.set_logic('save', export_scene)
    gui.set_button_text('save', 'Save Scene')
    gui.set_label('index', 'Polygon Index:')
    gui.set_logic('select', select_button)
    gui.set_button_text('select', 'Select Polygon')
    gui.set_label('resolution', 'Resolution')
    gui.set_field('resolution', str(resolution))
    gui.set_logic('resolution', set_resolution)
    gui.set_button_text('resolution', 'Set Resolution')
    gui.set_label('innerRadius', 'Inner Radius')
    gui.set_label('blockNumber', 'Number of Blocks')
    gui.set_label('blockHeight', 'Height of Blocks')
    gui.set_button_text('addCircularRoom', 'Add Circular Room')
    gui.set_logic('addCircularRoom', add_circular_room)
    gui.set_button_text('searchScene', '..')
    gui.set_logic('searchScene', browse_input_file)
    gui.set_label('radius', 'Disc radius')
    gui.set_field('radius', str(radius))
    gui.set_button_text('radius', 'Set radius')
    gui.set_logic('radius', set_radius)
    gui.set_button_text('clear', 'Clear scene')
    gui.set_logic('clear', clear)
    gui.set_button_text('help', 'help')
    gui.set_logic('help', display_help)
    set_up()
    # add_circular_room([-1, 2], 4, 50, 1)
    gui.mainWindow.show()
    sys.exit(app.exec_())
