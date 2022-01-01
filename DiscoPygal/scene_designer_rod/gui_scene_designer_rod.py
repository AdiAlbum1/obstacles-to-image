from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal
from gui.gui import GUI, MainWindowPlus

POINT_RADIUS = 0.1
GRID_SIZE = 200

MODE_POLYGON_OBSTACLE = 0
MODE_ROD_START = 1
MODE_ROD_END = 2

MODE_NAMES = {
    MODE_POLYGON_OBSTACLE: 'Polygonal Obstacles',
    MODE_ROD_START: 'Rod Start Position',
    MODE_ROD_END: 'Rod End Position'
}

MODE_CURRENT_STR = '(Current Mode: {})'

class MainWindowSceneDesigner(MainWindowPlus):
    signal_ctrl_z = pyqtSignal()
    signal_delete = pyqtSignal()
    signal_esc = pyqtSignal()
    signal_drop = pyqtSignal(str)

    def __init__(self, gui):
        super().__init__(gui)

    # Adjust zoom level/scale on +/- key press
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Plus:
            self.gui.zoom /= 0.9
        if event.key() == QtCore.Qt.Key_Minus:
            self.gui.zoom *= 0.9
        if event.modifiers() & QtCore.Qt.ControlModifier and event.key() == QtCore.Qt.Key_Z:
            self.signal_ctrl_z.emit()
        if event.key() == QtCore.Qt.Key_Escape:
            self.signal_esc.emit()
        if event.key() == QtCore.Qt.Key_Delete:
            self.signal_delete.emit()
        self.gui.redraw()


class GUI_scene_designer_rod(GUI):
    # Variables that are added to the given class
    grid = []
    resolution = 1.0
    radius = 1.0

    polygon_obstacles = []
    gui_polygon_obstacles = []

    # Represented as a disc/point
    rod_start = ()
    gui_rod_start = None
    rod_end = ()
    gui_rod_end = None

    rod_start_angle = 0.0
    rod_end_angle = 0.0
    rod_length = 1.0

    drawing_mode = MODE_POLYGON_OBSTACLE
    polyline = []
    gui_current_polygon_vertices = []
    gui_current_polygon_edges = []

    selected = {
        'index': None,
        'type': None
    }


    def __init__(self):
        super().__init__()
        self.zoom = 50.0
        self.redraw()

    def setupUi(self):
        self.mainWindow = MainWindowSceneDesigner(self)
        MainWindow = self.mainWindow

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1118, 894)
        MainWindow.setStyleSheet("QMainWindow { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }\n"
"#centralwidget { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }\n"
"QLabel { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 3, 1, 1, 1)
        self.gridLayout_0 = QtWidgets.QGridLayout()
        self.gridLayout_0.setObjectName("gridLayout_0")
        self.modeRodEndButton = QtWidgets.QPushButton(self.centralwidget)
        self.modeRodEndButton.setObjectName("modeRodEndButton")
        self.gridLayout_0.addWidget(self.modeRodEndButton, 28, 0, 1, 1)
        self.fileBrowseButton = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.fileBrowseButton.setFont(font)
        self.fileBrowseButton.setObjectName("fileBrowseButton")
        self.gridLayout_0.addWidget(self.fileBrowseButton, 3, 1, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_0.addItem(spacerItem, 33, 0, 1, 1)
        self.updateRodButton = QtWidgets.QPushButton(self.centralwidget)
        self.updateRodButton.setObjectName("updateRodButton")
        self.gridLayout_0.addWidget(self.updateRodButton, 23, 0, 1, 1)
        self.polygonEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.polygonEdit.setFont(font)
        self.polygonEdit.setObjectName("polygonEdit")
        self.gridLayout_0.addWidget(self.polygonEdit, 12, 0, 1, 1)
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setObjectName("loadButton")
        self.gridLayout_0.addWidget(self.loadButton, 4, 0, 1, 1)
        self.writerEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.writerEdit.setObjectName("writerEdit")
        self.gridLayout_0.addWidget(self.writerEdit, 10, 0, 1, 1)
        self.outputEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.outputEdit.setFont(font)
        self.outputEdit.setObjectName("outputEdit")
        self.gridLayout_0.addWidget(self.outputEdit, 6, 0, 1, 1)
        self.polygonButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.polygonButton.setFont(font)
        self.polygonButton.setObjectName("polygonButton")
        self.gridLayout_0.addWidget(self.polygonButton, 13, 0, 1, 1)
        self.rodAngleStartEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.rodAngleStartEdit.setObjectName("rodAngleStartEdit")
        self.gridLayout_0.addWidget(self.rodAngleStartEdit, 18, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_0.addWidget(self.label_2, 11, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_0.addWidget(self.label_3, 14, 0, 1, 1)
        self.modeRodStartButton = QtWidgets.QPushButton(self.centralwidget)
        self.modeRodStartButton.setObjectName("modeRodStartButton")
        self.gridLayout_0.addWidget(self.modeRodStartButton, 27, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout_0.addWidget(self.label, 25, 0, 1, 1)
        self.clearButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.clearButton.setFont(font)
        self.clearButton.setObjectName("clearButton")
        self.gridLayout_0.addWidget(self.clearButton, 1, 0, 1, 1)
        self.saveButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.saveButton.setFont(font)
        self.saveButton.setObjectName("saveButton")
        self.gridLayout_0.addWidget(self.saveButton, 9, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout_0.addWidget(self.label_6, 21, 0, 1, 1)
        self.helpButton = QtWidgets.QPushButton(self.centralwidget)
        self.helpButton.setObjectName("helpButton")
        self.gridLayout_0.addWidget(self.helpButton, 0, 0, 1, 1)
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.gridLayout_0.addWidget(self.label_1, 5, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_0.addWidget(self.label_4, 24, 0, 1, 1)
        self.inputEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.inputEdit.setFont(font)
        self.inputEdit.setObjectName("inputEdit")
        self.gridLayout_0.addWidget(self.inputEdit, 3, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(300, 20, QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_0.addItem(spacerItem1, 36, 0, 1, 1)
        self.modePolyObstButton = QtWidgets.QPushButton(self.centralwidget)
        self.modePolyObstButton.setObjectName("modePolyObstButton")
        self.gridLayout_0.addWidget(self.modePolyObstButton, 26, 0, 1, 1)
        self.resolutionEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.resolutionEdit.setFont(font)
        self.resolutionEdit.setObjectName("resolutionEdit")
        self.gridLayout_0.addWidget(self.resolutionEdit, 15, 0, 1, 1)
        self.resolutionButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.resolutionButton.setFont(font)
        self.resolutionButton.setObjectName("resolutionButton")
        self.gridLayout_0.addWidget(self.resolutionButton, 16, 0, 1, 1)
        self.label_0 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_0.setFont(font)
        self.label_0.setObjectName("label_0")
        self.gridLayout_0.addWidget(self.label_0, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.gridLayout_0.addWidget(self.label_7, 19, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setObjectName("label_5")
        self.gridLayout_0.addWidget(self.label_5, 17, 0, 1, 1)
        self.rodLengthEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.rodLengthEdit.setObjectName("rodLengthEdit")
        self.gridLayout_0.addWidget(self.rodLengthEdit, 22, 0, 1, 1)
        self.rodAngleEndEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.rodAngleEndEdit.setObjectName("rodAngleEndEdit")
        self.gridLayout_0.addWidget(self.rodAngleEndEdit, 20, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_0, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.lineEdits['input_path'] = self.inputEdit
        self.lineEdits['output_path'] = self.outputEdit
        self.lineEdits['polygon_index'] = self.polygonEdit
        self.lineEdits['resolution'] = self.resolutionEdit
        self.lineEdits['rod_start_angle'] = self.rodAngleStartEdit
        self.lineEdits['rod_end_angle'] = self.rodAngleEndEdit
        self.lineEdits['rod_length'] = self.rodLengthEdit
        
        self.pushButtons['help'] = self.helpButton
        self.pushButtons['clear'] = self.clearButton
        self.pushButtons['load'] = self.loadButton
        self.pushButtons['save'] = self.saveButton
        self.pushButtons['select_polygon'] = self.polygonButton
        self.pushButtons['set_resolution'] = self.resolutionButton
        self.pushButtons['update_rod'] = self.updateRodButton
        self.pushButtons['file_browser'] = self.fileBrowseButton

        # Drawing modes
        self.pushButtons['mode_polygonal_obstacles'] = self.modePolyObstButton
        self.pushButtons['mode_rod_start'] = self.modeRodStartButton
        self.pushButtons['mode_rod_end'] = self.modeRodEndButton

        self.labels['current_mode'] = self.label

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.modeRodEndButton.setText(_translate("MainWindow", "Rod End Position"))
        self.fileBrowseButton.setText(_translate("MainWindow", "..."))
        self.updateRodButton.setText(_translate("MainWindow", "Update Rod"))
        self.loadButton.setText(_translate("MainWindow", "Load Scene"))
        self.polygonButton.setText(_translate("MainWindow", "Select Polygon"))
        self.label_2.setText(_translate("MainWindow", "Polygon Index:"))
        self.label_3.setText(_translate("MainWindow", "Resolution:"))
        self.modeRodStartButton.setText(_translate("MainWindow", "Rod Start Position"))
        self.label.setText(_translate("MainWindow", "(Current Mode: Polygonal Obstacles)"))
        self.clearButton.setText(_translate("MainWindow", "Clear Scene"))
        self.saveButton.setText(_translate("MainWindow", "Save Scene"))
        self.label_6.setText(_translate("MainWindow", "Rod Length:"))
        self.helpButton.setText(_translate("MainWindow", "Help"))
        self.label_1.setText(_translate("MainWindow", "Output Path:"))
        self.label_4.setText(_translate("MainWindow", "Drawing Mode:"))
        self.modePolyObstButton.setText(_translate("MainWindow", "Polygonal Obstacles"))
        self.resolutionButton.setText(_translate("MainWindow", "Set Resolution"))
        self.label_0.setText(_translate("MainWindow", "Input Path:"))
        self.label_7.setText(_translate("MainWindow", "Rod End Angle:"))
        self.label_5.setText(_translate("MainWindow", "Rod Start Angle:"))
