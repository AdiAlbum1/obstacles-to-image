from PyQt5 import QtCore, QtWidgets, QtGui
from gui.gui import GUI
from gui.logger import Logger


class GUI_mrmp(GUI):
    comboBoxes = {}

    def __init__(self):
        super().__init__()
        self.zoom = 50.0
        self.redraw()

    def setupUi(self):
        MainWindow = self.mainWindow
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.width, self.height)
        MainWindow.setStyleSheet("QMainWindow { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }\n"
                                 "#centralwidget { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }\n"
                                 "QLabel { background-color : rgb(54, 57, 63); color : rgb(220, 221, 222); }")
        MainWindow.setObjectName("MainWindow")
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
        self.gridLayout.addWidget(self.graphicsView, 3, 2, 1, 1)
        self.gridLayout_0 = QtWidgets.QGridLayout()
        self.gridLayout_0.setObjectName("gridLayout_0")
        self.label_0 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_0.setFont(font)
        self.label_0.setObjectName("label_0")
        self.gridLayout_0.addWidget(self.label_0, 3, 0, 1, 1)
        self.toolButton_0 = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.toolButton_0.setFont(font)
        self.toolButton_0.setObjectName("toolButton_0")
        self.gridLayout_0.addWidget(self.toolButton_0, 4, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout_0.addWidget(self.label_2, 13, 0, 1, 1)
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout_0.addWidget(self.pushButton_8, 18, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.gridLayout_0.addWidget(self.label_7, 1, 0, 1, 1)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.gridLayout_0.addWidget(self.lineEdit_5, 17, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_0.addWidget(self.pushButton_2, 15, 0, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_0.addWidget(self.lineEdit_6, 24, 0, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout_0.addWidget(self.pushButton_6, 27, 0, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_0.addWidget(self.pushButton_3, 19, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_0.addWidget(self.label_3, 21, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_0.addWidget(self.lineEdit_3, 22, 0, 1, 1)
        self.toolButton_1 = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.toolButton_1.setFont(font)
        self.toolButton_1.setObjectName("toolButton_1")
        self.gridLayout_0.addWidget(self.toolButton_1, 7, 1, 1, 1)
        self.lineEdit_1 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_1.setFont(font)
        self.lineEdit_1.setObjectName("lineEdit_1")
        self.gridLayout_0.addWidget(self.lineEdit_1, 7, 0, 1, 1)
        self.toolButton_2 = QtWidgets.QToolButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.toolButton_2.setFont(font)
        self.toolButton_2.setObjectName("toolButton_2")
        self.gridLayout_0.addWidget(self.toolButton_2, 14, 1, 1, 1)
        self.lineEdit_0 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_0.setFont(font)
        self.lineEdit_0.setObjectName("lineEdit_0")
        self.gridLayout_0.addWidget(self.lineEdit_0, 4, 0, 1, 1)
        self.pushButton_1 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.gridLayout_0.addWidget(self.pushButton_1, 11, 0, 1, 1)
        self.label_1 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.gridLayout_0.addWidget(self.label_1, 6, 0, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_0.addWidget(self.pushButton_4, 20, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_0.addWidget(self.label_5, 16, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_0.addWidget(self.lineEdit_2, 14, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.gridLayout_0.addWidget(self.label_6, 23, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.gridLayout_0.addWidget(self.label_4, 8, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(300, 0, QtWidgets.QSizePolicy.MinimumExpanding,
                                           QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_0.addItem(spacerItem, 28, 0, 3, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_0.addWidget(self.pushButton_7, 0, 0, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout_0.addWidget(self.pushButton_5, 26, 0, 1, 1)
        self.pushButton_0 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton_0.setFont(font)
        self.pushButton_0.setObjectName("pushButton_0")
        self.gridLayout_0.addWidget(self.pushButton_0, 5, 0, 1, 1)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.gridLayout_0.addWidget(self.lineEdit_4, 10, 0, 1, 1)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setObjectName("comboBox")
        self.gridLayout_0.addWidget(self.comboBox, 2, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_0, 3, 1, 1, 1)
        self.textEdit = Logger(self.centralwidget)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 3, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # end of eq designer generated code

        self.lineEdits['scene'] = self.lineEdit_0
        self.lineEdits['planner'] = self.lineEdit_1
        self.lineEdits['argument'] = self.lineEdit_4
        self.lineEdits['solution'] = self.lineEdit_2
        self.lineEdits['export'] = self.lineEdit_5
        self.lineEdits['index'] = self.lineEdit_3
        self.lineEdits['graphPercent'] = self.lineEdit_6
        self.pushButtons['scene'] = self.pushButton_0
        self.pushButtons['planner'] = self.pushButton_1
        self.pushButtons['solution'] = self.pushButton_2
        self.pushButtons['animate'] = self.pushButton_3
        self.pushButtons['verify'] = self.pushButton_4
        self.pushButtons['graph'] = self.pushButton_5
        self.pushButtons['clearGraph'] = self.pushButton_6
        self.pushButtons['help'] = self.pushButton_7
        self.pushButtons['export'] = self.pushButton_8
        self.pushButtons['searchScene'] = self.toolButton_0
        self.pushButtons['searchPlanner'] = self.toolButton_1
        self.pushButtons['searchSolution'] = self.toolButton_2
        self.labels['scene'] = self.label_0
        self.labels['planner'] = self.label_1
        self.labels['solution'] = self.label_2
        self.labels['index'] = self.label_3
        self.labels['argument'] = self.label_4
        self.labels['export'] = self.label_5
        self.labels['graphPercent'] = self.label_6
        self.labels['mode'] = self.label_7

        self.comboBoxes['mode'] = self.comboBox
