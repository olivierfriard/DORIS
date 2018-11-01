#!/usr/bin/env python3
"""
DORIS
Detection of Objects Research Interactive Software
Copyright 2017-2018 Olivier Friard

This file is part of DORIS.

  DORIS is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  any later version.

  DORIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not see <http://www.gnu.org/licenses/>.


Requirements:
pyqt5
opencv
numpy
matplotlib

optional:
mpl_scatter_density (pip3 install mpl_scatter_density)

TODO:

* add choice for backgroung algo
* implement check of position when 1 object must be detected
* automatic determination

match shape
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html

"""


THRESHOLD_DEFAULT = 50
VIEWER_WIDTH = 480

# BGR colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
MAROON = (128, 0, 0)
TEAL = (0, 128, 128)
PURPLE = (128, 0, 128)
WHITE = (255, 255, 255)

# colors list for marker
COLORS_LIST = [BLUE, MAGENTA, YELLOW, MAROON, TEAL, PURPLE, WHITE]

MARKER_TYPE = "contour"
MARKER_COLOR = GREEN
ARENA_COLOR = RED
AREA_COLOR = GREEN

CIRCLE_THICKNESS = 2
RECTANGLE_THICKNESS = 2

TOLERANCE_OUTSIDE_ARENA = 0.05  # fraction of size of object

__version__ = "0.0.3"
__version_date__ = "2018-10-29"

from PyQt5.QtCore import Qt, QT_VERSION_STR, PYQT_VERSION_STR, pyqtSignal, QEvent
from PyQt5.QtGui import (QPixmap, QImage, qRgb)
from PyQt5.QtWidgets import (QMainWindow, QApplication,QStatusBar, QMenu, QFileDialog, QMessageBox, QInputDialog,
                             QWidget, QVBoxLayout, QLabel, QSpacerItem, QSizePolicy)

import os
import platform
import numpy as np
#np.set_printoptions(threshold="nan")
import cv2
import copy

import sys
import time
import pathlib
import math
import matplotlib
matplotlib.use("Qt4Agg" if QT_VERSION_STR[0] == "4" else "Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

try:
    import mpl_scatter_density
    flag_mpl_scatter_density = True
except:
    flag_mpl_scatter_density = False

import argparse

import doris_functions

from doris_ui import Ui_MainWindow

class Click_label(QLabel):

    mouse_pressed_signal = pyqtSignal(QEvent)

    def __init__(self, parent= None):
        QLabel.__init__(self, parent)

    def mousePressEvent(self, event):
        """
        label clicked
        """
        self.mouse_pressed_signal.emit(event)


class FrameViewer(QWidget):
    """
    widget for visualizing frame
    """
    def __init__(self):
        super(FrameViewer, self).__init__()

        self.setWindowTitle("")

        self.vbox = QVBoxLayout()

        self.lb_frame = Click_label()
        self.lb_frame.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.vbox.addWidget(self.lb_frame)

        self.setLayout(self.vbox)

    def pbOK_clicked(self):
        self.close()


font = cv2.FONT_HERSHEY_SIMPLEX



def frame2pixmap(frame):
    """
    convert np.array (frame) to QT pixmap
    """
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        pass
    height, width = frame.shape[:2]

    img = QImage(frame, width, height, QImage.Format_RGB888)

    return QPixmap.fromImage(img)


def toQImage(frame, copy=False):

    gray_color_table = [qRgb(i, i, i) for i in range(256)]
    if frame is None:
        return QImage()

    im = np.asarray(frame)
    if im.dtype == np.uint8:
        if len(im.shape) == 2:
            qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
            qim.setColorTable(gray_color_table)
            return qim.copy() if copy else qim
        elif len(im.shape) == 3:
            if im.shape[2] == 3:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888);
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32);
                return qim.copy() if copy else qim


def plot_density(x, y, x_lim=(0,0), y_lim=(0,0)):

    if flag_mpl_scatter_density:
        x = np.array(x)
        y = np.array(y)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
        ax.scatter_density(x, y)
        if x_lim != (0,0):
            ax.set_xlim(x_lim)
        if y_lim != (0,0):
            ax.set_ylim(y_lim[::-1])

        plt.show()

    else:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("DORIS")
        msg.setText("the mpl_scatter_density module is required to plot density")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        return


def plot_path(verts, x_lim, y_lim, color):

    print("verts", verts)

    # invert verts y
    verts = [(x[0], y_lim[1] - x[1]) for x in verts]

    codes = [Path.MOVETO]
    codes.extend([Path.LINETO] * (len(verts)-1))

    path = Path(verts, codes)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, edgecolor=tuple((x/255 for x in color)), facecolor="none", lw=1)
    ax.add_patch(patch)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    plt.show()








    '''
    (p,t), (q,u), (s,z) = points[0], points[1], points[2]
    A=((u-t)*z**2+(-u**2+t**2-q**2+p**2)*z+t*u**2+(-t**2+s**2-p**2)*u+(q**2-s**2)*t)/((q-p)*z+(p-s)*u+(s-q)*t)

    B=-((q-p)*z**2+(p-s)*u**2+(s-q)*t**2+(q-p)*s**2+(p**2-q**2)*s+p*q**2-p**2*q)/((q-p)*z+(p-s)*u+(s-q)*t)

    C=-((p*u-q*t)*z**2+(-p*u**2+q*t**2-p*q**2+p**2*q)*z+s*t*u**2+(-s*t**2+p*s**2-p**2*s)*u+(q**2*s-q*s**2)*t)/((q-p)*z+(p-s)*u+(s-q)*t)

    print(A,B,C)
    '''

    '''
    x = complex(points[0][0],points[0][1])
    y = complex(points[1][0],points[1][1])
    z = complex(points[2][0],points[2][1])
    print(x,y,z)

    w = z-x
    w /= y-x
    c = (x-y)*(w-abs(w)**2)/2j/w.imag-x
    print(c.real,c.imag)
    print(abs(c+x))
    return c.real, c.imag, abs(c+x)
    '''

    '''(x%+.3f)^2+(y%+.3f)^2 = %.3f^2' % (c.real, c.imag, abs(c+x))'''


class Ui_MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):

        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.setWindowTitle("DORIS v. {} - (c) Olivier Friard".format(__version__))
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("font-size:24px")
        self.setStatusBar(self.statusBar)
        self.actionAbout.triggered.connect(self.about)
        
        self.action1.triggered.connect(lambda: self.frame_viewer_scale(0, 1))
        self.action1_2.triggered.connect(lambda: self.frame_viewer_scale(0, 0.5))
        self.action1_4.triggered.connect(lambda: self.frame_viewer_scale(0, 0.25))

        self.action_treated_1.triggered.connect(lambda: self.frame_viewer_scale(1, 1))
        self.action_treated_1_2.triggered.connect(lambda: self.frame_viewer_scale(1, 0.5))
        self.action_treated_1_4.triggered.connect(lambda: self.frame_viewer_scale(1, 0.25))
        

        # self.label1.mousePressEvent = self.frame_mousepressed

        self.actionOpen_video.triggered.connect(lambda: self.open_video(""))
        self.actionLoad_directory_of_images.triggered.connect(self.load_dir_images)
        self.actionQuit.triggered.connect(self.close)

        # self.actionFrame_width.triggered.connect(self.frame_width)

        self.pb_next_frame.clicked.connect(self.next_frame)
        #self.pb_next_frame.clicked.connect(lambda: self.pb(1))
        self.pb_1st_frame.clicked.connect(self.reset)

        self.pb_goto_frame.clicked.connect(self.go_to_frame)

        #self.pb_forward.clicked.connect(lambda: self.pb(self.sb_frame_offset.value()))
        #self.pb_backward.clicked.connect(lambda: self.pb(-self.sb_frame_offset.value()))
        self.pb_forward.clicked.connect(lambda: self.for_back_ward("forward"))
        self.pb_backward.clicked.connect(lambda: self.for_back_ward("backward"))

        # menu for arena button
        menu1 = QMenu()
        menu1.addAction("Rectangle arena", lambda: self.define_arena("rectangle"))
        menu1.addAction("Circle arena (3 points)", lambda: self.define_arena("circle (3 points)"))
        menu1.addAction("Circle arena (center radius)", lambda: self.define_arena("circle (center radius)"))
        menu1.addAction("Polygon arena", lambda: self.define_arena("polygon"))

        self.pb_define_arena.setMenu(menu1)
        self.pb_clear_arena.clicked.connect(self.clear_arena)

        self.pbGo.clicked.connect(self.run_analysis)
        self.pb_stop.clicked.connect(self.stop)

        self.sb_threshold.valueChanged.connect(self.treat_and_show)
        self.sb_blur.valueChanged.connect(self.treat_and_show)
        self.cb_invert.stateChanged.connect(self.treat_and_show)
        self.cb_background.stateChanged.connect(self.background)

        self.sbMin.valueChanged.connect(self.treat_and_show)
        self.sbMax.valueChanged.connect(self.treat_and_show)

        self.sb_largest_number.valueChanged.connect(self.treat_and_show)
        self.sb_max_extension.valueChanged.connect(self.treat_and_show)

        self.pb_show_all_objects.clicked.connect(self.show_all_objects)

        # coordinates analysis
        self.pb_reset_xy.clicked.connect(self.reset_xy_analysis)
        self.pb_save_xy.clicked.connect(self.save_xy)
        self.pb_plot_path.clicked.connect(self.plot_path)
        self.pb_plot_xy_density.clicked.connect(self.plot_xy_density)
        self.pb_plot_xy_density.setEnabled(flag_mpl_scatter_density)

        # menu for area button
        menu = QMenu()
        menu.addAction("Rectangle", lambda: self.add_area_func("rectangle"))
        menu.addAction("Circle (center radius)", lambda: self.add_area_func("circle (center radius)"))
        menu.addAction("Circle (3 points)", lambda: self.add_area_func("circle (3 points)"))
        menu.addAction("Polygon", lambda: self.add_area_func("polygon"))

        self.pb_add_area.setMenu(menu)
        self.pb_remove_area.clicked.connect(self.remove_area)
        self.pb_open_areas.clicked.connect(self.open_areas)
        self.pb_reset_areas.clicked.connect(self.reset_areas_analysis)
        self.pb_save_areas.clicked.connect(self.save_areas)
        self.pb_active_areas.clicked.connect(self.activate_areas)
        self.pb_save_objects_number.clicked.connect(self.save_objects_number)

        self.frame = None
        self.capture = None
        self.output = ""
        self.videoFileName = ""
        self.fgbg = None
        self.flag_stop_analysis = False
        self.positions = []
        self.video_height = 0
        self.video_width = 0
        self.frame_width = VIEWER_WIDTH
        self.total_frame_nb = 0
        self.fps = 0
        self.areas =  {}
        self.flag_define_arena = False
        #self.flag_add_area = ""
        self.add_area = {}
        self.arena = []
        self.mem_filtered_objects = {}

        self.dir_images = []
        self.dir_images_index = 0

        self.positions, self.objects_number = [], []

        # default
        self.sb_threshold.setValue(THRESHOLD_DEFAULT)

        self.fw = []
        self.fw.append(FrameViewer())
        self.fw[0].lb_frame.mouse_pressed_signal.connect(self.frame_mousepressed)
        self.fw[0].setGeometry(100, 100, 512, 512)
        self.fw[0].show()

        self.fw.append(FrameViewer())
        self.fw[1].setGeometry(640, 100, 512, 512)
        self.fw[1].show()




    def click_on_frame(self, id_, event):
        print(event)


    def about(self):

        modules = []
        modules.append("OpenCV")
        modules.append("version {}".format(cv2.__version__))

        # matplotlib
        modules.append("\nMatplotlib")
        modules.append("version {}".format(matplotlib.__version__))

        about_dialog = msg = QMessageBox()
        #about_dialog.setIconPixmap(QPixmap(os.path.dirname(os.path.realpath(__file__)) + "/logo_eye.128px.png"))
        about_dialog.setWindowTitle("About DORIS")
        about_dialog.setStandardButtons(QMessageBox.Ok)
        about_dialog.setDefaultButton(QMessageBox.Ok)
        about_dialog.setEscapeButton(QMessageBox.Ok)

        about_dialog.setInformativeText(("<b>DORIS</b> v. {ver} - {date}"
        "<p>Copyright &copy; 2017-2018 Olivier Friard<br>"
        "Department of Life Sciences and Systems Biology<br>"
        "University of Torino - Italy<br>").format(ver=__version__,
                                                   date=__version_date__))

        details = ("Python {python_ver} ({architecture}) - Qt {qt_ver} - PyQt{pyqt_ver} on {system}\n"
        "CPU type: {cpu_info}\n\n"
        "{modules}").format(python_ver=platform.python_version(),
                            architecture="64-bit" if sys.maxsize > 2**32 else "32-bit",
                            pyqt_ver=PYQT_VERSION_STR,
                            system=platform.system(),
                            qt_ver=QT_VERSION_STR,
                            cpu_info=platform.machine(),
                            modules="\n".join(modules))

        about_dialog.setDetailedText(details)

        _ = about_dialog.exec_()


    def frame_viewer_scale(self, fw_idx, scale):
        """
        change scale of frame viewer
        """
        self.fw[fw_idx].lb_frame.clear()
        self.fw[fw_idx].lb_frame.resize(int(self.frame.shape[1] * scale), int(self.frame.shape[0] * scale))
        if fw_idx == 0:
            self.fw[fw_idx].lb_frame.setPixmap(frame2pixmap(self.frame).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                               Qt.KeepAspectRatio))
        if fw_idx == 1:
            treated_frame = self.treatment(self.frame)            
            self.fw[1].lb_frame.setPixmap(QPixmap.fromImage(toQImage(treated_frame)).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                                         Qt.KeepAspectRatio))
        self.fw[fw_idx].setFixedSize(self.fw[fw_idx].vbox.sizeHint())


    def for_back_ward(self, direction="forward"):

        step = self.sb_frame_offset.value() if direction == "forward" else -self.sb_frame_offset.value()

        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1 + step)
        self.pb()


    def go_to_frame(self):

        if self.le_goto_frame.text():
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.le_goto_frame.text()) - 1)
            except:
                pass
            self.pb()


    def add_area_func(self, shape):
        if shape:
            text, ok = QInputDialog.getText(self, "New area", "Area name:")
            if ok:
                self.add_area = {"type": shape, "name": text}
                msg = ""
                if shape == "circle (3 points)":
                    msg = "New circle area: Click on the video to define 3 points belonging to the circle"
                if shape == "circle (center radius)":
                    msg = "New circle area: click on the video to define the center of the circle and then a point belonging to the circle"
                if shape == "polygon":
                    msg = "New polygon area: click on the video to define the edges of the polygon. Right click to finish"
                if shape == "rectangle":
                    msg = "New rectangle area: click on the video to define the top-lef and bottom-right edges of the rectangle."

                self.statusBar.showMessage(msg)


    def remove_area(self):
        """
        remove the selected area
        """

        for selected_item in self.lw_area_definition.selectedItems():
            self.lw_area_definition.takeItem(self.lw_area_definition.row(selected_item))
            self.activate_areas()


    def define_arena(self, shape):
        """
        switch to define area mode
        """
        if self.flag_define_arena:
            self.flag_define_arena = ""
        else:
            self.flag_define_arena = shape
            self.pb_define_arena.setEnabled(False)
            msg = ""
            if shape == "rectangle":
                msg = "New arena: click on the video to define the top-lef and bottom-right edges of the rectangle."
            if shape == "circle (3 points)":
                msg = "New arena: click on the video to define 3 points belonging to the circle"
            if shape == "circle (center radius)":
                msg = "New arena: click on the video to define the center of the circle and then a point belonging to the circle"
            if shape == "polygon":
                msg = "New arena: click on the video to define the edges of the polygon"


            self.statusBar.showMessage(msg)


    def reload_frame(self):
        if self.dir_images:
            self.dir_images_index -= 1
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.pb()


    def clear_arena(self):
        """
        clear the arena
        """
        self.flag_define_arena = False
        self.arena = []
        self.le_arena.setText("")

        self.pb_define_arena.setEnabled(True)
        self.pb_clear_arena.setEnabled(False)

        self.reload_frame()


    def ratio_thickness(self, video_width, frame_width):
        """
        return ratio and pen thickness for contours according to video resolution
        """

        ratio = video_width / frame_width
        if ratio <= 1:
            drawing_thickness = 1
        else:
            drawing_thickness = round(ratio)
        return ratio, drawing_thickness


    def frame_mousepressed(self, event):
        """
        record clicked coordinates if arena or area mode activated
        """

        conversion, drawing_thickness = self.ratio_thickness(self.video_width, self.fw[0].lb_frame.pixmap().width())

        #print("area type:", self.add_area["type"])

        if self.add_area:

            if event.button() == 4:
                self.add_area = {}
                self.statusBar.showMessage("New area canceled")
                self.reload_frame()
                return

            if event.button() == 1:
                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

            if self.add_area["type"] == "circle (center radius)":
                if "center" not in self.add_area:
                    self.add_area["center"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["type"] = "circle"
                    self.add_area["radius"] = int((( event.pos().x() * conversion - self.add_area["center"][0] )**2 + ( event.pos().y() * conversion - self.add_area["center"][1] )**2)**0.5)
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.statusBar.showMessage("New circle area created")
                    return

            if self.add_area["type"] == "circle (3 points)":
                if "points" not in self.add_area:
                    self.add_area["points"] = []
                if len(self.add_area["points"]) < 3:
                    self.add_area["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                    print(len(self.add_area["points"]))

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=AREA_COLOR, lineType=8, thickness=drawing_thickness)

                if len(self.add_area["points"]) == 3:
                    cx, cy, radius = doris_functions.find_circle(self.add_area["points"])
                    self.add_area["type"] = "circle"
                    self.add_area["center"] = [int(cx), int(cy)]
                    self.add_area["radius"] = int(radius)
                    print(self.add_area)
                    del self.add_area["points"]
                    print(self.add_area)
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.statusBar.showMessage("New circle area created")
                    return

            if self.add_area["type"] == "rectangle":
                if "pt1" not in self.add_area:
                    self.add_area["pt1"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["pt2"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.statusBar.showMessage("New rectangle area created")
                    return

            if self.add_area["type"] == "polygon":

                if event.button() == 2: # right click to finish
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.statusBar.showMessage("The new polygon area is defined")
                    return

                if event.button() == 1: # left button
                    cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                    if "points" not in self.add_area:
                        self.add_area["points"] = [[int(event.pos().x() * conversion), int(event.pos().y() * conversion)]]
                    else:
                        self.add_area["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                        cv2.line(self.frame, tuple(self.add_area["points"][-2]), tuple(self.add_area["points"][-1]), color=AREA_COLOR, lineType=8, thickness=drawing_thickness)

                    self.statusBar.showMessage("Polygon area: {} point(s) selected. Right click to finish".format(len(self.add_area["points"])))
                    self.display_frame(self.frame)

        # arena
        if self.flag_define_arena:

            # cancel arena creation (mid button)
            if event.button() == 4:
                self.flag_define_arena = ""
                self.pb_define_arena.setEnabled(True)
                self.statusBar.showMessage("Arena creation canceled")
                self.reload_frame()
                return

            if self.flag_define_arena == "rectangle":
                self.arena.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)
                self.statusBar.showMessage("Rectangle arena: {} point(s) selected.".format(len( self.arena)))


                if len(self.arena) == 2:
                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.pb_define_arena.setText("Define arena")
                    self.arena = {"type": "rectangle", "points": self.arena, "name": "arena"}
                    self.le_arena.setText("{}".format(self.arena))

                    cv2.rectangle(self.frame, tuple(self.arena["points"][0]), tuple(self.arena["points"][1]), color=ARENA_COLOR, thickness=drawing_thickness)
                    self.display_frame(self.frame)

                    self.statusBar.showMessage("The rectangle arena is defined")

            if self.flag_define_arena == "polygon":

                if event.button() == 2: # right click to finish

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.pb_define_arena.setText("Define arena")

                    self.arena = {'type': 'polygon', 'points': self.arena, 'name': 'arena'}
                    self.le_arena.setText("{}".format(self.arena))

                    self.statusBar.showMessage("The new polygon arena is defined")

                else:

                    self.arena.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                    cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                    self.display_frame(self.frame)

                    self.statusBar.showMessage("Polygon arena: {} point(s) selected. Right click to finish".format(len( self.arena)))

                    if len(self.arena) >= 2:
                        cv2.line(self.frame, tuple(self.arena[-2]), tuple(self.arena[-1]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                        self.display_frame(self.frame)

            if self.flag_define_arena == "circle (3 points)":

                self.arena.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

                self.statusBar.showMessage("Circle arena: {} point(s) selected.".format(len( self.arena)))

                if len(self.arena) == 3:
                    cx, cy, r = doris_functions.find_circle(self.arena)
                    cv2.circle(self.frame, (int(abs(cx)), int(abs(cy))), int(r), color=ARENA_COLOR, thickness=drawing_thickness)
                    self.display_frame(self.frame)

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.arena = {'type': 'circle', 'center': [round(cx), round(cy)], 'radius': round(r), 'name': 'arena'}
                    self.le_arena.setText("{}".format(self.arena))

                    self.statusBar.showMessage("The new circle arena is defined")

            if self.flag_define_arena == "circle (center radius)":

                self.arena.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4, color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

                if len(self.arena) == 2:
                    cx, cy = self.arena[0]
                    radius = doris_functions.euclidean_distance(self.arena[0], self.arena[1])
                    cv2.circle(self.frame, (int(abs(cx)), int(abs(cy))), int(radius), color=ARENA_COLOR, thickness=drawing_thickness)

                    self.display_frame(self.frame)

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.arena = {'type': 'circle', 'center': [round(cx), round(cy)], 'radius': round(radius), 'name': 'arena'}
                    self.le_arena.setText("{}".format(self.arena))

                    self.statusBar.showMessage("The new circle arena is defined")


    '''
    def frame_width(self):
        """
        let user choose a width for frame image
        """
        i, ok_pressed = QInputDialog.getInt(self, "Set frame width", "Frame width", self.frame_width, 20, 2000, 1)
        if ok_pressed:
            self.frame_width = i

        self.treat_and_show()
    '''


    def background(self):
        if self.cb_background.isChecked():
            self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            print("backgound substraction activated")
        else:
            self.fgbg = None
        for w in [self.lb_threshold, self.sb_threshold]:
            w.setEnabled(not self.cb_background.isChecked())


    def reset(self):
        "go to 1st frame"

        if self.dir_images:
            self.dir_images_index = 0
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.pb()


    def next_frame(self):
        """
        go to next frame
        """
        r, results = self.pb(1)
        if not r:
            return

        self.analysis(results)


    def reset_xy_analysis(self):
        """
        reset coordinates analysis
        """
        self.positions = []
        self.te_xy.clear()


    def save_xy(self):
        """
        save results of recorded positions in TSV file
        """
        if self.positions:
            file_name, _ = QFileDialog(self).getSaveFileName(self, "Save objects coordinates", "", "All files (*)")
            out = ""
            if file_name:
                for row in self.positions:
                    for obj in row:
                        out += "{},{}\t".format(obj[0], obj[1])
                    out = out.strip() + "\n"
                with open(file_name, "w") as f_in:
                    f_in.write(out)
        else:
            print("no positions to be saved")


    def open_areas(self, file_name):
        """
        load areas from disk
        format required:
        area_name; circle; x_center,y_center; radius; red_level, green_level, blue_level
        area_name; rectangle; x_min,y_min; x_max,y_max; red_level, green_level, blue_level
        """
        if not file_name:
            file_name, _ = QFileDialog(self).getOpenFileName(self, "Load areas from file", "", "All files (*)")
        if file_name:
            self.lw_area_definition.clear()
            with open(file_name, "r") as f_in:
                for line in f_in.readlines():
                    self.lw_area_definition.addItem(line.strip())
            self.activate_areas()


    def reset_areas_analysis(self):
        """
        reset areas analysis
        """
        self.objects_number = []
        self.te_number_objects.clear()


    def save_areas(self):
        """
        save defined areas to file
        """
        file_name, _ = QFileDialog(self).getSaveFileName(self, "Save areas to file", "", "All files (*)")
        if file_name:
            with open(file_name, "w") as f_out:
                for idx in range(self.lw_area_definition.count()):
                    f_out.write(self.lw_area_definition.item(idx).text() + "\n")


    def save_objects_number(self):
        """
        save results of objects number by area
        """
        if self.objects_number:
            file_name, _ = QFileDialog(self).getSaveFileName(self, "Save objects number", "", "All files (*)")
            out = "\t".join(list(sorted(self.areas.keys()))) + "\n"
            if file_name:
                for row in self.objects_number:
                    out += "\t".join([str(x) for x in row]) + "\n"
                with open(file_name, "w") as f_in:
                    f_in.write(out)
        else:
            self.statusBar.showMessage("no results to be saved")


    def plot_xy_density(self):

        if self.positions:
            for n_object in range(len(self.positions[0])):
                x, y = [], []
                for row in self.positions:
                    x.append(row[n_object][0])
                    y.append(row[n_object][1])
                plot_density(x, y, x_lim=(0, self.video_width), y_lim=(0, self.video_height))
        else:
            self.statusBar.showMessage("no positions to be plotted")


    def plot_path(self):

        print(self.positions)
        if self.positions:
            for n_object in range(len(self.positions[0])):
                verts = []
                for row in self.positions:
                    verts.append(row[n_object])
                plot_path(verts, x_lim=(0, self.video_width), y_lim=(0, self.video_height), color=COLORS_LIST[n_object % len(COLORS_LIST)])
        else:
            self.statusBar.showMessage("no positions to be plotted")


    def open_video(self, file_name):
        """
        let user select a video
        """

        if not file_name:
            file_name, _ = QFileDialog(self).getOpenFileName(self, "Open video", "", "All files (*)")
        if file_name:
            self.capture = cv2.VideoCapture(file_name)

            if not self.capture.isOpened():
                QMessageBox.critical(self, "DORIS", "Could not open {}".format(file_name))
                return

            self.total_frame_nb = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.lb_total_frames_nb.setText("Total number of frames: <b>{}</b>".format(self.total_frame_nb))

            self.fps = self.capture.get(cv2.CAP_PROP_FPS)

            self.frame_idx = 0
            self.pb(1)
            self.video_height, self.video_width, _ = self.frame.shape
            self.videoFileName = file_name

            self.statusBar.showMessage("video loaded (w: {} h: {})".format(self.video_width, self.video_height))


    def load_dir_images(self, dir_images):
        """
        Load directory of images
        """
        if not dir_images:
            dir_images = QFileDialog(self).getExistingDirectory(self, "Select Directory")
        if dir_images:
            p = pathlib.Path(dir_images)
            self.dir_images = sorted(list(p.glob('*.jpg')) + list(p.glob('*.JPG')) + list(p.glob("*.png")))

            self.lb_total_frames_nb.setText("Total number of images: <b>{}</b>".format(len(self.dir_images)))

            self.dir_images_index = 0
            self.pb(1)

            print("self.frame.shape", self.frame.shape)

            self.video_height, self.video_width, _ = self.frame.shape

            self.statusBar.showMessage("{} image(s) found".format(len(self.dir_images)))


    def update_frame_index(self):
        """
        update frame index
        """
        if self.dir_images:
            self.frame_idx = self.dir_images_index
        else:
            self.frame_idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.leFrame.setText(str(self.frame_idx))


    def treatment(self, frame):
        """
        apply treament to frame
        returns treated frame
        """

        threshold_method = {"name": "adaptive",
                            "block_size": 81,
                            "offset": 38}

        # self.sb_threshold.value()

        frame = doris_functions.image_treatment(frame,
                                        blur=self.sb_blur.value(),
                                        threshold_method=threshold_method,
                                        invert=self.cb_invert.isChecked())
        '''
        if self.frame is None:
            return None

        if self.fgbg:
            frame = self.fgbg.apply(frame)

        # blur
        if self.sb_blur.value():
            frame = cv2.blur(frame, (self.sb_blur.value(), self.sb_blur.value()))

        # threshold
        if self.sb_threshold.value():
        # color to gray levels
            if not self.fgbg:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, frame = cv2.threshold(frame, self.sb_threshold.value(), 255, cv2.THRESH_BINARY)

        # invert
        if self.cb_invert.isChecked():
            frame = (255 - frame)
        '''

        return frame


    def update_objects(self, frame):
        """
        returns filtered objects
        """

        all_objects, filtered_objects = doris_functions.detect_and_filter_objects(frame=frame,
                                                                  min_size=self.sbMin.value(),
                                                                  max_size=self.sbMax.value(),
                                                                  largest_number=self.sb_largest_number.value(),
                                                                  arena=self.arena,
                                                                  max_extension=self.sb_max_extension.value(),
                                                                  tolerance_outside_arena=TOLERANCE_OUTSIDE_ARENA
                                                                 )

        '''
        if self.mem_filtered_objects:
            self.reorder_objects(mem_objects, objects)
        '''

        #self.mem_filtered_objects = copy.deepcopy(filtered_objects)
        print("===============")

        return filtered_objects


    def draw_marker_on_objects(self, frame, objects, marker_type=MARKER_TYPE):
        """
        draw marker (rectangle or contour) around objects
        marker color from index of object in COLORS_LIST
        """
        ratio, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)
        for idx in objects:
            print(idx)
            marker_color = COLORS_LIST[(idx - 1) % len(COLORS_LIST)]
            if marker_type == "rectangle":
                cv2.rectangle(frame, objects[idx]["min"], objects[idx]["max"] , marker_color, drawing_thickness)
            if marker_type == "contour":
                cv2.drawContours(frame, [objects[idx]["contour"]], 0, marker_color, drawing_thickness)

            cv2.putText(frame, str(idx), objects[idx]["max"], font, ratio, marker_color, drawing_thickness, cv2.LINE_AA)

        return frame


    def display_frame(self, frame):
        """
        display the current frame in label pixmap
        """

        self.fw[0].lb_frame.setPixmap(frame2pixmap(frame).scaled(self.fw[0].lb_frame.size(), Qt.KeepAspectRatio))


    def display_treated_frame(self, frame):
        """
        show treated frame
        """

        self.fw[1].lb_frame.setPixmap(QPixmap.fromImage(toQImage(frame)).scaled(self.fw[1].lb_frame.size(), Qt.KeepAspectRatio))


    def treat_and_show(self):
        """
        apply treament to frame and show results
        """

        if self.frame is None:
            return

        treated_frame = self.treatment(self.frame)

        frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                        self.update_objects(treated_frame),
                                        marker_type="contour")

        self.display_frame(frame_with_objects)
        self.display_treated_frame(treated_frame)


    def show_all_objects(self):
        """

        """
        if self.frame is None:
            return

        all_objects, _ = doris_functions.detect_and_filter_objects(frame=self.treatment(self.frame),
                                                                  min_size=self.sbMin.value(),
                                                                  max_size=self.sbMax.value(),
                                                                  largest_number=self.sb_largest_number.value(),
                                                                  arena=self.arena,
                                                                  max_extension=self.sb_max_extension.value(),
                                                                  tolerance_outside_arena=TOLERANCE_OUTSIDE_ARENA
                                                                 )

        frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                         all_objects,
                                                         marker_type="contour")
        self.display_frame(frame_with_objects)


    def closeEvent(self, event):
        try:
            self.capture.release()
            cv2.destroyAllWindows()
        except:
            pass

        try:
            self.fw[0].close()
            self.fw[1].close()
        except:
            pass
        print("close")


    def activate_areas(self):
        """
        create areas for object in areas function from te_area_definition
        """
        areas = {}
        for idx in range(self.lw_area_definition.count()):
            print(self.lw_area_definition.item(idx).text())
            d = eval(self.lw_area_definition.item(idx).text())
            if "name" in d:
                areas[d["name"]] = eval(self.lw_area_definition.item(idx).text())
        self.areas = areas

        self.pb()


    def pb(self, nf=1):
        """
        read 'nf' frames and do some analysis
        """
        if self.dir_images:
            if self.dir_images_index < len(self.dir_images) - 1:
                self.dir_images_index += 1
            else:
                self.flag_stop_analysis = False
                self.statusBar.showMessage("Last image of dir")
                return False, {}
            self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)

        else:

            if self.capture is not None:
                ret, self.frame = self.capture.read()
                if not ret:
                    return False, {}
            else:
                return False, {}

        self.update_frame_index()

        treated_frame = self.treatment(self.frame)

        (all_objects, filtered_objects) = doris_functions.detect_and_filter_objects(frame=treated_frame,
                                                                  min_size=self.sbMin.value(),
                                                                  max_size=self.sbMax.value(),
                                                                  largest_number=self.sb_largest_number.value(),
                                                                  arena=self.arena,
                                                                  max_extension=self.sb_max_extension.value(),
                                                                  tolerance_outside_arena=TOLERANCE_OUTSIDE_ARENA
                                                                 )

        print(filtered_objects)

        if filtered_objects:
            filtered_objects = doris_functions.reorder_objects(self.mem_filtered_objects, filtered_objects)
            self.mem_filtered_objects = dict(filtered_objects)

        print(filtered_objects)

        if self.cb_display_analysis.isChecked():

            # update information on GUI
            self.lb_all.setText("All objects detected ({})".format(len(all_objects)))
            out = ""
            for idx in sorted(all_objects.keys()):
                out += "Object #{}: {} pixels\n".format(idx, all_objects[idx]["area"])
            self.te_all_objects.setText(out)

            self.lb_filtered.setText("Filtered objects ({})".format(len(filtered_objects)))
            if self.sb_largest_number.value() and len(filtered_objects) < self.sb_largest_number.value():
                self.lb_filtered.setStyleSheet('color: red')
            else:
                self.lb_filtered.setStyleSheet("")

            out = ""
            for idx in filtered_objects:
                out += "Object #{}: {} pixels\n".format(idx, filtered_objects[idx]["area"])
            self.te_objects.setText(out)


            # draw contour of objects
            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                         filtered_objects,
                                                         marker_type=MARKER_TYPE)

            _, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)

            # draw areas
            for area in self.areas:
                if "type" in self.areas[area]:
                    if self.areas[area]["type"] == "circle":
                        cv2.circle(frame_with_objects, tuple(self.areas[area]["center"]), self.areas[area]["radius"], color=AREA_COLOR, thickness=drawing_thickness)
                        cv2.putText(frame_with_objects, self.areas[area]["name"], tuple(self.areas[area]["center"]), font, 1, AREA_COLOR, drawing_thickness, cv2.LINE_AA)

                    if self.areas[area]["type"] == "rectangle":
                        cv2.rectangle(frame_with_objects, tuple(self.areas[area]["pt1"]), tuple(self.areas[area]["pt2"]), color=AREA_COLOR, thickness=drawing_thickness)
                        cv2.putText(frame_with_objects, self.areas[area]["name"], tuple(self.areas[area]["pt1"]), font, 1, AREA_COLOR, drawing_thickness, cv2.LINE_AA)

                    if self.areas[area]["type"] == "polygon":
                        for idx, point in enumerate(self.areas[area]["points"][:-1]):
                            cv2.line(frame_with_objects, tuple(point), tuple(self.areas[area]["points"][idx + 1]), color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                        cv2.line(frame_with_objects, tuple(self.areas[area]["points"][-1]), tuple(self.areas[area]["points"][0]), color=RED, lineType=8, thickness=drawing_thickness)
                        cv2.putText(frame_with_objects, self.areas[area]["name"], tuple(self.areas[area]["points"][0]), font, 1, AREA_COLOR, drawing_thickness, cv2.LINE_AA)

            # draw arena
            if self.arena:

                if self.arena["type"] == "polygon":
                    for idx, point in enumerate(self.arena["points"][:-1]):
                        cv2.line(frame_with_objects, tuple(point), tuple(self.arena["points"][idx + 1]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                    cv2.line(frame_with_objects, tuple(self.arena["points"][-1]), tuple(self.arena["points"][0]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                    #cv2.putText(modified_frame, self.arena["name"], tuple(self.arena["points"][0]), font, 0.5, ARENA_COLOR, 1, cv2.LINE_AA)

                if self.arena["type"] == "circle":
                    cv2.circle(frame_with_objects, tuple(self.arena["center"]), self.arena["radius"], color=ARENA_COLOR, thickness=drawing_thickness)
                    #cv2.putText(modified_frame, self.arena["name"], tuple(self.arena["center"]), font, 0.5, ARENA_COLOR, 1, cv2.LINE_AA)

                if self.arena["type"] == "rectangle":
                    cv2.rectangle(frame_with_objects, tuple(self.arena["points"][0]), tuple(self.arena["points"][1]), color=ARENA_COLOR, thickness=drawing_thickness)


                '''
                for idx, point in enumerate(self.arena[:-1]):
                    cv2.line(modified_frame, tuple(point), tuple(self.arena[idx + 1]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                cv2.line(modified_frame, tuple(self.arena[-1]), tuple(self.arena[0]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                '''

            # display frames
            self.display_frame(frame_with_objects)
            self.display_treated_frame(treated_frame)

            app.processEvents()

        return True, {"frame": self.frame_idx, "objects": filtered_objects}


    def analysis(self, results):
        """
        analyze current frame
        """

        if self.cb_record_xy.isChecked():
            pos =  []

            for idx in sorted(list(results["objects"].keys())):
                pos.append(results["objects"][idx]["centroid"])

            ''''
            if self.positions:
                pos = [""] * len(results["objects"])
                for idx in results["objects"]:
                    r_dist = []

                    print("idx", idx, "centroid", results["objects"][idx]["centroid"])
                    print("previous pos:", self.positions[-1])

                    for idx_obj, obj in enumerate(self.positions[-1]):
                        print(idx, idx_obj, euclidean_distance(results["objects"][idx]["centroid"], obj))
                        r_dist.append([euclidean_distance(results["objects"][idx]["centroid"], obj), idx_obj, idx])
                    r_dist = sorted(r_dist)

                    print("r_dist", r_dist)

                    for j in range(len(pos)):
                        if pos[r_dist[j][1]] == "":
                            pos[r_dist[j][1]] = results["objects"][idx]["centroid"]
                            break
                print("pos", pos)
                self.positions.append(pos)

            if not self.positions:
                for idx in results["objects"]:
                    pos.append(results["objects"][idx]["centroid"])
                self.positions.append(pos)

            print("============")
            print()
            '''

            self.positions.append(pos)

            out = ""
            for p in pos:
                out += "{},{}\t".format(p[0], p[1])
            self.te_xy.append(out.strip())

            '''
            if self.cb_display_analysis.isChecked():
                self.te_xy.setText("{} positions recorded".format(len(self.positions)))
            '''


        if self.cb_record_number_objects.isChecked():

            nb = {}

            for area in sorted(self.areas.keys()):

                nb[area] = 0

                if self.areas[area]["type"] == "circle":
                    cx, cy = self.areas[area]["center"]
                    radius = self.areas[area]["radius"]

                    for idx in results["objects"]:
                        x, y = results["objects"][idx]["centroid"]

                        if ((cx - x)**2 + (cy - y)**2)**.5 <= radius:
                            print("object #{} in area {}".format(idx, area))
                            nb[area] += 1

                if self.areas[area]["type"] == "rectangle":
                    minx, miny = self.areas[area]["pt1"]
                    maxx, maxy = self.areas[area]["pt2"]

                    for idx in results["objects"]:
                        x, y = results["objects"][idx]["centroid"]

                        if minx <= x <= maxx and miny <= y <= maxy:
                            print("object #{} in area {}".format(idx, area))
                            nb[area] += 1

                if self.areas[area]["type"] == "polygon":
                    for idx in results["objects"]:
                        x, y = results["objects"][idx]["centroid"]
                        if cv2.pointPolygonTest(np.array(self.areas[area]["points"]), (x, y), False) >= 0:
                            nb[area] += 1

            self.objects_number.append(nb)

            out = "frame: {}\t".format(self.frame_idx)
            for area in sorted(self.areas.keys()):
                out += "{area}: {nb}\t".format(area=area, nb=nb[area])
            self.te_number_objects.append(out)


    def run_analysis(self):
        """
        run analysis from current frame to end
        """

        if self.flag_stop_analysis:
            return

        if not self.dir_images:
            try:
                self.capture
            except:
                QMessageBox.warning(self, "DORIS", "No video")
                return

        #self.positions, self.objects_number = [], []

        while True:
            r, results = self.pb()
            if not r:
                break

            app.processEvents()
            if self.flag_stop_analysis:
                break

            self.analysis(results)

        '''
        if self.cb_record_xy.isChecked():
            self.te_xy.setText("{} positions recorded".format(len(self.positions)))

        if self.cb_record_number_objects.isChecked():
            self.te_number_objects.setText("{} records".format(len(self.objects_number)))
        '''



        #if not self.flag_stop_analysis:
        #    QMessageBox.information(self, "Objects tracker", "All frames were processed")

        self.flag_stop_analysis = False


    def stop(self):
        """
        stop analysis
        """
        self.flag_stop_analysis = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DORIS (Detection of Objects Research Interactive Software)")

    parser.add_argument("-v", action="store_true", default=False, dest="version", help="Print version")
    parser.add_argument("-i", action="store",  dest="video_file", help="path of video file")
    parser.add_argument("-d", action="store",  dest="directory", help="path of images directory")
    parser.add_argument("--areas", action="store", dest="areas_file", help="path of file containing the areas definition")
    parser.add_argument("--arena", action="store", dest="arena_file", help="path of file containing the arena definition")
    parser.add_argument("--threshold", action="store", default=50, dest="threshold", help="Threshold value")
    parser.add_argument("--blur", action="store", dest="blur", help="Blur value")
    parser.add_argument("--invert", action="store_true", dest="invert", help="Invert B/W")

    results = parser.parse_args()
    if results.version:
        print("version {} release date: {}".format(__version__, __version_date__))
        sys.exit()

    app = QApplication(sys.argv)
    w = Ui_MainWindow()

    if results.blur:
        w.sb_blur.setValue(int(results.blur))

    if results.threshold:
        w.sb_threshold.setValue(int(results.threshold))

    print("results.invert", results.invert)
    w.cb_invert.setChecked(results.invert)


    if results.video_file:
        if os.path.isfile(results.video_file):
            w.open_video(results.video_file)
        else:
            print("{} not found".format(results.video_file))
            sys.exit()

    if results.directory:
        if os.path.isdir(results.directory):
            w.load_dir_images(results.directory)
        else:
            print("{} directory not found".format(results.directory))
            sys.exit()

    if results.areas_file:
        if os.path.isfile(results.areas_file):
            w.open_areas(results.areas_file)
        else:
            print("{} not found".format(results.areas_file))
            sys.exit()

    if results.arena_file:
        if os.path.isfile(results.arena_file):
            with open(results.arena_file)  as f:
                content = f.read()
            w.le_arena.setText(content)

            w.arena = eval(content)

            w.pb_define_arena.setEnabled(False)
            w.pb_clear_arena.setEnabled(True)
        else:
            print("{} not found".format(results.arena_file))
            sys.exit()

    w.show()
    w.raise_()
    w.pb()
    sys.exit(app.exec_())
