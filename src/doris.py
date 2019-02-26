#!/usr/bin/env python3
"""
DORIS
Detection of Objects Research Interactive Software
Copyright 2017-2019 Olivier Friard

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


from PyQt5.QtCore import Qt, QT_VERSION_STR, PYQT_VERSION_STR, pyqtSignal, QEvent
from PyQt5.QtGui import (QPixmap, QImage, qRgb)
from PyQt5.QtWidgets import (QMainWindow, QApplication, QStatusBar, QDialog,
                             QMenu, QFileDialog, QMessageBox, QInputDialog,
                             QWidget, QVBoxLayout, QLabel, QSpacerItem,
                             QSizePolicy, QCheckBox, QHBoxLayout, QPushButton,
                             QMessageBox)

import logging
import os
import platform
import json
import numpy as np
import pandas as pd
import cv2
import copy

import sys
import time
import pathlib
import datetime as dt
import math
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.figure import Figure

import argparse
import itertools

import doris_functions
import version
from config import *
import dialog
from doris_ui import Ui_MainWindow


logging.basicConfig(format='%(asctime)s,%(msecs)d  %(module)s l.%(lineno)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

COLORS_LIST = doris_functions.COLORS_LIST


class Input_dialog(QDialog):
    """
    dialog for user input. Elements can be checkbox, lineedit, spinbox

    """

    def __init__(self, label_caption, elements_list):
        super().__init__()

        hbox = QVBoxLayout()
        self.label = QLabel()
        self.label.setText(label_caption)
        hbox.addWidget(self.label)

        self.elements = {}
        for element in elements_list:
            if element[0] == "cb":
                self.elements[element[1]] = QCheckBox(element[1])
                self.elements[element[1]].setChecked(element[2])
                hbox.addWidget(self.elements[element[1]])
            if element[0] == "le":
                lb = QLabel(element[1])
                hbox.addWidget(lb)
                self.elements[element[1]] = QLineEdit()
                hbox.addWidget(self.elements[element[1]])
            if element[0] == "sb":
                lb = QLabel(element[1])
                hbox.addWidget(lb)
                self.elements[element[1]] = QSpinBox()
                self.elements[element[1]].setRange(element[2], element[3])
                self.elements[element[1]].setSingleStep(element[4])
                self.elements[element[1]].setValue(element[5])
                hbox.addWidget(self.elements[element[1]])

        hbox2 = QHBoxLayout()

        self.pbCancel = QPushButton("Cancel")
        self.pbCancel.clicked.connect(self.reject)
        hbox2.addWidget(self.pbCancel)

        self.pbOK = QPushButton("OK")
        self.pbOK.clicked.connect(self.accept)
        self.pbOK.setDefault(True)
        hbox2.addWidget(self.pbOK)

        hbox.addLayout(hbox2)

        self.setLayout(hbox)

        self.setWindowTitle("title")


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
        super().__init__()

        self.vbox = QVBoxLayout()

        self.lb_frame = Click_label()
        self.lb_frame.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.vbox.addWidget(self.lb_frame)

        self.setLayout(self.vbox)


    def closeEvent(self, event):
        #qm = QMessageBox
        #ret = qm.question(self,'', "Are you sure to close this window?", qm.Yes | qm.No)

        #if ret == qm.Yes:
        event.accept()

    '''
    def pbOK_clicked(self):
        self.close()
    '''

font = FONT

def frame2pixmap(frame):
    """
    convert np.array (frame) to QT pixmap
    """
    try:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
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
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                return qim.copy() if copy else qim
            elif im.shape[2] == 4:
                qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                return qim.copy() if copy else qim


class Ui_MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):

        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.setWindowTitle("DORIS v. {} - (c) Olivier Friard".format(version.__version__))
        self.statusBar = QStatusBar()
        self.statusBar.setStyleSheet("font-size:24px")
        self.setStatusBar(self.statusBar)
        self.actionAbout.triggered.connect(self.about)

        self.action1.triggered.connect(lambda: self.frame_viewer_scale(0, 1))
        self.action1_2.triggered.connect(lambda: self.frame_viewer_scale(0, 0.5))
        self.action1_4.triggered.connect(lambda: self.frame_viewer_scale(0, 0.25))
        self.action2.triggered.connect(lambda: self.frame_viewer_scale(0, 2))

        self.action_treated_1.triggered.connect(lambda: self.frame_viewer_scale(1, 1))
        self.action_treated_1_2.triggered.connect(lambda: self.frame_viewer_scale(1, 0.5))
        self.action_treated_1_4.triggered.connect(lambda: self.frame_viewer_scale(1, 0.25))
        self.action_treated_2.triggered.connect(lambda: self.frame_viewer_scale(1, 2))

        self.actionDraw_reference.triggered.connect(self.process_and_show)
        self.actionShow_centroid_of_object.triggered.connect(self.process_and_show)
        self.actionShow_contour_of_object.triggered.connect(self.process_and_show)
        self.actionOrigin_from_point.triggered.connect(self.define_coordinate_center_1point)
        self.actionOrigin_from_center_of_3_points_circle.triggered.connect(self.define_coordinate_center_3points_circle)
        self.actionSelect_objects_to_track.triggered.connect(lambda: self.select_objects_to_track(all_=False))
        self.actionDefine_scale.triggered.connect(self.define_scale)

        self.actionOpen_video.triggered.connect(lambda: self.open_video(""))
        self.actionLoad_directory_of_images.triggered.connect(self.load_dir_images)
        self.actionOpen_project.triggered.connect(self.open_project)
        self.actionSave_project.triggered.connect(self.save_project)
        self.actionSave_project_as.triggered.connect(self.save_project_as)
        self.actionQuit.triggered.connect(self.close)

        self.hs_frame.setMinimum(1)
        self.hs_frame.setMaximum(100)
        self.hs_frame.setValue(1)
        self.hs_frame.sliderMoved.connect(self.hs_frame_moved)

        self.pb_next_frame.clicked.connect(self.next_frame)
        self.pb_1st_frame.clicked.connect(self.reset)

        self.pb_define_scale.clicked.connect(self.define_scale)
        self.pb_reset_scale.clicked.connect(self.reset_scale)

        # menu for Define origin button
        menu = QMenu()
        menu.addAction("Origin from center of 3 points circle", self.define_coordinate_center_3points_circle)
        menu.addAction("Origin from a point", self.define_coordinate_center_1point)
        self.pb_define_origin.setMenu(menu)

        self.pb_reset_origin.clicked.connect(self.reset_origin)

        self.pb_goto_frame.clicked.connect(self.pb_go_to_frame)

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

        self.pb_run_tracking.clicked.connect(self.run_tracking)
        self.pb_run_tracking_frame_interval.clicked.connect(self.run_tracking_frames_interval)
        self.pb_stop.clicked.connect(self.stop)

        self.cb_threshold_method.currentIndexChanged.connect(self.threshold_method_changed)

        self.sb_blur.valueChanged.connect(self.process_and_show)
        self.cb_invert.stateChanged.connect(self.process_and_show)

        self.sb_threshold.valueChanged.connect(self.process_and_show)

        self.sb_block_size.valueChanged.connect(self.process_and_show)
        self.sb_offset.valueChanged.connect(self.process_and_show)

        self.cb_background.stateChanged.connect(self.background)

        self.sbMin.valueChanged.connect(self.process_and_show)
        self.sbMax.valueChanged.connect(self.process_and_show)

        self.sb_max_extension.valueChanged.connect(self.process_and_show)

        self.sb_percent_out_of_arena.valueChanged.connect(self.process_and_show)

        self.pb_show_all_objects.clicked.connect(self.show_all_objects)
        self.pb_show_all_filtered_objects.clicked.connect(self.show_all_filtered_objects)
        self.pb_separate_objects.clicked.connect(self.force_objects_number)
        self.pb_select_objects_to_track.clicked.connect(self.select_objects_to_track)
        self.pb_track_all_filtered.clicked.connect(lambda: self.select_objects_to_track(all_=True))

        # coordinates analysis
        self.pb_reset_xy.clicked.connect(self.reset_xy_analysis)
        self.pb_save_xy.clicked.connect(self.save_objects_positions)
        self.pb_plot_path.clicked.connect(lambda: self.plot_path_clicked("path"))
        self.pb_plot_positions.clicked.connect(lambda: self.plot_path_clicked("positions"))
        self.pb_plot_xy_density.clicked.connect(self.plot_xy_density)
        self.pb_distances.clicked.connect(self.distances)

        # menu for area button
        menu = QMenu()
        menu.addAction("Rectangle", lambda: self.add_area_func("rectangle"))
        menu.addAction("Circle (center radius)", lambda: self.add_area_func("circle (center radius)"))
        menu.addAction("Circle (3 points)", lambda: self.add_area_func("circle (3 points)"))
        menu.addAction("Polygon", lambda: self.add_area_func("polygon"))

        self.pb_add_area.setMenu(menu)
        self.pb_remove_area.clicked.connect(self.remove_area)
        self.pb_reset_areas.clicked.connect(self.reset_areas_analysis)
        self.pb_save_areas.clicked.connect(self.save_areas)
        self.pb_active_areas.clicked.connect(self.activate_areas)
        self.pb_save_objects_number.clicked.connect(self.save_objects_areas)
        self.pb_time_in_areas.clicked.connect(self.time_in_areas)

        self.always_skip_frame = False
        self.frame = None
        self.capture = None
        self.output = ""
        self.videoFileName = ""
        self.coord_df = None
        self.areas_df = None
        self.fgbg = None
        self.flag_stop_analysis = False
        self.video_height = 0
        self.video_width = 0
        self.frame_width = VIEWER_WIDTH
        self.total_frame_nb = 0
        self.fps = 0
        self.areas = {}
        self.flag_define_arena = False
        self.flag_define_coordinate_center_1point = False
        self.flag_define_coordinate_center_3points = False
        self.flag_define_scale = False
        self.coordinate_center = [0, 0]
        self.coordinate_center_def = []
        self.scale_points = []
        self.add_area = {}
        self.arena = {}
        '''self.mem_filtered_objects = {}'''
        self.all_objects = {}
        self.objects_to_track = {}
        self.scale = 1

        self.dir_images = []
        self.dir_images_index = 0

        self.objects_number = []

        self.mem_position_objects = {}

        self.project_path = ""

        # default
        self.sb_threshold.setValue(THRESHOLD_DEFAULT)

        self.fw = []
        self.fw.append(FrameViewer())
        self.fw[0].setWindowTitle("Original frame")
        self.fw[0].lb_frame.mouse_pressed_signal.connect(self.frame_mousepressed)
        self.fw[0].setGeometry(100, 100, 512, 512)
        # self.fw[0].show()

        self.fw.append(FrameViewer())
        self.fw[1].setGeometry(640, 100, 512, 512)
        self.fw[1].setWindowTitle("Processed frame")

        self.running_tracking = False

        self.threshold_method_changed()

        self.sb_percent_out_of_arena.setValue(int(TOLERANCE_OUTSIDE_ARENA * 100))

        self.frame_scale = DEFAULT_FRAME_SCALE
        self.processed_frame_scale = DEFAULT_FRAME_SCALE


    def about(self):
        """
        About dialog box
        """

        modules = []
        modules.append("OpenCV")
        modules.append(f"version {cv2.__version__}")

        # matplotlib
        modules.append("\nMatplotlib")
        modules.append(f"version {matplotlib.__version__}")

        about_dialog = msg = QMessageBox()
        # about_dialog.setIconPixmap(QPixmap(os.path.dirname(os.path.realpath(__file__)) + "/logo_eye.128px.png"))
        about_dialog.setWindowTitle("About DORIS")
        about_dialog.setStandardButtons(QMessageBox.Ok)
        about_dialog.setDefaultButton(QMessageBox.Ok)
        about_dialog.setEscapeButton(QMessageBox.Ok)

        about_dialog.setInformativeText((f"<b>DORIS</b> v. {version.__version__} - {version.__version_date__}"
        "<p>Copyright &copy; 2017-2018 Olivier Friard<br>"
        "Department of Life Sciences and Systems Biology<br>"
        "University of Torino - Italy<br>"))

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


    def hs_frame_moved(self):
        """
        slider moved by user
        """
        self.le_goto_frame.setText(str(self.hs_frame.value()))
        self.pb_go_to_frame()


    def threshold_method_changed(self):
        """
        threshold method changed
        """

        for w in [self.sb_threshold]:
            w.setEnabled(self.cb_threshold_method.currentIndex() == 2)  # Simple threshold

        for w in [self.sb_block_size, self.sb_offset]:
            w.setEnabled(self.cb_threshold_method.currentIndex() != 2)  # Simple threshold

        self.process_and_show()


    def save_project_as(self):
        """
        save project as
        """
        project_file_path, _ = QFileDialog().getSaveFileName(self, "Save project", "",
                                                             "DORIS projects (*.doris);All files (*)")
        if not project_file_path:
            return
        self.project_path = project_file_path
        self.save_project()


    def save_project(self):
        """
        save parameters of current project in a text file
        """
        if not self.project_path:
            project_file_path, _ = QFileDialog().getSaveFileName(self, "Save project", "",
                                                                 "DORIS projects (*.doris);All files (*)")
            if not project_file_path:
                return
        else:
            project_file_path = self.project_path

        config = {}
        if self.videoFileName:
            config["video_file_path"] = self.videoFileName
        if self.dir_images:
            config["dir_images"] = str(self.dir_images[0].parent)

        config["blur"] = self.sb_blur.value()
        config["invert"] = self.cb_invert.isChecked()
        config["arena"] = self.arena
        config["min_object_size"] = self.sbMin.value()
        config["max_object_size"] = self.sbMax.value()
        config["percent_out_of_arena"] = self.sb_percent_out_of_arena.value()
        config["object_max_extension"] = self.sb_max_extension.value()
        config["threshold_method"] = THRESHOLD_METHODS[self.cb_threshold_method.currentIndex()]
        config["block_size"] = self.sb_block_size.value()
        config["offset"] = self.sb_offset.value()
        config["cut_off"] = self.sb_threshold.value()
        config["normalize_coordinates"] = self.cb_normalize_coordinates.isChecked()
        config["areas"] = self.areas
        config["referential_system_origin"] = self.coordinate_center
        config["scale"] = self.scale
        config["show_centroid"] = self.actionShow_centroid_of_object.isChecked()
        config["show_contour"] = self.actionShow_contour_of_object.isChecked()
        config["show_reference"] = self.actionDraw_reference.isChecked()
        config["max_distance"] = self.sb_max_distance.value()
        config["frame_scale"] = self.frame_scale
        config["processed_frame_scale"] = self.processed_frame_scale

        '''
        config["record_number_of_objects_by_area"] = self.cb_record_number_objects.isChecked()
        config["record_objects_coordinates"] = self.cb_record_xy.isChecked()
        '''

        try:
            with open(project_file_path, "w") as f_out:
                f_out.write(json.dumps(config))
            self.project_path = project_file_path
            self.setWindowTitle(f"DORIS v. {version.__version__} - {self.project_path}")
        except:
            logging.critical(f"project not saved: {project_file_path}")
            QMessageBox.critical(self, "DORIS", f"project not saved: {project_file_path}")


    def frame_viewer_scale(self, fw_idx, scale):
        """
        change scale of frame viewer
        """

        self.fw[fw_idx].show()
        self.fw[fw_idx].lb_frame.clear()
        self.fw[fw_idx].lb_frame.resize(int(self.frame.shape[1] * scale), int(self.frame.shape[0] * scale))
        if fw_idx == 0:
            self.fw[fw_idx].lb_frame.setPixmap(frame2pixmap(self.frame).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                               Qt.KeepAspectRatio))
            self.frame_width = self.fw[fw_idx].lb_frame.width()
            self.frame_scale = scale

        if fw_idx == 1:
            processed_frame = self.frame_processing(self.frame)
            self.fw[1].lb_frame.setPixmap(QPixmap.fromImage(toQImage(processed_frame)).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                                              Qt.KeepAspectRatio))
            self.processed_frame_scale = scale

        self.fw[fw_idx].setFixedSize(self.fw[fw_idx].vbox.sizeHint())
        self.process_and_show()


    def for_back_ward(self, direction="forward"):

        logging.debug("function: for_back_ward")

        step = self.sb_frame_offset.value() - 1 if direction == "forward" else -self.sb_frame_offset.value() - 1

        if self.dir_images:
            if 0 < self.dir_images_index + step < len(self.dir_images):
                self.dir_images_index += step
            self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)
        elif self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) + step)

        self.pb()


    def go_to_frame(self, frame_nb: int):
        """
        load frame and visualize it
        """
        if self.dir_images:
            self.dir_images_index = frame_nb
            self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)
        elif self.capture is not None:
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
            except Exception:
                logging.debug("exception in function go_to_frame")
                pass


    def pb_go_to_frame(self):

        logging.debug("function: pb_go_to_frame")

        if self.le_goto_frame.text():
            try:
                int(self.le_goto_frame.text())
            except ValueError:
                return
            self.go_to_frame(frame_nb=int(self.le_goto_frame.text()) - 1)
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
                    msg = "New polygon area: click on the video to define the vertices of the polygon. Right click to finish"
                if shape == "rectangle":
                    msg = "New rectangle area: click on the video to define 2 opposite vertices."

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
        switch to define arena mode
        """

        logging.debug("function: define_arena")

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
        """
        reload frame and show
        """

        logging.debug("function: reload_frame")

        if self.dir_images:
            self.dir_images_index -= 1
        elif self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.pb()


    def clear_arena(self):
        """
        clear the arena
        """
        self.flag_define_arena = False
        self.arena = {}
        self.le_arena.setText("")

        self.pb_define_arena.setEnabled(True)
        self.pb_clear_arena.setEnabled(False)

        self.reload_frame()


    def ratio_thickness(self, video_width: int, frame_width: int) -> (float, int):
        """
        return ratio and pen thickness for contours according to video resolution
        """

        logging.debug(f"video_width: {video_width}, frame_width: {frame_width}")

        ratio = video_width / frame_width
        if ratio <= 1:
            drawing_thickness = 1
        else:
            drawing_thickness = round(ratio)

        logging.debug(f"ratio: {ratio}, drawing_thickness: {drawing_thickness}")

        return ratio, drawing_thickness


    def draw_point_origin(self, frame, position, color, drawing_thickness):
        """
        draw a point (circle and cross) on frame
        """

        position = tuple(position)
        cv2.circle(frame, position, 8,
                   color=color, lineType=8, thickness=drawing_thickness)

        cv2.line(frame, (position[0], position[1] - 0),
                        (position[0], position[1] + 30),
                 color=color, lineType=8, thickness=drawing_thickness)

        cv2.line(frame, (position[0], position[1]),
                        (position[0] + 30, position[1]),
                 color=color, lineType=8, thickness=drawing_thickness)

        return frame


    def draw_circle_cross(self, frame, position, color, drawing_thickness):
        """
        draw a cross with circle on frame
        """

        cross_length = 20
        position = tuple(position)
        cv2.circle(frame, position, cross_length // 4,
                   color=color, lineType=8, thickness=drawing_thickness)

        cv2.line(frame, (position[0], position[1] - cross_length),
                        (position[0], position[1] + cross_length),
                 color=color, lineType=8, thickness=drawing_thickness)

        cv2.line(frame, (position[0] - cross_length, position[1]),
                        (position[0] + cross_length, position[1]),
                 color=color, lineType=8, thickness=drawing_thickness)

        return frame



    def frame_mousepressed(self, event):
        """
        record clicked coordinates if arena or area mode activated
        """

        logging.debug("function: frame_mousepressed")
        conversion, drawing_thickness = self.ratio_thickness(self.video_width, self.fw[0].lb_frame.pixmap().width())

        # frame = self.frame.copy()

        # set coordinate center with 1 point
        if self.flag_define_coordinate_center_1point:
            self.coordinate_center = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
            self.frame = self.draw_point_origin(self.frame, self.coordinate_center, BLUE, drawing_thickness)

            #self.display_frame(self.frame)
            self.flag_define_coordinate_center_1point = False
            self.actionOrigin_from_point.setText("from point")
            self.le_coordinates_center.setText(f"{self.coordinate_center}")
            self.reload_frame()
            self.statusBar.showMessage(f"Referential origin defined")

        # set coordinate center with 3 points circle
        if self.flag_define_coordinate_center_3points:
            self.coordinate_center_def.append((int(event.pos().x() * conversion), int(event.pos().y() * conversion)))
            cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=BLUE, lineType=8, thickness=drawing_thickness)

            self.display_frame(self.frame)

            if len(self.coordinate_center_def) == 3:

                x, y, _ = doris_functions.find_circle(self.coordinate_center_def)
                self.coordinate_center = [int(x), int(y)]
                self.frame = self.draw_point_origin(self.frame, self.coordinate_center, BLUE, drawing_thickness)

                self.coordinate_center_def = []
                self.flag_define_coordinate_center_3point3 = False
                self.actionOrigin_from_center_of_3_points_circle.setText("from center of 3 points circle")
                self.le_coordinates_center.setText(f"{self.coordinate_center}")
                self.reload_frame()
                self.statusBar.showMessage(f"Referential origin defined")

        # set scale
        if self.flag_define_scale:
            if len(self.scale_points) < 2:
                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

                self.scale_points.append((int(event.pos().x() * conversion), int(event.pos().y() * conversion)))

                if len(self.scale_points) == 2:
                    cv2.line(self.frame, self.scale_points[0], self.scale_points[1],
                             color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                    self.display_frame(self.frame)
                    self.flag_define_scale = False
                    self.actionDefine_scale.setText("Define scale")
                    while True:
                        real_length_str, ok_pressed = QInputDialog.getText(self, "Real length", "Value (w/o unit):")
                        if not ok_pressed:
                            return
                        try:
                            float(real_length_str)
                            break
                        except:
                            QMessageBox.warning(self, "DORIS", f"{real_length_str} was not recognized as length")
                    self.scale = float(real_length_str) / doris_functions.euclidean_distance(self.scale_points[0], self.scale_points[1])
                    self.le_scale.setText(f"{self.scale:0.5f}")
                    self.scale_points = []
                    self.reload_frame()
                    self.statusBar.showMessage(f"Scale defined: {self.scale:0.5f}")


        # add area
        if self.add_area:

            if event.button() == 4:
                self.add_area = {}
                self.statusBar.showMessage("New area canceled")
                self.reload_frame()
                return

            if event.button() == 1:
                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=AREA_COLOR, lineType=8, thickness=drawing_thickness)

                # arena
                self.frame = self.draw_arena(self.frame, drawing_thickness)

                # draw areas
                self.frame = self.draw_areas(self.frame, drawing_thickness)

                # origin
                if self.coordinate_center != [0, 0]:
                    self.frame = self.draw_point_origin(self.frame, self.coordinate_center, BLUE, drawing_thickness)

                self.display_frame(self.frame)


            if self.add_area["type"] == "circle (center radius)":
                if "center" not in self.add_area:
                    self.add_area["center"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["type"] = "circle"
                    self.add_area["radius"] = int(((event.pos().x() * conversion - self.add_area["center"][0]) ** 2
                                                    + (event.pos().y() * conversion - self.add_area["center"][1]) ** 2) ** 0.5)
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

                if len(self.add_area["points"]) == 3:
                    cx, cy, radius = doris_functions.find_circle(self.add_area["points"])
                    self.add_area["type"] = "circle"
                    self.add_area["center"] = [int(cx), int(cy)]
                    self.add_area["radius"] = int(radius)
                    del self.add_area["points"]
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.reload_frame()
                    self.statusBar.showMessage("New circle area created")
                    return

            if self.add_area["type"] == "rectangle":
                if "pt1" not in self.add_area:
                    self.add_area["pt1"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["pt2"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                    # reorder vortex
                    print(self.add_area)
                    self.add_area["pt1"], self.add_area["pt2"] = ([min(self.add_area["pt1"][0], self.add_area["pt2"][0]), min(self.add_area["pt1"][1],self.add_area["pt2"][1])],
                                                                 [max(self.add_area["pt1"][0], self.add_area["pt2"][0]), max(self.add_area["pt1"][1], self.add_area["pt2"][1])])
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.reload_frame()
                    self.statusBar.showMessage("New rectangle area created")
                    return

            if self.add_area["type"] == "polygon":

                if event.button() == 2:  # right click to finish
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.reload_frame()
                    self.statusBar.showMessage("The new polygon area is defined")
                    return

                if event.button() == 1:  # left button
                    '''
                    cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                               color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                    '''
                    if "points" not in self.add_area:
                        self.add_area["points"] = [[int(event.pos().x() * conversion), int(event.pos().y() * conversion)]]
                    else:
                        self.add_area["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                        cv2.line(self.frame, tuple(self.add_area["points"][-2]), tuple(self.add_area["points"][-1]),
                                 color=AREA_COLOR, lineType=8, thickness=drawing_thickness)

                    self.statusBar.showMessage(("Polygon area: {} point(s) selected."
                                                " Right click to finish").format(len(self.add_area["points"])))
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
                if "points" not in self.arena:
                    self.arena["points"] = []
                self.arena["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)

                self.display_frame(self.frame)
                self.statusBar.showMessage("Rectangle arena: {} point(s) selected.".format(len(self.arena["points"])))

                if len(self.arena["points"]) == 2:
                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.pb_define_arena.setText("Define arena")
                    self.arena = {**self.arena, **{"type": "rectangle", "name": "arena"}}
                    self.le_arena.setText("{}".format(self.arena))

                    cv2.rectangle(self.frame, tuple(self.arena["points"][0]), tuple(self.arena["points"][1]),
                                  color=ARENA_COLOR, thickness=drawing_thickness)
                    #self.display_frame(self.frame)
                    self.process_and_show()
                    self.statusBar.showMessage("The rectangle arena is defined")

            if self.flag_define_arena == "polygon":

                if event.button() == 2:  # right click to finish

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.pb_define_arena.setText("Define arena")

                    self.arena = {**self.arena, **{"type": "polygon", "name": "arena"}}

                    self.le_arena.setText("{}".format(self.arena))
                    self.process_and_show()
                    self.statusBar.showMessage("The new polygon arena is defined")

                else:
                    if "points" not in self.arena:
                        self.arena["points"] = []

                    self.arena["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                    cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                               color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                    self.display_frame(self.frame)

                    self.statusBar.showMessage(("Polygon arena: {} point(s) selected. "
                                                "Right click to finish").format(len(self.arena["points"])))

                    if len(self.arena["points"]) >= 2:
                        cv2.line(self.frame, tuple(self.arena["points"][-2]), tuple(self.arena["points"][-1]),
                                 color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                        self.display_frame(self.frame)

            if self.flag_define_arena == "circle (3 points)":
                if "points" not in self.arena:
                    self.arena["points"] = []

                self.arena["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

                self.statusBar.showMessage("Circle arena: {} point(s) selected.".format(len(self.arena["points"])))

                if len(self.arena["points"]) == 3:
                    cx, cy, r = doris_functions.find_circle(self.arena["points"])
                    cv2.circle(self.frame, (int(abs(cx)), int(abs(cy))), int(r), color=ARENA_COLOR, thickness=drawing_thickness)
                    # self.display_frame(self.frame)

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.arena = {'type': 'circle', 'center': [round(cx), round(cy)], 'radius': round(r), 'name': 'arena'}

                    self.le_arena.setText("{}".format(self.arena))

                    self.process_and_show()
                    self.statusBar.showMessage("The new circle arena is defined")

            if self.flag_define_arena == "circle (center radius)":
                if "points" not in self.arena:
                    self.arena["points"] = []

                self.arena["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

                if len(self.arena["points"]) == 2:
                    cx, cy = self.arena["points"][0]
                    radius = doris_functions.euclidean_distance(self.arena["points"][0], self.arena["points"][1])
                    cv2.circle(self.frame, (int(abs(cx)), int(abs(cy))), int(radius), color=ARENA_COLOR, thickness=drawing_thickness)

                    # self.display_frame(self.frame)

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.arena = {"type": "circle", "center": [round(cx), round(cy)], "radius": round(radius), "name": "arena"}
                    self.le_arena.setText("{}".format(self.arena))
                    self.process_and_show()
                    self.statusBar.showMessage("The new circle arena is defined")


    def background(self):
        if self.cb_background.isChecked():
            self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
            print("backgound substraction activated")
        else:
            self.fgbg = None
        for w in [self.lb_threshold, self.sb_threshold]:
            w.setEnabled(not self.cb_background.isChecked())


    def reset_origin(self):
        """
        reset referential origin to (0, 0)
        """
        self.coordinate_center = [0, 0]
        self.le_coordinates_center.setText(f"{self.coordinate_center}")
        self.reload_frame()
        self.statusBar.showMessage(f"Referential origin reset")


    def reset_scale(self):
        """
        reset scale to 1
        """
        logging.debug("function: reset scale")
        self.flag_define_scale = False
        self.scale_points = []
        self.reload_frame()
        self.scale = 1
        self.le_scale.setText("1")
        self.statusBar.showMessage(f"Scale reset")


    def reset(self):
        """
        go to 1st frame
        """

        logging.debug("function: reset")

        if self.dir_images:
            self.dir_images_index = 0
        else:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.objects_to_track = {}
        self.te_tracked_objects.clear()
        self.mem_position_objects = {}

        self.reset_xy_analysis()

        self.always_skip_frame = False

        self.pb()


    def next_frame(self):
        """
        go to next frame
        """

        logging.debug("function: next_frame")

        if not self.pb():
            return


    def reset_xy_analysis(self):
        """
        reset recorded positions
        """
        if self.coord_df is not None and self.objects_to_track:
            # init dataframe for recording objects coordinates
            self.initialize_positions_dataframe()
            self.te_xy.clear()


    def save_objects_positions(self):
        """
        save results of recorded positions in TSV file
        """
        if self.coord_df is not None:
            file_name, _ = QFileDialog().getSaveFileName(self, "Save objects coordinates", "", "All files (*)")
            if file_name:
                self.coord_df.to_csv(file_name, sep="\t", decimal=".")
        else:
            QMessageBox.warning(self, "DORIS", "no positions to be saved")


    def reset_areas_analysis(self):
        """
        reset areas analysis
        """
        if self.areas_df is not None and self.objects_to_track:
            self.initialize_areas_dataframe()

            self.te_number_objects.clear()


    def save_areas(self):
        """
        save defined areas to file
        """
        file_name, _ = QFileDialog().getSaveFileName(self, "Save areas to file", "", "All files (*)")
        if file_name:
            with open(file_name, "w") as f_out:
                for idx in range(self.lw_area_definition.count()):
                    f_out.write(self.lw_area_definition.item(idx).text() + "\n")


    def time_in_areas(self):
        """
        time of objects in each area
        """
        if self.areas_df is None:
            return

        time_objects_in_areas = pd.DataFrame(index=sorted(list(self.objects_to_track)), columns = sorted(list(self.areas.keys())))

        areas_non_nan_df = self.areas_df.dropna(thresh=1)
        for area in self.areas:
            for idx in self.objects_to_track:
                time_objects_in_areas.ix[idx, area] = areas_non_nan_df[f"area {area} object #{idx}"].sum() / self.fps

        logging.debug(f"{time_objects_in_areas}")

        file_name, _ = QFileDialog().getSaveFileName(self, "Save time in areas", "", "All files (*)")
        if file_name:
            time_objects_in_areas.to_csv(file_name, sep="\t", decimal=".")


    def save_objects_areas(self):
        """
        save presence of objects in areas
        """
        if self.areas_df is None:
            QMessageBox.warning(self, "DORIS", "no objects to be saved")
            return
        file_name, _ = QFileDialog().getSaveFileName(self, "Save objects in areas", "", "All files (*)")
        if file_name:
            self.areas_df.to_csv(file_name, sep="\t", decimal=".")


    def plot_xy_density(self):

        if self.coord_df is None:
            QMessageBox.warning(self, "DORIS", "no positions recorded")
            return

        x_lim = np.array([0 - self.coordinate_center[0], self.video_width - self.coordinate_center[0]])
        y_lim = np.array([0 - self.coordinate_center[1], self.video_height - self.coordinate_center[1]])

        if self.cb_normalize_coordinates.isChecked():
            x_lim = x_lim / self.video_width
            y_lim = y_lim / self.video_width

        x_lim = x_lim * self.scale
        y_lim = y_lim * self.scale

        doris_functions.plot_density(self.coord_df,
                                     x_lim=x_lim,
                                     y_lim=y_lim)


    def distances(self):
        """
        save distances
        """
        if self.coord_df is None:
            QMessageBox.warning(self, "DORIS", "no positions recorded")
            return

        results = pd.DataFrame(index=sorted(list(self.objects_to_track)), columns = ["distance"])

        for idx in sorted(list(self.objects_to_track.keys())):
            dx = self.coord_df[f"x{idx}"] - self.coord_df[f"x{idx}"].shift(1)
            dy = self.coord_df[f"y{idx}"] - self.coord_df[f"y{idx}"].shift(1)
            dist = (dx*dx + dy*dy) ** 0.5
            results.ix[idx, "distance"] = dist.sum()

        file_name, _ = QFileDialog().getSaveFileName(self, "Save distances", "", "All files (*)")
        if file_name:
            results.to_csv(file_name, sep="\t", decimal=".")




    def plot_path_clicked(self, plot_type="path"):
        """
        plot the path or positions based on recorded coordinates
        """

        if self.coord_df is None:
            QMessageBox.warning(self, "DORIS", "no positions recorded")
            return

        x_lim = np.array([0 - self.coordinate_center[0], self.video_width - self.coordinate_center[0]])
        y_lim = np.array([0 - self.coordinate_center[1], self.video_height - self.coordinate_center[1]])

        if self.cb_normalize_coordinates.isChecked():
            x_lim = x_lim / self.video_width
            y_lim = y_lim / self.video_width

        x_lim = x_lim * self.scale
        y_lim = y_lim * self.scale

        if plot_type == "path":
            doris_functions.plot_path(self.coord_df,
                                      x_lim=x_lim,
                                      y_lim=y_lim)
        if plot_type == "positions":
            doris_functions.plot_positions(self.coord_df,
                                           x_lim=x_lim,
                                           y_lim=y_lim)


    def open_video(self, file_name):
        """
        open a video
        if file_name not provided ask user to select a file
        """

        logging.debug("function: open_video")

        if not file_name:
            file_name, _ = QFileDialog(self).getOpenFileName(self, "Open video", "", "All files (*)")
        if file_name:
            if not os.path.isfile(file_name):
                QMessageBox.critical(self, "DORIS", f"{file_name} not found")
                return

            self.capture = cv2.VideoCapture(file_name)

            if not self.capture.isOpened():
                QMessageBox.critical(self, "DORIS", "Could not open {}".format(file_name))
                return

            self.total_frame_nb = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.hs_frame.setMinimum(1)
            self.hs_frame.setMaximum(self.total_frame_nb)
            self.hs_frame.setTickInterval(10)

            # self.lb_total_frames_nb.setText("Total number of frames: <b>{}</b>".format(self.total_frame_nb))

            self.fps = self.capture.get(cv2.CAP_PROP_FPS)

            self.frame_idx = 0
            # self.update_frame_index()
            self.pb()
            self.video_height, self.video_width, _ = self.frame.shape
            self.videoFileName = file_name

            # default scale
            for idx in range(2):
                self.frame_viewer_scale(idx, DEFAULT_FRAME_SCALE)
                self.fw[idx].show()

            self.fw[0].setWindowTitle(f"Original frame - {file_name}")
            self.statusBar.showMessage("video loaded ({}x{})".format(self.video_width, self.video_height))


    def load_dir_images(self, dir_images):
        """
        Load directory of images
        """
        logging.debug("function: load_dir_images")

        if not dir_images:
            dir_images = QFileDialog(self).getExistingDirectory(self, "Select Directory")
        if dir_images:
            p = pathlib.Path(dir_images)
            self.dir_images = sorted(list(p.glob('*.jpg')) + list(p.glob('*.JPG')) + list(p.glob("*.png")) + + list(p.glob("*.PNG")))

            self.total_frame_nb = len(self.dir_images)
            self.hs_frame.setMinimum(1)
            self.hs_frame.setMaximum(self.total_frame_nb)
            self.hs_frame.setTickInterval(10)

            self.lb_frames.setText(f"<b>{self.total_frame_nb}</b> images")

            self.dir_images_index = 0
            self.pb()

            logging.info(f"self.frame.shape: {self.frame.shape}")

            self.video_height, self.video_width, _ = self.frame.shape

            # default scale
            for idx in range(2):
                self.frame_viewer_scale(idx, 0.5)
                self.fw[idx].show()

            self.fw[0].setWindowTitle(f"Original frame - {dir_images}")
            self.statusBar.showMessage(f"{len(self.dir_images)} image(s) found")


    def update_frame_index(self):
        """
        update frame index
        """
        if self.dir_images:
            self.frame_idx = self.dir_images_index
        else:
            self.frame_idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))

        self.lb_frames.setText((f"Frame: <b>{self.frame_idx}</b> / {self.total_frame_nb}&nbsp;&nbsp;&nbsp;&nbsp;"
                                f"Time: <b>{dt.timedelta(seconds=round(self.frame_idx/self.fps, 3))}</b>&nbsp;&nbsp;&nbsp;&nbsp;"
                                f"({self.frame_idx/self.fps:.3f} seconds)"))


    def frame_processing(self, frame):
        """
        apply treament to frame
        returns treated frame
        """

        threshold_method = {"name": THRESHOLD_METHODS[self.cb_threshold_method.currentIndex()],
                            "block_size": self.sb_block_size.value(),
                            "offset": self.sb_offset.value(),
                            "cut-off": self.sb_threshold.value(),
                            }

        return doris_functions.image_processing(frame,
                                                blur=self.sb_blur.value(),
                                                threshold_method=threshold_method,
                                                invert=self.cb_invert.isChecked())


    def define_coordinate_center_1point(self):
        """
        define origin of coordinates with a point
        """
        if self.frame is not None:
            self.flag_define_coordinate_center_1point = not self.flag_define_coordinate_center_1point
            self.actionOrigin_from_point.setText("from point" if not self.flag_define_coordinate_center_1point
                                                        else "Cancel origin definition")
            self.statusBar.showMessage("You have to select the origin of the referential system" * self.flag_define_coordinate_center_1point)


    def define_coordinate_center_3points_circle(self):
        """
        define origin of coordinates with a 3 points circle
        """
        if self.frame is not None:
            self.flag_define_coordinate_center_3points = not self.flag_define_coordinate_center_3points
            self.actionOrigin_from_center_of_3_points_circle.setText("from center of 3 points circle" if not self.flag_define_coordinate_center_3points
                                                        else "Cancel origin definition")
            self.statusBar.showMessage("You have to select the origin of the referential system with a 3 points circle" * self.flag_define_coordinate_center_3points)


    def define_scale(self):
        """
        define scale. from pixels to real unit
        """
        if self.frame is not None:
            self.flag_define_scale = not self.flag_define_scale
            self.actionDefine_scale.setText("Define scale" if not self.flag_define_scale else "Cancel scale definition")
            self.statusBar.showMessage("You have to select 2 points on the video" * self.flag_define_scale)


    def initialize_positions_dataframe(self):
        """
        initialize dataframe for recording objects coordinates
        """
        columns = ["tag", "frame"]
        for idx in self.objects_to_track:
            columns.extend([f"x{idx}", f"y{idx}"])
        self.coord_df = pd.DataFrame(index=range(self.total_frame_nb), columns=columns)


    def initialize_areas_dataframe(self):
        """
        initialize dataframe for recording presence of objects in areas
        """
        columns = ["tag", "frame"]
        for area in sorted(self.areas.keys()):
            for idx in self.objects_to_track:
                columns.append(f"area {area} object #{idx}")
        self.areas_df = pd.DataFrame(index=range(self.total_frame_nb), columns=columns)



    def select_objects_to_track(self, all_=False):
        """
        select objects to track and create the dataframes for recording objects positions and presence in area
        """

        if all_:
            self.objects_to_track = {}
            for idx in self.filtered_objects:
                self.objects_to_track[len(self.objects_to_track) + 1] = dict(self.filtered_objects[idx])

        else:
            elements = []
            for idx in self.filtered_objects:
                elements.append(("cb", f"Object # {idx}", False))
            ib = Input_dialog("Select the objects to track", elements)

            if not ib.exec_():
                return

            self.objects_to_track = {}
            for idx in ib.elements:
                if ib.elements[idx].isChecked():
                    self.objects_to_track[len(self.objects_to_track) + 1] = dict(self.filtered_objects[int(idx.replace("Object # ", ""))])

        self.initialize_positions_dataframe()

        self.initialize_areas_dataframe()

        self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)

        logging.debug(f"coord_df: {self.coord_df.head()}")

        logging.info(f"objects to track: {list(self.objects_to_track.keys())}")

        self.process_and_show()


    def draw_reference_clicked(self):
        """
        click on draw reference menu option
        """
        self.process_and_show()


    def draw_reference(self, frame):
        """
        draw reference (100px square) on frame
        """

        if frame is not None:
            ratio, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)
            text_height = cv2.getTextSize(text=str("1"), fontFace=FONT, fontScale=FONT_SIZE, thickness=drawing_thickness)[0][1]
            cv2.rectangle(frame, (10, 10), (110, 110), RED, 1)
            cv2.putText(frame, "100x100 px", (10, 10 + text_height), font, FONT_SIZE, RED, drawing_thickness, cv2.LINE_AA)

        return frame


    def draw_marker_on_objects(self, frame, objects, marker_type=MARKER_TYPE):
        """
        draw marker (rectangle or contour) around objects
        marker color from index of object in COLORS_LIST
        """

        ratio, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)
        for idx in objects:
            marker_color = COLORS_LIST[(idx - 1) % len(COLORS_LIST)]
            logging.debug(f"idx: {idx}, marker_color: {marker_color}")

            if self.actionShow_contour_of_object.isChecked():
                if marker_type == RECTANGLE:
                    cv2.rectangle(frame, objects[idx]["min"], objects[idx]["max"], marker_color, drawing_thickness)
                if marker_type == CONTOUR:
                    cv2.drawContours(frame, [objects[idx]["contour"]], 0, marker_color, drawing_thickness)

                cv2.putText(frame, str(idx), objects[idx]["max"], font, FONT_SIZE, marker_color, drawing_thickness, cv2.LINE_AA)

            if self.actionShow_centroid_of_object.isChecked():
                self.draw_circle_cross(frame, objects[idx]["centroid"], marker_color, drawing_thickness)
                if not self.actionShow_contour_of_object.isChecked():
                    cv2.putText(frame, str(idx), objects[idx]["centroid"], font, FONT_SIZE, marker_color, drawing_thickness, cv2.LINE_AA)

        return frame


    def display_frame(self, frame):
        """
        display the current frame in viewer
        """

        self.fw[0].lb_frame.setPixmap(frame2pixmap(frame).scaled(self.fw[0].lb_frame.size(), Qt.KeepAspectRatio))


    def display_processed_frame(self, frame):
        """
        show treated frame in viewer
        """

        self.fw[1].lb_frame.setPixmap(QPixmap.fromImage(toQImage(frame)).scaled(self.fw[1].lb_frame.size(), Qt.KeepAspectRatio))


    def draw_areas(self, frame, drawing_thickness):
        """
        draw the user defined areas
        """

        text_height = cv2.getTextSize(text=str("1"), fontFace=FONT, fontScale=FONT_SIZE, thickness=drawing_thickness)[0][1]

        for area in self.areas:
            if "type" in self.areas[area]:
                if self.areas[area]["type"] == "circle":
                    cv2.circle(frame, tuple(self.areas[area]["center"]), self.areas[area]["radius"],
                               color=AREA_COLOR, thickness=drawing_thickness)
                    cv2.putText(frame, self.areas[area]["name"], tuple((self.areas[area]["center"][0] + self.areas[area]["radius"], self.areas[area]["center"][1])),
                                font, FONT_SIZE, AREA_COLOR, drawing_thickness, cv2.LINE_AA)

                if self.areas[area]["type"] == "rectangle":
                    cv2.rectangle(frame, tuple(self.areas[area]["pt1"]), tuple(self.areas[area]["pt2"]),
                                  color=AREA_COLOR, thickness=drawing_thickness)
                    cv2.putText(frame, self.areas[area]["name"], tuple((self.areas[area]["pt1"][0], self.areas[area]["pt1"][1] + text_height)),
                                font, FONT_SIZE, AREA_COLOR, drawing_thickness, cv2.LINE_AA)

                if self.areas[area]["type"] == "polygon":
                    for idx, point in enumerate(self.areas[area]["points"][:-1]):
                        cv2.line(frame, tuple(point), tuple(self.areas[area]["points"][idx + 1]),
                                 color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                    cv2.line(frame, tuple(self.areas[area]["points"][-1]), tuple(self.areas[area]["points"][0]),
                             color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                    cv2.putText(frame, self.areas[area]["name"], tuple(self.areas[area]["points"][0]),
                                font, FONT_SIZE, AREA_COLOR, drawing_thickness, cv2.LINE_AA)
        return frame



    def draw_arena(self, frame, drawing_thickness):
        """
        draw arena
        """
        if self.arena:
            if self.arena["type"] == "polygon":
                for idx, point in enumerate(self.arena["points"][:-1]):
                    cv2.line(frame, tuple(point), tuple(self.arena["points"][idx + 1]),
                             color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                cv2.line(frame, tuple(self.arena["points"][-1]), tuple(self.arena["points"][0]),
                         color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)

            if self.arena["type"] == "circle":
                cv2.circle(frame, tuple(self.arena["center"]), self.arena["radius"],
                           color=ARENA_COLOR, thickness=drawing_thickness)

            if self.arena["type"] == "rectangle":
                cv2.rectangle(frame, tuple(self.arena["points"][0]), tuple(self.arena["points"][1]),
                              color=ARENA_COLOR, thickness=drawing_thickness)

        return frame


    def process_and_show(self):
        """
        process frame and show results
        """

        logging.debug(f"function: process_and_show    self.frame is None: {self.frame is None}")

        if self.frame is None:
            return

        processed_frame = self.frame_processing(self.frame)

        (all_objects, filtered_objects) = doris_functions.detect_and_filter_objects(frame=processed_frame,
                                                                  min_size=self.sbMin.value(),
                                                                  max_size=self.sbMax.value(),
                                                                  arena=self.arena,
                                                                  max_extension=self.sb_max_extension.value(),
                                                                  tolerance_outside_arena=self.sb_percent_out_of_arena.value()/100
                                                                 )

        logging.info(f"number of all filtered objects: {len(filtered_objects)}")
        logging.info(f"self.objects_to_track: {list(self.objects_to_track.keys())}")

        # check filtered objects number
        # no filtered object
        if len(filtered_objects) == 0 and len(self.objects_to_track):
            logging.info("No filtered objects")
            self.statusBar.showMessage("No filtered objects!")
            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                             {},
                                                             marker_type=MARKER_TYPE)
            self.display_frame(frame_with_objects)
            self.display_processed_frame(processed_frame)
            self.flag_stop_analysis = True
            QMessageBox.critical(self, "DORIS", "No object detected")
            return

        # filtered object are less than objects to track
        # apply clustering
        if len(filtered_objects) < len(self.objects_to_track):

            # if self.frame_idx - 1 not in self.mem_position_objects:
            if True:  # disabled aggregation of points to previous centroid due to a bug
                logging.info("Filtered object are less than objects to track: applying k-means clustering")

                contours_list = [filtered_objects[x]["contour"] for x in filtered_objects]
                new_contours = doris_functions.apply_k_means(contours_list, len(self.objects_to_track))

                new_filtered_objects = {}
                # add info to objects: centroid, area ...
                for idx, cnt in enumerate(new_contours):
                    # print("cnt", type(cnt))

                    cnt = cv2.convexHull(cnt)
                    M = cv2.moments(cnt)

                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    n = np.vstack(cnt).squeeze()
                    try:
                        x, y = n[:, 0], n[:, 1]
                    except Exception:
                        x = n[0]
                        y = n[1]

                    new_filtered_objects[idx + 1] = {"centroid": (cx, cy),
                                                     "contour": cnt,
                                                     "area": cv2.contourArea(cnt),
                                                     "min": (int(np.min(x)), int(np.min(y))),
                                                     "max": (int(np.max(x)), int(np.max(y)))
                                                    }
                filtered_objects = dict(new_filtered_objects)

            else: # previous centroids known
                logging.info("filtered object are less than objects to track: group by distances to centroids")
                contours_list = [filtered_objects[x]["contour"] for x in filtered_objects]
                myarray = np.vstack(contours_list)
                myarray = myarray.reshape(myarray.shape[0], myarray.shape[2])
                centroids = [self.mem_position_objects[self.frame_idx - 1][k]["centroid"] for k in  self.mem_position_objects[self.frame_idx - 1]]

                logging.debug(f"Known centroids: {centroids}")

                new_contours = doris_functions.group(myarray, centroids)
                logging.debug(f"number of new contours after group: {len(new_contours)}")

                new_filtered_objects = {}
                # add info to objects: centroid, area ...
                for idx, cnt in enumerate(new_contours):
                    # print("cnt", type(cnt))

                    #cnt = cv2.convexHull(cnt)
                    M = cv2.moments(cnt)

                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = 0, 0
                    n = np.vstack(cnt).squeeze()
                    try:
                        x, y = n[:, 0], n[:, 1]
                    except Exception:
                        x = n[0]
                        y = n[1]

                    new_filtered_objects[idx + 1] = {"centroid": (cx, cy),
                                                     "contour": cnt,
                                                     "area": cv2.contourArea(cnt),
                                                     "min": (int(np.min(x)), int(np.min(y))),
                                                     "max": (int(np.max(x)), int(np.max(y)))
                                                    }
                filtered_objects = dict(new_filtered_objects)


        if self.objects_to_track:
            mem_costs = {}
            obj_indexes = list(filtered_objects.keys())
            # iterate all combinations of detected objects of length( self.objects_to_track)
            logging.debug(f"combinations of filtered objects: {obj_indexes}")

            if self.frame_idx -1 in self.mem_position_objects:

                for indexes in itertools.combinations(obj_indexes, len(self.mem_position_objects[self.frame_idx - 1])):
                    cost = doris_functions.cost_sum_assignment(self.mem_position_objects[self.frame_idx - 1], dict([(idx, filtered_objects[idx]) for idx in indexes]))
                    logging.debug(f"index: {indexes} cost: {cost}")
                    mem_costs[cost] = indexes

            else:
                for indexes in itertools.combinations(obj_indexes, len(self.objects_to_track)):
                    cost = doris_functions.cost_sum_assignment(self.objects_to_track, dict([(idx, filtered_objects[idx]) for idx in indexes]))
                    logging.debug(f"index: {indexes} cost: {cost}")
                    mem_costs[cost] = indexes

            min_cost = min(list(mem_costs.keys()))
            logging.debug(f"minimal cost: {min_cost}")

            # select new objects to track

            new_objects_to_track = dict([(i + 1, filtered_objects[idx]) for i, idx in enumerate(mem_costs[min_cost])])
            logging.debug(f"new objects to track : {list(new_objects_to_track.keys())}")

            self.objects_to_track = doris_functions.reorder_objects(self.objects_to_track, new_objects_to_track)

            self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)


        # check max distance from previous detected objects
        if self.sb_max_distance.value():
            if self.frame_idx - 1 in self.mem_position_objects:
                p1 = np.array([self.mem_position_objects[self.frame_idx - 1][k]["centroid"] for k in self.mem_position_objects[self.frame_idx - 1]])
                p2 = np.array([self.objects_to_track[k]["centroid"] for k in self.objects_to_track])
                dist_max = int(round(np.max(doris_functions.distances(p1, p2))))

                logging.debug(f"dist max: {dist_max}")

                if dist_max > self.sb_max_distance.value():

                    logging.debug(f"distance is greater then allowed: {dist_max}")

                    # self.flag_stop_analysis = True
                    self.update_info(all_objects, filtered_objects, self.objects_to_track)
                    frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                                     self.objects_to_track,
                                                                     marker_type=MARKER_TYPE)
                    self.display_frame(frame_with_objects)
                    self.display_processed_frame(processed_frame)

                    if not self.always_skip_frame:
                        buttons = ["Accept movement", "Skip frame", "Always skip frame", "Close", "Go to previous frame"]
                        if self.running_tracking:
                            buttons.append("Stop tracking")

                        response = dialog.MessageDialog("DORIS",
                                                        (f"Object moved of {dist_max} pixels.\n"
                                                         f"The maximum distance allowed between frames is {self.sb_max_distance.value()}.\n\n"
                                                         "What do you want to do?"),
                                                        buttons)
                    else:
                        response = "Skip frame"

                    if response in ["Skip frame", "Always skip frame"]:
                        self.objects_to_track = self.mem_position_objects[self.frame_idx - 1]
                        self.mem_position_objects[self.frame_idx] = dict(self.mem_position_objects[self.frame_idx - 1])
                        if response == "Always skip frame":
                            self.always_skip_frame = True

                    if response == "Go to previous frame":
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 2)
                        self.pb()
                        return

                    if response == "Close":
                        return

                    if response == "Stop tracking":
                        self.flag_stop_analysis = True
                        return

        self.filtered_objects = filtered_objects

        if self.cb_display_analysis.isChecked():

            self.update_info(all_objects, filtered_objects, self.objects_to_track)

            _, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)

            # draw arena
            frame_with_objects = self.draw_arena(self.frame.copy(), drawing_thickness)

            # draw reference (100 px square)
            if self.actionDraw_reference.isChecked():
                frame_with_objects = self.draw_reference(frame_with_objects)

            # draw coordinate center if defined
            if self.coordinate_center != [0, 0]:
                frame_with_objects = self.draw_point_origin(frame_with_objects, self.coordinate_center, BLUE, drawing_thickness)

            # draw areas
            frame_with_objects = self.draw_areas(frame_with_objects, drawing_thickness)

            # draw contour of objects
            if self.objects_to_track:
                # draw contours of tracked objects
                frame_with_objects = self.draw_marker_on_objects(frame_with_objects,
                                                                 self.objects_to_track,
                                                                 marker_type=MARKER_TYPE)
            else:
                # draw contours of filtered objects
                frame_with_objects = self.draw_marker_on_objects(frame_with_objects,
                                                                 filtered_objects,
                                                                 marker_type=MARKER_TYPE)


            # display frames
            self.display_frame(frame_with_objects)
            self.display_processed_frame(processed_frame)

        #  record objects data
        self.record_objects_data(self.frame_idx, self.objects_to_track)



    def show_all_objects(self):
        """
        display all objects on frame
        """
        if self.frame is None:
            return

        all_objects, _ = doris_functions.detect_and_filter_objects(frame=self.frame_processing(self.frame),
                                                                   min_size=self.sbMin.value(),
                                                                   max_size=self.sbMax.value(),
                                                                   arena=self.arena,
                                                                   max_extension=self.sb_max_extension.value(),
                                                                   tolerance_outside_arena=self.sb_percent_out_of_arena.value()/100
                                                                   )

        frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                         all_objects,
                                                         marker_type="contour")
        self.display_frame(frame_with_objects)



    def show_all_filtered_objects(self):
        """
        display all filtered objects on frame
        """
        if self.frame is None:
            return

        _, filtered_objects = doris_functions.detect_and_filter_objects(frame=self.frame_processing(self.frame),
                                                                        min_size=self.sbMin.value(),
                                                                        max_size=self.sbMax.value(),
                                                                        arena=self.arena,
                                                                        max_extension=self.sb_max_extension.value(),
                                                                        tolerance_outside_arena=self.sb_percent_out_of_arena.value()/100
                                                                        )

        frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                         filtered_objects,
                                                         marker_type="contour")
        self.display_frame(frame_with_objects)


    def force_objects_number(self):
        """
        separate initial aggregated objects using k-means clustering to arbitrary number of objects
        """

        logging.debug("function: force_objects_number")

        logging.debug(f"filtered objects: {list(self.filtered_objects.keys())}")

        nb_obj, ok_pressed = QInputDialog.getInt(self, "Get number of objects to filter", "Number of objects:", 1, 1, 1000, 1)
        if (not ok_pressed) or (not nb_obj):
            return

        contours_list = [self.filtered_objects[x]["contour"] for x in self.filtered_objects]
        new_contours = doris_functions.apply_k_means(contours_list, nb_obj)

        logging.debug(f"Number of new contours: {len(new_contours)}")

        new_filtered_objects = {}
        # add info to objects: centroid, area ...
        for idx, cnt in enumerate(new_contours):
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0
            n = np.vstack(cnt).squeeze()
            try:
                x, y = n[:, 0], n[:, 1]
            except Exception:
                x, y = n[0], n[1]

            new_filtered_objects[idx + 1] = {"centroid": (cx, cy),
                                             "contour": cnt,
                                             "area": cv2.contourArea(cnt),
                                             "min": (int(np.min(x)), int(np.min(y))),
                                             "max": (int(np.max(x)), int(np.max(y)))
                                            }
        self.filtered_objects = dict(new_filtered_objects)

        logging.debug(f"number filtered objects after k-means: {len(self.filtered_objects)}")
        logging.debug(f"filtered objects after k-means: {list(self.filtered_objects.keys())}")

        self.update_info(all_objects=None,
                         filtered_objects=self.filtered_objects,
                         tracked_objects=self.objects_to_track)

        # draw contour of objects
        frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                         self.filtered_objects,
                                                         marker_type=MARKER_TYPE)

        self.display_frame(frame_with_objects)


    def closeEvent(self, event):
        try:
            self.capture.release()
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            self.fw[0].close()
            self.fw[1].close()
        except Exception:
            pass


    def activate_areas(self):
        """
        create areas for object in areas function from te_area_definition
        """

        logging.debug("function: activate_areas")

        areas = {}
        for idx in range(self.lw_area_definition.count()):
            d = eval(self.lw_area_definition.item(idx).text())
            if "name" in d:
                areas[d["name"]] = eval(self.lw_area_definition.item(idx).text())
        self.areas = areas

        self.process_and_show()


    def update_info(self, all_objects, filtered_objects, tracked_objects=None):
        """
        update info about objects in text edit boxes
        """

        # update information on GUI
        if all_objects is not None:
            self.lb_all.setText(f"All detected objects ({len(all_objects)})")
            out = ""
            for idx in sorted(all_objects.keys()):
                out += f"Object #{idx}: {all_objects[idx]['area']} px\n"
            self.te_all_objects.setText(out)

        self.lb_filtered.setText(f"Filtered objects ({len(filtered_objects)})")

        out = ""
        for idx in filtered_objects:
            out += f"Object #{idx}: {filtered_objects[idx]['area']} px\n"
        self.te_filtered_objects.setText(out)

        if tracked_objects:
            self.lb_tracked_objects.setStyleSheet("")
            out = ""
            for idx in tracked_objects:
                out += f"Object #{idx}: {tracked_objects[idx]['area']} px\n"
            self.te_tracked_objects.setText(out)
        else:
            self.lb_tracked_objects.setStyleSheet("color: red")



    def pb(self) -> bool:
        """
        read next frame and do some analysis

        Returns:
            bool: True if frame else False
        """

        logging.debug("function: pb")

        if self.dir_images:
            if self.dir_images_index < len(self.dir_images) - 1:
                self.dir_images_index += 1
            else:
                self.flag_stop_analysis = False
                self.statusBar.showMessage("Last image of dir")
                return False
            self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)

        else:

            if self.capture is not None:
                ret, self.frame = self.capture.read()
                if not ret:
                    return False
            else:
                return False

        self.update_frame_index()

        self.process_and_show()

        app.processEvents()

        return True


    def record_objects_data(self, frame_idx, objects):
        """
        record objects positions and presence in areas defined by user
        """

        # objects positions
        if self.cb_record_xy.isChecked():

            if self.coord_df is None:
                QMessageBox.warning(self, "DORIS", "No objects to track")
                return

            logging.debug(f"sorted objects to record: {sorted(list(objects.keys()))}")

            # frame idx
            self.coord_df.ix[frame_idx, "frame"] = frame_idx
            # tag
            self.coord_df.ix[frame_idx, "tag"] = self.le_tag.text()

            for idx in sorted(list(objects.keys())):
                if self.cb_normalize_coordinates.isChecked():
                    self.coord_df.ix[frame_idx, f"x{idx}"] = (objects[idx]["centroid"][0] - self.coordinate_center[0]) / self.video_width
                    self.coord_df.ix[frame_idx, f"y{idx}"] = (objects[idx]["centroid"][1] - self.coordinate_center[1]) / self.video_width
                else:
                    self.coord_df.ix[frame_idx, f"x{idx}"] = self.scale * (objects[idx]["centroid"][0] - self.coordinate_center[0])
                    self.coord_df.ix[frame_idx, f"y{idx}"] = self.scale * (objects[idx]["centroid"][1] - self.coordinate_center[1])

            logging.debug(f"coord_df: {self.coord_df}")

            if self.cb_display_analysis.isChecked():
                self.te_xy.clear()
                self.te_xy.append(str(self.coord_df[frame_idx - 3: frame_idx + 3 + 1]))

        # presence in areas
        if self.cb_record_number_objects.isChecked():

            nb = {}
            if self.areas_df is None:
                QMessageBox.warning(self, "DORIS", "No objects to track")
                return

            # frame idx
            self.areas_df.ix[frame_idx, "frame"] = frame_idx
            # tag
            self.areas_df.ix[frame_idx, "tag"] = self.le_tag.text()


            for area in sorted(list(self.areas.keys())):

                nb[area] = 0

                if self.areas[area]["type"] == "circle":
                    cx, cy = self.areas[area]["center"]
                    radius = self.areas[area]["radius"]

                    for idx in objects:
                        x, y = objects[idx]["centroid"]
                        if ((cx - x) ** 2 + (cy - y) ** 2) ** .5 <= radius:
                            nb[area] += 1

                        self.areas_df.ix[frame_idx, f"area {area} object #{idx}"] = int(((cx - x) ** 2 + (cy - y) ** 2) ** .5 <= radius)


                if self.areas[area]["type"] == "rectangle":
                    minx, miny = self.areas[area]["pt1"]
                    maxx, maxy = self.areas[area]["pt2"]

                    for idx in objects:
                        x, y = objects[idx]["centroid"]
                        self.areas_df.ix[frame_idx, f"area {area} object #{idx}"] = int(minx <= x <= maxx and miny <= y <= maxy)

                        if minx <= x <= maxx and miny <= y <= maxy:
                            nb[area] += 1

                if self.areas[area]["type"] == "polygon":
                    for idx in objects:
                        x, y = objects[idx]["centroid"]
                        self.areas_df.ix[frame_idx, f"area {area} object #{idx}"] = int(cv2.pointPolygonTest(np.array(self.areas[area]["points"]), (x, y), False) >= 0)

                        if cv2.pointPolygonTest(np.array(self.areas[area]["points"]), (x, y), False) >= 0:
                            nb[area] += 1

            if self.cb_display_analysis.isChecked():
                self.te_number_objects.clear()
                self.te_number_objects.append(str(self.areas_df[frame_idx - 3: frame_idx + 3 + 1]))

            self.objects_number.append(nb)


    def run_tracking(self):
        """
        run tracking from current frame to end
        """

        # check if objects to track are defined
        if not self.objects_to_track :
            QMessageBox.warning(self, "DORIS", "No objects to track.\nSelect objects to track before running tracking")
            return

        if self.flag_stop_analysis:
            return
        if self.running_tracking:
            return

        if not self.dir_images:
            try:
                self.capture
            except:
                QMessageBox.warning(self, "DORIS", "No video")
                return

        self.running_tracking = True
        while True:
            if not self.pb():
                self.running_tracking = False
                logging.info("tracking finished")
                break

            app.processEvents()
            if self.flag_stop_analysis:
                self.running_tracking = False
                logging.info("tracking stopped")
                break

        self.flag_stop_analysis = False


    def run_tracking_frames_interval(self):
        """
        run tracking in a frames interval
        """
        # check if objects to track are defined
        if not self.objects_to_track :
            QMessageBox.warning(self, "DORIS", "No objects to track.\nSelect objects to track before running tracking")
            return
        if self.running_tracking:
            QMessageBox.warning(self, "DORIS", "A tracking task is already running")
            return


        try:
            text, ok = QInputDialog.getText(self, "Run tracking", "Frames interval: (ex. 123-456)")
            if not ok:
                return
            start_frame, stop_frame = [int(x) for x in text.split("-")]
            if start_frame >= stop_frame:
                raise
        except:
            QMessageBox.warning(self, "DORIS", f"{text} is not a valid interval")
            return

        logging.info(f"start_frame: {start_frame} stop frame: {stop_frame}")
        self.go_to_frame(start_frame - 1)

        self.running_tracking = True
        while True:
            if not self.pb():
                self.running_tracking = False
                logging.info("tracking finished")
                break

            app.processEvents()

            if self.frame_idx >= stop_frame:
                logging.info("tracking finished")
                self.running_tracking = False
                break

            if self.flag_stop_analysis:
                self.running_tracking = False
                logging.info("tracking stopped")
                break

        self.flag_stop_analysis = False



    def stop(self):
        """
        stop analysis
        """
        self.flag_stop_analysis = True


    def open_project(self, file_name=""):
        """
        open a project file and load parameters
        """

        logging.debug("open_project")
        if not file_name:
            file_name, _ = QFileDialog().getOpenFileName(self, "Open project", "", "All files (*)")

        if not file_name:
            return

        if not os.path.isfile(file_name):
            QMessageBox.critical(self, "DORIS", f"{file_name} not found")
            return

        try:
            with open(file_name) as f_in:
                config = json.loads(f_in.read())

            self.sb_blur.setValue(config.get("blur", BLUR_DEFAULT_VALUE))

            if "invert" in config:
                self.cb_invert.setChecked(config["invert"])

            if "normalize_coordinates" in config:
                self.cb_normalize_coordinates.setChecked(config["normalize_coordinates"])

            try:
                self.arena = config["arena"]
                if self.arena:
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.le_arena.setText(str(config["arena"]))
            except KeyError:
                print("arena not found")

            if "min_object_size" in config:
                self.sbMin.setValue(config["min_object_size"])
            if "max_object_size" in config:
                self.sbMax.setValue(config["max_object_size"])
            if "object_max_extension" in config:
                self.sb_max_extension.setValue(config["object_max_extension"])
            if "percent_out_of_arena" in config:
                self.sb_percent_out_of_arena.setValue(config["percent_out_of_arena"])
            if "threshold_method" in config:
                self.cb_threshold_method.setCurrentIndex(THRESHOLD_METHODS.index(config["threshold_method"]))
            if "block_size" in config:
                self.sb_block_size.setValue(config["block_size"])
            if "offset" in config:
                self.sb_offset.setValue(config["offset"])
            if "cut_off" in config:
                self.sb_threshold.setValue(config["cut_off"])

            if "scale" in config:
                self.le_scale.setText(f"{config['scale']:0.5f}")
                self.scale = config["scale"]

            try:
                self.areas = config["areas"]
                if self.areas:
                    self.lw_area_definition.clear()
                    for area in self.areas:
                        self.lw_area_definition.addItem(str(self.areas[area]))
            except KeyError:
                self.areas = {}

            '''
            if "record_number_of_objects_by_area" in config:
                self.cb_record_number_objects.setChecked(config["record_number_of_objects_by_area"])

            if "record_objects_coordinates" in config:
                self.cb_record_xy.setChecked(config["record_objects_coordinates"])
            '''

            if "video_file_path" in config:
                try:
                    if os.path.isfile(config["video_file_path"]):
                        self.open_video(config["video_file_path"])
                    else:
                        QMessageBox.critical(self, "DORIS", f"File {config['video_file_path']} not found")
                except Exception:
                    pass

            if "dir_images" in config:
                try:
                    if os.path.isdir(config["dir_images"]):
                        self.load_dir_images(config["dir_images"])
                    else:
                        QMessageBox.critical(self, "DORIS", f"Directory {config['dir_images']} not found")
                except Exception:
                    pass

            self.coordinate_center = config.get("referential_system_origin", [0,0])
            self.le_coordinates_center.setText(f"{self.coordinate_center}")

            self.actionShow_centroid_of_object.setChecked(config.get("show_centroid", SHOW_CENTROID_DEFAULT))
            self.actionShow_contour_of_object.setChecked(config.get("show_contour", SHOW_CONTOUR_DEFAULT))
            self.actionDraw_reference.setChecked(config.get("show_reference", False))
            self.sb_max_distance.setValue(config.get("max_distance", 0))
            self.frame_scale = config.get("frame_scale", DEFAULT_FRAME_SCALE)
            self.frame_viewer_scale(0, self.frame_scale)
            self.processed_frame_scale = config.get("processed_frame_scale", DEFAULT_FRAME_SCALE)
            self.frame_viewer_scale(1, self.processed_frame_scale)

            self.process_and_show()

        except Exception:
            print("Error in project file")
            raise

        self.project_path = file_name
        self.setWindowTitle(f"DORIS v. {version.__version__} - {self.project_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DORIS (Detection of Objects Research Interactive Software)")

    parser.add_argument("-v", action="store_true", default=False, dest="version", help="Print version")
    parser.add_argument("-p", action="store", dest="project_file", help="path of project file")
    parser.add_argument("-i", action="store", dest="video_file", help="path of video file")
    parser.add_argument("-d", action="store", dest="directory", help="path of images directory")
    '''parser.add_argument("--areas", action="store", dest="areas_file", help="path of file containing the areas definition")'''
    '''parser.add_argument("--arena", action="store", dest="arena_file", help="path of file containing the arena definition")'''
    parser.add_argument("--threshold", action="store", default=50, dest="threshold", help="Threshold value")
    parser.add_argument("--blur", action="store", dest="blur", help="Blur value")
    parser.add_argument("--invert", action="store_true", dest="invert", help="Invert B/W")


    options = parser.parse_args()
    if options.version:
        print(f"version {version.__version__} release date: {version.__version_date__}")
        sys.exit()

    app = QApplication(sys.argv)
    w = Ui_MainWindow()

    if options.project_file:
        if os.path.isfile(options.project_file):
            w.open_project(options.project_file)
        else:
            print(f"{options.project_file} not found!")
            sys.exit()
    else:

        if options.blur:
            w.sb_blur.setValue(int(options.blur))

        if options.threshold:
            w.sb_threshold.setValue(int(options.threshold))

        w.cb_invert.setChecked(options.invert)

        if options.video_file:
            if os.path.isfile(options.video_file):
                w.open_video(options.video_file)
            else:
                print(f"{options.video_file} not found")
                sys.exit()

        if options.directory:
            if os.path.isdir(options.directory):
                w.load_dir_images(options.directory)
            else:
                print(f"{options.directory} directory not found")
                sys.exit()

    w.show()
    w.raise_()
    sys.exit(app.exec_())
