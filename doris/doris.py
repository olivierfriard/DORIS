"""
DORIS
Detection of Objects Research Interactive Software
Copyright 2017-2020 Olivier Friard

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
pandas
numpy
matplotlib
sklearn

optional:
mpl_scatter_density (pip3 install mpl_scatter_density)

match shape
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html

"""


import argparse
import collections
import copy
import datetime as dt
import itertools
import json
import logging
import math
import os
import pathlib
import platform
import sys
import time
from io import StringIO
from doris import doris_qrc

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from sklearn import __version__ as sklearn_version
from matplotlib.figure import Figure
from matplotlib.path import Path
from PyQt5.QtCore import (PYQT_VERSION_STR, QT_VERSION_STR, QEvent, QSettings,
                          Qt, pyqtSignal, QPoint)
from PyQt5.QtGui import QFont, QImage, QPixmap, qRgb
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
                             QFileDialog, QHBoxLayout, QInputDialog, QLabel,
                             QMainWindow, QMenu, QMessageBox, QPushButton,
                             QSizePolicy, QSpacerItem, QStatusBar, QVBoxLayout,
                             QWidget)

import cv2

from doris import doris_functions, version, dialog
from doris.config import *
from doris.doris_ui import Ui_MainWindow






logging.basicConfig(format='%(asctime)s,%(msecs)d  %(module)s l.%(lineno)d %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

COLORS_LIST = doris_functions.COLORS_LIST




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

    zoom_changed_signal = pyqtSignal(str)
    show_contour_changed_signal = pyqtSignal(bool)
    change_frame_signal = pyqtSignal(str)


    def __init__(self, idx):
        super().__init__()

        self.vbox = QVBoxLayout()
        self.idx = idx

        # some widgets
        hbox = QHBoxLayout()

        # zoom
        hbox.addWidget(QLabel("Zoom"))
        self.zoom = QComboBox()
        self.zoom.addItems(ZOOM_LEVELS)
        self.zoom.setCurrentIndex(1)
        self.zoom.currentIndexChanged.connect(self.zoom_changed)
        hbox.addWidget(self.zoom)

        # show contour
        if idx == 0:
            self.cb_show_contour = QCheckBox("Show object contour")
            self.cb_show_contour.setChecked(True)
            self.cb_show_contour.clicked.connect(self.cb_show_contour_clicked)
            hbox.addWidget(self.cb_show_contour)


        # stay on top
        self.cb_stay_on_top = QCheckBox("Stay on top")
        self.cb_stay_on_top.setChecked(True)
        self.cb_stay_on_top.stateChanged.connect(self.cb_stay_on_top_clicked)
        hbox.addWidget(self.cb_stay_on_top)

        hbox.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.vbox.addLayout(hbox)

        self.lb_frame = Click_label()
        self.lb_frame.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.vbox.addWidget(self.lb_frame)

        self.setLayout(self.vbox)

        self.setWindowFlags(Qt.WindowStaysOnTopHint)


    def keyPressEvent(self, event):
        ek = event.key()

        if ek == Qt.Key_Left:
            self.change_frame_signal.emit("backward")
        if ek == Qt.Key_Right:
            self.change_frame_signal.emit("forward")


    def zoom_changed(self):
        """
        zoom changed
        """
        self.zoom_changed_signal.emit(self.zoom.currentText())


    def cb_show_contour_clicked(self):
        """

        """
        self.show_contour_changed_signal.emit(self.cb_show_contour.isChecked())


    def cb_stay_on_top_clicked(self):
        """
        manage the window z - position (stay on top)
        """
        if self.cb_stay_on_top.isChecked():
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()


    def closeEvent(self, event):
        event.accept()


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

        class Flag():
            def __init__(self, define=False, shape="", color=BLACK):
                self.define = define
                self.shape = shape
                self.color = color

        super(Ui_MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.setWindowTitle(f"DORIS v. {version.__version__} - (c) Olivier Friard")
        self.tab_tracking_results.setCurrentIndex(0)
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
        self.actionShow_contour_of_object.triggered.connect(self.actionShow_contour_changed)
        self.actionOrigin_from_point.triggered.connect(self.define_coordinate_center_1point)
        self.actionOrigin_from_center_of_3_points_circle.triggered.connect(self.define_coordinate_center_3points_circle)
        self.actionSelect_objects_to_track.triggered.connect(lambda: self.select_objects_to_track(all_=False))
        self.actionDefine_scale.triggered.connect(self.define_scale)

        self.actionOpen_video.triggered.connect(lambda: self.open_video(""))
        self.actionLoad_directory_of_images.triggered.connect(self.load_dir_images)
        self.actionNew_project.triggered.connect(self.new_project)
        self.actionOpen_project.triggered.connect(self.open_project)
        self.actionSave_project.triggered.connect(self.save_project)
        self.actionSave_project_as.triggered.connect(self.save_project_as)
        self.actionQuit.triggered.connect(self.close)

        self.hs_frame.setMinimum(1)
        self.hs_frame.setMaximum(100)
        self.hs_frame.setValue(1)
        self.hs_frame.sliderMoved.connect(self.hs_frame_moved)

        # self.pb_next_frame.clicked.connect(self.next_frame)
        self.pb_1st_frame.clicked.connect(self.reset)

        self.pb_define_scale.clicked.connect(self.define_scale)
        self.pb_reset_scale.clicked.connect(self.reset_scale)

        self.lw_masks.doubleClicked.connect(self.lw_masks_doubleclicked)
        self.lw_masks.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lw_masks.customContextMenuRequested.connect(self.lw_masks_right_clicked)

        # menu for adding a mask
        menu0 = QMenu()
        menu0.addAction("Rectangle", lambda: self.add_mask(RECTANGLE))
        menu0.addAction("Circle (3 points)", lambda: self.add_mask(CIRCLE_3PTS))
        menu0.addAction("Circle (center radius)", lambda: self.add_mask(CIRCLE_CENTER_RADIUS))
        menu0.addAction("Polygon", lambda: self.add_mask(POLYGON))
        self.pb_add_mask.setMenu(menu0)

        #self.pb_add_mask.clicked.connect(self.add_mask)

        # menu for Define origin button
        menu = QMenu()
        menu.addAction("Origin from center of 3 points circle", self.define_coordinate_center_3points_circle)
        menu.addAction("Origin from a point", self.define_coordinate_center_1point)
        self.pb_define_origin.setMenu(menu)

        self.pb_reset_origin.clicked.connect(self.reset_origin)

        self.pb_goto_frame.clicked.connect(self.pb_go_to_frame)

        self.pb_forward.clicked.connect(lambda: self.for_back_ward("forward"))
        self.pb_backward.clicked.connect(lambda: self.for_back_ward("backward"))
        self.tb_plus1.pressed.connect(lambda: self.for_back_ward("+1"))
        self.tb_minus1.pressed.connect(lambda: self.for_back_ward("-1"))
        self.tb_plus10.pressed.connect(lambda: self.for_back_ward("+10"))
        self.tb_minus10.pressed.connect(lambda: self.for_back_ward("-10"))

        # menu for arena button
        menu1 = QMenu()
        menu1.addAction("Rectangle arena", lambda: self.define_arena(RECTANGLE))
        menu1.addAction("Circle arena (3 points)", lambda: self.define_arena("circle (3 points)"))
        menu1.addAction("Circle arena (center radius)", lambda: self.define_arena("circle (center radius)"))
        menu1.addAction("Polygon arena", lambda: self.define_arena("polygon"))
        self.pb_define_arena.setMenu(menu1)

        self.pb_clear_arena.clicked.connect(self.clear_arena)

        self.pb_run_tracking.clicked.connect(self.run_tracking)
        self.pb_run_tracking_frame_interval.clicked.connect(self.run_tracking_frames_interval)
        '''
        self.pb_stop.clicked.connect(self.stop_button)
        '''

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
        self.sb_max_distance.setValue(DIST_MAX)

        self.pb_show_all_objects.clicked.connect(self.show_all_objects)
        self.pb_show_all_filtered_objects.clicked.connect(self.show_all_filtered_objects)
        self.pb_separate_objects.clicked.connect(self.force_objects_number)
        self.pb_select_objects_to_track.clicked.connect(self.select_objects_to_track)
        self.pb_track_all_filtered.clicked.connect(lambda: self.select_objects_to_track(all_=True))
        self.pb_repick_objects.clicked.connect(lambda: self.repick_objects(mode="tracked"))
        self.pb_repick_objects_from_all.clicked.connect(lambda: self.repick_objects(mode="all"))

        # coordinates analysis
        self.pb_view_coordinates.clicked.connect(self.view_coordinates)
        self.pb_delete_coordinates.clicked.connect(self.delete_coordinates)
        self.pb_save_xy.clicked.connect(self.export_objects_coordinates)
        self.pb_plot_path.clicked.connect(lambda: self.plot_path_clicked("path"))
        self.pb_plot_positions.clicked.connect(lambda: self.plot_path_clicked("positions"))
        self.pb_plot_xy_density.clicked.connect(self.plot_xy_density)
        self.pb_distances.clicked.connect(self.distances)

        # menu for area button
        menu = QMenu()
        menu.addAction("Rectangle", lambda: self.add_area_func(RECTANGLE))
        menu.addAction("Circle (center radius)", lambda: self.add_area_func("circle (center radius)"))
        menu.addAction("Circle (3 points)", lambda: self.add_area_func("circle (3 points)"))
        menu.addAction("Polygon", lambda: self.add_area_func("polygon"))

        self.pb_add_area.setMenu(menu)
        self.pb_remove_area.clicked.connect(self.remove_area)
        self.pb_delete_area_analysis.clicked.connect(self.delete_areas_analysis)
        # self.pb_active_areas.clicked.connect(self.activate_areas)
        self.pb_save_objects_number.clicked.connect(self.save_objects_areas)
        self.pb_time_in_areas.clicked.connect(self.time_in_areas)

        self.coordinate_center = [0, 0]
        self.le_coordinates_center.setText(f"{self.coordinate_center}")

        self.always_skip_frame = False
        self.frame, self.previous_frame = None, None
        self.capture = None
        self.output = ""
        self.video_file_name = ""
        self.coord_df = None
        self.areas_df = None
        self.fgbg = None
        self.flag_stop_tracking = False
        #self.continue_when_no_objects = False
        self.video_height = 0
        self.video_width = 0
        self.frame_width = VIEWER_WIDTH
        self.total_frame_nb = 0
        self.fps = 0
        self.areas = {}
        self.flag_define_arena = False
        self.flag_define_coordinate_center_1point = False
        self.flag_define_coordinate_center_3points = False
        
        self.flag_add_mask = Flag(define=False, shape="", color=BLACK)
        self.flag_define_scale = False
        self.coordinate_center = [0, 0]
        self.coordinate_center_def = []
        self.scale_points = []
        self.mask_points = []
        self.masks = []
        self.add_area = {}
        self.arena = {}
        self.filtered_objects = {}
        self.all_objects = {}
        self.objects_to_track = {}
        self.scale = 1

        self.dir_images = []
        self.dir_images_path = ""
        self.dir_images_index = 0

        self.objects_number = []

        self.mem_position_objects = {}

        self.project = ""
        self.project_path = ""

        # default
        self.sb_threshold.setValue(THRESHOLD_DEFAULT)

        self.fw = []

        self.running_tracking = False

        self.threshold_method_changed()

        self.frame_scale = DEFAULT_FRAME_SCALE
        self.processed_frame_scale = DEFAULT_FRAME_SCALE

        
        # screen_size = app.primaryScreen().size()
        screen_size = QApplication.primaryScreen().size()
        if screen_size.width() >= 1200 and screen_size.height() >= 750:
            self.setGeometry(0, 0, 1200, 750)
        else:
            self.setGeometry(0, 0, 1000, 700)
        

        self.pick_point = None

        self.repicked_objects = None

        self.read_config()

        self.menu_update()


    def about(self):
        """About dialog box."""

        modules = []
        modules.append(f"OpenCV version {cv2.__version__}")
        modules.append(f"\nNumpy version {np.__version__}")
        modules.append(f"\nMatplotlib version {matplotlib.__version__}")
        modules.append(f"\nPandas version {pd.__version__}")
        modules.append(f"\nSciPy version {scipy_version}")
        modules.append(f"\nsklearn version {sklearn_version}")
        modules_str = "\n".join(modules)

        about_dialog = QMessageBox()
        about_dialog.setIconPixmap(QPixmap(":/logo_256px"))
        about_dialog.setWindowTitle("About DORIS")
        about_dialog.setStandardButtons(QMessageBox.Ok)
        about_dialog.setDefaultButton(QMessageBox.Ok)
        about_dialog.setEscapeButton(QMessageBox.Ok)

        about_dialog.setInformativeText((f"<b>DORIS</b> v. {version.__version__} - {version.__version_date__}"
        "<p>Copyright &copy; 2017-2020 Olivier Friard<br><br>"
        '<a href="http://www.boris.unito.it/pages/doris">www.boris.unito.it/pages/doris</a> for more details.<br><br>'
        "Department of Life Sciences and Systems Biology<br>"
        "University of Torino - Italy<br>"))

        architecture = "64-bit" if sys.maxsize > 2**32 else "32-bit"
        details = (f"Python {platform.python_version()} ({architecture}) "
                   f"- Qt {QT_VERSION_STR} - PyQt{PYQT_VERSION_STR} on {platform.system()}\n"
                   f"CPU type: {platform.machine()}\n\n"
                   f"{modules_str}")

        about_dialog.setDetailedText(details)

        _ = about_dialog.exec_()


    def create_viewers(self):
        """Crete the frame viewers"""
        self.fw.append(FrameViewer(ORIGINAL_FRAME_VIEWER_IDX))
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].setWindowTitle("Original frame")
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].lb_frame.mouse_pressed_signal.connect(self.frame_mousepressed)
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].zoom_changed_signal.connect(lambda: self.frame_viewer_scale2(ORIGINAL_FRAME_VIEWER_IDX))
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].show_contour_changed_signal.connect(self.cb_show_contour_changed)
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].change_frame_signal.connect(self.for_back_ward)
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].setGeometry(10, 10, 512, 512)

        self.fw.append(FrameViewer(PROCESSED_FRAME_VIEWER_IDX))
        self.fw[PROCESSED_FRAME_VIEWER_IDX].zoom_changed_signal.connect(lambda: self.frame_viewer_scale2(PROCESSED_FRAME_VIEWER_IDX))
        self.fw[PROCESSED_FRAME_VIEWER_IDX].setGeometry(560, 10, 512, 512)
        self.fw[PROCESSED_FRAME_VIEWER_IDX].setWindowTitle("Processed frame")

        self.fw.append(FrameViewer(PREVIOUS_FRAME_VIEWER_IDX))
        self.fw[PREVIOUS_FRAME_VIEWER_IDX].zoom_changed_signal.connect(lambda: self.frame_viewer_scale2(PREVIOUS_FRAME_VIEWER_IDX))
        self.fw[PREVIOUS_FRAME_VIEWER_IDX].setGeometry(800, 10, 512, 512)
        self.fw[PREVIOUS_FRAME_VIEWER_IDX].setWindowTitle("Previous frame")


    def menu_update(self):
        """Update the menu"""
        
        self.actionOpen_video.setEnabled(self.project != "")
        self.actionLoad_directory_of_images.setEnabled(self.project != "")
        self.actionSave_project.setEnabled(self.project != "")
        self.actionSave_project_as.setEnabled(self.project != "")


    def lw_masks_right_clicked(self, QPos):
        """right click menu for masks listwidget"""

        if self.lw_masks.currentItem():
            self.listMenu= QMenu()
            menu_item = self.listMenu.addAction("Remove Item")
            menu_item.triggered.connect(self.mask_menu_item_clicked)

            parentPosition = self.lw_masks.mapToGlobal(QPoint(0, 0))        
            self.listMenu.move(parentPosition + QPos)

            self.listMenu.show() 


    def mask_menu_item_clicked(self):
        """remove mask from listwidget"""
        self.masks.pop(self.lw_masks.row(self.lw_masks.currentItem()))
        self.lw_masks.takeItem(self.lw_masks.row(self.lw_masks.currentItem()))
        self.reload_frame()


    def lw_masks_doubleclicked(self):
        """
        remove the mask form listwidget
        """
        if dialog.MessageDialog("DORIS", "Remove the mask?", ["Yes", "Cancel"]) == "Yes":
            self.masks.pop(self.lw_masks.row(self.lw_masks.selectedItems()[0]))
            self.lw_masks.takeItem(self.lw_masks.row(self.lw_masks.selectedItems()[0]))

            print(self.masks)

    def actionShow_contour_changed(self):
        self.fw[ORIGINAL_FRAME_VIEWER_IDX].cb_show_contour.setChecked(self.actionShow_contour_of_object.isChecked())
        self.process_and_show()


    def cb_show_contour_changed(self):
        self.actionShow_contour_of_object.setChecked(self.fw[ORIGINAL_FRAME_VIEWER_IDX].cb_show_contour.isChecked())
        self.process_and_show()


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

        for w in [self.lb_threshold, self.sb_threshold]:
            w.setEnabled(self.cb_threshold_method.currentIndex() == THRESHOLD_METHODS.index("Simple"))  # Simple threshold

        for w in [self.lb_adaptive_threshold, self.lb_block_size, self.lb_offset,
                  self.sb_block_size, self.sb_offset]:
            w.setEnabled(self.cb_threshold_method.currentIndex() != THRESHOLD_METHODS.index("Simple"))  # Simple threshold

        self.process_and_show()


    def hide_viewers(self):
        """
        Hide frame viewers
        """
        mem_visible = {}
        for w in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]:
            mem_visible[w] = self.fw[w].isVisible()
            if self.fw[w].isVisible():
                self.fw[w].hide()
        return mem_visible


    def show_viewers(self, mem_visible):
        """ Show frame viewers """
        for w in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]:
            if mem_visible[w]:
                self.fw[w].show()


    def save_project_as(self):
        """Save project as."""

        mem_visible = self.hide_viewers()

        project_file_path, _ = QFileDialog().getSaveFileName(self, "Save project", "",
                                                             "DORIS projects (*.doris);;All files (*)")
        self.show_viewers(mem_visible)

        if not project_file_path:
            return
        self.project_path = project_file_path
        self.save_project()


    def save_project(self):
        """ Save parameters of current project in a JSON file """

        if not self.project_path:

            mem_visible = self.hide_viewers()

            if self.video_file_name:
                project_path_suggestion = str(pathlib.Path(self.video_file_name).with_suffix(".doris"))
            elif self.dir_images_path:
                project_path_suggestion = str(pathlib.Path(self.dir_images_path).with_suffix(".doris"))
            else:
                project_path_suggestion = ""

            project_file_path, _ = QFileDialog().getSaveFileName(self, "Save project",
                                                                 project_path_suggestion,
                                                                 "DORIS projects (*.doris);;All files (*)")
            self.show_viewers(mem_visible)
            if not project_file_path:
                return
        else:
            project_file_path = self.project_path

        config = {}
        if self.video_file_name:
            config["video_file_path"] = self.video_file_name
        if self.dir_images_path:
            config["dir_images_path"] = str(self.dir_images_path)

        config["start_from"] = self.sb_start_from.value()
        config["stop_to"] = self.sb_stop_to.value()
        config["blur"] = self.sb_blur.value()
        config["invert"] = self.cb_invert.isChecked()
        config["arena"] = self.arena
        config["min_object_size"] = self.sbMin.value()
        config["max_object_size"] = self.sbMax.value()
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

        config["original_frame_viewer_position"] = [int(self.fw[ORIGINAL_FRAME_VIEWER_IDX].x()),
                                                    int(self.fw[ORIGINAL_FRAME_VIEWER_IDX].y())]
        config["processed_frame_viewer_position"] = [int(self.fw[PROCESSED_FRAME_VIEWER_IDX].x()),
                                                     int(self.fw[PROCESSED_FRAME_VIEWER_IDX].y())]
        config["masks"] = self.masks
        config["record_coordinates"] = self.cb_record_xy.isChecked()
        config["record_presence_area"] = self.cb_record_presence_area.isChecked()
        config["apply_scale"] = self.cb_apply_scale.isChecked()
        config["apply_origin"] = self.cb_apply_origin.isChecked()

        # save coordinates
        if self.coord_df is not None:
            # self.coord_df[self.coord_df.columns] = self.coord_df[self.coord_df.columns].fillna(0).astype(int)
            config["coordinates"] = self.coord_df.to_csv()

        # current frame
        config["current_frame"] = self.frame_idx

        # objects to track
        if self.objects_to_track:
            obj_dict = copy.deepcopy(self.objects_to_track)
            for idx in obj_dict:
                obj_dict[idx]["contour"] = obj_dict[idx]["contour"].tolist()
            config["objects_to_track"] = copy.deepcopy(obj_dict)

        try:
            with open(project_file_path, "w") as f_out:
                f_out.write(json.dumps(config))
            self.project_path = project_file_path
            self.setWindowTitle(f"DORIS v. {version.__version__} - {self.project_path}")
        except:
            logging.critical(f"project not saved: {project_file_path}")
            QMessageBox.critical(self, "DORIS", f"project not saved: {project_file_path}")


    def frame_viewer_scale2(self, fw_idx):
        """
        change scale of frame viewer
        """
        logging.debug("function: frame_viewer_scale")

        self.fw[fw_idx].show()
        try:
            self.fw[fw_idx].lb_frame.clear()
            scale = eval(self.fw[fw_idx].zoom.currentText())
            self.fw[fw_idx].lb_frame.resize(int(self.frame.shape[1] * scale), int(self.frame.shape[0] * scale))
            if fw_idx in [ORIGINAL_FRAME_VIEWER_IDX, PREVIOUS_FRAME_VIEWER_IDX]:
                self.fw[fw_idx].lb_frame.setPixmap(frame2pixmap(self.frame).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                                   Qt.KeepAspectRatio))
                self.frame_width = self.fw[fw_idx].lb_frame.width()
                self.frame_scale = scale

            if fw_idx == PROCESSED_FRAME_VIEWER_IDX:
                processed_frame = self.frame_processing(self.frame)
                self.fw[fw_idx].lb_frame.setPixmap(QPixmap.fromImage(toQImage(processed_frame)).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                                                  Qt.KeepAspectRatio))
                self.processed_frame_scale = scale

            self.fw[fw_idx].setFixedSize(self.fw[fw_idx].vbox.sizeHint())
            if fw_idx in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]:
                self.process_and_show()
        except Exception:
            logging.critical("error")


    def frame_viewer_scale(self, fw_idx, scale):
        """
        change scale of frame viewer
        """
        logging.debug("function: frame_viewer_scale")

        self.fw[fw_idx].show()
        try:
            self.fw[fw_idx].lb_frame.clear()
            self.fw[fw_idx].lb_frame.resize(int(self.frame.shape[1] * scale), int(self.frame.shape[0] * scale))
            if fw_idx in [ORIGINAL_FRAME_VIEWER_IDX, PREVIOUS_FRAME_VIEWER_IDX]:
                self.fw[fw_idx].lb_frame.setPixmap(frame2pixmap(self.frame).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                                   Qt.KeepAspectRatio))
                self.frame_width = self.fw[fw_idx].lb_frame.width()
                self.frame_scale = scale

            if fw_idx == PROCESSED_FRAME_VIEWER_IDX:
                processed_frame = self.frame_processing(self.frame)
                self.fw[fw_idx].lb_frame.setPixmap(QPixmap.fromImage(toQImage(processed_frame)).scaled(self.fw[fw_idx].lb_frame.size(),
                                                                                                  Qt.KeepAspectRatio))
                self.processed_frame_scale = scale

            self.fw[fw_idx].setFixedSize(self.fw[fw_idx].vbox.sizeHint())
            if fw_idx in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]:
                self.process_and_show()
        except Exception:
            logging.critical("error")


    def for_back_ward(self, direction="forward"):
        """go back and forward"""

        logging.debug("function: for_back_ward")
        if direction in ["+1", "-1", "-10", "+10"]:
            step = int(direction)

        if direction == "forward":
            step = self.sb_frame_offset.value()

        if direction == "backward":
            step = - self.sb_frame_offset.value()

        while True:
            if self.dir_images:
                logging.info(f"self.dir_images_index + step: {self.dir_images_index + step}")
                self.dir_images_index += step
                if self.dir_images_index >= len(self.dir_images):
                    self.dir_images_index = len(self.dir_images) - 1
                if self.dir_images_index < 0:
                    self.dir_images_index = 0
                self.previous_frame = self.frame
                self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)

            elif self.capture is not None:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) + step - 1)
                self.previous_frame = self.frame
                ret, self.frame = self.capture.read()

            if self.dir_images or self.capture is not None:
                self.update_frame_index()
                self.process_and_show()
            QApplication.processEvents()

            if (not self.tb_plus1.isDown() and not self.tb_minus1.isDown() 
                and not self.tb_plus10.isDown() and not self.tb_minus10.isDown()):
                break





    def go_to_frame(self, frame_nb: int):
        """
        load frame and visualize it
        """
        if self.dir_images:
            self.dir_images_index = frame_nb
            if self.dir_images_index >= len(self.dir_images):
                self.dir_images_index = len(self.dir_images) - 1
            if self.dir_images_index < 0:
                self.dir_images_index = 0

            self.previous_frame = cv2.imread(str(self.dir_images[self.dir_images_index - 1]), -1)
            self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)

        if self.capture is not None:
            try:
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
                ret, self.frame = self.capture.read()

            except Exception:
                logging.debug("exception in function go_to_frame")
                pass

        self.update_frame_index()

        self.process_and_show()


    def pb_go_to_frame(self):

        logging.debug("function: pb_go_to_frame")

        if not self.dir_images and self.capture is None:
            return

        if self.le_goto_frame.text():
            try:
                int(self.le_goto_frame.text())
            except ValueError:
                return
            self.go_to_frame(frame_nb=int(self.le_goto_frame.text()) - 1)


    def add_area_func(self, shape):

        if self.frame is None:
            return

        if shape:
            # disable the stay on top property for frame viewers
            self.disable_viewers_stay_on_top

            text, ok = QInputDialog.getText(self, "New area", "Area name:")
            if ok:
                self.add_area = {"type": shape, "name": text}
                msg = ""
                if shape == "circle (3 points)":
                    msg = "New circle area: Click on the video to define 3 points belonging to the circle"
                if shape == "circle (center radius)":
                    msg = ("New circle area: click on the video to define the center of the circle "
                           "and a point belonging to the circle")
                if shape == "polygon":
                    msg = ("New polygon area: click on the video to define the vertices of the polygon. "
                           "Right click to close the polygon")
                if shape == RECTANGLE:
                    msg = "New rectangle area: click on the video to define 2 opposite vertices."

                self.statusBar.showMessage(msg)


    def remove_area(self):
        """ Remove the selected area """

        for selected_item in self.lw_area_definition.selectedItems():
            self.lw_area_definition.takeItem(self.lw_area_definition.row(selected_item))
            self.activate_areas()


    def define_arena(self, shape):
        """ Switch to define arena mode """

        logging.debug("function: define_arena")

        if not self.dir_images and self.capture is None:
            return

        if self.flag_define_arena:
            self.flag_define_arena = ""
        else:
            self.flag_define_arena = shape
            self.pb_define_arena.setEnabled(False)
            msg = ""
            if shape == RECTANGLE:
                msg = "New arena: click on the video to define the top-lef and bottom-right edges of the rectangle."
            if shape == "circle (3 points)":
                msg = "New arena: click on the video to define 3 points belonging to the circle"
            if shape == "circle (center radius)":
                msg = "New arena: click on the video to define the center of the circle and then a point belonging to the circle"
            if shape == "polygon":
                msg = "New arena: click on the video to define the edges of the polygon"

            self.statusBar.showMessage(msg)


    def reload_frame(self):
        """ Reload frame and show. """

        logging.debug("function: reload_frame")

        if self.dir_images:
            self.dir_images_index -= 1
        elif self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.pb()


    def clear_arena(self):
        """ Clear the arena """

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

        #logging.debug(f"video_width: {video_width}, frame_width: {frame_width}")

        ratio = video_width / frame_width
        if ratio <= 1:
            drawing_thickness = 1
        else:
            drawing_thickness = round(ratio)

        #logging.debug(f"ratio: {ratio}, drawing_thickness: {drawing_thickness}")

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
        record clicked coordinates for:
            * arena definition
            * area definition
            * origin definition
            * setting scale
            * re-picking objects
            * masks definition

        """

        logging.debug("function: frame_mousepressed")

        conversion, drawing_thickness = self.ratio_thickness(self.video_width,
                                                             self.fw[ORIGINAL_FRAME_VIEWER_IDX].lb_frame.pixmap().width())

        ''' pick object
        if self.pick_point:
            print([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
            self.pick_point = False
        '''
        if self.repicked_objects is not None:

            # cancel repick
            if event.button() in [Qt.RightButton]:
                self.repicked_objects = None
                return

            # pick object by clicking inside the object contour
            '''
            if self.repick_mode == "tracked":
                for o in self.objects_to_track:
                    # check if clicked point is inside an object
                    if int(cv2.pointPolygonTest(np.array(self.objects_to_track[o]["contour"]),
                                                (int(event.pos().x() * conversion), int(event.pos().y() * conversion)),
                                                False) >= 0):
                        self.repicked_objects.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                        self.statusBar.showMessage(f"Click object #{len(self.repicked_objects) + 1} on the video (right-click to cancel)")

            if self.repick_mode == "all": 
                for o in self.filtered_objects:
                    print("o", o)
                    # check if clicked point is inside an object
                    if int(cv2.pointPolygonTest(np.array(self.filtered_objects[o]["contour"]),
                                                (int(event.pos().x() * conversion), int(event.pos().y() * conversion)),
                                                False) >= 0):
                        self.repicked_objects.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                        self.statusBar.showMessage(f"Click object #{len(self.repicked_objects) + 1} on the video (right-click to cancel)")
            '''

            
            # pick object by clicking nearly (not inside)
            self.repicked_objects.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
            self.statusBar.showMessage(f"Click object #{len(self.repicked_objects) + 1} on the video (right-click to cancel)")
            

        # set coordinates of referential origin with 1 point
        if self.flag_define_coordinate_center_1point:
            self.coordinate_center = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
            self.frame = self.draw_point_origin(self.frame, self.coordinate_center, BLUE, drawing_thickness)

            #self.display_frame(self.frame)
            self.flag_define_coordinate_center_1point = False
            self.actionOrigin_from_point.setText("from point")
            self.le_coordinates_center.setText(f"{self.coordinate_center}")
            self.reload_frame()
            self.statusBar.showMessage(f"Referential origin defined")

        # set coordinates of referential origin with the center of a 3 points defined circle
        if self.flag_define_coordinate_center_3points:
            self.coordinate_center_def.append((int(event.pos().x() * conversion), int(event.pos().y() * conversion)))
            cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=BLUE, lineType=8, thickness=drawing_thickness)

            self.display_frame(self.frame)

            if len(self.coordinate_center_def) == 3:

                x, y, radius = doris_functions.find_circle(self.coordinate_center_def)
                if radius == -1:
                    self.disable_viewers_stay_on_top()
                    QMessageBox.warning(self, "DORIS",
                                        ("A circle can not be defined with the selected points. "
                                         "Please retry selecting different points.")
                                       )
                    self.statusBar.showMessage("")
                    self.coordinate_center_def = []
                    self.reload_frame()
                    return

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

                    self.disable_viewers_stay_on_top()

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

        # add mask
        if self.flag_add_mask.define:

            # cancel mask creation (mid button)
            if event.button() == Qt.MidButton:
                self.mask_points = []
                self.flag_add_mask.define = False
                self.reload_frame()
                self.statusBar.showMessage("Mask creation canceled")

                return

            if event.button() == Qt.LeftButton:
                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                self.display_frame(self.frame)

                self.mask_points.append((int(event.pos().x() * conversion), int(event.pos().y() * conversion)))

            if self.flag_add_mask.shape in [RECTANGLE, CIRCLE_CENTER_RADIUS] and len(self.mask_points) == 2:

                if (self.flag_add_mask.shape == RECTANGLE):
                    min_x = min(self.mask_points[0][0], self.mask_points[1][0])
                    max_x = max(self.mask_points[0][0], self.mask_points[1][0])
                    min_y = min(self.mask_points[0][1], self.mask_points[1][1])
                    max_y = max(self.mask_points[0][1], self.mask_points[1][1])

                    self.masks.append({"type": RECTANGLE, "color": self.flag_add_mask.color,
                                        "coordinates": ((min_x, min_y), (max_x, max_y))})
                    self.lw_masks.addItem(str(self.masks[-1]))

                    cv2.rectangle(self.frame, (min_x, min_y), (max_x, max_y),
                                color=self.flag_add_mask.color, thickness=cv2.FILLED)

                if (self.flag_add_mask.shape == CIRCLE_CENTER_RADIUS):
                    radius = int(doris_functions.euclidean_distance(self.mask_points[0], self.mask_points[1]))
                    self.masks.append({"type": CIRCLE, "color": self.flag_add_mask.color,
                                        "center": self.mask_points[0], "radius": int(radius)})
                    self.lw_masks.addItem(str(self.masks[-1]))

                    cv2.circle(self.frame, self.mask_points[0], int(radius),
                               color=self.flag_add_mask.color, thickness=cv2.FILLED)

                self.mask_points = []
                self.display_frame(self.frame)
                self.flag_add_mask.define = False
                self.reload_frame()
                self.statusBar.showMessage(f"Mask added")

            if (self.flag_add_mask.shape in [CIRCLE_3PTS]) and len(self.mask_points) == 3:
                cx, cy, radius = doris_functions.find_circle(self.mask_points)
                if radius == -1:
                    self.disable_viewers_stay_on_top()
                    QMessageBox.warning(self, "DORIS",
                                        ("A circle can not be defined with the selected points. "
                                         "Please retry selecting different points.")
                                       )
                    self.mask_points = []
                    self.statusBar.showMessage("")
                    self.flag_add_mask.define = False
                    self.reload_frame()
                    return


                self.masks.append({"type": CIRCLE, "color": self.flag_add_mask.color,
                                   "center": (round(cx), round(cy)), "radius": int(radius)})
                self.lw_masks.addItem(str(self.masks[-1]))

                cv2.circle(self.frame, (round(cx), round(cy)), int(radius),
                           color=self.flag_add_mask.color, thickness=cv2.FILLED)

                self.mask_points = []
                self.display_frame(self.frame)
                self.flag_add_mask.define = False
                self.reload_frame()
                self.statusBar.showMessage(f"Mask added")

            if self.flag_add_mask.shape == POLYGON:

                if event.button() == Qt.RightButton:  # right click to finish

                    self.masks.append({"type": POLYGON, "color": self.flag_add_mask.color,
                                        "coordinates": self.mask_points})
                    self.lw_masks.addItem(str(self.masks[-1]))

                    cv2.fillPoly(self.frame, np.array([self.mask_points]), color=self.flag_add_mask.color)

                    self.mask_points = []
                    self.display_frame(self.frame)
                    self.flag_add_mask.define = False
                    self.reload_frame()
                    self.statusBar.showMessage(f"Mask added")


        # add area
        if self.add_area:

            if event.button() == Qt.MidButton:
                self.add_area = {}
                self.statusBar.showMessage("New area canceled")
                self.reload_frame()
                return

            if event.button() == Qt.LeftButton:
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
                    if radius == -1:
                        self.disable_viewers_stay_on_top()
                        QMessageBox.warning(self, "DORIS",
                                            ("A circle can not be defined with the selected points. "
                                            "Please retry selecting different points.")
                                        )
                        self.statusBar.showMessage("")
                        del self.add_area["points"]
                        self.add_area = {}
                        self.reload_frame()
                        return

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

            if self.add_area["type"] == RECTANGLE:
                if "pt1" not in self.add_area:
                    self.add_area["pt1"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["pt2"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                    # reorder vortex
                    logging.debug(self.add_area)
                    self.add_area["pt1"], self.add_area["pt2"] = ([min(self.add_area["pt1"][0], self.add_area["pt2"][0]), min(self.add_area["pt1"][1],self.add_area["pt2"][1])],
                                                                 [max(self.add_area["pt1"][0], self.add_area["pt2"][0]), max(self.add_area["pt1"][1], self.add_area["pt2"][1])])
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.reload_frame()
                    self.statusBar.showMessage("New rectangle area created")
                    return

            if self.add_area["type"] == "polygon":

                if event.button() == Qt.RightButton:  # right click to finish
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}
                    self.reload_frame()
                    self.statusBar.showMessage("The new polygon area is defined")
                    return

                if event.button() == Qt.LeftButton:
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

                    self.statusBar.showMessage((f"Polygon area: {len(self.add_area['points'])} point(s) selected."
                                                " Right click to finish"))
                    self.display_frame(self.frame)

        # arena
        if self.flag_define_arena:

            # cancel arena creation (mid button)
            if event.button() == Qt.MidButton:
                self.flag_define_arena = ""
                self.pb_define_arena.setEnabled(True)
                self.statusBar.showMessage("Arena creation canceled")
                self.reload_frame()
                return

            if self.flag_define_arena == RECTANGLE:
                if "points" not in self.arena:
                    self.arena["points"] = []
                self.arena["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                           color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)

                self.display_frame(self.frame)
                self.statusBar.showMessage(f"Rectangle arena: {len(self.arena['points'])} point(s) selected.")

                if len(self.arena["points"]) == 2:  # rectangle area finished
                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.pb_define_arena.setText("Define arena")
                    # check top-left and right-bottom
                    min_x = min(self.arena["points"][0][0], self.arena["points"][1][0])
                    max_x = max(self.arena["points"][0][0], self.arena["points"][1][0])
                    min_y = min(self.arena["points"][0][1], self.arena["points"][1][1])
                    max_y = max(self.arena["points"][0][1], self.arena["points"][1][1])
                    self.arena["points"] = [(min_x, min_y), (max_x, max_y)]

                    self.arena = {**self.arena, **{"type": "rectangle", "name": "arena"}}
                    self.le_arena.setText(f"{self.arena}")

                    cv2.rectangle(self.frame, self.arena["points"][0], self.arena["points"][1],
                                  color=ARENA_COLOR, thickness=drawing_thickness)

                    self.display_frame(self.frame)
                    self.reload_frame()
                    '''self.process_and_show()'''
                    self.statusBar.showMessage("The rectangle arena is defined")

            if self.flag_define_arena == "polygon":

                if event.button() == Qt.RightButton:  # right click to finish

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.pb_define_arena.setText("Define arena")

                    self.arena = {**self.arena, **{"type": "polygon", "name": "arena"}}

                    self.le_arena.setText(f"{self.arena}")
                    self.display_frame(self.frame)
                    self.reload_frame()

                    '''self.process_and_show()'''
                    self.statusBar.showMessage("The new polygon arena is defined")

                else:
                    if "points" not in self.arena:
                        self.arena["points"] = []

                    self.arena["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                    cv2.circle(self.frame, (int(event.pos().x() * conversion), int(event.pos().y() * conversion)), 4,
                               color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                    self.display_frame(self.frame)

                    self.statusBar.showMessage((f"Polygon arena: {len(self.arena['points'])} point(s) selected. "
                                                "Right click to finish"))

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

                self.statusBar.showMessage(f"Circle arena: {len(self.arena['points'])} point(s) selected.")

                if len(self.arena["points"]) == 3:
                    cx, cy, radius = doris_functions.find_circle(self.arena["points"])
                    if radius == -1:
                        self.disable_viewers_stay_on_top()
                        QMessageBox.warning(self, "DORIS",
                                            ("A circle can not be defined with the selected points. "
                                            "Please retry selecting different points.")
                                        )
                        self.statusBar.showMessage("")
                        self.flag_define_arena = ""
                        self.arena = {}
                        self.reload_frame()
                        return

                    cv2.circle(self.frame, (int(abs(cx)), int(abs(cy))), int(radius), color=ARENA_COLOR,
                               thickness=drawing_thickness)
                    # self.display_frame(self.frame)

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.arena = {'type': 'circle', 'center': [round(cx), round(cy)],
                                  'radius': round(radius), 'name': 'arena'}

                    self.le_arena.setText(f"{self.arena}")

                    self.display_frame(self.frame)
                    self.reload_frame()

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
                    self.le_arena.setText(f"{self.arena}")
                    self.display_frame(self.frame)
                    self.reload_frame()

                    '''self.process_and_show()'''
                    self.statusBar.showMessage("The new circle arena is defined")

    def background(self):
        if self.cb_background.isChecked():
            self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
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
        reset analysis and go to 1st frame
        """

        logging.debug("function: reset")

        if not self.dir_images and self.capture is None:
            return

        if dialog.MessageDialog("DORIS", "Confirm reset?", ["Yes", "Cancel"]) == "Cancel":
            return

        self.flag_stop_tracking = True

        if self.dir_images:
            self.dir_images_index = -1
        if self.capture is not None:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.objects_to_track = {}
        self.te_tracked_objects.clear()
        self.mem_position_objects = {}
        self.coord_df = None
        self.initialize_positions_dataframe()

        self.areas_df = None
        self.initialize_areas_dataframe()

        self.te_xy.clear()
        self.te_number_objects.clear()

        self.always_skip_frame = False
        #self.continue_when_no_objects = False
        self.cb_continue_when_no_objects.setChecked(False)

        self.pb()


    def disable_viewers_stay_on_top(self):
        """
        disable the stay on top property on viewers
        to allow a correct visualization of the message dialog
        """
        if self.fw:
            for viewer in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]: #  PREVIOUS_FRAME_VIEWER_IDX
                self.fw[viewer].cb_stay_on_top.setChecked(False)


    def view_coordinates(self):
        """
        view dataframe of recorded coordinates
        """

        # print(f"self.coord_df\n {self.coord_df}")

        if self.coord_df is not None and self.objects_to_track:

            w = dialog.Results_dialog()
            w.setWindowFlags(Qt.WindowStaysOnTopHint)

            w.setWindowTitle("DORIS - Object's coordinates")
            w.ptText.setReadOnly(True)
            font = QFont("Monospace")
            w.ptText.setFont(font)

            if self.cb_apply_scale.isChecked():
                scale = self.scale
            else:
                scale = 1

            if self.cb_apply_origin.isChecked():
                origin_x, origin_y = self.coordinate_center
            else:
                origin_x, origin_y = 0, 0

            df = self.coord_df.copy()
            for idx in sorted(list(self.objects_to_track.keys())):
                df[f"x{idx}"] = scale * (df[f"x{idx}"] - origin_x)
                df[f"y{idx}"] = scale * (df[f"y{idx}"] - origin_y)

            w.ptText.appendPlainText(df.iloc[:, :].to_string(index=False))

            w.exec_()

        else:
            self.disable_viewers_stay_on_top()
            QMessageBox.warning(self, "DORIS", "No objects to track were selected")


    def delete_coordinates(self):
        """
        reset recorded coordinates
        """

        if self.coord_df is not None and self.objects_to_track:

            if dialog.MessageDialog("DORIS", "Confirm deletion of coordinates?", ["Yes", "Cancel"]) == "Cancel":
                return
            # init dataframe for recording objects coordinates
            self.initialize_positions_dataframe()
            self.te_xy.clear()


    def export_objects_coordinates(self):
        """
        Export objects coordinates in TSV file
        """
        if self.coord_df is not None:

            mem_visible = self.hide_viewers()
            if self.video_file_name:
                path_suggestion = str(pathlib.Path(self.video_file_name).with_suffix(".tsv"))
            elif self.dir_images_path:
                project_path_suggestion = str(pathlib.Path(self.dir_images_path).with_suffix(".doris"))
            else:
                path_suggestion = ""

            file_name, _ = QFileDialog().getSaveFileName(self, "Export objects coordinates",
                                                         path_suggestion,
                                                         "TSV files (*.tsv);;All files (*)")

            if file_name:
                apply_scale, apply_origin = False, False
                if self.scale != 1:
                    apply_scale = dialog.MessageDialog("DORIS", "Apply scale to exported coordinates", ["Yes", "No"]) == "Yes"

                if self.coordinate_center != [0, 0]:
                    apply_origin = dialog.MessageDialog("DORIS", "Apply origin to exported coordinates", ["Yes", "No"]) == "Yes"

                if apply_scale:
                    scale = self.scale
                else:
                    scale = 1

                if apply_origin:
                    origin_x, origin_y = self.coordinate_center
                else:
                    origin_x, origin_y = 0, 0

                #obj_col_coord = [] # list of objects coordinates columns 
                df = self.coord_df.copy()
                for idx in sorted(list(self.objects_to_track.keys())):
                    df[f"x{idx}"] = scale * (df[f"x{idx}"] - origin_x)
                    df[f"y{idx}"] = scale * (df[f"y{idx}"] - origin_y)
                    #obj_col_coord.extend([f"x{idx}", f"y{idx}"]) 


                df.to_csv(file_name, sep="\t", decimal=".")
                # self.coord_df.to_pickle(file_name  + ".pickle")
            self.show_viewers(mem_visible)
        else:
            QMessageBox.warning(self, "DORIS", "No coordinates to save")


    def delete_areas_analysis(self):
        """
        reset areas analysis
        """
        if self.areas_df is not None and self.objects_to_track:
            if dialog.MessageDialog("DORIS", "Confirm deletion of area analysis?", ["Yes", "Cancel"]) == "Cancel":
                return
            self.initialize_areas_dataframe()

            self.te_number_objects.clear()


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
                time_objects_in_areas[area][idx] = areas_non_nan_df[f"area {area} object #{idx}"].sum() / self.fps

        logging.debug(f"{time_objects_in_areas}")

        mem_visible = self.hide_viewers()
        file_name, _ = QFileDialog().getSaveFileName(self, "Save time in areas", "", "All files (*)")
        self.show_viewers(mem_visible)

        if file_name:
            time_objects_in_areas.to_csv(file_name, sep="\t", decimal=".")


    def save_objects_areas(self):
        """
        save presence of objects in areas
        """
        if self.areas_df is None:
            QMessageBox.warning(self, "DORIS", "no objects to be saved")
            return

        mem_visible = self.hide_viewers()
        file_name, _ = QFileDialog().getSaveFileName(self, "Save objects in areas", "", "All files (*)")
        self.show_viewers(mem_visible)

        if file_name:
            self.areas_df.to_csv(file_name, sep="\t", decimal=".")


    def plot_xy_density(self):

        if self.coord_df is None:
            QMessageBox.warning(self, "DORIS", "no positions recorded")
            return

        try:
            # set origin
            x_lim = np.array([0 - self.coordinate_center[0], self.video_width - self.coordinate_center[0]])
            y_lim = np.array([0 - self.coordinate_center[1], self.video_height - self.coordinate_center[1]])

            # normalize (if required)
            if self.cb_normalize_coordinates.isChecked():
                x_lim = x_lim / self.video_width
                y_lim = y_lim / self.video_width

            # apply scale
            x_lim = x_lim * self.scale
            y_lim = y_lim * self.scale

            doris_functions.plot_density(self.coord_df,
                                         x_lim=x_lim,
                                         y_lim=y_lim)
        except Exception:
            error_type, _, _ = dialog.error_message("plot density", sys.exc_info())
            logging.debug(error_type)


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
            results["distance"][idx] = dist.sum()

        file_name, _ = QFileDialog().getSaveFileName(self, "Save distances", "", "All files (*)")
        if file_name:
            try:
                results.to_csv(file_name, sep="\t", decimal=".")
            except Exception:
                error_type, _, _ = dialog.error_message("save distances", sys.exc_info())
                logging.debug(error_type)



    def plot_path_clicked(self, plot_type="path"):
        """
        plot the path or positions based on recorded coordinates
        """

        if self.coord_df is None:
            QMessageBox.warning(self, "DORIS", "No Coordinates were recorded")
            return

        apply_scale, apply_origin = False, False
        if self.scale != 1:
            apply_scale = dialog.MessageDialog("DORIS", "Apply scale to exported coordinates", ["Yes", "No"]) == "Yes"

        if self.coordinate_center != [0, 0]:
            apply_origin = dialog.MessageDialog("DORIS", "Apply origin to exported coordinates", ["Yes", "No"]) == "Yes"

        if apply_scale:
            scale = self.scale
        else:
            scale = 1

        if apply_origin:
            origin_x, origin_y = self.coordinate_center
        else:
            origin_x, origin_y = 0, 0

        df = self.coord_df.copy()
        for idx in sorted(list(self.objects_to_track.keys())):
            df[f"x{idx}"] = scale * (df[f"x{idx}"] - origin_x)
            df[f"y{idx}"] = scale * (df[f"y{idx}"] - origin_y)

        x_lim = np.array([0 - origin_x, self.video_width - origin_x])
        y_lim = np.array([0 - origin_y, self.video_height - origin_y])

        x_lim = x_lim * scale
        y_lim = y_lim * scale


        '''
        x_lim = np.array([0 - self.coordinate_center[0], self.video_width - self.coordinate_center[0]])
        y_lim = np.array([0 - self.coordinate_center[1], self.video_height - self.coordinate_center[1]])

        if self.cb_normalize_coordinates.isChecked():
            x_lim = x_lim / self.video_width
            y_lim = y_lim / self.video_width

        x_lim = x_lim * self.scale
        y_lim = y_lim * self.scale
        '''

        if plot_type == "path":
            doris_functions.plot_path(df,
                                      x_lim=x_lim,
                                      y_lim=y_lim)
        if plot_type == "positions":
            doris_functions.plot_positions(df,
                                           x_lim=x_lim,
                                           y_lim=y_lim)


    def open_video(self, file_name):
        """
        open a video
        if file_name not provided ask user to select a file
        """

        logging.debug("function: open_video")

        if not file_name:
            file_name, _ = QFileDialog().getOpenFileName(self, "Open video", "", "All files (*)")

        if file_name:
            if not os.path.isfile(file_name):
                QMessageBox.critical(self, "DORIS", f"{file_name} not found")
                return

            if self.capture:
                self.capture.release()

            self.capture = cv2.VideoCapture(file_name)

            if not self.capture.isOpened():
                QMessageBox.critical(self, "DORIS", f"Could not open {pathlib.Path(file_name).name}")
                self.capture.release()
                return

            self.total_frame_nb = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.total_frame_nb < 0:
                QMessageBox.critical(self, "DORIS", f"{pathlib.Path(file_name).name} has an unknown format")
                self.capture.release()
                return

            self.create_viewers()

            # frames slider
            self.hs_frame.setMinimum(1)
            self.hs_frame.setMaximum(self.total_frame_nb)
            self.hs_frame.setTickInterval(10)

            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            logging.debug(f"FPS: {self.fps}")

            self.frame_idx = 0

            self.pb()
            self.video_height, self.video_width, _ = self.frame.shape
            self.video_file_name = file_name

            # default scale
            for idx in range(2):
                self.fw[idx].zoom.setCurrentIndex(ZOOM_LEVELS.index(str(DEFAULT_FRAME_SCALE)))
                self.frame_viewer_scale(idx, DEFAULT_FRAME_SCALE)
                self.fw[idx].show()

            '''
            # moved after selection of objects to track
            self.initialize_positions_dataframe()
            self.initialize_areas_dataframe()
            '''

            self.fw[ORIGINAL_FRAME_VIEWER_IDX].setWindowTitle(f"Original frame - {pathlib.Path(file_name).name}")
            self.statusBar.showMessage(f"video loaded ({self.video_width}x{self.video_height})")


    def load_dir_images(self):
        """Load directory of images"""

        logging.debug("function: load_dir_images")

        if self.dir_images_path == "":
            self.dir_images_path = QFileDialog().getExistingDirectory(self, "Select Directory")

        if self.dir_images_path:
            p = pathlib.Path(self.dir_images_path)
            self.dir_images = sorted(list(p.glob('*')))
            print(f"self.dir_images: {self.dir_images}")

            '''
                                     + list(p.glob('*.jpg'))
                                     + list(p.glob('*.JPG'))
                                     + list(p.glob("*.png"))
                                     + list(p.glob("*.PNG")))
            '''
            self.total_frame_nb = len(self.dir_images)

            logging.info(f"images number: {self.total_frame_nb}")

            
            if not self.total_frame_nb:
                QMessageBox.critical(self, "DORIS", f"No images were found in {self.dir_images_path}")
                self.dir_images_path = ""
                return
            
            self.dir_images_index = -1
            r = self.pb()
            if r == False:
                QMessageBox.critical(self, "DORIS",
                                     f"The file <br>{self.dir_images[self.dir_images_index]}<br> can not be loaded")
                self.dir_images_path = ""
                self.dir_images = []
                return

            self.create_viewers()

            logging.debug(f"self.frame.shape: {self.frame.shape}")

            self.hs_frame.setMinimum(1)
            self.hs_frame.setMaximum(self.total_frame_nb)
            self.hs_frame.setTickInterval(10)

            self.lb_frames.setText(f"<b>{self.total_frame_nb}</b> images")

            self.video_height, self.video_width, _ = self.frame.shape

            '''
            # moved after selection of objects to track
            self.initialize_positions_dataframe()
            self.initialize_areas_dataframe()
            '''

            # default scale
            for idx in range(2):
                self.frame_viewer_scale(idx, 0.5)
                self.fw[idx].show()

            self.fw[ORIGINAL_FRAME_VIEWER_IDX].setWindowTitle(f"Original frame - {pathlib.Path(self.dir_images_path).name}")
            self.statusBar.showMessage(f"{self.total_frame_nb} image(s) found")


    def update_frame_index(self):
        """update frame index in label"""

        if self.dir_images:
            self.frame_idx = self.dir_images_index
            self.lb_frames.setText(f"Frame: <b>{self.frame_idx + 1}</b> / {self.total_frame_nb}")

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
                                                invert=self.cb_invert.isChecked(),
                                                arena=self.arena,
                                                masks=self.masks
                                                )


    def add_mask(self, shape):
        """
        define a mask to apply on image
        """

        if not self.dir_images and self.capture is None:
            return

        if self.frame is not None:
            self.flag_add_mask.define = not self.flag_add_mask.define
            self.statusBar.showMessage("")
            if self.flag_add_mask.define:
                self.flag_add_mask.color = WHITE if dialog.MessageDialog("DORIS", "Mask color",
                                                                         ["White", "Black"]) == "White" else BLACK
                self.flag_add_mask.shape = shape
                if shape in [RECTANGLE, CIRCLE_CENTER_RADIUS]:
                    self.statusBar.showMessage("You have to select 2 points on the video" * self.flag_add_mask.define)
                if shape == CIRCLE_3PTS:
                    self.statusBar.showMessage(("New circle mask: Click on the video to define 3 points "
                                               "belonging to the circle") * self.flag_add_mask.define)
                if shape == POLYGON:
                    self.statusBar.showMessage(("New polygon mask: click on the video to define the vertices of the polygon."
                                                "Right click to close the polygon") * self.flag_add_mask.define)
                    


    def define_coordinate_center_1point(self):
        """
        define origin of coordinates with a point
        """
        if self.frame is not None:
            self.flag_define_coordinate_center_1point = not self.flag_define_coordinate_center_1point
            self.actionOrigin_from_point.setText("from point" if not self.flag_define_coordinate_center_1point
                                                        else "Cancel origin definition")
            self.statusBar.showMessage(("You have to select the origin "
                                        "of the referential system") * self.flag_define_coordinate_center_1point)


    def define_coordinate_center_3points_circle(self):
        """
        define origin of coordinates with a 3 points circle
        """
        if self.frame is not None:
            self.flag_define_coordinate_center_3points = not self.flag_define_coordinate_center_3points
            self.actionOrigin_from_center_of_3_points_circle.setText("from center of 3 points circle" if not self.flag_define_coordinate_center_3points
                                                        else "Cancel origin definition")
            self.statusBar.showMessage(("You have to select the origin "
                                        "of the referential system with a 3 points circle") * self.flag_define_coordinate_center_3points)


    def define_scale(self):
        """
        define scale. from pixels to real unit
        """
        if self.frame is not None:
            # disable the stay on top property for frame viewers
            #for viewer in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]: #  PREVIOUS_FRAME_VIEWER_IDX
            #    self.fw[viewer].cb_stay_on_top.setChecked(False)

            self.flag_define_scale = not self.flag_define_scale
            self.actionDefine_scale.setText("Define scale" if not self.flag_define_scale else "Cancel scale definition")
            self.statusBar.showMessage("You have to select 2 points on the video" * self.flag_define_scale)


    def initialize_positions_dataframe(self):
        """
        initialize dataframe for recording objects coordinates
        """

        logging.debug("function: initialize_positions_dataframe" )

        columns = ["frame"]
        for idx in sorted(list(self.objects_to_track.keys())):
            columns.extend([f"x{idx}", f"y{idx}"])
        self.coord_df = pd.DataFrame(index=range(self.total_frame_nb), columns=columns)
        # set frame index
        self.coord_df["frame"] = range(1, self.total_frame_nb + 1)

        #self.coord_df.loc[:, ] = pd.NA
        #self.coord_df["y1"] = pd.NA

        logging.debug(f"self.coord_df: {self.coord_df}")


    def initialize_areas_dataframe(self):
        """
        initialize dataframe for recording presence of objects in areas
        """
        logging.debug("function: initialize_areas_dataframe" )

        columns = ["frame"]
        for area in sorted(self.areas.keys()):
            for idx in self.objects_to_track:
                columns.append(f"area {area} object #{idx}")
        self.areas_df = pd.DataFrame(index=range(self.total_frame_nb), columns=columns)
        # set frame index
        self.areas_df["frame"] = range(1, self.total_frame_nb + 1)


    def select_objects_to_track(self, all_=False):
        """
        Select objects to track and create the dataframes
        for recording objects positions and presence in area

        Args:
            all_ (boolean): True -> track all filtered objects
                            False (default) ->  let user choose the objects to track from filtered objects
        """

        logging.debug(f"function select_objects_to_track")

        if not self.dir_images and self.capture is None:
            return

        self.show_all_filtered_objects()

        if all_:
            self.objects_to_track = {}
            for idx in self.filtered_objects:
                self.objects_to_track[len(self.objects_to_track) + 1] = dict(self.filtered_objects[idx])
        else:
            elements = []
            for idx in self.filtered_objects:
                elements.append(f"Object # {idx}")
            w = dialog.CheckListWidget(elements)
            w.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            if w.exec_():

                logging.debug(f"objects checked: {w.checked}")

                self.objects_to_track = {}
                for el in w.checked:
                    self.objects_to_track[len(self.objects_to_track) + 1] = dict(self.filtered_objects[int(el.replace("Object # ", ""))])
                if not self.objects_to_track:
                    self.te_tracked_objects.clear()
            else:
                return

        logging.debug(f"self.objects_to_track\n {self.objects_to_track}")

        if self.coord_df is None:
            self.initialize_positions_dataframe()
        else:
            if dialog.MessageDialog("DORIS", "Do you want to reset the coordinates?",
                                    [YES, NO]) == YES: 
                self.initialize_positions_dataframe()
        if self.areas_df is None:
            self.initialize_areas_dataframe()
        else:
            if dialog.MessageDialog("DORIS", "Do you want to reset the coordinates?",
                                    [YES, NO]) == YES: 
                self.initialize_areas_dataframe()

        # delete positions on last frame
        if self.frame_idx - 1 in self.mem_position_objects:
            del self.mem_position_objects[self.frame_idx - 1]
        self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)

        logging.debug(f"objects to track: {list(self.objects_to_track.keys())}")

        self.cb_record_xy.setChecked(True)
        self.process_and_show()


    def repick_objects(self, mode="tracked"):
        """
        allow user to manually repick objects from image by clicking

        Args:
            mode (str): mode for showing objects: "all" -> all objects, "tracked" -> tracked objects
        """

        if not self.objects_to_track:
            QMessageBox.critical(self, "DORIS", "No object(s) to track")
            return

        # display objects on previous frame
        if self.previous_frame is not None:
            try:
                frame_with_objects = self.draw_marker_on_objects(self.previous_frame.copy(),
                                                             self.mem_position_objects[self.frame_idx - 1],
                                                             marker_type=MARKER_TYPE)

                # self.frame_viewer_scale2(2)
                # self.display_frame(frame_with_objects, 2)
            except Exception:
                error_type, _, _ = dialog.error_message("display objects on previous frame", sys.exc_info())
                logging.debug(error_type)

        self.repick_mode = mode
        logging.debug(f"repick mode: {self.repick_mode}")

        if mode == "all":
            self.show_all_filtered_objects()

            print(f"filtered objects: {list(self.filtered_objects.keys())}")
        


        self.statusBar.showMessage(f"Click object #1 on the original frame viewer (right-click to cancel)")
        self.repicked_objects = []
        while True:
            QApplication.processEvents()
            if self.repicked_objects is None:
                break
            if len(self.repicked_objects) == len(self.objects_to_track):  # all objects clicked
                break

        self.statusBar.showMessage(f"Done")

        # close previous frame
        self.fw[2].close()

        if self.repicked_objects is None:
            return

        new_order = {}
        for idx, (x, y) in enumerate(self.repicked_objects):
            if mode == "all":
                distances = [doris_functions.euclidean_distance(self.filtered_objects[o]["centroid"], (x, y)) for o in self.filtered_objects]
            elif mode == "tracked":
                distances = [doris_functions.euclidean_distance(self.objects_to_track[o]["centroid"], (x, y)) for o in self.objects_to_track]

            print(distances)
            print(min(distances))
            print(distances.index(min(distances)))

            new_order[idx + 1] = distances.index(min(distances)) + 1

        print(f"new order: {new_order}")

        # test if objects clicked more than one time
        print(list(new_order.values()))

        if sorted(list(set(new_order.values()))) == sorted(list(new_order.values())):
            new_objects_to_track = {}
            for idx in new_order:
                if mode == "all":
                    new_objects_to_track[idx] = copy.deepcopy(self.filtered_objects[new_order[idx]])
                elif mode == "tracked":
                    new_objects_to_track[idx] = copy.deepcopy(self.objects_to_track[new_order[idx]])

            self.objects_to_track = copy.deepcopy(new_objects_to_track)

            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                             self.objects_to_track,
                                                             marker_type=MARKER_TYPE)
            self.display_frame(frame_with_objects)

            self.repicked_objects = None

            self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)
            self.record_objects_data(self.frame_idx, self.objects_to_track)

            return

        else:

            clicked_objects = list(set(new_order.values()))
            if mode == "all":
                contours_list = [self.filtered_objects[x]["contour"] for x in clicked_objects]
            elif mode == "tracked":
                contours_list = [self.objects_to_track[x]["contour"] for x in clicked_objects]

            all_points = np.vstack(contours_list)
            all_points = all_points.reshape(all_points.shape[0], all_points.shape[2])

            centroids_list0 = self.repicked_objects
            new_contours = doris_functions.group_of(all_points, centroids_list0)

            new_objects = {}
            # add info to objects: centroid, area ...
            for idx, cnt in enumerate(new_contours):

                logging.debug(f"idx: {idx} len cnt {len(cnt)}")

                #cnt = cv2.convexHull(cnt)

                n = np.vstack(cnt).squeeze()
                try:
                    x, y = n[:, 0], n[:, 1]
                except Exception:
                    x = n[0]
                    y = n[1]

                # centroid
                cx = int(np.mean(x))
                cy = int(np.mean(y))

                new_objects[idx + 1] = {"centroid": (cx, cy),
                                                 "contour": cnt,
                                                 "area": cv2.contourArea(cnt),
                                                 "min": (int(np.min(x)), int(np.min(y))),
                                                 "max": (int(np.max(x)), int(np.max(y)))
                                                }

            # print("new_filtered_objects", [(x, new_filtered_objects[x]["centroid"]) for x in new_filtered_objects])

            self.objects_to_track = copy.deepcopy(new_objects)

            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                             self.objects_to_track,
                                                             marker_type=MARKER_TYPE)

            self.display_frame(frame_with_objects)


            self.repicked_objects = None

            self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)
            self.record_objects_data(self.frame_idx, self.objects_to_track)

            return



        '''

        if max([len(new_order2[x]) for x in new_order2]) == 1:  # 1 new object by old object

            new_objects_to_track = {}
            new_objects_to_track2 = {}
            for o in new_order2:
                new_objects_to_track2[new_order2[o][0]] = copy.deepcopy(self.objects_to_track[o])

            self.objects_to_track = copy.deepcopy(new_objects_to_track2)

            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                             self.objects_to_track,
                                                             marker_type=MARKER_TYPE)
            self.display_frame(frame_with_objects)


        else:  # more new positions than old objects (grouped object clicked twice)


            # limit contours to contours of clicked objects
            contours_list = [self.objects_to_track[x]["contour"] for x in list(new_order2.keys())]

            all_points = np.vstack(contours_list)
            all_points = all_points.reshape(all_points.shape[0], all_points.shape[2])

            centroids_list0 = self.repicked_objects

            logging.debug(f"Known centroids: {centroids_list0}")

            new_contours = doris_functions.group_of(all_points, centroids_list0)

            new_filtered_objects = {}
            # add info to objects: centroid, area ...
            for idx, cnt in enumerate(new_contours):

                logging.debug(f"idx: {idx} len cnt {len(cnt)}")

                n = np.vstack(cnt).squeeze()
                try:
                    x, y = n[:, 0], n[:, 1]
                except Exception:
                    x = n[0]
                    y = n[1]

                # centroid
                cx = int(np.mean(x))
                cy = int(np.mean(y))

                new_filtered_objects[idx + 1] = {"centroid": (cx, cy),
                                                 "contour": cnt,
                                                 "area": cv2.contourArea(cnt),
                                                 "min": (int(np.min(x)), int(np.min(y))),
                                                 "max": (int(np.max(x)), int(np.max(y)))
                                                }

            # print("new_filtered_objects", [(x, new_filtered_objects[x]["centroid"]) for x in new_filtered_objects])

            self.objects_to_track = copy.deepcopy(new_filtered_objects)

            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                             self.objects_to_track,
                                                             marker_type=MARKER_TYPE)

            self.display_frame(frame_with_objects)


        self.repicked_objects = None

        self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)
        self.record_objects_data(self.frame_idx, self.objects_to_track)
        '''


    def draw_reference_clicked(self):
        """
        click on draw reference menu option
        """
        self.process_and_show()


    def draw_reference(self, frame):
        """
        draw reference (a 100 px square) on frame
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

        Args:
            frame np.array(): image where to draw markers
            objects (dict): objects to draw
            marker_type (str): select the marker type to draw: rectangle or contour

        Returns:
            np.array: frame with objects drawn
        """

        logging.debug("function: draw_maker_on_objects")

        # print([(x, objects[x]["centroid"]) for x in objects])

        ratio, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)
        for idx in objects:

            marker_color = COLORS_LIST[(idx - 1) % len(COLORS_LIST)]

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

            if self.actionShow_object_path.isChecked():
                try:
                    for i in range(-OBJECT_PATH_LENGTH, -1):
                        cv2.line(frame,
                                 self.mem_position_objects[self.frame_idx + i][idx]["centroid"],
                                 self.mem_position_objects[self.frame_idx + i + 1][idx]["centroid"],
                                 marker_color, drawing_thickness + 1)
                except:
                    # positions history not long enough
                    pass

        return frame


    def display_frame(self, frame, viewer_idx=0):
        """
        display the current frame in viewer of index viewer_idx
        """
        if viewer_idx in [ORIGINAL_FRAME_VIEWER_IDX, PREVIOUS_FRAME_VIEWER_IDX]:
            self.fw[viewer_idx].lb_frame.setPixmap(frame2pixmap(frame).scaled(self.fw[viewer_idx].lb_frame.size(),
                                                                              Qt.KeepAspectRatio))
        if viewer_idx in [PROCESSED_FRAME_VIEWER_IDX]:
            self.fw[viewer_idx].lb_frame.setPixmap(QPixmap.fromImage(toQImage(frame)).scaled(self.fw[viewer_idx].lb_frame.size(),
                                                                                              Qt.KeepAspectRatio))


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

                if self.areas[area]["type"] == RECTANGLE:
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

            if self.arena["type"] == RECTANGLE:
                cv2.rectangle(frame, tuple(self.arena["points"][0]), tuple(self.arena["points"][1]),
                              color=ARENA_COLOR, thickness=drawing_thickness)

        return frame


    def draw_masks(self, frame, drawing_thickness):
        """
        draw masks on frame
        """
        for mask in self.masks:
            print(mask)
            if mask["type"] == RECTANGLE:
                cv2.rectangle(frame, tuple(mask["coordinates"][0]), tuple(mask["coordinates"][1]),
                              color=mask["color"], thickness=-1)
            if mask["type"] == CIRCLE:
                cv2.circle(frame, tuple(mask["center"]), mask["radius"],
                           color=mask["color"], thickness=-1)



        return frame


    def display_coordinates(self, frame_idx):
        """
        display coordinates
        """
        if self.cb_display_analysis.isChecked():

            if self.cb_apply_scale.isChecked():
                scale = self.scale
            else:
                scale = 1

            if self.cb_apply_origin.isChecked():
                origin_x, origin_y = self.coordinate_center
            else:
                origin_x, origin_y = 0, 0

            df = self.coord_df.copy()
            for idx in sorted(list(self.objects_to_track.keys())):
                df[f"x{idx}"] = scale * (df[f"x{idx}"] - origin_x)
                df[f"y{idx}"] = scale * (df[f"y{idx}"] - origin_y)

            self.te_xy.clear()
            if frame_idx >= NB_ROWS_COORDINATES_VIEWER // 2:
                start = frame_idx - NB_ROWS_COORDINATES_VIEWER // 2
            else:
                start = 0

            self.te_xy.append(df.iloc[start: frame_idx + NB_ROWS_COORDINATES_VIEWER // 2 + 1, :].to_string(index=False))


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
                                                                 )

        logging.debug(f"number of all filtered objects: {len(filtered_objects)}")

        contours_list = [filtered_objects[x]["contour"] for x in filtered_objects]

        logging.debug(f"self.objects_to_track: {list(self.objects_to_track.keys())}")

        # check filtered objects number
        # no filtered object
        '''
        if ((self.cb_record_xy.isChecked() or self.cb_record_presence_area.isChecked()) 
            and (len(filtered_objects) == 0) and (len(self.objects_to_track))):
        '''
        if len(filtered_objects) == 0 and len(self.objects_to_track):
            logging.debug("No object detected")

            frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                             {},
                                                             marker_type=MARKER_TYPE)

            self.update_info(all_objects,
                             filtered_objects,
                             {})

            _, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)

            # draw arena
            frame_with_objects = self.draw_arena(self.frame.copy(), drawing_thickness)

            # draw reference (100 px square)
            if self.actionDraw_reference.isChecked():
                frame_with_objects = self.draw_reference(frame_with_objects)

            # draw coordinate center if defined
            if self.coordinate_center != [0, 0]:
                frame_with_objects = self.draw_point_origin(frame_with_objects,
                                                            self.coordinate_center,
                                                            BLUE, drawing_thickness)

            # draw areas
            frame_with_objects = self.draw_areas(frame_with_objects, drawing_thickness)

            self.display_frame(frame_with_objects, 0)
            self.display_frame(processed_frame, 1)

            #if not self.continue_when_no_objects:
            if not self.cb_continue_when_no_objects.isChecked():
                choices = ["OK", "Skip frame", "Always skip frame when no objects found"]
                if self.running_tracking:
                    choices.append("Stop tracking")
                response = dialog.MessageDialog("DORIS",
                                                f"No objects detected.",
                                                choices)

                if response == "Stop tracking":
                    self.flag_stop_tracking = True
                if response == "Always skip frame when no objects found":
                    #self.continue_when_no_objects = True
                    self.cb_continue_when_no_objects.setChecked(True)

            if self.cb_record_xy.isChecked():
                # frame idx
                self.coord_df["frame"][self.frame_idx] = self.frame_idx + 1
                # tag
                for idx in sorted(list(self.objects_to_track.keys())):
                    # self.coord_df.loc[self.frame_idx, (f"x{idx}", f"y{idx}")] = [pd.NA, pd.NA]
                    self.coord_df.loc[self.frame_idx, (f"x{idx}", f"y{idx}")] = [np.nan, np.nan]

                # reset next frames to nan
                # self.coord_df.loc[self.frame_idx + 1:] = np.nan
                # self.coord_df.loc[self.frame_idx + 1:] = pd.NA

                self.display_coordinates(self.frame_idx)

            return

        # cancel continue_when_no_objects mode

        # self.continue_when_no_objects = False


        # filtered object are less than objects to track
        # apply clustering
        if len(filtered_objects) < len(self.objects_to_track):

            if self.frame_idx - 1 not in self.mem_position_objects:  # k-means
            #if True:  # disabled aggregation of points to previous centroid due to a bug
                logging.debug("Filtered object(s) are less than objects to track: applying k-means clustering")

                contours_list = [filtered_objects[x]["contour"] for x in filtered_objects]
                new_contours = doris_functions.apply_k_means(contours_list, len(self.objects_to_track))

                logging.debug(f"new contours after kmeans: {new_contours}")

                new_filtered_objects = {}
                # add info to objects: centroid, area ...
                for idx, cnt in enumerate(new_contours):
                    # cnt = cv2.convexHull(cnt)

                    n = np.vstack(cnt).squeeze()
                    try:
                        x, y = n[:, 0], n[:, 1]
                    except Exception:
                        x = n[0]
                        y = n[1]

                    # centroid
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        cx, cy = np.mean(x), np.mean(y)

                    new_filtered_objects[idx + 1] = {"centroid": (cx, cy),
                                                     "contour": cnt,
                                                     "area": cv2.contourArea(cnt),
                                                     "min": (int(np.min(x)), int(np.min(y))),
                                                     "max": (int(np.max(x)), int(np.max(y)))
                                                    }
                filtered_objects = dict(new_filtered_objects)

            else: # previous centroids known

                logging.debug("filtered object are less than objects to track: group by distances to centroids")

                contours_list1 = [filtered_objects[x]["contour"] for x in filtered_objects]
                centroids_list1 = [filtered_objects[obj_idx]["centroid"] for obj_idx in filtered_objects]
                # FIXME: ValueError: need at least one array to concatenate
                points1 = np.vstack(contours_list1)
                points1 = points1.reshape(points1.shape[0], points1.shape[2])

                centroids_list0 = [self.mem_position_objects[self.frame_idx - 1][k]["centroid"] for k in self.mem_position_objects[self.frame_idx - 1]]

                logging.debug(f"Known centroids: {centroids_list0}")
                logging.debug(f"Detected centroids: {centroids_list1}")

                #new_contours = doris_functions.group_sc(points1, centroids_list0, centroids_list1)

                new_contours = doris_functions.group_of(points1, centroids_list0)

                if [True for x in new_contours if len(x) == 0]:
                    logging.debug("one contour is null. Applying k-means")
                    contours_list = [filtered_objects[x]["contour"] for x in filtered_objects]
                    new_contours = doris_functions.apply_k_means(contours_list, len(self.objects_to_track))

                logging.debug(f"number of new contours after group: {len(new_contours)}")

                new_filtered_objects = {}
                # add info to objects: centroid, area ...
                for idx, cnt in enumerate(new_contours):

                    #cnt = cv2.convexHull(cnt)

                    n = np.vstack(cnt).squeeze()
                    try:
                        x, y = n[:, 0], n[:, 1]
                    except Exception:
                        x = n[0]
                        y = n[1]

                    # centroid
                    cx = int(np.mean(x))
                    cy = int(np.mean(y))

                    new_filtered_objects[idx + 1] = {"centroid": (cx, cy),
                                                     "contour": cnt,
                                                     "area": cv2.contourArea(cnt),
                                                     "min": (int(np.min(x)), int(np.min(y))),
                                                     "max": (int(np.max(x)), int(np.max(y)))
                                                    }
                    # print(idx, "centroid", (cx, cy))
                filtered_objects = dict(new_filtered_objects)

        self.filtered_objects = filtered_objects


        # assign filtered objects to objects to track
        if self.objects_to_track:
            try:
                mem_costs = {}
                obj_indexes = list(filtered_objects.keys())
                # iterate all combinations of detected objects of length( self.objects_to_track)

                # logging.debug(f"combinations of filtered objects: {obj_indexes}")

                if self.frame_idx -1 in self.mem_position_objects:

                    for indexes in itertools.combinations(obj_indexes, len(self.mem_position_objects[self.frame_idx - 1])):
                        cost = doris_functions.cost_sum_assignment(self.mem_position_objects[self.frame_idx - 1],
                                                                dict([(idx, filtered_objects[idx]) for idx in indexes]))

                        # logging.debug(f"index: {indexes} cost: {cost}")

                        mem_costs[cost] = indexes

                else:
                    for indexes in itertools.combinations(obj_indexes, len(self.objects_to_track)):
                        cost = doris_functions.cost_sum_assignment(self.objects_to_track,
                                                                dict([(idx, filtered_objects[idx]) for idx in indexes]))
                        # logging.debug(f"index: {indexes} cost: {cost}")
                        mem_costs[cost] = indexes

                min_cost = min(list(mem_costs.keys()))
                # logging.debug(f"minimal cost: {min_cost}")

                # select new objects to track

                new_objects_to_track = dict([(i + 1, filtered_objects[idx]) for i, idx in enumerate(mem_costs[min_cost])])

                # logging.debug(f"new objects to track : {list(new_objects_to_track.keys())}")

                self.objects_to_track = doris_functions.reorder_objects(self.objects_to_track, new_objects_to_track)

                self.mem_position_objects[self.frame_idx] = dict(self.objects_to_track)
            except Exception:
                self.error_message("assign filtered objects to objects to track", sys.exc_info())


        # check max distance from previous detected objects
        if (self.cb_record_xy.isChecked() or self.cb_record_presence_area.isChecked()) and self.sb_max_distance.value():

            if self.frame_idx - 1 in self.mem_position_objects:

                p1 = np.array([self.mem_position_objects[self.frame_idx - 1][k]["centroid"]
                               for k in self.mem_position_objects[self.frame_idx - 1]])

                p2 = np.array([self.objects_to_track[k]["centroid"] for k in self.objects_to_track])

                distances = doris_functions.distances(p1, p2)
                logging.debug(f"distances: {distances}")

                distant_objects = [idx_object for idx_object, distance in enumerate(distances) if distance > self.sb_max_distance.value()]

                dist_max = int(round(np.max(doris_functions.distances(p1, p2))))

                logging.debug(f"dist max: {dist_max}")

                #if dist_max > self.sb_max_distance.value():
                if distant_objects:

                    logging.debug(f"distance is greater than allowed by user: {distant_objects}")

                    self.update_info(all_objects, filtered_objects, self.objects_to_track)
                    frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                                     self.objects_to_track,
                                                                     marker_type=MARKER_TYPE)

                    # show previous positions
                    ratio, drawing_thickness = self.ratio_thickness(self.video_width, self.frame_width)
                    for distant_object in distant_objects:
                        marker_color = COLORS_LIST[(distant_object) % len(COLORS_LIST)]
                        self.draw_circle_cross(frame_with_objects,
                                               self.mem_position_objects[self.frame_idx - 1][distant_object + 1]["centroid"],
                                               marker_color, drawing_thickness)

                        cv2.putText(frame_with_objects,
                                    str(f" ({distant_object + 1})"),
                                    self.mem_position_objects[self.frame_idx - 1][distant_object + 1]["centroid"],
                                    font, FONT_SIZE, marker_color, drawing_thickness, cv2.LINE_AA)

                    self.display_frame(frame_with_objects, 0)
                    self.display_frame(processed_frame, 1)

                    if not self.always_skip_frame:
                        buttons = [REPICK_OBJECTS,
                                   ACCEPT_MOVEMENT,
                                   SKIP_FRAME,
                                   ALWAYS_SKIP_FRAME
                                   ]
                        if not self.running_tracking:
                            buttons.append(GO_TO_PREVIOUS_FRAME)

                        if self.running_tracking:
                            buttons.append("Stop tracking")

                        response = dialog.MessageDialog("DORIS",
                                        (f"The object(s) # <b>{', '.join([str(x + 1) for x in distant_objects])}</b> "
                                         "moved more than allowed.<br>"
                                         f"Maximum distance allowed: {self.sb_max_distance.value()}.<br><br>"
                                         "What do you want to do?"),
                                         buttons)
                    else:
                        response = SKIP_FRAME

                    if response in [SKIP_FRAME, ALWAYS_SKIP_FRAME]:
                        # frame skipped and positions are taken from previous frame

                        if self.cb_record_xy.isChecked():
                            # frame idx
                            self.coord_df.loc[self.frame_idx, ("frame")] = self.frame_idx + 1
                            for idx in sorted(list(self.objects_to_track.keys())):
                                self.coord_df.loc[self.frame_idx, (f"x{idx}", f"y{idx}")] = [np.nan, np.nan]

                            # reset next frames to nan
                            if self.cb_reset_following_coordinates.isChecked():
                                self.coord_df.loc[self.frame_idx + 1:] = np.nan

                            self.display_coordinates(self.frame_idx)

                        # last tracked objects
                        self.objects_to_track = self.mem_position_objects[self.frame_idx - 1]
                        self.mem_position_objects[self.frame_idx] = dict(self.mem_position_objects[self.frame_idx - 1])
                        if response == ALWAYS_SKIP_FRAME:
                            self.always_skip_frame = True

                        return

                    if response == REPICK_OBJECTS:
                        self.repick_objects("tracked")

                    if response == GO_TO_PREVIOUS_FRAME:
                        self.for_back_ward(direction="backward")
                        return

                    if response == "Stop tracking":
                        self.flag_stop_tracking = True
                        return


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
                frame_with_objects = self.draw_point_origin(frame_with_objects,
                                                            self.coordinate_center,
                                                            BLUE,
                                                            drawing_thickness)

            # draw areas
            frame_with_objects = self.draw_areas(frame_with_objects, drawing_thickness)

            # draw masks
            frame_with_objects = self.draw_masks(frame_with_objects, drawing_thickness)

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
            self.display_frame(frame_with_objects, 0)
            self.display_frame(processed_frame, 1)

        #  record objects data
        self.record_objects_data(self.frame_idx, self.objects_to_track)
        self.always_skip_frame = False



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
                                                                   max_extension=self.sb_max_extension.value()
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
                                                                        )

        frame_with_objects = self.draw_marker_on_objects(self.frame.copy(),
                                                         filtered_objects,
                                                         marker_type="contour")
        self.display_frame(frame_with_objects)


    def force_objects_number(self):
        """separate initial aggregated objects using k-means clustering to arbitrary number of objects"""

        logging.debug("function: force_objects_number")

        if not self.dir_images and self.capture is None:
            return

        if not self.filtered_objects:
            return

        logging.debug(f"filtered objects: {list(self.filtered_objects.keys())}")

        nb_obj, ok_pressed = QInputDialog.getInt(self, "Get number of objects to filter",
                                                 "Number of objects:", 1, 1, 1000, 1)
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


    def save_config(self):
        """
        save configuration file
        """

        config_file_path = str(pathlib.Path(os.path.expanduser("~")) / ".doris")
        settings = QSettings(config_file_path, QSettings.IniFormat)
        settings.setValue("geometry", self.saveGeometry())


    def read_config(self):
        """
        Read configuration file
        """

        config_file_path = str(pathlib.Path(os.path.expanduser("~")) / ".doris")
        if os.path.isfile(config_file_path):
            settings = QSettings(config_file_path, QSettings.IniFormat)
            try:
                self.restoreGeometry(settings.value("geometry"))
            except Exception:
                logging.warning("Error trying to restore geometry")
                pass


    def closeEvent(self, event):

        if self.coord_df is not None:
            if dialog.MessageDialog("DORIS",
                                    ("Check if your data are saved and confirm close."),
                                    ["Cancel", "Close DORIS"]) == "Cancel":
                event.ignore()
                return

        self.repicked_objects = None

        if self.capture:
            self.capture.release()

        # close the frame viewers
        try:
            for i in range(3):
               self.fw[i].close()
        except Exception:
            pass

        self.save_config()


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
            self.te_tracked_objects.clear()



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
                self.flag_stop_tracking = False
                self.statusBar.showMessage("Last image of dir")
                return False
            self.previous_frame = self.frame
            self.frame = cv2.imread(str(self.dir_images[self.dir_images_index]), -1)
            if self.frame is None:
                return False
        else:
            if self.capture is not None:

                self.previous_frame = self.frame
                ret, self.frame = self.capture.read()
                if not ret:
                    self.flag_stop_tracking = False
                    return False
            else:
                self.flag_stop_tracking = False
                return False

        self.update_frame_index()

        self.process_and_show()

        QApplication.processEvents()

        return True


    def record_objects_data(self, frame_idx, objects):
        """
        Record objects coordinates and presence in areas defined by user
        """

        if not self.objects_to_track:
            return

        # objects positions
        if self.cb_record_xy.isChecked():

            if self.coord_df is None:
                return

            logging.debug(f"sorted objects to record: {sorted(list(objects.keys()))}")

            # frame idx
            self.coord_df.loc[frame_idx - 1, "frame"] = frame_idx  # + 1

            obj_col_coord = [] # list of objects coordinates columns 
            for idx in sorted(list(objects.keys())):
                obj_col_coord.extend([f"x{idx}", f"y{idx}"]) 
                # record centroid of objects without modifications (oring, scale)
                self.coord_df.loc[frame_idx - 1, (f"x{idx}", f"y{idx}")] = (objects[idx]["centroid"][0], objects[idx]["centroid"][1])

                '''
                if self.cb_normalize_coordinates.isChecked():
                    self.coord_df.loc[frame_idx - 1, (f"x{idx}", f"y{idx}")] = ((objects[idx]["centroid"][0] - self.coordinate_center[0]) / self.video_width,
                                                                                (objects[idx]["centroid"][1] - self.coordinate_center[1]) / self.video_width
                                                                               ) 
                else:
                    if self.cb_apply_scale.isChecked():
                        scale = self.scale
                    else:
                        scale = 1

                    if self.cb_apply_origin.isChecked():
                        origin_x, origin_y = self.coordinate_center
                    else:
                        origin_x, origin_y = 0, 0

                    self.coord_df.loc[frame_idx - 1, (f"x{idx}", f"y{idx}")] = (scale * (objects[idx]["centroid"][0] - origin_x),
                                                                                scale * (objects[idx]["centroid"][1] - origin_y)
                                                                               )
                '''
            # set NaN as coordinates to next frames
            if self.cb_reset_following_coordinates.isChecked():
                self.coord_df.loc[frame_idx:, obj_col_coord] = np.nan
                # self.coord_df.loc[frame_idx:, obj_col_coord] = pd.NA

            self.display_coordinates(frame_idx)

        # presence in areas
        if self.cb_record_presence_area.isChecked():

            nb = {}
            if self.areas_df is None:
                QMessageBox.warning(self, "DORIS", "No objects to track")
                return

            # frame idx
            self.areas_df["frame"][frame_idx - 1] = frame_idx

            for area in sorted(list(self.areas.keys())):

                nb[area] = 0

                if self.areas[area]["type"] == "circle":
                    cx, cy = self.areas[area]["center"]
                    radius = self.areas[area]["radius"]

                    for idx in objects:
                        x, y = objects[idx]["centroid"]
                        if ((cx - x) ** 2 + (cy - y) ** 2) ** .5 <= radius:
                            nb[area] += 1

                        self.areas_df[f"area {area} object #{idx}"][frame_idx] = int(((cx - x) ** 2 + (cy - y) ** 2) ** .5 <= radius)

                if self.areas[area]["type"] == RECTANGLE:
                    minx, miny = self.areas[area]["pt1"]
                    maxx, maxy = self.areas[area]["pt2"]

                    for idx in objects:
                        x, y = objects[idx]["centroid"]
                        self.areas_df[f"area {area} object #{idx}"][frame_idx] = int(minx <= x <= maxx and miny <= y <= maxy)

                        if minx <= x <= maxx and miny <= y <= maxy:
                            nb[area] += 1

                if self.areas[area]["type"] == "polygon":
                    for idx in objects:
                        x, y = objects[idx]["centroid"]
                        self.areas_df[f"area {area} object #{idx}"][frame_idx] = int(cv2.pointPolygonTest(np.array(self.areas[area]["points"]),
                                                                                                             (x, y), False) >= 0)

                        if cv2.pointPolygonTest(np.array(self.areas[area]["points"]), (x, y), False) >= 0:
                            nb[area] += 1

            if self.cb_display_analysis.isChecked():
                self.te_number_objects.clear()
                self.te_number_objects.append(str(self.areas_df[frame_idx - 3: frame_idx + 3 + 1]))

            self.objects_number.append(nb)


    def stop_tracking(self):
        """ Stop running tracking """

        self.running_tracking = False
        self.flag_stop_tracking = False
        self.pb_run_tracking.setChecked(False)
        self.pb_run_tracking_frame_interval.setChecked(False)
        self.always_skip_frame = False
        logging.info("tracking stopped")


    def run_tracking(self):
        """ Run tracking from current frame to end """

        # check if objects to track are defined
        if not self.objects_to_track :

            self.disable_viewers_stay_on_top()
            QMessageBox.warning(self, "DORIS", "No objects to track.\nSelect objects to track before running tracking")
            self.pb_run_tracking.setChecked(False)
            return

        if self.running_tracking:
            self.flag_stop_tracking = True
            return

        if not self.dir_images:
            try:
                self.capture
            except:
                self.stop_tracking()
                self.disable_viewers_stay_on_top()
                QMessageBox.warning(self, "DORIS", "No video")
                return

        self.running_tracking = True
        self.pb_run_tracking.setChecked(True)
        while True:
            if not self.pb():
                self.stop_tracking()
                break

            QApplication.processEvents()
            if self.flag_stop_tracking:
                self.stop_tracking()
                break

        self.flag_stop_tracking = False


    def run_tracking_frames_interval(self):
        """
        run tracking in a frames interval
        """
        # check if objects to track are defined
        if not self.objects_to_track:
            self.disable_viewers_stay_on_top()
            QMessageBox.warning(self, "DORIS", "No objects to track.\nSelect objects to track before running tracking")
            self.pb_run_tracking_frame_interval.setChecked(False)
            return

        if self.running_tracking:
            self.flag_stop_tracking = True
            return

        # disable the stay on top property for frame viewers
        for viewer in [ORIGINAL_FRAME_VIEWER_IDX, PROCESSED_FRAME_VIEWER_IDX]: #  PREVIOUS_FRAME_VIEWER_IDX
            self.fw[viewer].cb_stay_on_top.setChecked(False)

        try:
            text, ok = QInputDialog.getText(self, "Run tracking", "Frames interval: (ex. 123-456)")
            if not ok:
                return
            start_frame, stop_frame = [int(x) for x in text.split("-")]
            if start_frame >= stop_frame:
                raise
        except:
            self.disable_viewers_stay_on_top()
            QMessageBox.warning(self, "DORIS", f"{text} is not a valid interval")
            return

        logging.info(f"start_frame: {start_frame} stop frame: {stop_frame}")

        self.go_to_frame(start_frame - 1)

        self.running_tracking = True
        while True:
            if not self.pb():
                self.stop_tracking()
                break

            QApplication.processEvents()

            if self.frame_idx >= stop_frame:
                self.stop_tracking()
                break

            if self.flag_stop_tracking:
                self.stop_tracking()
                break

        self.flag_stop_tracking = False


    def new_project(self):
        """
        initialize program for new project
        """

        self.frame = None
        self.frame_idx = 0
        self.total_frame_nb = 0
        if self.capture:
            self.capture.release()
            self.capture = None
        self.dir_images = ""
        self.fw = []

        self.lb_frames.clear()
        self.objects_to_track = {}
        self.te_tracked_objects.clear()
        self.mem_position_objects = {}
        self.cb_record_xy.setChecked(False)
        self.cb_record_presence_area.setChecked(False)
        self.coord_df = None
        self.areas_df = None

        self.te_xy.clear()
        self.te_number_objects.clear()

        self.always_skip_frame = False
        self.sb_start_from.setValue(0)
        self.sb_stop_to.setValue(0)
        self.sb_frame_offset.setValue(1)

        self.coordinate_center = [0, 0]
        self.le_coordinates_center.setText(f"{self.coordinate_center}")

        self.flag_define_scale = False
        self.scale_points = []

        self.scale = 1
        self.le_scale.setText("1")

        self.sbMin.setValue(0)
        self.sbMax.setValue(0)
        self.sb_max_extension.setValue(0)
        self.sb_max_distance.setValue(DIST_MAX)
        self.sb_blur.setValue(BLUR_DEFAULT_VALUE)

        self.cb_threshold_method.setCurrentIndex(0)
        self.sb_threshold.setValue(THRESHOLD_DEFAULT)
        self.sb_block_size.setValue(ADAPTIVE_BLOCK_SIZE)
        self.sb_offset.setValue(ADAPTIVE_OFFSET)


        self.cb_normalize_coordinates.setChecked(False)
        self.cb_display_analysis.setChecked(True)

        self.clear_arena()

        # masks
        self.masks = []
        self.lw_masks.clear()

        self.te_all_objects.clear()
        self.te_filtered_objects.clear()

        self.project_path = ""
        self.setWindowTitle(f"DORIS v. {version.__version__}")

        self.video_height, self.video_width = 0, 0
        self.video_file_name = ""

        self.lw_area_definition.clear()
        self.areas = {}

        #self.__init__()

        try:
            for i in range(3):
               self.fw[i].close()
        except Exception:
            pass

        self.project = "NO NAME"
        self.menu_update()


    def open_project(self, file_name=""):
        """
        open a project file and load parameters
        """

        logging.debug("function: open_project")

        if not file_name:
            file_name, _ = QFileDialog().getOpenFileName(self, "Open project", "", "DORIS projects (*.doris);;All files (*)")

        if not file_name:
            return

        if not os.path.isfile(file_name):
            QMessageBox.critical(self, "DORIS", f"{file_name} not found")
            return

        try:
            with open(file_name) as f_in:
                config = json.loads(f_in.read())

            # load coordinates
            if "coordinates" in config:
                self.coord_df = pd.read_csv(StringIO(config["coordinates"]), sep=",", index_col=0)

            self.sb_start_from.setValue(config.get("start_from", 0))
            self.sb_stop_to.setValue(config.get("stop_to", 0))
            self.sb_blur.setValue(config.get("blur", BLUR_DEFAULT_VALUE))
            self.cb_invert.setChecked(config.get("invert", False))
            self.cb_normalize_coordinates.setChecked(config.get("normalize_coordinates", False))

            self.cb_record_xy.setChecked(config.get("record_coordinates", False))
            self.cb_record_presence_area.setChecked(config.get("record_presence_area", False))
            self.cb_apply_scale.setChecked(config.get("apply_scale", False))
            self.cb_apply_origin.setChecked(config.get("apply_origin", False))

            try:
                self.arena = config["arena"]
                if self.arena:
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)
                    self.le_arena.setText(str(config["arena"]))
            except KeyError:
                logging.info("arena not found")

            self.sbMin.setValue(config.get("min_object_size", 0))
            self.sbMax.setValue(config.get("max_object_size", 0))
            self.sb_max_extension.setValue(config.get("object_max_extension", 0))
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


            if "video_file_path" in config:
                try:
                    if os.path.isfile(config["video_file_path"]):
                        self.open_video(config["video_file_path"])
                    # check if video file is on same dir than project file
                    elif (pathlib.Path(file_name).parent / pathlib.Path(config["video_file_path"]).name).is_file():
                        self.open_video(str(pathlib.Path(file_name).parent / pathlib.Path(config["video_file_path"]).name))
                    else:
                        QMessageBox.critical(self, "DORIS", f"File {config['video_file_path']} not found")
                        return None
                except Exception:
                    pass

            self.dir_images_path = config.get("dir_images_path", "")
            if self.dir_images_path:
                try:
                    if os.path.isdir(self.dir_images_path):
                        self.load_dir_images()
                    else:
                        QMessageBox.critical(self, "DORIS", f"Directory {self.dir_images_path} not found")
                        self.dir_images_path = ""
                        return None
                except Exception:
                    pass

            self.coordinate_center = config.get("referential_system_origin", [0,0])
            self.le_coordinates_center.setText(f"{self.coordinate_center}")
            self.masks = config.get("masks", [])
            if self.masks:
                for mask in self.masks:
                    self.lw_masks.addItem(str(mask))

            self.actionShow_centroid_of_object.setChecked(config.get("show_centroid", SHOW_CENTROID_DEFAULT))
            self.actionShow_contour_of_object.setChecked(config.get("show_contour", SHOW_CONTOUR_DEFAULT))
            self.actionDraw_reference.setChecked(config.get("show_reference", False))
            self.sb_max_distance.setValue(config.get("max_distance", 0))

            '''
            # original frame viewer
            self.fw[ORIGINAL_FRAME_VIEWER_IDX].move(*config.get("original_frame_viewer_position", [20, 20]))
            self.frame_scale = config.get("frame_scale", DEFAULT_FRAME_SCALE)
            self.fw[ORIGINAL_FRAME_VIEWER_IDX].zoom.setCurrentText(str(self.frame_scale))
            self.frame_viewer_scale(ORIGINAL_FRAME_VIEWER_IDX, self.frame_scale)

            # processed frame viewer
            self.fw[PROCESSED_FRAME_VIEWER_IDX].move(*config.get("processed_frame_viewer_position", [40, 40]))
            self.processed_frame_scale = config.get("processed_frame_scale", DEFAULT_FRAME_SCALE)
            self.fw[PROCESSED_FRAME_VIEWER_IDX].zoom.setCurrentText(str(self.processed_frame_scale))
            self.frame_viewer_scale(PROCESSED_FRAME_VIEWER_IDX, self.processed_frame_scale)
            '''

            # objects to track
            if "objects_to_track" in config:
                obj_dict = dict(config["objects_to_track"])
                for idx in obj_dict:
                    obj_dict[idx]["contour"] = np.asarray(obj_dict[idx]["contour"])

                self.objects_to_track = dict(obj_dict)

            self.frame_idx = config.get("current_frame", 0)
            if self.frame_idx:
                self.go_to_frame(self.frame_idx)
            elif self.sb_start_from.value():
                self.go_to_frame(self.sb_start_from.value())
            else:
                self.process_and_show()



        except Exception:
            raise
            logging.warning("Error in project file")
            QMessageBox.critical(self, "DORIS", f"Error in project file: {file_name}")
            return

        self.project_path = file_name
        self.setWindowTitle(f"DORIS v. {version.__version__} - {self.project_path}")

        self.project = pathlib.Path(self.project_path).name
        self.menu_update()

        return True


    def error_message(self, task: str, exc_info: tuple) -> None:
        """
        show details about the error

        """
        error_type, error_file_name, error_lineno = doris_functions.error_info(exc_info)
        QMessageBox.critical(None, "DORIS",
                            (f"An error occured during {task}.<br>"
                            f"DORIS version: {version.__version__}<br>"
                            f"Error: {error_type}<br>"
                            f"in {error_file_name} "
                            f"at line # {error_lineno}<br><br>"
                            "Please report this problem to improve the software at:<br>"
                            '<a href="https://github.com/olivierfriard/DORIS/issues">https://github.com/olivierfriard/DORIS/issues</a>'
                            ))



def main():

    parser = argparse.ArgumentParser(description="DORIS (Detection of Objects Research Interactive Software)")

    parser.add_argument("-v", action="store_true", default=False, dest="version", help="Print version")
    parser.add_argument("-p", action="store", dest="project_file", help="path of project file")
    parser.add_argument("--threshold", action="store", default=THRESHOLD_DEFAULT, dest="threshold", help="Threshold value")
    parser.add_argument("--blur", action="store", default=BLUR_DEFAULT_VALUE, dest="blur", help="Blur value")
    parser.add_argument("--invert", action="store_true", dest="invert", help="Invert B/W")

    options = parser.parse_args()
    if options.version:
        print(f"version {version.__version__} release date: {version.__version_date__}")
        sys.exit()

    app = QApplication(sys.argv)
    w = Ui_MainWindow()

    if options.project_file:
        if os.path.isfile(options.project_file):
            if w.open_project(options.project_file) is None:
                sys.exit()
        else:
            print(f"{options.project_file} not found!")
            sys.exit()
    else:
        if options.blur:
            w.sb_blur.setValue(int(options.blur))

        if options.threshold:
            w.sb_threshold.setValue(int(options.threshold))

        w.cb_invert.setChecked(options.invert)

    w.show()
    w.raise_()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()