#!/usr/bin/env python3
"""
DORIS
Detection of Objects Research Interactive Software
Copyright 2017 Olivier Friard

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
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
MAGENTA = (255,0,255)
YELLOW = (255,255,0)
BLACK = (0,0,0)
MAROON = (128,0,0)
TEAL = (0,128,128)
PURPLE = (128,0,128)
WHITE = (255,255,255)

# colors list for marker
COLORS_LIST = [BLUE, MAGENTA, YELLOW, MAROON, TEAL, PURPLE, WHITE]

MARKER_TYPE = "contour"
MARKER_COLOR = GREEN
ARENA_COLOR = RED
AREA_COLOR = GREEN

CIRCLE_THICKNESS = 2
RECTANGLE_THICKNESS = 2

TOLERANCE_OUTSIDE_ARENA = 0.05

__version__ = "0.0.1"
__version_date__ = "2017-11-07"

from PyQt5.QtCore import *

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import os
import platform
import numpy as np
np.set_printoptions(threshold="nan")
import cv2
import copy

import sys
import time
import math
import matplotlib
matplotlib.use("Qt4Agg" if QT_VERSION_STR[0] == "4" else "Qt5Agg")
import matplotlib.pyplot as plt

try:
    import mpl_scatter_density
    flag_mpl_scatter_density = True
except:
    flag_mpl_scatter_density = False

import argparse

from doris_ui import *

font = cv2.FONT_HERSHEY_SIMPLEX

'''
def distance_point_line(P, M1, M2):
    """
    return:
    * distance from point P to segment M1-M2
    * coordinates of nearest point on segment
    """

    EPS = 0
    EPSEPS = EPS * EPS

    def SqLineMagnitude(x1, y1, x2, y2):
        return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

    px, py = P
    x1, y1 = M1
    x2, y2 = M2

    result = 0

    SqLineMag = SqLineMagnitude(x1, y1, x2, y2)
    if SqLineMag < EPSEPS:
        return - 1.0

    u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / SqLineMag

    if (u < EPS) or (u > 1):
        ###  Closest point does not fall within the line segment,
        ###  take the shorter distance to an endpoint
        d1 = SqLineMagnitude(px, py, x1, y1)
        d2 = SqLineMagnitude(px, py, x2, y2)
        if d1 <= d2:
            result = d1
            ix, iy = x1, y1
        else:
            result = d2
            ix, iy = x2, y2

    else:

        #  Intersecting point is on the line, use the formula
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        result = SqLineMagnitude(px, py, ix, iy)


    return math.sqrt(result), (ix, iy)
'''

def frame2pixmap(frame):
    try:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
    
    from matplotlib.path import Path
    import matplotlib.patches as patches
    
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
    

def extract_objects(frame, threshold, min_size=0, max_size=0, largest_number=0, arena=[], max_extension=50):
    """
    returns all detected objects and filtered objects
    """

    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nr_objects = len(contours)

    print("nr objects", nr_objects)

    all_objects = {}
    for idx, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        area = cv2.contourArea(cnt)
        all_objects[idx] = {"centroid": (cx, cy), "contour": cnt, "area": area}

    # record objects that not match conditions
    obj_to_del_idx = []
    for idx in all_objects:

        # check if object size is >= of minimal size
        if min_size and all_objects[idx]["area"] < min_size:
            print("deleted #{} for min size".format(idx))
            obj_to_del_idx.append(idx)
            continue

        # check if object size is <= of maximal size
        if max_size and all_objects[idx]["area"] > max_size:
            print("deleted #{} for max size".format(idx))
            obj_to_del_idx.append(idx)
            continue
            
        # check if object extension <= max extension
        if max_extension:
            n = np.vstack(all_objects[idx]["contour"]).squeeze()
            if (max(n[:,0]) - min(n[:,0]) > max_extension) or (max(n[:,1]) - min(n[:,1]) > max_extension):
                print("deleted #{} for max extension".format(idx))
                obj_to_del_idx.append(idx)
                continue

        if arena:

            # check if centroid of object in arena
            if arena["type"] == "polygon":
                
                np_arena = np.array(arena["points"])
                if cv2.pointPolygonTest(np_arena, all_objects[idx]["centroid"], False) < 0:
                    obj_to_del_idx.append(idx)
                    continue
                # check if all contour points are in polygon arena (5% tolerance)
                n = np.vstack(all_objects[idx]["contour"]).squeeze()
                nl = len(n)
                count_out = 0
                for pt in n:
                    if cv2.pointPolygonTest(np_arena, tuple(pt), False) < 0:
                        count_out += 1
                        if count_out / nl > TOLERANCE_OUTSIDE_ARENA:
                            break
                if count_out / nl > TOLERANCE_OUTSIDE_ARENA:
                    obj_to_del_idx.append(idx)
                    continue

            if arena["type"] == "circle":
                if dist(all_objects[idx]["centroid"], arena["center"]) > arena["radius"]:
                    obj_to_del_idx.append(idx)
                    continue

                # check if all contour points are in circle arena (5% tolerance)
                n = np.vstack(all_objects[idx]["contour"]).squeeze()
                dist_ = ((n[:,0] - arena["center"][0])**2 + (n[:,1] - arena["center"][1])**2)**0.5

                '''
                print("distance")
                print("contour size:", len(dist_))
                print("% >", np.count_nonzero(dist_ > arena["radius"]) / len(dist_))
                '''
                if np.count_nonzero(dist_ > arena["radius"]) / len(dist_) > TOLERANCE_OUTSIDE_ARENA:
                    obj_to_del_idx.append(idx)
                    continue

                #print("perimeter", cv2.arcLength(all_objects[idx]["contour"], True))


    # delete objects
    '''
    for idx in obj_to_del_idx:
        if idx in all_objects:
            del all_objects[idx]
    '''

    # sizes
    sizes = sorted([all_objects[idx]["area"] for idx in all_objects if idx not in obj_to_del_idx], reverse=True)

    filtered_objects = {}
    new_idx = 0
    for idx in all_objects:
        if idx in obj_to_del_idx:
            continue
        obj_size = all_objects[idx]["area"]

        if (sizes.index(obj_size) < largest_number):
            new_idx += 1
            # min/max
            n = np.vstack(all_objects[idx]["contour"]).squeeze()
            x, y = n[:,0], n[:,1]

            filtered_objects[new_idx] = {"centroid": all_objects[idx]["centroid"],
                                         "contour": all_objects[idx]["contour"],
                                         "size": obj_size,
                                         "min": (int(np.min(x)), int(np.min(y))),
                                         "max": (int(np.max(x)), int(np.max(y)))}


    return all_objects, filtered_objects
    

def dist(p1, p2):
    """
    euclidean distance of two points
    """
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5


def find_circle(points):
    
    
    x1, y1 = points[0]
    x2,y2 = points[1]
    x3,y3 = points[2]

    ma = (y2-y1)/(x2-x1)
    mb = (y3-y2)/(x3-x2)

    x = (ma*mb*(y1-y3) + mb*(x1+x2) - ma*(x2+x3))/(2*(mb-ma))

    print(x)

    y = -(1/ma)*(x-(x1+x2)/2)+(y1+y2)/2

    print(y)
    
    return x, y, dist((x, y), (x1, y1))

    
    
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
        
        self.actionAbout.triggered.connect(self.about)
        
        self.label1.mousePressEvent = self.label1_mousepressed

        self.actionOpen_video.triggered.connect(lambda: self.open_video(""))
        self.actionQuit.triggered.connect(self.close)
        
        self.actionFrame_width.triggered.connect(self.frame_width)

        self.pb_next_frame.clicked.connect(self.next_frame)
        #self.pb_next_frame.clicked.connect(lambda: self.pb(1))
        self.pb_1st_frame.clicked.connect(self.reset)
        
        self.pb_goto_frame.clicked.connect(self.go_to_frame)
        
        #self.pb_forward.clicked.connect(lambda: self.pb(self.sb_frame_offset.value()))
        #self.pb_backward.clicked.connect(lambda: self.pb(-self.sb_frame_offset.value()))
        self.pb_forward.clicked.connect(lambda: self.for_back_ward("forward"))
        self.pb_backward.clicked.connect(lambda: self.for_back_ward("backward"))


        menu1 = QMenu()
        menu1.addAction("Polygon arena", lambda: self.define_arena("polygon"))
        menu1.addAction("Circle arena", lambda: self.define_arena("circle"))

        self.pb_define_arena.setMenu(menu1)

        #self.pb_define_arena.clicked.connect(self.define_arena)

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

        # coordinates analysis
        self.pb_reset_xy.clicked.connect(self.reset_xy_analysis)
        self.pb_save_xy.clicked.connect(self.save_xy)
        self.pb_plot_path.clicked.connect(self.plot_path)
        self.pb_plot_xy_density.clicked.connect(self.plot_xy_density)
        self.pb_plot_xy_density.setEnabled(flag_mpl_scatter_density)

        # areas analysis
        menu = QMenu()
        menu.addAction("Circle", lambda: self.add_area_func("circle"))
        menu.addAction("Rectangle", lambda: self.add_area_func("rectangle"))
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
        
        self.positions, self.objects_number = [], []

        # default
        self.sb_threshold.setValue(THRESHOLD_DEFAULT)

    def about(self):

        players = []
        players.append("OpenCV")
        players.append("version {}".format(cv2.__version__))
        
        # matplotlib
        players.append("\nMatplotlib")
        players.append("version {}".format(matplotlib.__version__))

        about_dialog = msg = QMessageBox()
        #about_dialog.setIconPixmap(QPixmap(os.path.dirname(os.path.realpath(__file__)) + "/logo_eye.128px.png"))
        about_dialog.setWindowTitle("About DORIS")
        about_dialog.setStandardButtons(QMessageBox.Ok)
        about_dialog.setDefaultButton(QMessageBox.Ok)
        about_dialog.setEscapeButton(QMessageBox.Ok)

        about_dialog.setInformativeText(("<b>DORIS</b> v. {ver} - {date}"
        "<p>Copyright &copy; 2017 Olivier Friard<br>"
        "Department of Life Sciences and Systems Biology<br>"
        "University of Torino - Italy<br>").format(ver=__version__,
                                              date=__version_date__
                                              ))

        details = ("Python {python_ver} ({architecture}) - Qt {qt_ver} - PyQt{pyqt_ver} on {system}\n"
        "CPU type: {cpu_info}\n\n"
        "{players}").format(python_ver=platform.python_version(),
                            architecture="64-bit" if sys.maxsize > 2**32 else "32-bit",
                            pyqt_ver=PYQT_VERSION_STR,
                            system=platform.system(),
                            qt_ver=QT_VERSION_STR,
                            cpu_info=platform.machine(),
                            players="\n".join(players))

        about_dialog.setDetailedText(details)

        _ = about_dialog.exec_()


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
            text, ok = QInputDialog.getText(self, 'New area', 'Area name:')
            if ok:
                self.add_area = {"type": shape, "name": text}


    def remove_area(self):
        """
        remove the selected area
        """
        
        for selected_item in self.lw_area_definition.selectedItems():
            self.lw_area_definition.takeItem(self.lw_area_definition.row(selected_item))
            self.activate_areas()


    def define_arena(self, shape):
        """
        swith to define area mode
        """
        if self.flag_define_arena:
            self.flag_define_arena = ""
        else:
            self.flag_define_arena = shape
            self.pb_define_arena.setEnabled(False)


    def clear_arena(self):
        """
        clear the arena
        """
        self.flag_define_arena = False
        self.arena = []
        self.le_arena.setText("")
        
        self.pb_define_arena.setEnabled(True)
        self.pb_clear_arena.setEnabled(False)
        
        
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        self.pb()


    def conversion_thickness(self, video_width, frame_width):
        conversion = video_width / frame_width
        if conversion <= 1: 
            drawing_thickness = 1
        else:
            drawing_thickness = round(conversion)
        return conversion, drawing_thickness


    def label1_mousepressed(self, event):
        """
        record clicked coordinates if arena mode activated
        """

        conversion, drawing_thickness = self.conversion_thickness(self.video_width, self.frame_width)

        
        if self.add_area:
            if self.add_area["type"] == "circle":
                if "center" not in self.add_area:
                    self.add_area["center"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["radius"] = int((( event.pos().x() * conversion - self.add_area["center"][0] )**2 + ( event.pos().y() * conversion - self.add_area["center"][1] )**2)**0.5)
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}

        if self.add_area:
            if self.add_area["type"] == "rectangle":
                if "pt1" not in self.add_area:
                    self.add_area["pt1"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                else:
                    self.add_area["pt2"] = [int(event.pos().x() * conversion), int(event.pos().y() * conversion)]
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}

        if self.add_area:

            if self.add_area["type"] == "polygon":
                
                if event.button() == 2: # right click to finish
                    self.lw_area_definition.addItem(str(self.add_area))
                    self.activate_areas()
                    self.add_area = {}

                if "points" not in self.add_area:
                    self.add_area["points"] = [[int(event.pos().x() * conversion), int(event.pos().y() * conversion)]]
                else:
                    self.add_area["points"].append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                    cv2.line(self.frame, tuple(self.add_area["points"][-2]), tuple(self.add_area["points"][-1]), color=AREA_COLOR, lineType=8, thickness=drawing_thickness)

                    self.show_frame(self.frame)



        if self.flag_define_arena:
 
            if self.flag_define_arena == "polygon":
                
                if event.button() == 2: # right click to finish

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.pb_define_arena.setText("Define arena")

                    self.arena = {'type': 'polygon', 'points': self.arena, 'name': 'arena'}
                    self.le_arena.setText("{}".format(self.arena))

                else:
                    
                    self.arena.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])

                    if len(self.arena) >= 2:
                        cv2.line(self.frame, tuple(self.arena[-2]), tuple(self.arena[-1]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                        self.show_frame(self.frame)


            if self.flag_define_arena == "circle":
                
                self.arena.append([int(event.pos().x() * conversion), int(event.pos().y() * conversion)])
                
                if len(self.arena) == 3:
                    cx, cy, r = find_circle(self.arena)
                    cv2.circle(self.frame, (int(abs(cx)), int(abs(cy))), int(r), color=ARENA_COLOR, thickness=drawing_thickness)
        
                    self.show_frame(self.frame)

                    self.flag_define_arena = ""
                    self.pb_define_arena.setEnabled(False)
                    self.pb_clear_arena.setEnabled(True)

                    self.arena = {'type': 'circle', 'center': [round(cx), round(cy)], 'radius': round(r), 'name': 'arena'}
                    self.le_arena.setText("{}".format(self.arena))



    def frame_width(self):
        """
        let user choose a width for frame image
        """
        i, ok_pressed = QInputDialog.getInt(self, "Set frame width", "Frame width", self.frame_width, 20, 2000, 1)
        if ok_pressed:
            self.frame_width = i
        
        self.treat_and_show()
        

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
            print("no results to be saved")
        


    def plot_xy_density(self):

        if self.positions:
            for n_object in range(len(self.positions[0])):
                x, y = [], []
                for row in self.positions:
                    x.append(row[n_object][0])
                    y.append(row[n_object][1])
                plot_density(x, y, x_lim=(0, self.video_width), y_lim=(0, self.video_height))
        else:
            print("no positions to be plotted")

    def plot_path(self):

        if self.positions:
            
            for n_object in range(len(self.positions[0])):
                verts = []
                for row in self.positions:
                    verts.append(row[n_object])
                plot_path(verts, x_lim=(0, self.video_width), y_lim=(0, self.video_height), color=COLORS_LIST[n_object % len(COLORS_LIST)])
        else:
            print("no positions to be plotted")


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
            
            self.statusBar().showMessage("video dim {}x{}".format(self.video_width, self.video_height))


    def update_frame(self):
        """
        update frame number
        """

        self.frame_idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        self.leFrame.setText(str(self.frame_idx))


    def treatment(self, frame):
        """
        apply treament to frame
        returns treated frame
        """
        if self.frame is None:
            return
        
        if self.fgbg:
            frame = self.fgbg.apply(frame)

        # threshold
        if self.sb_threshold.value():
        # color to gray levels
            if not self.fgbg:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, frame = cv2.threshold(frame, self.sb_threshold.value(), 255, cv2.THRESH_BINARY)

        # invert
        if self.cb_invert.isChecked():
            frame = (255 - frame)

        # blur
        if self.sb_blur.value():
            frame = cv2.blur(frame, (self.sb_blur.value(), self.sb_blur.value()))

        return frame


    def update_objects(self):
        """
        returns filtered objects
        """

        all_objects, filtered_objects = extract_objects(frame=self.treatedFrame,
                                                        threshold=0,
                                                        min_size=self.sbMin.value(),
                                                        max_size=self.sbMax.value(),
                                                        largest_number=self.sb_largest_number.value(),
                                                        arena=self.arena,
                                                        max_extension=self.sb_max_extension.value()
                                                        )

        if self.mem_filtered_objects:

            if len(filtered_objects) == len(self.mem_filtered_objects):

                new_filtered = dict([[x, {}] for x in filtered_objects])

                for new_idx in filtered_objects:
                    r_dist = []

                    for mem_idx in self.mem_filtered_objects:
                        r_dist.append([dist(filtered_objects[new_idx]["centroid"], self.mem_filtered_objects[mem_idx]["centroid"]), new_idx, mem_idx])
                        print("dist",new_idx ,mem_idx, dist(filtered_objects[new_idx]["centroid"], self.mem_filtered_objects[mem_idx]["centroid"]) ) 

                    r_dist = sorted(r_dist)

                    print(r_dist)
                    print(filtered_objects[ r_dist[0][2] ]["size"])
                    print(r_dist[0][2])

                    new_filtered[r_dist[0][2]] = copy.deepcopy(filtered_objects[ new_idx ])
                    del self.mem_filtered_objects[r_dist[0][2]]

            
                filtered_objects = copy.deepcopy(new_filtered)

        print("===============")


        if self.cb_display_analysis.isChecked():
            self.lb_all.setText("All (number of objects: {})".format(len(all_objects)))
            out = ""
            for idx in sorted(all_objects.keys()):
                out += "Object #{}: {} pixels\n".format(idx, all_objects[idx]["area"])
            self.te_all_objects.setText(out)
    
            self.lb_filtered.setText("Filtered (number of objects: {})".format(len(filtered_objects)))
            if self.sb_largest_number.value() and len(filtered_objects) < self.sb_largest_number.value():
                self.lb_filtered.setStyleSheet('color: red')
            else:
                self.lb_filtered.setStyleSheet('')
                
            out = ""
            for idx in filtered_objects:
                out += "Object #{}: {} pixels\n".format(idx, filtered_objects[idx]["size"])
            self.te_objects.setText(out)
        
        self.mem_filtered_objects = copy.deepcopy(filtered_objects)
        return filtered_objects


    def draw_marker_on_objects(self, frame, objects, marker_type=MARKER_TYPE):
        """
        draw marker (rectangle or contour) around objects
        marker color from index of object in COLORS_LIST 
        """
        _, drawing_thickness = self.conversion_thickness(self.video_width, self.frame_width)
        for idx in objects:
            marker_color = COLORS_LIST[(idx - 1) % len(COLORS_LIST)]
            if marker_type == "rectangle":
                cv2.rectangle(frame, objects[idx]["min"], objects[idx]["max"] , marker_color, drawing_thickness)
            if marker_type == "contour":
                cv2.drawContours(frame, [objects[idx]["contour"]], 0, marker_color, drawing_thickness)

            cv2.putText(frame, str(idx), objects[idx]["max"], font, 0.5, marker_color, 1, cv2.LINE_AA)

        return frame


    def show_frame(self, frame):
        """
        show the current frame in label pixmap
        """
        # frame from video
        pm = frame2pixmap(frame)
        pm_resized = pm.scaled(self.frame_width, self.frame_width, QtCore.Qt.KeepAspectRatio)
        self.label1.setPixmap(pm_resized)


    def show_treated_frame(self, frame):
        """
        show treated frame
        """
        pm = QPixmap.fromImage(toQImage(frame))
        pm_resized = pm.scaled(self.frame_width, self.frame_width, QtCore.Qt.KeepAspectRatio)
        self.label2.setPixmap(pm_resized)


    def treat_and_show(self):
        """
        apply treament to frame and show results
        """

        if self.frame is None:
            return

        self.treatedFrame = self.treatment(self.frame)

        modified_frame = self.frame.copy()

        _ = self.draw_marker_on_objects(modified_frame, self.update_objects(), marker_type="contour")

        # frame from video
        self.show_frame(modified_frame)

        # treated frame
        self.show_treated_frame(self.treatedFrame)
        
        '''
        pm = frame2pixmap(self.treatedFrame)
        pm_resized = pm.scaled(self.frame_width, self.frame_width, QtCore.Qt.KeepAspectRatio)
        self.label2.setPixmap(pm_resized)
        '''


    def closeEvent(self, event):
        try:
            self.capture.release()
            cv2.destroyAllWindows()
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
        
        if self.capture is not None:
            ret, self.frame = self.capture.read()
            if not ret:
                return False, {}
        else:
            return False, {}
    
        self.update_frame()

        self.treatedFrame = self.treatment(self.frame)

        filtered_objects = self.update_objects()

        if self.cb_display_analysis.isChecked():
            modified_frame = self.frame.copy()
            _ = self.draw_marker_on_objects(modified_frame, filtered_objects, marker_type=MARKER_TYPE)

            conversion, drawing_thickness = self.conversion_thickness(self.video_width, self.frame_width)

            # draw areas
            for area in self.areas:
                if "type" in self.areas[area]:
                    if self.areas[area]["type"] == "circle":
                        cv2.circle(modified_frame, tuple(self.areas[area]["center"]), self.areas[area]["radius"], color=AREA_COLOR, thickness=drawing_thickness)
                        cv2.putText(modified_frame, self.areas[area]["name"], tuple(self.areas[area]["center"]), font, 0.5, AREA_COLOR, 1, cv2.LINE_AA)

                    if self.areas[area]["type"] == "rectangle":
                        cv2.rectangle(modified_frame, tuple(self.areas[area]["pt1"]), tuple(self.areas[area]["pt2"]), color=AREA_COLOR, thickness=drawing_thickness)
                        cv2.putText(modified_frame, self.areas[area]["name"], tuple(self.areas[area]["pt1"]), font, 0.5, AREA_COLOR, 1, cv2.LINE_AA)

                    if self.areas[area]["type"] == "polygon":
                        for idx, point in enumerate(self.areas[area]["points"][:-1]):
                            cv2.line(modified_frame, tuple(point), tuple(self.areas[area]["points"][idx + 1]), color=AREA_COLOR, lineType=8, thickness=drawing_thickness)
                        cv2.line(modified_frame, tuple(self.areas[area]["points"][-1]), tuple(self.areas[area]["points"][0]), color=RED, lineType=8, thickness=drawing_thickness)                        
                        cv2.putText(modified_frame, self.areas[area]["name"], tuple(self.areas[area]["points"][0]), font, 0.5, AREA_COLOR, 1, cv2.LINE_AA)

            
            # draw arena
            if self.arena:

                if self.arena["type"] == "polygon":
                    for idx, point in enumerate(self.arena["points"][:-1]):
                        cv2.line(modified_frame, tuple(point), tuple(self.arena["points"][idx + 1]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                    cv2.line(modified_frame, tuple(self.arena["points"][-1]), tuple(self.arena["points"][0]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)                        
                    #cv2.putText(modified_frame, self.arena["name"], tuple(self.arena["points"][0]), font, 0.5, ARENA_COLOR, 1, cv2.LINE_AA)

                if self.arena["type"] == "circle":
                    cv2.circle(modified_frame, tuple(self.arena["center"]), self.arena["radius"], color=ARENA_COLOR, thickness=drawing_thickness)
                    #cv2.putText(modified_frame, self.arena["name"], tuple(self.arena["center"]), font, 0.5, ARENA_COLOR, 1, cv2.LINE_AA)


                '''
                for idx, point in enumerate(self.arena[:-1]):
                    cv2.line(modified_frame, tuple(point), tuple(self.arena[idx + 1]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                cv2.line(modified_frame, tuple(self.arena[-1]), tuple(self.arena[0]), color=ARENA_COLOR, lineType=8, thickness=drawing_thickness)
                '''


        # frame from video
        if self.cb_display_analysis.isChecked():
            self.show_frame(modified_frame)
            self.show_treated_frame(self.treatedFrame)
        
        if self.cb_display_analysis.isChecked():
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
                        print(idx, idx_obj, dist(results["objects"][idx]["centroid"], obj))
                        r_dist.append([dist(results["objects"][idx]["centroid"], obj), idx_obj, idx])
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
    parser.add_argument("--areas", action="store", dest="areas_file", help="path of file containing the areas definition")
    parser.add_argument("--arena", action="store", dest="arena_file", help="path of file containing the arena definition")
    parser.add_argument("--threshold", action="store", dest="threshold", help="Threshold value")
    parser.add_argument("--invert", action="store_true", default=50, dest="invert", help="Invert B/W")

    results = parser.parse_args()
    if results.version:
        print("version {} release date: {}".format(__version__, __version_date__))
        sys.exit()

    app = QApplication(sys.argv)
    w = Ui_MainWindow()

    if results.threshold:
        w.sb_threshold.setValue(int(results.threshold))


    if results.invert:
        w.cb_invert.setChecked(True)

    if results.video_file:
        if os.path.isfile(results.video_file):
            w.open_video(results.video_file)
        else:
            print("{} not found".format(results.video_file))
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
