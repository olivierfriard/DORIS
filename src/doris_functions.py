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


"""

import cv2
import numpy as np
import struct
#np.set_printoptions(threshold="nan")
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

try:
    import mpl_scatter_density
except ModuleNotFoundError:
    print("mpl_scatter_density not found")

'''
COLORS_LIST = ["#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",

        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58"]
'''

RGBSTR_COLORS_LIST = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF",
        "800000", "008000", "000080", "808000", "800080", "008080", "808080",
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
        "400000", "004000", "000040", "404000", "400040", "004040", "404040",
        "200000", "002000", "000020", "202000", "200020", "002020", "202020",
        "600000", "006000", "000060", "606000", "600060", "006060", "606060",
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]


def rgbstr_to_tuple(rgb_str):
    return struct.unpack('BBB', bytes.fromhex(rgb_str))

COLORS_LIST = [rgbstr_to_tuple(x) for x in RGBSTR_COLORS_LIST]


def plot_path(verts, x_lim, y_lim, color):


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


def plot_density(x, y, x_lim=(0, 0), y_lim=(0,0)):

    try:
        x = np.array(x)
        y = np.array(y)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        ax.scatter_density(x, y)
        if x_lim != (0, 0):
            ax.set_xlim(x_lim)
        if y_lim != (0, 0):
            ax.set_ylim(y_lim[::-1])

        plt.show()
        return True

    except:
        return False


def apply_k_means(contours, n_inds):
    """
    This function applies the k-means clustering algorithm to separate merged
    contours. The algorithm is applied when detected contours are fewer than
    expected objects(number of animals) in the scene.

    Parameters
    ----------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    n_inds: int
        total number of individuals being tracked
    meas_now: array_like, dtype=float
        individual's location on current frame

    Returns
    -------
    contours: list
        a list of all detected contours that pass the area based threhold criterion
    meas_now: array_like, dtype=float
        individual's location on current frame
    """
    #del meas_now[:]
    print("kmeans")
    centroids = []
    print("contours[0]", type(contours[0]))
    # Clustering contours to separate individuals
    myarray = np.vstack(contours)
    myarray = myarray.reshape(myarray.shape[0], myarray.shape[2])

    kmeans = KMeans(n_clusters=n_inds, random_state=0, n_init = 50).fit(myarray)
    l = len(kmeans.cluster_centers_)

    new_contours = []
    for i in range(n_inds):
        new_contours.append(myarray[kmeans.labels_==i])

    '''
    for i in range(l):
        x = int(tuple(kmeans.cluster_centers_[i])[0])
        y = int(tuple(kmeans.cluster_centers_[i])[1])
        centroids.append([x,y])
    '''

    return new_contours #, centroids


def image_processing(frame,
                    blur=5,
                    threshold_method={},
                    invert=False):
    """
    apply treament to frame
    returns treated frame
    """

    if frame is None:
        return None

    '''
    if self.fgbg:
        frame = self.fgbg.apply(frame)
    '''

    # blur
    if blur:
        frame = cv2.blur(frame, (blur, blur))

    # threshold
    # color to gray levels
    '''
    if not self.fgbg:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    '''

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if threshold_method["name"] == "Simple":
        ret, frame = cv2.threshold(frame, threshold_method["cut-off"], 255, cv2.THRESH_BINARY)

    if threshold_method["name"] in ["Adaptive (mean)", "Adaptive (Gaussian)"]:
        tm = cv2.ADAPTIVE_THRESH_MEAN_C if threshold_method["name"] == "Adaptive (mean)" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        frame = cv2.adaptiveThreshold(frame,
                                      255,
                                      tm,
                                      cv2.THRESH_BINARY,
                                      threshold_method["block_size"] if threshold_method["block_size"] % 2 else threshold_method["block_size"] + 1,
                                      threshold_method["offset"])

    if invert:
        frame = (255 - frame)

    return frame


def euclidean_distance(p1, p2):
    """
    euclidean distance between two points
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def find_circle(points):
    """
    Find circle that pass by 3 points

    Args:
        points (list): list of points

    Returns:
        float: x of circle center
        float: y of circle center
        float: radius of circle
    """

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    ma = (y2 - y1) / (x2 - x1)
    mb = (y3 - y2) / (x3 - x2)

    x = (ma * mb * (y1 - y3) + mb * (x1 + x2) - ma * (x2 + x3)) / (2 * (mb - ma))

    y = - (1 / ma) * (x - (x1 + x2) / 2) + (y1 + y2) / 2

    return x, y, euclidean_distance((x, y), (x1, y1))


def detect_and_filter_objects(frame,
                              min_size=0,
                              max_size=0,
                              largest_number=0,
                              arena=[],
                              max_extension=50,
                              tolerance_outside_arena=0.05):
    """
    returns all detected objects and filtered objects

    Args:
        frame: np.array
        min_size (int): minimum size of objects to detect
        max_size (int): maximum size of objects to detect
        largest_number (int): number of largest objects to select
        arena (list): list of arena
        max_extension (int): maximum extension of object to select

    Returns:
        dict: all detected objects
        dict: filtered objects
    """

    _, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    nr_objects = len(contours)

    all_objects = {}
    for idx, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        n = np.vstack(cnt).squeeze()
        try:
            x, y = n[:,0], n[:,1]
        except:
            x = n[0]
            y = n[1]

        all_objects[idx] = {"centroid": (cx, cy),
                            "contour": cnt,
                            "area": cv2.contourArea(cnt),
                            "min": (int(np.min(x)), int(np.min(y))),
                            "max": (int(np.max(x)), int(np.max(y)))
                            }

    # record objects that not match conditions
    obj_to_del_idx = []
    for idx in all_objects:

        # check if object area is >= of minimal size
        if min_size and all_objects[idx]["area"] < min_size:
            # print("skipped #{} for min area".format(idx))
            obj_to_del_idx.append(idx)
            continue

        # check if object area is <= of maximal size
        if max_size and all_objects[idx]["area"] > max_size:
            # print("skipped #{} for max area".format(idx))
            obj_to_del_idx.append(idx)
            continue

        # check if object extension <= max extension
        if max_extension:
            '''
            n = np.vstack(all_objects[idx]["contour"]).squeeze()
            if (max(n[:,0]) - min(n[:,0]) > max_extension) or (max(n[:,1]) - min(n[:,1]) > max_extension):
            '''
            if ((all_objects[idx]["max"][0] - all_objects[idx]["min"][0]) > max_extension
                or (all_objects[idx]["max"][1] - all_objects[idx]["min"][1]) > max_extension):
                # print("skipped #{} for max extension".format(idx))
                obj_to_del_idx.append(idx)
                continue

        # check if objects in arena
        if arena:

            if arena["type"] == "rectangle":
                np_arena = np.array(arena["points"])
                if not (np_arena[0][0] <= all_objects[idx]["centroid"][0] <= np_arena[1][0]
                    and np_arena[0][1] <= all_objects[idx]["centroid"][1] <= np_arena[1][1]):
                    obj_to_del_idx.append(idx)
                    continue

            # check if all contour points are in polygon arena (with TOLERANCE_OUTSIDE_ARENA tolerance)
            if arena["type"] == "polygon":

                np_arena = np.array(arena["points"])
                if cv2.pointPolygonTest(np_arena, all_objects[idx]["centroid"], False) < 0:
                    obj_to_del_idx.append(idx)
                    continue

                n = np.vstack(all_objects[idx]["contour"]).squeeze()
                nl = len(n)
                count_out = 0
                for pt in n:
                    if cv2.pointPolygonTest(np_arena, tuple(pt), False) < 0:
                        count_out += 1
                        if count_out / nl > tolerance_outside_arena:
                            break
                if count_out / nl > tolerance_outside_arena:
                    obj_to_del_idx.append(idx)
                    continue

            # check if all contour points are in circle arena (with TOLERANCE_OUTSIDE_ARENA tolerance)
            if arena["type"] == "circle":
                if euclidean_distance(all_objects[idx]["centroid"], arena["center"]) > arena["radius"]:
                    obj_to_del_idx.append(idx)
                    continue

                n = np.vstack(all_objects[idx]["contour"]).squeeze()
                dist_ = ((n[:,0] - arena["center"][0])**2 + (n[:,1] - arena["center"][1])**2)**0.5

                '''
                print("distance")
                print("contour size:", len(dist_))
                print("% >", np.count_nonzero(dist_ > arena["radius"]) / len(dist_))
                '''
                if np.count_nonzero(dist_ > arena["radius"]) / len(dist_) > tolerance_outside_arena:
                    obj_to_del_idx.append(idx)
                    continue


    # sizes
    sorted_areas = sorted([all_objects[idx]["area"] for idx in all_objects if idx not in obj_to_del_idx], reverse=True)

    filtered_objects = {}
    new_idx = 0
    for idx in all_objects:
        if idx in obj_to_del_idx:
            continue
        if (sorted_areas.index(all_objects[idx]["area"]) < largest_number):
            new_idx += 1
            # min/max
            n = np.vstack(all_objects[idx]["contour"]).squeeze()
            x, y = n[:,0], n[:,1]

            filtered_objects[new_idx] = {"centroid": all_objects[idx]["centroid"],
                                         "contour": all_objects[idx]["contour"],
                                         "area": all_objects[idx]["area"],
                                         "min": all_objects[idx]["min"],
                                         "max": all_objects[idx]["max"]}

    return all_objects, filtered_objects


def reorder_objects(mem_objects: dict, objects: dict) -> dict:

    if len(objects) == len(mem_objects):

        mem_positions = [mem_objects[k]["centroid"] for k in mem_objects]
        positions = [objects[k]["centroid"] for k in objects]

        p1 = np.array(mem_positions)
        p2 = np.array(positions)
        print(p1)
        print(p2)

        distances = cdist(p1, p2)

        row_ind, col_ind = linear_sum_assignment(distances)

        if not np.array_equal(col_ind, list(range(len(col_ind)))):


            reordered_object = dict([(idx + 1, objects[k + 1]) for idx, k in enumerate([x for x in col_ind])])
            return reordered_object
            '''
            current_ids = col_ind.copy()
            reordered = [i[0] for i in sorted(enumerate(current_ids), key=lambda x:x[1])]
            p2 = [list(x) for (y,x) in sorted(zip(reordered, p2))]
            '''
        else:
            return objects

    else:
        print("len !=")

        return objects



































