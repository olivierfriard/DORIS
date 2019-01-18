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


"""

import config
import cv2
import numpy as np
import struct
import itertools
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

'''
try:
    import mpl_scatter_density
except ModuleNotFoundError:
    print("mpl_scatter_density not found")
'''


def rgbstr_to_bgr_tuple(rgb_str):
    return struct.unpack('BBB', bytes.fromhex(rgb_str))[::-1]


COLORS_LIST = [rgbstr_to_bgr_tuple(x) for x in config.RGBSTR_COLORS_LIST]


def plot_path(df, x_lim, y_lim):
    """
    plot path
    """

    plt.figure()
    axes = plt.gca()
    axes.set_aspect("equal", adjustable="box")

    for idx in range(1, int((len(df.columns) - 1)/2) + 1):
        plot_color = "#" + config.RGBSTR_COLORS_LIST[(idx - 1) % len(config.RGBSTR_COLORS_LIST)]
        plt.plot(f"x{idx}", f"y{idx}", plot_color, data=df, alpha=1)

    plt.xlabel("x")
    plt.ylabel("y")

    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    axes.set_ylim(axes.get_ylim()[::-1])

    plt.tight_layout()
    plt.show()




def plot_positions(df, x_lim, y_lim,):
    """
    plot positions

    Args:
        df (pandas.dataframe):
        x_lim (tuple of int): plot x limits
        y_lim (tuple of int): plot y limits
    """

    #plt.figure(figsize=(5,5))

    plt.figure()
    axes = plt.gca()
    axes.set_aspect("equal", adjustable="box")

    for idx in range(1, int((len(df.columns) - 1)/2) +1):
        plot_color = "#" + config.RGBSTR_COLORS_LIST[(idx - 1) % len(config.RGBSTR_COLORS_LIST)]
        plt.scatter(df[f"x{idx}"], df[f"y{idx}"], c=plot_color, alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')

    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    axes.set_ylim(axes.get_ylim()[::-1])

    plt.tight_layout()
    plt.show()



def plot_density_old(x, y, x_lim=(0, 0), y_lim=(0,0)):

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


def plot_density(df, x_lim, y_lim):

    df = df.dropna(thresh=1)

    for idx in range(1, int((len(df.columns) - 1)/2) +1):
        plt.figure()
        axes = plt.gca()
        axes.set_aspect("equal", adjustable="box")

        plt.hist2d(df[f"x{idx}"], df[f"y{idx}"], bins=20, cmap=plt.cm.Reds)

        plt.xlabel("x")
        plt.ylabel("y")

        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)
        axes.set_ylim(axes.get_ylim()[::-1])

        plt.tight_layout()
        plt.show()



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

    logging.debug("apply kmeans")
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


def euclidean_distance(p1: tuple, p2: tuple) -> float:
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
                              arena={},
                              max_extension=50,
                              tolerance_outside_arena=0.05,
                              previous_objects={}):
    """
    returns all detected objects and filtered objects

    Args:
        frame: np.array
        min_size (int): minimum size of objects to detect
        max_size (int): maximum size of objects to detect
        number_of_objects (int): number of objects to return
        arena (list): list of arena
        max_extension (int): maximum extension of object to select
        previous_objects (dict): object(s) detected in previous frame

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

    # record objects that not match conditions (for deleting)
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
                dist_ = ((n[:,0] - arena["center"][0]) ** 2 + (n[:,1] - arena["center"][1]) ** 2) ** 0.5

                '''
                print("distance")
                print("contour size:", len(dist_))
                print("% >", np.count_nonzero(dist_ > arena["radius"]) / len(dist_))
                '''
                '''
                print("% out", idx, np.count_nonzero(dist_ > arena["radius"]) / len(dist_) )
                '''
                if np.count_nonzero(dist_ > arena["radius"]) / len(dist_) > tolerance_outside_arena:
                    obj_to_del_idx.append(idx)
                    continue

    filtered_objects = {}
    for obj_idx in all_objects:
        if obj_idx not in obj_to_del_idx:
            filtered_objects[obj_idx] = dict(all_objects[obj_idx])



    # remove objects to delete
    '''
    for obj_idx in obj_to_del_idx:
        all_objects.pop(obj_idx, None)
    '''

    # reorder filtered objects starting from #1
    filtered_objects = dict([(i + 1, filtered_objects[idx]) for i, idx in enumerate(filtered_objects.keys())])


    return all_objects, filtered_objects

    '''
    # check distances from previous detected objects
    if previous_objects:
        print("previous objects")
        mem_costs = {}
        obj_indexes = list(all_objects.keys())
        for indexes in itertools.combinations(obj_indexes, number_of_objects):
            cost = cost_sum_assignment(previous_objects, dict([(idx, all_objects[idx]) for idx in indexes]))
            print(indexes, cost)
            mem_costs[cost] = indexes

        min_cost = min(list(mem_costs.keys()))
        print("min cost", min_cost)

        filtered_objects = dict([(i + 1, all_objects[idx]) for i, idx in enumerate(mem_costs[min_cost])])


    else:

        # select 'number_of_objects' objects
        print("select objects")
        sorted_areas = sorted([all_objects[idx]["area"] for idx in all_objects if idx not in obj_to_del_idx], reverse=True)
        filtered_objects = {}
        new_idx = 0
        for idx in all_objects:
            if (sorted_areas.index(all_objects[idx]["area"]) < number_of_objects):
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
    '''


def cost_sum_assignment(mem_objects: dict, objects: dict) -> int:
    """
    reorder objects to assign object index to closer previous one
    """

    if len(objects) == len(mem_objects):
        mem_positions = [mem_objects[k]["centroid"] for k in mem_objects]
        positions = [objects[k]["centroid"] for k in objects]

        p1 = np.array(mem_positions)
        p2 = np.array(positions)

        distances = cdist(p1, p2)

        row_ind, col_ind = linear_sum_assignment(distances)

        return int(round(distances[row_ind, col_ind].sum()))

    else:
        print("len !=")

        return None


def reorder_objects(mem_objects: dict, objects: dict) -> dict:
    """
    reorder objects to assign object index to closer previous one
    """

    if len(objects) == len(mem_objects):
        mem_positions = [mem_objects[k]["centroid"] for k in mem_objects]
        positions = [objects[k]["centroid"] for k in objects]

        p1 = np.array(mem_positions)
        p2 = np.array(positions)
        # print(p1, p2)

        distances = cdist(p1, p2)

        row_ind, col_ind = linear_sum_assignment(distances)

        if not np.array_equal(col_ind, list(range(len(col_ind)))):
            reordered_object = dict([(idx + 1, objects[k + 1]) for idx, k in enumerate([x for x in col_ind])])
            return reordered_object
        else:
            return objects

    else:
        print("len !=")

        return objects
