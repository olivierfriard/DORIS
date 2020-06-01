#!/usr/bin/env python3
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


"""

import itertools
import logging
import os.path
import struct
import sys

import cv2
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.stats import kde
from sklearn.cluster import KMeans
import pandas as pd

from doris import config


def rgbstr_to_bgr_tuple(rgb_str):
    return struct.unpack('BBB', bytes.fromhex(rgb_str))[::-1]


COLORS_LIST = [rgbstr_to_bgr_tuple(x) for x in config.RGBSTR_COLORS_LIST]

def plot_path(df, x_lim, y_lim):
    """
    plot path
    """

    print("df", df)

    plt.figure()
    axes = plt.gca()
    axes.set_aspect("equal", adjustable="box")

    for idx in range(1, int((len(df.columns) - 1)/2) + 1):
        plot_color = "#" + config.RGBSTR_COLORS_LIST[(idx - 1) % len(config.RGBSTR_COLORS_LIST)]
        
        plt.plot(df[f"x{idx}"].to_numpy(float, na_value=np.nan),
                 df[f"y{idx}"].to_numpy(float, na_value=np.nan),
                 color=plot_color,
                 alpha=1)
        

        '''
        plt.plot(df[f"x{idx}"],
                 df[f"y{idx}"],
                 color=plot_color,
                 alpha=1)
        '''


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

    print("df", df)

    plt.figure()
    axes = plt.gca()
    axes.set_aspect("equal", adjustable="box")

    for idx in range(1, int((len(df.columns) - 1)/2) +1):
        plot_color = "#" + config.RGBSTR_COLORS_LIST[(idx - 1) % len(config.RGBSTR_COLORS_LIST)]
        plt.scatter(df[f"x{idx}"].to_numpy(float, na_value=np.nan),
                    df[f"y{idx}"].to_numpy(float, na_value=np.nan),
                    c=plot_color,
                    alpha=0.5)

    plt.xlabel('x')
    plt.ylabel('y')

    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    axes.set_ylim(axes.get_ylim()[::-1])

    plt.tight_layout()
    plt.show()


def plot_density(df, x_lim, y_lim):

    nbins=100
    i = 1
    while True:
    
        if f"x{i}" not in df:
            return
        
        df[f"x{i}"] = pd.to_numeric(df[f"x{i}"], errors='coerce')
        df[f"y{i}"] = pd.to_numeric(df[f"y{i}"], errors='coerce')

        k = kde.gaussian_kde((df[f"x{i}"].dropna(), df[f"y{i}"].dropna()))

        xi, yi = np.mgrid[df[f"x{i}"].dropna().min():df[f"x{i}"].dropna().max():nbins*1j,
                          df[f"y{i}"].dropna().min():df[f"y{i}"].dropna().max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        plt.figure()
        axes = plt.gca()
        axes.set_aspect("equal", adjustable="box")

        plt.xlabel("x")
        plt.ylabel("y")

        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)
        axes.set_ylim(axes.get_ylim()[::-1])

        plt.title(f"Density plot for object #{i}")
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape))  # , cmap=plt.cm.BuGn_r
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        i += 1



def group_of(points, centroids):

    '''
    print(points)
    print(centroids)
    '''

    distances = np.zeros([len(points), len(centroids)])

    for idx1, p in enumerate(points):
        for idx2, c in enumerate(centroids):
            distances[idx1, idx2] = ((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) ** .5

    # distances2 = [((points[:,0] - centroid[0]) **2 + (points[:,1] - centroid[1]) **2)**.5 for centroid in centroids]

    '''
    print("distances")
    print(distances)
    '''
    # print("distances2")
    # print(distances2)

    # mini = np.minimum(distances[:,0], distances[:,1])
    mini = np.min(distances, axis=1)

    # print(mini)

    new_ctr = []
    for idx, _ in enumerate(centroids):
        new_ctr.append([])

    for idx1 in range(len(points)):
        for idx2 in range(len(centroids)):
            if distances[idx1, idx2] == mini[idx1]:
                new_ctr[idx2].append(tuple(points[idx1, :]))

    return [np.array(x) for x in new_ctr]



def group(points, centroids):
    """
    group points by distance to centroids
    NOT WORKING
    """

    def f_clu(cc, ctrOK):
        # assegna ciascun punto al centroide piu vicino
        nclu = np.shape(ctrOK)[0]
        nobj = np.shape(cc)[0]
        clu = np.zeros(nobj)
        for i in np.arange(nobj):
            zz = np.sum((np.dot(np.ones((nclu, 1)), np.reshape(cc[i, :], (1, 2))) - ctrOK) ** 2, axis=1)
            clu[i] = (zz == np.min(zz)).nonzero()[0][0]
        return clu

    clu = f_clu(points, centroids)
    '''
    print(f"clu: {clu}")
    '''
    clusters = []
    for idx, centroid in enumerate(centroids):
        clusters.append(points[(clu == idx).nonzero()[0], :])



    # distances = [((points[:,0] - centroid[0]) **2 + (points[:,1] - centroid[1]) **2)**.5 for centroid in centroids]

    # results = [points[d==np.minimum(*distances)] for d in distances]

    # return results

    return clusters



def group_sc(points, centroids_list0, centroids_list1):
    """
    group points by distance to centroids
    author: Sergio Castellano
    """

    def f_obj(ctr0, nobj0, ctr1, nobj1):
        # trova i centroidi dei "nobj1" oggetti

        dd = np.zeros((nobj0, nobj1))
        riga = []
        colonna = []
        ctrOK = np.zeros((nobj1, 2))
        for i in np.arange(nobj0):
            for ii in np.arange(nobj1):
                dd[i, ii] = distance.euclidean(ctr0[i, :], ctr1[ii, :])

        dv = np.sort(np.reshape(dd, nobj0 * nobj1, 0))
        for i in np.arange(nobj1):
            r0, c0 = (dd == dv[i]).nonzero()

            riga = np.concatenate((riga, r0))
            colonna = np.concatenate((colonna, c0))
        conta = 0
        for i in np.arange(nobj0):
            x = (riga == i).nonzero()[0]
            '''
            print("x", x)
            '''
            if len(x) == 1:
                '''
                print("conta", conta)
                '''
                ctrOK[conta, 0] = ctr0[int(riga[x]), 0]
                ctrOK[conta, 1] = ctr0[int(riga[x]), 1]
                conta += 1
            else:
                for j in np.arange(len(x)):
                    '''
                    print("conta", conta)
                    '''
                    ctrOK[conta, 0] = (ctr0[int(riga[x[j]]), 0] + ctr1[int(colonna[x[j]]), 0]) / 2
                    ctrOK[conta, 1] = (ctr0[int(riga[x[j]]), 1] + ctr1[int(colonna[x[j]]), 1]) / 2
                    conta += 1

        return ctrOK


    def f_clu(cc, ctrOK):
        # assegna ciascun punto al centroide piu vicino
        nclu = np.shape(ctrOK)[0]
        nobj = np.shape(cc)[0]
        clu = np.zeros(nobj)
        for i in np.arange(nobj):
            zz = np.sum((np.dot(np.ones((nclu, 1)), np.reshape(cc[i,:], (1, 2))) - ctrOK) ** 2,axis = 1)
            clu[i] = (zz == np.min(zz)).nonzero()[0][0]
        return clu

    '''
    print(f"centroids_list0 {centroids_list0}")
    print(f"centroids_list1 {centroids_list1}")
    '''

    '''
    problem with:
    centroids_list0 [(508, 556), (555, 552), (511, 503), (555, 491)]
    centroids_list1 [(555, 552), (511, 515), (555, 491)]
    '''

    ctrOK = f_obj(np.array(centroids_list1), len(centroids_list1), np.array(centroids_list0), len(centroids_list0))
    '''
    print("ctrOK", ctrOK)
    '''

    clu = f_clu(points, ctrOK)

    '''
    print(f"clu: {clu}")
    '''

    clusters = []
    for idx, _ in enumerate(ctrOK):
        clusters.append(points[(clu == idx).nonzero()[0],:]  )

    '''
    distances = [((points[:,0] - centroid[0]) **2 + (points[:,1] - centroid[1]) **2)**.5 for centroid in centroids]

    results = [points[d==np.minimum(*distances)] for d in distances]

    return results
    '''
    return clusters



def apply_k_means(contours, n_inds):
    """
    This function applies the k-means clustering algorithm to separate merged
    contours. The algorithm is applied when detected contours are fewer than
    expected objects in the scene.

    see https://stackoverflow.com/questions/38355153/initial-centroids-for-scikit-learn-kmeans-clustering

    Args:
        contours (list): list of contours
        n_inds (int): total number of individuals being tracked

    Returns:
        list: contours
    """

    logging.debug("function: apply_k_means")
    try:
        centroids = []

        logging.debug(f"len contours: {len(contours)}")
        # Clustering contours to separate individuals
        myarray = np.vstack(contours)
        myarray = myarray.reshape(myarray.shape[0], myarray.shape[2])

        kmeans = KMeans(n_clusters=n_inds, random_state=0, n_init = 50).fit(myarray)
        '''l = len(kmeans.cluster_centers_)'''

        new_contours = [myarray[kmeans.labels_==i] for i in range(n_inds)]

        '''
        for i in range(n_inds):
            new_contours.append(myarray[kmeans.labels_==i])
        '''

        return new_contours

    except Exception:
        error_type, error_file_name, error_lineno = error_info(sys.exc_info())
        logging.error(f"error in apply_k_means function. Error: {error_type} in {error_file_name} {error_lineno}")
        return []


def image_processing(frame,
                     blur=5,
                     threshold_method={},
                     invert=False,
                     arena={},
                     masks=[]):
    """
    apply treament to frame
    returns treated frame
    """

    if frame is None:
        return None

    # apply masks (black for selected mask areas)
    for mask in masks:
        if mask["type"] == config.RECTANGLE:
            cv2.rectangle(frame, tuple(mask["coordinates"][0]), tuple(mask["coordinates"][1]),
                          color=mask["color"], thickness=cv2.FILLED)
        if mask["type"] == config.CIRCLE:
            cv2.circle(frame, tuple(mask["center"]), mask["radius"],
                          color=mask["color"], thickness=cv2.FILLED)
        if mask["type"] == config.POLYGON:
            cv2.fillPoly(frame, np.array([mask["coordinates"]]),
                          color=mask["color"])

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

    # set out of arena to black
    if arena:
        mask = np.zeros(frame.shape[:2], np.uint8)

        if arena["type"] == config.POLYGON:
            cv2.fillPoly(mask, [np.array(arena["points"])], 255)

        if arena["type"] == config.CIRCLE:
            cv2.circle(mask, tuple(arena["center"]), arena["radius"],
                       color=255, thickness=cv2.FILLED)

        if arena["type"] == config.RECTANGLE:
            cv2.rectangle(mask, tuple(arena["points"][0]), tuple(arena["points"][1]),
                          color=255, thickness=cv2.FILLED)
        inverted_mask = 255 - mask

        frame = np.where(inverted_mask==255, 0, frame)

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
        float: x of circle center or 0 in case of error
        float: y of circle center or 0 in case of error
        float: radius of circle or -1 in case of error

    """

    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    if (x2 - x1) == 0:
        x2 += 1
    ma = (y2 - y1) / (x2 - x1)
    if (x3 - x2) == 0:
        x3 += 1

    mb = (y3 - y2) / (x3 - x2)

    try:
        x = (ma * mb * (y1 - y3) + mb * (x1 + x2) - ma * (x2 + x3)) / (2 * (mb - ma))
        y = - (1 / ma) * (x - (x1 + x2) / 2) + (y1 + y2) / 2
    except ZeroDivisionError:
        return (0, 0, -1)


    return (x, y, euclidean_distance((x, y), (x1, y1)))


def detect_and_filter_objects(frame,
                              min_size=0,
                              max_size=0,
                              arena={},
                              max_extension=50):
    """
    returns all detected objects and filtered objects

    Args:
        frame: np.array
        min_size (int): minimum size of objects to detect
        max_size (int): maximum size of objects to detect
        number_of_objects (int): number of objects to return
        arena (list): list of arena
        max_extension (int): maximum extension of object to select (in pixels)

    Returns:
        dict: all detected objects
        dict: filtered objects
    """

    #_, contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    nr_objects = len(contours)

    all_objects = {}
    for idx, cnt in enumerate(contours):

        n = np.vstack(cnt).squeeze()
        try:
            x, y = n[:,0], n[:,1]
        except:
            x, y = n[0], n[1]

        # centroid
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = np.mean(x), np.mean(y)

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
            obj_to_del_idx.append(idx)
            continue

        # check if object area is <= of maximal size
        if max_size and all_objects[idx]["area"] > max_size:
            obj_to_del_idx.append(idx)
            continue

        # check if object extension <= max extension
        if max_extension:
            if ((all_objects[idx]["max"][0] - all_objects[idx]["min"][0]) > max_extension
                or (all_objects[idx]["max"][1] - all_objects[idx]["min"][1]) > max_extension):
                obj_to_del_idx.append(idx)
                continue

    filtered_objects = {}
    for obj_idx in all_objects:
        if obj_idx not in obj_to_del_idx:
            filtered_objects[obj_idx] = dict(all_objects[obj_idx])

    # reorder filtered objects starting from #1
    filtered_objects = dict([(i + 1, filtered_objects[idx]) for i, idx in enumerate(filtered_objects.keys())])

    return all_objects, filtered_objects


def distances(a1, a2):
    return cdist(a1, a2).diagonal()


def cost_sum_assignment(mem_objects: dict, objects: dict) -> int:
    """
    sum of distances between centroids of objects
    """

    if len(objects) == len(mem_objects):

        p1 = np.array([mem_objects[k]["centroid"] for k in mem_objects])
        p2 = np.array([objects[k]["centroid"] for k in objects])

        distances = cdist(p1, p2)

        row_ind, col_ind = linear_sum_assignment(distances)

        return int(round(distances[row_ind, col_ind].sum()))

    else:
        raise
        return -1


def reorder_objects(mem_objects: dict, objects: dict) -> dict:
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

        if not np.array_equal(col_ind, list(range(len(col_ind)))):
            reordered_object = dict([(idx + 1, objects[k + 1]) for idx, k in enumerate([x for x in col_ind])])
            return reordered_object
        else:
            return objects

    else:

        return objects


def error_info(exc_info: tuple) -> tuple:
    """
    return details about error
    usage: error_info(sys.exc_info())

    Args:
        sys.exc_info() (tuple):

    Returns:
        tuple: error type, error file name, error line number
    """

    exc_type, exc_obj, exc_tb = exc_info
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    return (exc_obj, fname, exc_tb.tb_lineno)
