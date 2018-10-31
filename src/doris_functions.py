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
#np.set_printoptions(threshold="nan")
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def image_treatment(frame,
                    blur=5,
                    threshold_method={"name": "adaptive",
                                      "block_size": 81,
                                      "offset": 38},
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
    #ret, frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

    # invert

    if threshold_method["name"] == "adaptive":
        frame = cv2.adaptiveThreshold(frame,
                                      255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY,
                                      threshold_method["block_size"],
                                      threshold_method["offset"])


    if invert:
        frame = (255 - frame)


    return frame


def euclidean_distance(p1, p2):
    """
    euclidean distance between two points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


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

    print("nr objects", nr_objects)

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
            print("skipped #{} for min area".format(idx))
            obj_to_del_idx.append(idx)
            continue

        # check if object area is <= of maximal size
        if max_size and all_objects[idx]["area"] > max_size:
            print("skipped #{} for max area".format(idx))
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
                print("skipped #{} for max extension".format(idx))
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



































