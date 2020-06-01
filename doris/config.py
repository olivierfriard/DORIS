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

DEFAULT_FRAME_SCALE = 0.5
ZOOM_LEVELS = ["2", "1", "0.5", "0.25"]

DIST_MAX = 0

THRESHOLD_METHODS = ["Simple", "Adaptive (mean)", "Adaptive (Gaussian)"]

BLUR_DEFAULT_VALUE = 2
SHOW_CENTROID_DEFAULT = False
SHOW_CONTOUR_DEFAULT = True

THRESHOLD_DEFAULT = 128
ADAPTIVE_BLOCK_SIZE = 10
ADAPTIVE_OFFSET = 10


VIEWER_WIDTH = 480

NB_ROWS_COORDINATES_VIEWER = 20

YES = "Yes"
NO = "No"

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

CONTOUR = "contour"
RECTANGLE = "rectangle"
CIRCLE = "circle"
CIRCLE_CENTER_RADIUS = "circle (center radius)"
CIRCLE_3PTS = "circle (3 points)"
POLYGON = "polygon"

MARKER_TYPE = CONTOUR
MARKER_COLOR = GREEN
ARENA_COLOR = PURPLE
AREA_COLOR = GREEN

CIRCLE_THICKNESS = 2
RECTANGLE_THICKNESS = 2

OBJECT_PATH_LENGTH = 20

ORIGINAL_FRAME_VIEWER_IDX = 0
PROCESSED_FRAME_VIEWER_IDX = 1
PREVIOUS_FRAME_VIEWER_IDX = 2

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SIZE = 0.5

RGBSTR_COLORS_LIST = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF",
        "800000", "008000", "000080", "808000", "800080", "008080", "808080",
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
        "400000", "004000", "000040", "404000", "400040", "004040", "404040",
        "200000", "002000", "000020", "202000", "200020", "002020", "202020",
        "600000", "006000", "000060", "606000", "600060", "006060", "606060",
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]



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

SKIP_FRAME = "Skip frame"
ALWAYS_SKIP_FRAME = "Always skip frame"
REPICK_OBJECTS = "Repick objects"
ACCEPT_MOVEMENT = "Accept movement"
GO_TO_PREVIOUS_FRAME = "Go to previous frame"