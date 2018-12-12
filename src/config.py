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

THRESHOLD_METHODS = ["Adaptive (mean)", "Adaptive (Gaussian)", "Simple"]


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


MARKER_TYPE = "contour"
MARKER_COLOR = GREEN
ARENA_COLOR = RED
AREA_COLOR = GREEN

CIRCLE_THICKNESS = 2
RECTANGLE_THICKNESS = 2

TOLERANCE_OUTSIDE_ARENA = 0.05  # fraction of size of object


FONT = cv2.FONT_HERSHEY_SIMPLEX
