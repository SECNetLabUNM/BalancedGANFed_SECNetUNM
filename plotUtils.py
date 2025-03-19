
import os
from matplotlib import rcParams, pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib as mpl
from matplotlib.lines import Line2D

def getGradColors(mainColor, n_lines):
  color_list = [mainColor,'gray']
  cmap = LinearSegmentedColormap.from_list("",color_list)
  colors = cmap(np.linspace(0, 1, n_lines))
  pass
  return colors

def getColorsCmap(colormapStr, n_lines):
  cmap = mpl.colormaps[colormapStr]
  colors = cmap(np.linspace(0, 1, n_lines))
  pass
  return colors

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
linestyle_tuple = [
     ('loosely dotted',        (0, (1, 10))),
     ('dotted',                (0, (1, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('long dash with offset', (5, (10, 3))),
     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
nlinestyles = len(linestyle_tuple)

# markers = list( Line2D.markers.keys() )[:-4].remove('*')
markers = list( Line2D.markers.keys() )[:-4]
markers.remove('*')
nmarkers = len(markers)



if __name__ == "__main__":
  pass
