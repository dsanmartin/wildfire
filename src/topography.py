import numpy as np
from utils import G2D, G3D
from parameters import hill_center_x, hill_center_y, hill_height, hill_width, hill_length

# A hill using a Gaussian function
hill2D = lambda x: G2D(x, 0, hill_center_x, 0, hill_length, 1, hill_height)
hill3D = lambda x, y: G3D(x, y, 0, hill_center_x, hill_center_y, 0, hill_length, hill_width, 1, hill_height)
# Flat terrain
flat2D = lambda x: x * 0 
flat3D = lambda x, y: x * y * 0
