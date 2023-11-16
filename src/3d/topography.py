import numpy as np
from utils import G
from parameters import hill_center_x, hill_center_y, hill_height, hill_width, hill_length

# A hill in the middle of domain
hill = lambda x, y: G(x, y, 0, hill_center_x, hill_center_y, 0, hill_length, hill_width, 1, hill_height)
# Hill
# topo = lambda x: hill(x, y_c)
# Flat terrain
flat = lambda x, y: x * y * 0 
