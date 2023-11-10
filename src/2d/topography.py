import numpy as np
from utils import G
from parameters import hill_center, hill_height, hill_width

# A hill in the middle of domain
hill = lambda x: G(x, 0, hill_center, 0, hill_width, 1, hill_height)
# Hill
# topo = lambda x: hill(x, y_c)
# Flat terrain
flat = lambda x: x * 0 
