import numpy as np
from utils import G
from parameters import hill_center, hill_height, sx

# A hill in the middle of domain
# topo_height = 30 
# sx = 100 #10000
# sy = 100 #10000
# x_c = (x_max + x_min) / 2
# y_c = (y_max + y_min) / 2
#top = lambda x, y: A_topo * np.exp(-((x - x_c) ** 2 / sx + (y - y_c) ** 2 / sy)) 
# hill = lambda x, y: G(x, y, x_c, y_c, sx, sy, topo_height)
hill = lambda x: G(x, 0, hill_center, 0, sx, 1, hill_height)
# top = lambda x, y: x * y * 0 #+ dy
# Hill
# topo = lambda x: hill(x, y_c)
# Flat terrain
flat = lambda x: x * 0 
