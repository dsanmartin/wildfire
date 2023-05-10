import numpy as np
from parameters import u_ast, kappa, d, z_0, T_inf, TA, x_min, x_max
from utils import create_plate

# To load data 
BASE_DIR = './output/'
sim_name = '20230502144131' 
path = BASE_DIR + sim_name + '/' 

# Initial fluid flow vector field $\mathbf{u}=(u, v)$
# $u(x,y,0)$
# u0 = lambda x, y: u_r * (y / y_r) ** alpha # Power-law
u0 = lambda x, y: np.piecewise(y, [y > 0, y == 0], [ # Piecewise is used for y=0
        lambda y: u_ast / kappa * np.log((y - d) / z_0), # Log wind profile if y > 0
        lambda y: y * 0 # 0 if y = 0
    ])
# $v(x,y, 0) = 0$
v0 = lambda x, y: x * 0 
# Initial fuel
Y0 = lambda x, y: x * 0 
# Initial temperature (create a 'plate')
# x_start = 250
# x_end = 550
# Experiments
x_end = (x_max + x_min) / 2
x_start = x_end - 300
y_start = 0
y_end = 5
# FDS
x_start = 10
x_end = x_start + 2 # 2
y_start = 0
y_end = .5
plate = create_plate(x_start, x_end, y_start, y_end) # Return True if x_min <= x <= x_max & y_min <= y <= y_max
T0 = lambda x, y: T_inf + (plate(x, y)) * (TA - T_inf)
# Initial pressure
p0 = lambda x, y: x * 0 #+ 1e-12
# Force term
fx = lambda x, y: x * 0
fy = lambda x, y: x * 0 
F = lambda x, y: np.array([fx(x, y), fy(x, y)])
# Extra source term
# ST = lambda x, y, T: (TA - T) * plate(x, y)

# Initial conditions from data
# data = np.load(path + 'data.npz')
# n = 331
# u0_data = data['u'][n]
# v0_data = data['v'][n]
# T0_data = data['T'][n]
# Y0_data = data['Y'][n]
# u0 = lambda x, y: u0_data[:, :-1]
# v0 = lambda x, y: v0_data[:, :-1]
# T0 = lambda x, y: T0_data[:, :-1]
# Y0 = lambda x, y: Y0_data[:, :-1]
