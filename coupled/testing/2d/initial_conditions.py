import numpy as np
from parameters import u_ast, kappa, d, z_0, u_r, y_r, alpha, T_inf, TA, x_min, x_max
from utils import create_plate, create_half_gaussian

# To load data 

# Initial fluid flow vector field $\mathbf{u}=(u, v)$ at t=0
# Log wind profile
log_wind = lambda x, y: np.piecewise(y, [y > 0, y == 0], [ # Piecewise is used for y=0
        lambda y: u_ast / kappa * np.log((y - d) / z_0), # Log wind profile if y > 0
        lambda y: y * 0 # 0 if y = 0
    ])
# Power law wind profile (FDS experiment)
power_law_wind = lambda x, y: u_r * (y / y_r) ** alpha 
initial_u = power_law_wind
u0 = lambda x, y: initial_u(x, y)
# $v(x,y, 0) = 0$
v0 = lambda x, y: x * 0 
# Initial fuel
Y0 = lambda x, y: x * 0 
# Initial temperature (create a 'plate')
# x_start = 250
# x_end = 550
# Experiments
x_start = 10
x_end = x_start + 2
y_start = 0
y_end = .5 
# FDS
x_start = 0
x_end = x_start + 3.3 
y_start = 0
y_end = .25
plate = create_plate(x_start, x_end, y_start, y_end) # Return True if x_min <= x <= x_max & y_min <= y <= y_max
half_gaussian = create_half_gaussian(x_start, 4, 1)
shape = plate
T0 = lambda x, y: T_inf + (shape(x, y)) * (TA - T_inf)
# Initial pressure
p0 = lambda x, y: x * 0 #+ 1e-12
# Force term
fx = lambda x, y: x * 0
fy = lambda x, y: x * 0 
F = lambda x, y: np.array([fx(x, y), fy(x, y)])
# Extra source term
# ST = lambda x, y, T: (TA - T) * plate(x, y)

# Initial conditions from data
# BASE_DIR = './output/'
# sim_name = '20230509091755' 
# path = BASE_DIR + sim_name + '/' 
# data = np.load(path + 'data.npz')
# n = 204
# u0_data = data['u'][n]
# v0_data = data['v'][n]
# T0_data = data['T'][n]
# Y0_data = data['Y'][n]
# u0 = lambda x, y: u0_data[:, :-1]
# v0 = lambda x, y: v0_data[:, :-1]
# T0 = lambda x, y: T0_data[:, :-1]
# Y0 = lambda x, y: Y0_data[:, :-1]
