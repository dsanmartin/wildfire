""" Initial conditions
"""
import numpy as np
from topography import flat, hill
from utils import create_plate, create_half_gaussian
from arguments import T_hot # Parameters from command line
from parameters import T_inf, u_ast, kappa, d, u_z_0, u_r, y_r, alpha, initial_u_type, topography_shape, fuel_height, T0_shape, T0_x_start, T0_x_end, T0_y_start, T0_y_end, T0_x_center, T0_width, T0_height

# Initial fluid flow vector field $\mathbf{u}=(u, v)$ at t=0 #
# Log wind profile
log_wind = lambda x, y: np.piecewise(y, [y > 0, y == 0], [ # Piecewise is used for y=0
        lambda y: u_ast / kappa * np.log((y - d) / u_z_0), # Log wind profile if y > 0
        lambda y: y * 0 # 0 if y = 0
    ])
# Power law wind profile (FDS experiment)
power_law_wind = lambda x, y: u_r * (y / y_r) ** alpha 
initial_u = power_law_wind if initial_u_type == 'power law' else log_wind
u0 = lambda x, y: initial_u(x, y)   #+ np.random.rand(*x.shape) * 0.5
# $v(x,y, 0) = 0$
v0 = lambda x, y: x * 0 

# Initial fuel $Y(x,y,0)$ #
topo = flat if topography_shape == 'flat' else hill
Y0 = lambda x, y: (y <= (topo(x) + fuel_height)).astype(int) # 1 if y <= topo(x) + fuel_height else 0

# Initial temperature $T(x,y,0)$ #
if T0_shape == 'plate':
    shape = create_plate(T0_x_start, T0_x_end, T0_y_start, T0_y_end) # Return True if x_min <= x <= x_max & y_min <= y <= y_max
else:
    shape = create_half_gaussian(T0_x_center, T0_width, T0_height) #
T0 = lambda x, y: T_inf + (shape(x, y)) * (T_hot - T_inf)

# Initial pressure $p(x, y, 0)$ #
p0 = lambda x, y: x * 0 

# Force term $F=(fx, fy)$ #
fx = lambda x, y: x * 0
fy = lambda x, y: x * 0 
F = lambda x, y: np.array([fx(x, y), fy(x, y)])

# Extra source term
# ST = lambda x, y, T: (TA - T) * plate(x, y)

# Initial conditions from data #
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
