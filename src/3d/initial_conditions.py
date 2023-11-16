""" Initial conditions
"""
import numpy as np
from topography import flat, hill
from utils import create_plate, create_half_gaussian
from arguments import T_hot # Parameters from command line
from parameters import T_inf, u_ast, k, d, u_z_0, u_r, z_r, alpha_u, initial_u_type, topography_shape, fuel_height, T0_shape, T0_x_start, T0_x_end, T0_y_start, T0_y_end, T0_z_start, T0_z_end, T0_x_center, T0_y_center, T0_length, T0_width, T0_height

def load_initial_condition(data_path: str) -> callable:
    data = np.load(data_path)
    return lambda x, y, z: data + x * y * z * 0

# Initial fluid flow vector field $\mathbf{u}=(u, v)$ at t=0 #
# Log wind profile
log_wind = lambda x, y, z: np.piecewise(z, [z > 0, z == 0], [ # Piecewise is used for y=0
        lambda z: u_ast / k * np.log((z - d) / u_z_0), # Log wind profile if y > 0
        lambda z: z * 0 # 0 if y = 0
    ])
# Power law wind profile (FDS experiment)
power_law_wind = lambda x, y, z: u_r * (z / z_r) ** alpha_u
initial_u = power_law_wind if initial_u_type == 'power law' else log_wind
u0 = lambda x, y, z: initial_u(x, y, z)   #+ np.random.rand(*x.shape) * 0.5
# $v(x,y, 0) = 0$
v0 = lambda x, y, z: x * 0 
# $w(x,y, 0) = 0$
w0 = lambda x, y, z: x * 0

# Initial fuel $Y(x,y,0)$ #
topo = flat if topography_shape == 'flat' else hill
Y0 = lambda x, y, z: (z <= (topo(x, y) + fuel_height)).astype(int) # 1 if y <= topo(x) + fuel_height else 0

# Initial temperature $T(x,y,0)$ #
if T0_shape == 'plate':
    shape = create_plate(T0_x_start, T0_x_end, T0_y_start, T0_y_end, T0_z_start, T0_z_end) # Return True if x_min <= x <= x_max & y_min <= y <= y_max
else:
    shape = create_half_gaussian(T0_x_center, T0_y_center, T0_length, T0_width, T0_height) #
T0 = lambda x, y, z: T_inf + (shape(x, y, z)) * (T_hot - T_inf)

# Initial pressure $p(x, y, 0)$ #
p0 = lambda x, y, z: x * 0 

# Force term $F=(fx, fy, fz)$ #
fx = lambda x, y, z: x * 0
fy = lambda x, y, z: x * 0 
fz = lambda x, y, z: x * 0
F = lambda x, y, z: np.array([fx(x, y, z), fy(x, y, z), fz(x, y, z)])
