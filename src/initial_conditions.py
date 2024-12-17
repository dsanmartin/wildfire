import numpy as np
from topography import flat2D, hill2D, flat3D, hill3D
from utils import create_plate, create_gaussian, create_plate_slope
from arguments import initial_u_type, T_hot, topography_shape, spatial_dims, Y_h, u_r, T0_shape, T0_x_start, T0_x_end, T0_y_start, T0_y_end, T0_z_start, T0_z_end, T0_x_center, T0_y_center, T0_z_center, T0_length, T0_width, T0_height, T_cold # Parameters from command line
from parameters import T_inf, u_ast, k, d, u_z_0, z_r, alpha_u

def load_initial_condition(data_path: str, ndim: int = 2) -> callable:
    data = np.load(data_path)
    if ndim == 2:
        return lambda x, y: data + x * y * 0
    elif ndim == 3:
        return lambda x, y, z: data + x * y * z * 0
    else:
        raise ValueError('ndim must be 2 or 3')

if spatial_dims == 2:
    # Initial fluid flow vector field $\mathbf{u}=(u, v)$ at t=0 #
    # Log wind profile
    log_wind = lambda x, y: np.piecewise(y, [y > 0, y == 0], [ # Piecewise is used for y=0
            lambda y: u_ast / k * np.log((y - d) / u_z_0), # Log wind profile if y > 0
            lambda y: y * 0 # 0 if y = 0
        ])
    # Power law wind profile (FDS experiment)
    power_law_wind = lambda x, y: u_r * (y / z_r) ** alpha_u
    constant = lambda x, y: u_r + x * y * 0
    # initial_u = power_law_wind if initial_u_type == 'power law' else log_wind
    if initial_u_type == 'constant':
        initial_u = constant
    elif initial_u_type == 'power law':
        initial_u = power_law_wind
    elif initial_u_type == 'log':
        initial_u = log_wind
    u0 = lambda x, y: initial_u(x, y)   #+ np.random.rand(*x.shape) * 0.5
    # $v(x,y, 0) = 0$
    v0 = lambda x, y: x * 0 
    U0 = np.array([u0, v0])

    # Initial fuel $Y(x,y,0)$ #
    if topography_shape == 'flat':
        topo = flat2D
    elif topography_shape == 'hill':
        topo = hill2D
    Y0 = lambda x, y: (y <= (topo(x) + Y_h)).astype(int) # 1 if y <= topo(x) + Y_h else 0

    # Initial temperature $T(x,y,0)$ #
    if T0_shape == 'plate':
        shape = create_plate((T0_x_start, T0_x_end), (T0_z_start, T0_z_end)) 
    elif T0_shape == 'gaussian':
        shape = create_gaussian((T0_x_center, 0), (T0_length, T0_height)) 
    # else:
    #     shape = create_plate((T0_x_start, T0_x_end), (T0_z_start, T0_z_end)) 
    T0 = lambda x, y: T_inf + (shape(x, y)) * (T_hot - T_inf)
    if T0_shape == 'cavity':
        shape1 = create_plate((T0_x_start, T0_x_end), (T0_y_start, T0_y_end))
        shape2 = create_plate((T0_x_start, T0_x_end), (T0_x_end - T0_y_end, T0_x_end - T0_y_start))
        # eps = 0.01
        # shape1 = create_plate_slope(T0_x_start, T0_x_end, T0_y_end, T0_y_end + eps)
        # shape2 = create_plate_slope(T0_x_start, T0_x_end, T0_x_end -T0_y_end - eps, T0_x_end - T0_y_end, True) 
        T0 = lambda x, y: T_inf + (shape1(x, y)) * (T_hot - T_inf) + (shape2(x, y)) * (T_cold - T_inf)

    # Initial pressure $p(x, y, 0)$ #
    p0 = lambda x, y: x * y * 0 

    # Force term $F=(fx, fy)$ #
    F = lambda x, y: np.array([
        x * 0, 
        y * 0
    ])

elif spatial_dims == 3:
    # Initial fluid flow vector field $\mathbf{u}=(u, v, w)$ at t=0 #
    # Log wind profile
    log_wind = lambda x, y, z: np.piecewise(z, [z > 0, z == 0], [ # Piecewise is used for y=0
            lambda z: u_ast / k * np.log((z - d) / u_z_0), # Log wind profile if y > 0
            lambda z: z * 0 # 0 if y = 0
        ])
    # Power law wind profile (FDS experiment)
    power_law_wind = lambda x, y, z: u_r * (z / z_r) ** alpha_u
    initial_u = power_law_wind if initial_u_type == 'power law' else log_wind
    u0 = lambda x, y, z: initial_u(x, y, z)   #+ np.random.rand(*x.shape) * 0.5
    # $v(x, y, z, 0) = 0$
    v0 = lambda x, y, z: np.zeros_like(y)
    # $w(x, y, z, 0) = 0$
    w0 = lambda x, y, z: np.zeros_like(z)
    U0 = np.array([u0, v0, w0])

    # Initial fuel $Y(x, y, z, 0)$ #
    topo = flat3D if topography_shape == 'flat' else hill3D
    Y0 = lambda x, y, z: (z <= (topo(x, y) + Y_h)).astype(np.float64) # 1 if y <= topo(x) + Y_h else 0

    # Initial temperature $T(x, y, z, 0)$ #
    if T0_shape == 'plate':
        shape = create_plate((T0_x_start, T0_x_end), (T0_y_start, T0_y_end), (T0_z_start, T0_z_end)) # Return True if x_min <= x <= x_max & y_min <= y <= y_max
    else:
        shape = create_gaussian((T0_x_center, T0_y_center, T0_z_center), (T0_length, T0_width, T0_height)) #
    T0 = lambda x, y, z: T_inf + (shape(x, y, z)) * (T_hot - T_inf)

    # Initial pressure $p(x, y, z, 0)$ #
    p0 = lambda x, y, z: x * y * z * 0 

    # Force term $F=(fx, fy, fz)$ #
    F = lambda x, y, z: np.array([
        x * 0, 
        y * 0, 
        z * 0
    ])
