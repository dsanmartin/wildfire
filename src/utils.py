import numpy as np
from arguments import T_act, A, H_R, h, alpha, S_top, S_bot, k, Y_D # Parameters from command line
from parameters import T_pc, T_inf, g, n_arrhenius, h_rad, C_p, rho, a_v, S_T_0, S_k_0, S_k, sigma, delta, sutherland_law, radiation, include_source, source_filter # Default parameters
#from parameters import T_act, A, H_R, h, alpha, S_top, S_bot, k, Y_D

# A lot of useful functions #
G2D = lambda x, y, x0, y0, sx, sy, A: A * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2)) # 2D Gaussian function
G3D = lambda x, y, z, x0, y0, z0, sx, sy, sz, A: A * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2 + (z - z0) ** 2 / sz ** 2)) # 3D Gaussian function
K = lambda T: A * np.exp(-T_act / T) # Arrhenius-like equation 
Km = lambda T: K(T) * T ** n_arrhenius # Modified Arrhenius-like equation
H = lambda T: T > T_pc # Step function
CV = lambda x, T_pc: 2 * x / T_pc - 1 # Variable change
Sg = lambda x, x0, k: 1 / (1 + np.exp(-2 * k * (x - x0))) # Sigmoid function (to smooth the step function)
HS2 = lambda x, x0, k: .5 * (1 + np.tanh(k * (x - x0))) # Hyperbolic tangent function (to smooth the step function) 
HS3 = lambda x, x0, k, T_pc: Sg(CV(x, T_pc), x0, k) # Hyperbolic tangent function (to smooth the step function)
Q_rad = lambda T: h_rad * (T ** 4 - T_inf ** 4) / (rho * C_p) # Radiative heat flux
source = lambda T, Y: H_R * Y * K(T) * H(T) / C_p # Source term
sink = lambda T: -h * a_v * (T - T_inf) / (rho * C_p) # Sink term
sutherland = lambda T: S_k_0 * (T / S_T_0) ** 1.5 * (S_T_0 + S_k) / (T + S_k) / (rho * C_p) # Sutherland's law
sutherland_T = lambda T: 1.5 * S_k_0 * (S_T_0 + S_k) / T ** 1.5 * (T ** .5 * (T + S_k) - T ** 1.5) / (T + S_k) ** 2 / (rho * C_p) # Sutherland's law derivative
stefan_radiation = lambda T: 4 * sigma * delta * T ** 3 / (rho * C_p) # Stefan-Boltzmann law
stefan_radiation_T = lambda T: 12 * sigma * delta * T ** 2 / (rho * C_p) # Stefan-Boltzmann law derivative
gamma = lambda r, s, dz, Nx, Ny: - (2 + (2 * np.pi * dz) ** 2 * ((r / Nx) ** 2 + (s / Ny) ** 2))
# # Convective heat transfer coefficient
if h < 0:
    hv = lambda v: np.piecewise(v, [v < 2, v >= 2], [
            lambda v: 0 * v, # No information
            lambda v: 12.12 - 1.16 * v + 11.6 * v ** 0.5 # Wind chill factor
    ])
else:
    hv = lambda v: h # Constant 17-18 W/m^2/K?

def domain_2D(x_min: float, x_max: float, y_min: float, y_max: float, t_min: float, t_max: float, Nx: int, Ny: int, Nt: int) -> tuple:
    """
    Generate a 2D domain for a given range and number of points in each dimension.

    Parameters
    ----------
    x_min : float
        Minimum value of x-axis.
    x_max : float
        Maximum value of x-axis.
    y_min : float
        Minimum value of y-axis.
    y_max : float
        Maximum value of y-axis.
    t_min : float
        Minimum value of time.
    t_max : float
        Maximum value of time.
    Nx : int
        Number of points in x-axis.
    Ny : int
        Number of points in y-axis.
    Nt : int
        Number of points in time.

    Returns
    -------
    x : numpy.ndarray (Nx,)
        1D array of x-axis values.
    y : numpy.ndarray (Ny,)
        1D array of y-axis values.
    t : numpy.ndarray (Nt,)
        1D array of time values.
    Xm : numpy.ndarray (Ny, Nx-1)
        2D meshgrid of x-axis values.
    Ym : numpy.ndarray (Ny, Nx-1)
        2D meshgrid of y-axis values.
    dx : float
        Interval size in x-axis.
    dy : float
        Interval size in y-axis.
    dt : float
        Interval size in time.
    """
    # 1D arrays
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    t = np.linspace(t_min, t_max, Nt)
    # Meshgrid
    Xm, Ym = np.meshgrid(x[:-1], y)
    # Interval size
    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
    return x, y, t, Xm, Ym, dx, dy, dt

def domain_3D(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float, t_min: float, t_max: float, Nx: int, Ny: int, Nz: int, Nt: int) -> tuple:
    """
    Generate a 3D domain for a given range and number of points in each dimension.

    Parameters
    ----------
    x_min : float
        Minimum value of x-axis.
    x_max : float
        Maximum value of x-axis.
    y_min : float
        Minimum value of y-axis.
    y_max : float
        Maximum value of y-axis.
    z_min : float
        Minimum value of z-axis.
    z_max : float
        Maximum value of z-axis.
    t_min : float
        Minimum value of time.
    t_max : float
        Maximum value of time.
    Nx : int
        Number of points in x-axis.
    Ny : int
        Number of points in y-axis.
    Nz : int
        Number of points in z-axis.
    Nt : int
        Number of points in time.

    Returns
    -------
    x : numpy.ndarray (Nx,)
        1D array of x-axis values.
    y : numpy.ndarray (Ny,)
        1D array of y-axis values.
    z : numpy.ndarray (Nz,)
        1D array of z-axis values.
    t : numpy.ndarray (Nt,)
        1D array of time values.
    Xm : numpy.ndarray (Nz, Ny, Nx-1)
        3D meshgrid of x-axis values.
    Ym : numpy.ndarray (Nz, Ny, Nx-1)
        3D meshgrid of y-axis values.
    Zm : numpy.ndarray (Nz, Ny, Nx-1)
        3D meshgrid of z-axis values.
    dx : float
        Interval size in x-axis.
    dy : float
        Interval size in y-axis.
    dz : float
        Interval size in z-axis.
    dt : float
        Interval size in time.
    """
    # 1D arrays
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    z = np.linspace(z_min, z_max, Nz)
    t = np.linspace(t_min, t_max, Nt)
    # Meshgrid
    Xm, Ym, Zm = np.meshgrid(x[:-1], y[:-1], z)
    # Interval size
    dx, dy, dz, dt = x[1] - x[0], y[1] - y[0], z[1] - z[0], t[1] - t[0]
    return x, y, z, t, Xm, Ym, Zm, dx, dy, dz, dt

def create_plate(x_lim: tuple[float, float], y_lim: tuple[float, float], z_lim: tuple[float, float] = None) -> callable:
    """
    Create a plate function that returns a binary mask indicating whether a point is inside the plate.
    This is used for temperature initial condition.

    Parameters:
    ----------
    x_lim : tuple[float, float]
        The x-axis limits of the plate.
    y_lim : tuple[float, float]
        The y-axis limits of the plate.
    z_lim : tuple[float, float], optional
        The z-axis limits of the plate. Default is None.

    Returns:
    -------
    callable
        A function that takes the coordinates of a point and returns a binary mask indicating whether the point is inside the plate.

    Notes:
    ------
    - If z_lim is not provided, the plate function will only consider the x and y coordinates.
    - If z_lim is provided, the plate function will consider the x, y, and z coordinates.
    """
    if z_lim is None:
        x_min, x_max = x_lim
        y_min, y_max = y_lim
        plate = lambda x, y: ((y >= y_min) & (y <= y_max) & (x <= x_max) & (x >= x_min)).astype(int)
    else:
        x_min, x_max = x_lim
        y_min, y_max = y_lim
        z_min, z_max = z_lim
        plate = lambda x, y, z: ((y >= y_min) & (y <= y_max) & (x <= x_max) & (x >= x_min) & (z <= z_max) & (z >= z_min)).astype(int)
    return plate

def create_gaussian(center: tuple, dimensions: tuple) -> callable:
    """
    Create a half Gaussian function based on the number of dimensions. 
    This is used for temperature initial condition.

    Parameters
    ----------
    center : tuple
        The center coordinates of the Gaussian function.
    dimensions : tuple
        The dimensions of the Gaussian function.

    Returns
    -------
    callable
        The half Gaussian function.

    Raises
    ------
    ValueError
        If the number of dimensions in center is invalid.
    """
    ndims = len(center)
    if ndims == 2: # For 2D
        x_center, y_center = center
        length, height = dimensions
        half_gaussian = lambda x, y: G2D(x, y, x_center, y_center, length, height, 1)
    elif ndims == 3: # For 3D
        x_center, y_center, z_center = center
        length, width, height = dimensions
        half_gaussian = lambda x, y, z: G3D(x, y, z, x_center, y_center, z_center, length, width, height, 1)
    else:
        raise ValueError('Invalid number of dimensions in center')
    return half_gaussian

def non_dimensional_numbers(parameters: dict) -> tuple[float, float, float, float, float, float, float]:
    """
    Calculate non-dimensional numbers based on the given parameters.

    Parameters:
    -----------
    parameters : dict
        A dictionary containing the following parameters:
        - 'x': numpy.ndarray : x-coordinates
        - 'y': numpy.ndarray : y-coordinates
        - 'z': numpy.ndarray : z-coordinates (optional)
        - 't': numpy.ndarray : time
        - 'nu': float : kinematic viscosity
        - 'Pr': float : Prandtl number
        - 'rho': float : density
        - 'C_p': float : specific heat capacity
        - 'h': float : heat transfer coefficient
        - 'A': float : characteristic length
        - 'g': List[float] : gravitational acceleration
        - 'u0': float : initial velocity in x-direction
        - 'v0': float : initial velocity in y-direction
        - 'w0': float : initial velocity in z-direction (optional)
        - 'T0': List[float] : initial temperature

    Returns:
    --------
    tuple[float, float, float, float, float, float, float]
        A tuple containing the following non-dimensional numbers:
        - Reynolds number (Re)
        - Grashof number (Gr)
        - Rayleigh number (Ra)
        - Strouhal number (Sr)
        - Stefan number (Ste)
        - Stanton number (St)
        - Zeldovich number (Ze)
    """
    x_min, x_max = parameters['x'][0], parameters['x'][-1]
    y_min, y_max = parameters['y'][0], parameters['y'][-1]
    x_dim, y_dim = x_max - x_min, y_max - y_min
    # t_min, t_max = parameters['t'][0], parameters['t'][-1]
    nu, Pr = parameters['nu'], parameters['Pr']
    rho, C_p, h = parameters['rho'], parameters['C_p'], parameters['h']
    A = parameters['A']
    g = parameters['g']
    alpha, k = parameters['alpha'], parameters['k']
    u0, v0, T0 = parameters['u0'], parameters['v0'], parameters['T0']
    L_v = y_dim
    L_c = x_dim * L_v
    L = (L_c) ** 0.5
    spt = u0 ** 2 + v0 ** 2
    if 'z' in parameters:
        z_min, z_max = parameters['z'][0], parameters['z'][-1]
        z_dim = z_max - z_min
        L_v = z_dim
        L_c = x_dim * y_dim * L_v
        L = (L_c) ** (1/3)
    if 'w0' in parameters:
        w0 = parameters['w0']
        spt += w0 ** 2
    dT = (np.max(T0) - np.min(T0))
    speed = np.sqrt(spt)
    U = np.max(speed)
    T = np.max(T0)
    T_avg = np.max(T0)
    alpha_T = 1 / T_avg
    # Reynolds
    Re = U * L / nu
    # Grashof
    Gr = abs(g[-1]) * alpha_T * dT * L_v ** 3 / nu ** 2 
    # Rayleigh
    Ra = Gr * Pr
    # Strouhal
    Sr = A * L / U
    # Stefan 
    Ste = C_p * dT / h
    # Stanton
    St = h * L / (rho * C_p * U)
    # Zeldovich
    Ze = T_avg * dT / T ** 2
    # Peclét
    Pe = U * L / alpha
    # Nusselt
    Nu = h * L / k
    # Return
    return Re, Gr, Ra, Sr, Ste, St, Ze, Pe, Nu

def f(U: tuple, T: np.ndarray, Y: np.ndarray) -> list:
    """
    Calculate the force in momentum equation.

    Parameters
    ----------
    U : tuple 
        Tuple containing the velocity components (u, v) or (u, v, w) depending on the number of dimensions.
    T : np.ndarray (Ny, Nx)
        Temperature array.
    Y : np.ndarray (Ny, Nx)
        Concentration array.

    Returns
    -------
    list
        List containing the components of the force.

    Raises
    ------
    ValueError
        If the number of dimensions of U is invalid.
    """
    ndims = len(U)
    g_x, g_y, g_z = g
    if ndims == 2:
        u, v = U
        g_y = g_z
        mod_U = np.sqrt(u ** 2 + v ** 2)
        return [
            - g_x * (T - T_inf) / T - Y_D * a_v * Y * mod_U * u,
            - g_y * (T - T_inf) / T - Y_D * a_v * Y * mod_U * v
        ]
    elif ndims == 3:
        u, v, w = U
        mod_U = np.sqrt(u ** 2 + v ** 2 + w ** 2)
        return [
            - g_x * (T - T_inf) / T - Y_D * a_v * Y * mod_U * u,
            - g_y * (T - T_inf) / T - Y_D * a_v * Y * mod_U * v,
            - g_z * (T - T_inf) / T - Y_D * a_v * Y * mod_U * w
        ]
    else:
        raise ValueError('Invalid number of dimensions of U')

def S(T: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate the value of souce and sink given temperature and fuel arrays.

    Parameters
    ----------
    T : np.ndarray (Ny, Nx)
        Array of temperatures.
    Y : np.ndarray (Ny, Nx)
        Array of fuel fraction.

    Returns
    -------
    np.ndarray
        Array of S values.

    """
    if include_source:
        S1 = source(T, Y)
        S2 = sink(T)
        if source_filter:
            S1[S1 >= S_top] = S_bot
        return S1 + S2
    else:
        return 0

def k(T: np.ndarray) -> np.ndarray:
    """
    Calculate the thermal conductivity.

    Parameters
    ----------
    T : np.ndarray
        Array of temperatures.

    Returns
    -------
    np.ndarray
        Array of thermal conductivities.

    Notes
    -----
    The thermal conductivity is calculated as the sum of the convective
    thermal conductivity (k_c) and the radiative thermal conductivity (k_r).
    The convective thermal conductivity is determined by the Sutherland's law
    if `sutherland_law` is True, otherwise it is set to `alpha`.
    The radiative thermal conductivity is calculated using the Stefan-Boltzmann
    law if `radiation` is True, otherwise it is set to 0.
    """
    k_c = sutherland(T) if sutherland_law else alpha
    k_r = stefan_radiation(T) if radiation else 0
    return k_c + k_r

def kT(T: np.ndarray) -> np.ndarray:
    """
    Calculate the thermal conductivity derivative with respect to temperature.

    Parameters
    ----------
    T : np.ndarray
        Array of temperatures.

    Returns
    -------
    np.ndarray
        Array of thermal conductivities derivatives.
    """
    kT_c = sutherland_T(T) if sutherland_law else 0
    kT_r = stefan_radiation_T(T) if radiation else 0
    return kT_c + kT_r