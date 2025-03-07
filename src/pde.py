import time
import numpy as np
from datetime import timedelta
from derivatives import compute_gradient, compute_laplacian, compute_first_derivative_upwind, compute_first_derivative
from poisson import solve_pressure
from turbulence import turbulence
from utils import f, S, k, kT, K, H, rho
from logs import log_time_step

def grad_pressure(p: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the gradient of pressure.

    2D case:
    .. math::
        \nabla p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}\right)

    3D case:
    .. math::
        \nabla p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}, \frac{\partial p}{\partial z}\right)

    Parameters
    ----------
    p : numpy.ndarray (Ny, Nx-1) or (Ny-1, Nx-1, Nz)
        Pressure field.
    params : dict
        Dictionary containing the interval sizes `dx`, `dy` and/or `dz`.

    Returns
    -------
    numpy.ndarray (2, Ny, Nx-1) or (3, Ny-1, Nx-1, Nz)
        Gradient of pressure.
    """
    # Get number of dimensions
    ndims = p.ndim
    if ndims == 2: # 2D case
        hs = (params['dx'], params['dy'])
        periodic = (True, False)
    elif ndims == 3: # 3D case
        hs = (params['dx'], params['dy'], params['dz'])
        periodic = (True, True, False)
    # Compute grad(p)
    grad_p = np.array(compute_gradient(p, hs, periodic))
    return grad_p

# def solve_tn(t_n: float, y_n: np.ndarray, dt: float, Phi: callable, boundary_conditions: callable, method: callable, params: dict) -> tuple[np.ndarray, np.ndarray]:
def solve_tn(t_n: float, y_n: np.ndarray, p: np.ndarray, dt: float, Phi: callable, boundary_conditions: callable, method: callable, params: dict, show_fp_iter: bool = False) -> tuple[np.ndarray, np.ndarray]:

    """
    Solve the PDE system for a single time step.

    Parameters
    ----------
    t_n : float
        Current time.
    y_n : numpy.ndarray (4, Ny, Nx-1)
        Array with the current solution.
    dt : float
        Time step.
    method : function
        Function that computes the next solution given the current one.
    params : dict
        Dictionary with the parameters of the PDE system.

    Returns
    -------
    y_np1 : np.ndarray (4, Ny, Nx-1)
        Array with the solution at the next time step.
    p : np.ndarray (Ny, Nx-1)
        Array with the pressure solution.

    Notes
    -----
    This function solves the PDE system for a single time step using the given
    method and parameters. It first computes the next solution using the given
    method, and then solves the pressure problem and applies a velocity correction
    using Chorin's projection method. Finally, it updates the boundary conditions
    and returns the solution at the next time step and the pressure solution.
    """
    # Get parameters
    T_mask = params['T_mask']
    t_source = params['t_source']
    T_source = params['T_source']
    bound = params['bound']
    T_min, T_max = params['T_min'], params['T_max']
    Y_min, Y_max = params['Y_min'], params['Y_max']
    # Density constant
    density_constant = params['density']
    # Poisson solver parameters
    max_iter = params['max_iter']
    tol = params['tol']
    log_fp = params['log_solver']
    # Solve time step 
    # if t_n <= 5:
    #     method = RK4
    # else:
    #     method = euler
    y_np1 = method(t_n, y_n, dt, Phi, params)
    # Solve Pressure problem
    if density_constant:
        p = solve_pressure(tuple(y_np1[:-2]), y_np1[-2], None, params)
    else:
        for i in range(max_iter):
            p_tmp = p.copy()
            p = solve_pressure(tuple(y_np1[:-2]), y_np1[-2], p, params)            
            if log_fp and show_fp_iter:
                l2_norm = np.linalg.norm(p.flatten() - p_tmp.flatten())
                l_inf_norm = np.linalg.norm(p.flatten() - p_tmp.flatten(), np.inf)
                print(f"Fixed-point iteration: {i}, L2: {l2_norm:.2e}, L-inf: {l_inf_norm:.2e}")
            if np.linalg.norm(p.flatten() - p_tmp.flatten(), np.inf) < tol:
                break
    grad_p = grad_pressure(p, params)
    # Velocity correction (Chorin's projection method)
    # T_inf = params['T_inf']
    # rho_v = rho_0 * T_inf / y_np1[-2]
    rho_v = rho(y_np1[-2])
    y_np1[:-2] = y_np1[:-2] - dt / rho_v * grad_p
    # y_np1[:-2] = y_np1[:-2] - dt / rho * grad_p
    # Update boundary conditions
    y_np1 = boundary_conditions(*y_np1, params)
    if bound:
        # Bound temperature
        y_np1[-2, y_np1[-2] < T_min] = T_min
        y_np1[-2, y_np1[-2] > T_max] = T_max
        # Bound mass fraction
        y_np1[-1, y_np1[-1] < Y_min] = Y_min 
        y_np1[-1, y_np1[-1] > Y_max] = Y_max 
    # Add temperature source if needed (permanent source up to t_source)
    if t_n <= t_source:
        y_np1[-2, T_mask] = T_source[T_mask]
    return y_np1, p

def euler(t_n: float, y_n: np.ndarray, dt: float, Phi: callable, params: dict) -> np.ndarray:
    """
    Implements the Euler method  step for solving a system of ordinary differential equations.

    Parameters
    ----------
    t_n : float
        The current time.
    y_n : numpy.ndarray (4, Ny, Nx)
        The current state of the system.
    dt : float
        The time step.
    params : dict
        A dictionary containing any additional parameters needed for the system.

    Returns
    -------
    numpy.ndarray (4, Ny, Nx)
        The updated state of the system at time t_n + dt.
    """
    y_np1 = y_n + dt * Phi(t_n, y_n, params)
    return y_np1

def RK2(t_n: float, y_n: np.ndarray, dt: float, Phi: callable, params: dict) -> np.ndarray:
    """
    Second-order Runge-Kutta method step for solving ordinary differential equations.

    Parameters
    ----------
    t_n : float
        Current time.
    y_n : np.ndarray (4, Ny, Nx)
        Current state vector.
    dt : float
        Time step.
    params : dict
        Dictionary containing parameters needed to evaluate the derivative function.

    Returns
    -------
    np.ndarray (4, Ny, Nx)
        State vector at the next time step.
    """
    k1 = Phi(t_n, y_n, params)
    k2 = Phi(t_n + dt, y_n + dt * k1, params)
    y_np1 = y_n + 0.5 * dt * (k1 + k2)
    return y_np1

def RK4(t_n: float, y_n: np.ndarray, dt: float, Phi: callable, params: dict) -> np.ndarray:
    """
    Fourth-order Runge-Kutta method step for solving ordinary differential equations.

    Parameters
    ----------
    t_n : float
        Current time.
    y_n : np.ndarray (4, Ny, Nx)
        Current state vector.
    dt : float
        Time step.
    params : dict
        Dictionary containing parameters needed to evaluate the derivative function.

    Returns
    -------
    np.ndarray (4, Ny, Nx)
        State vector at the next time step.
    """
    k1 = Phi(t_n, y_n, params)
    k2 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k1, params)
    k3 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k2, params)
    k4 = Phi(t_n + dt, y_n + dt * k3, params)
    y_np1 = y_n + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_np1

def data_post_processing(z: np.ndarray, p: np.ndarray, params: dict) -> dict:
    """
    Post-processes the input data by concatenating the last column to simulate periodic boundary in x.

    Parameters
    ----------
    z : numpy.ndarray (Nt, 4, Ny-1, Nx-1) or (Nt, 5, Ny-1, Nx-1, Nz)
        Variables u, v, (w), T, Y, without periodic BC.
    p : numpy.ndarray (Nt, Ny-1, Nx-1) or (Nt, Ny-1, Nx-1, Nz)
        Pressure without periodic BC.

    Returns
    -------
    dict
        Dictionary containing the post-processed data arrays u, v, (w), T, Y, and p.
        Each array has shape (Nt, Ny, Nx) or (Nt, Ny, Nx, Nz).

    """
    # Get ndims
    ndims = z.ndim
    if ndims == 4: # 2D case
        # Get data
        u, v, T, Y = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
        if params['experiment'] == 'cylinder':
            Nt, Ny, Nx = u.shape
            u_ = np.zeros((Nt, Ny+1, Nx+1))
            v_ = np.zeros((Nt, Ny+1, Nx+1))
            w_ = np.zeros((Nt, Ny+1, Nx+1))
            T_ = np.zeros((Nt, Ny+1, Nx+1))
            Y_ = np.zeros((Nt, Ny+1, Nx+1))
            p_ = np.zeros((Nt, Ny+1, Nx+1))
            # Copy data
            u_[:,:-1,:-1] = u
            u_[:,-1,:-1] = u[:, 0, :]
            u_[:, :, -1] = u_[:, :, 0]
            v_[:,:-1,:-1] = v
            v_[:,-1, :-1] = v[:, 0, :]
            v_[:, :, -1] = v_[:, :, 0]
            T_[:,:-1,:-1] = T
            T_[:,-1, :-1] = T[:, 0, :]
            T_[:, :, -1] = T_[:, :, 0]
            Y_[:,:-1,:-1] = Y
            Y_[:,-1, :-1] = Y[:, 0, :]
            Y_[:, :, -1] = Y_[:, :, 0]
            p_[:,:-1,:-1] = p
            p_[:,-1, :-1] = p[:, 0, :]
            p_[:, :, -1] = p_[:, :, 0]
            u, v, T, Y, p = u_, v_, T_, Y_, p_
        else:
            # Concatenate last column (periodic boundary in x)
            u = np.concatenate((u, u[:, :, 0].reshape(u.shape[0], u.shape[1], 1)), axis=2)
            v = np.concatenate((v, v[:, :, 0].reshape(v.shape[0], v.shape[1], 1)), axis=2)
            T = np.concatenate((T, T[:, :, 0].reshape(T.shape[0], T.shape[1], 1)), axis=2)
            Y = np.concatenate((Y, Y[:, :, 0].reshape(Y.shape[0], Y.shape[1], 1)), axis=2)
            p = np.concatenate((p, p[:, :, 0].reshape(p.shape[0], p.shape[1], 1)), axis=2)
        data = {
            'u': u,
            'v': v,
            'T': T,
            'Y': Y,
            'p': p
        }
    elif ndims == 5: # 3D case
        u_, v_, w_, T_, Y_ = z[:, 0], z[:, 1], z[:, 2], z[:, 3], z[:, 4]
        p_ = p.copy()
        # Get dimensions
        Nt, Ny, Nx, Nz = u_.shape
        u = np.zeros((Nt, Ny+1, Nx+1, Nz))
        v = np.zeros((Nt, Ny+1, Nx+1, Nz))
        w = np.zeros((Nt, Ny+1, Nx+1, Nz))
        T = np.zeros((Nt, Ny+1, Nx+1, Nz))
        Y = np.zeros((Nt, Ny+1, Nx+1, Nz))
        p = np.zeros((Nt, Ny+1, Nx+1, Nz))
        # Copy data
        u[:,:-1,:-1,:] = u_
        u[:,-1,:-1, :] = u_[:, 0, :, :]
        u[:, :, -1, :] = u[:, :, 0, :]
        v[:,:-1,:-1,:] = v_
        v[:,-1, :-1,:] = v_[:, 0, :, :]
        v[:, :, -1, :] = v[:, :, 0, :]
        w[:,:-1,:-1,:] = w_
        w[:,-1, :-1,:] = w_[:, 0, :, :]
        w[:, :, -1, :] = w[:, :, 0, :]
        T[:,:-1,:-1,:] = T_
        T[:,-1, :-1,:] = T_[:, 0, :, :]
        T[:, :, -1, :] = T[:, :, 0, :]
        Y[:,:-1,:-1,:] = Y_
        Y[:,-1, :-1,:] = Y_[:, 0, :, :]
        Y[:, :, -1, :] = Y[:, :, 0, :]
        p[:,:-1,:-1,:] = p_
        p[:, -1, :-1, :] = p_[:, 0, :, :]
        p[:, :, -1, :] = p[:, :, 0, :]
        data = {
            'u': u,
            'v': v,
            'w': w,
            'T': T,
            'Y': Y,
            'p': p,
        }
    return data

def Phi_2D(t: float, R: np.ndarray, params: dict) -> np.ndarray:
    """
    Computes the right-hand side of the partial differential equations (PDEs) for the 2D wildfire model.
    The PDEs are given by:

    Velocity:
    .. math::
        \dfrac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u} - (\mathbf{u}\cdot\nabla) \mathbf{u} + \mathbf{f}

    Temperature:
    .. math::
        \dfrac{\partial T}{\partial t} = \dfrac{\partial k(T)}{\partial T}||\nabla T||^2 + k(T)\nabla^2 T - (\mathbf{u}\cdot\nabla T) + S(T, Y)

    Combustion model:
    .. math::
        \dfrac{\partial Y}{\partial t} = -Y_f K(T) H(T) Y

    where \mathbf{u} is the velocity vector, \nu is the kinematic viscosity, \mathbf{f} is the external force vector,
    T is the temperature, Y is the fuel mass fraction, k(T) is the thermal conductivity, S(T, Y) is the source term
    for the temperature equation, Y_f is the fuel mass fraction of the wood, K(T) is the reaction rate coefficient,
    and H(T) is the heating value of the wood.

    Parameters
    ----------
    t : float
        Current time.
    R : array_like (4, Ny, Nx-1)
        Array of current values for the velocity components, temperature, and fuel mass fraction.
    params : dict
        Dictionary of parameters for the model.

    Returns
    -------
    numpy.ndarray (4, Ny, Nx-1)
        Array of the right-hand side values for the PDEs.

    Notes
    -----
    This function computes the right-hand side of the PDEs for the 2D wildfire model, which are used to update the
    velocity components, temperature, and fuel mass fraction at each time step. The PDEs are solved using a finite
    difference method.
    """
    dx, dy = params['dx'], params['dy']
    nu = params['nu']
    Y_f = params['Y_f']
    turb = params['turbulence']
    conservative = params['conservative']
    # Get variables
    u, v, T, Y = R
    # Forces
    F_x, F_y = f((u, v), T, Y)
    # Derivatives #
    # First partial derivatives 
    if conservative: # Conservative form for convection        
        uux = compute_first_derivative(u * u, dx, 1, (True, False)) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
        vuy = compute_first_derivative(u * v, dy, 0, (True, False)) # (u_{i, j+1} * v_{i, j+1} - u_{i, j-1} * v_{i, j-1}) / (2 * dy)
        uvx = compute_first_derivative(v * u, dx, 1, (True, False)) # (v_{i+1, j} * u_{i+1, j} - v_{i-1, j} * u_{i-1, j}) / (2 * dx)
        vvy = compute_first_derivative(v * v, dy, 0, (True, False)) # (v_{i, j+1}^2 - v_{i, j-1}^2) / (2 * dy)
    else: # Non-conservative form for convection        
        uux = compute_first_derivative_upwind(u, u, dx, 1) 
        vuy = compute_first_derivative_upwind(v, u, dy, 0, periodic=False)
        uvx = compute_first_derivative_upwind(u, v, dx, 1)
        vvy = compute_first_derivative_upwind(v, v, dy, 0, periodic=False)
    Tx, Ty = compute_gradient(T, (dx, dy), (True, False))
    uTx = compute_first_derivative_upwind(u, T, dx, 1)
    vTy = compute_first_derivative_upwind(v, T, dy, 0, periodic=False)
    # Second partial derivatives, compute Laplacian
    lap_u = compute_laplacian(u, (dx, dy), (True, False))
    lap_v = compute_laplacian(v, (dx, dy), (True, False))
    lap_T = compute_laplacian(T, (dx, dy), (True, False))
    # Turbulence
    sgs_x = sgs_y = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_T = turbulence((u, v), T, params)
    # PDE RHS
    # Velocity: \nu \nabla^2 \mathb{u} - (\mathbf{u}\cdot\nabla) \mathbf{u} + \mathbf{f}
    u_ = nu * lap_u - (uux + vuy) + F_x - sgs_x 
    v_ = nu * lap_v - (uvx + vvy) + F_y - sgs_y 
    # Temperature: \dfrac{\partial k(T)}{\partial T}||\nabla T||^2 + k(T)\nabla^2 T - (\mathbf{u}\cdot\nabla T) + S(T, Y) 
    T_ = kT(T) * (Tx ** 2 + Ty ** 2) + k(T) * lap_T - (u * Tx + v * Ty) + S(T, Y) - sgs_T 
    # T_ = kT(T) * (Tx ** 2 + Ty ** 2) + k(T) * lap_T - (uTx  + vTy) + S(T, Y) - sgs_T 
    # Combustion model: -Y_f K(T) H(T) Y
    Y_ = -Y_f * K(T) * H(T) * Y 
    # Boundary conditions
    u_, v_, T_, Y_ = boundary_conditions_2D(u_, v_, T_, Y_, params)
    return np.array([u_, v_, T_, Y_])

def Phi_3D(t: float, R: np.ndarray, params: dict) -> np.ndarray:
    """
    Computes the right-hand side of the partial differential equations (PDEs) for the 2D wildfire model.
    The PDEs are given by:

    Velocity:
    .. math::
        \dfrac{\partial \mathbf{u}}{\partial t} = \nu \nabla^2 \mathbf{u} - (\mathbf{u}\cdot\nabla) \mathbf{u} + \mathbf{f}

    Temperature:
    .. math::
        \dfrac{\partial T}{\partial t} = \dfrac{\partial k(T)}{\partial T}||\nabla T||^2 + k(T)\nabla^2 T - (\mathbf{u}\cdot\nabla T) + S(T, Y)

    Combustion model:
    .. math::
        \dfrac{\partial Y}{\partial t} = -Y_f K(T) H(T) Y

    where \mathbf{u} is the velocity vector, \nu is the kinematic viscosity, \mathbf{f} is the external force vector,
    T is the temperature, Y is the fuel mass fraction, k(T) is the thermal conductivity, S(T, Y) is the source term
    for the temperature equation, Y_f is the fuel mass fraction of the wood, K(T) is the reaction rate coefficient,
    and H(T) is the heating value of the wood.

    Parameters
    ----------
    t : float
        Current time.
    R : array_like (5, Ny-1, Nx-1, Nz)
        Array of current values for the velocity components, temperature, and fuel mass fraction.
    params : dict
        Dictionary of parameters for the model.

    Returns
    -------
    numpy.ndarray (5, Ny-1, Nx-1, Nz)
        Array of the right-hand side values for the PDEs.

    Notes
    -----
    This function computes the right-hand side of the PDEs for the 2D wildfire model, which are used to update the
    velocity components, temperature, and fuel mass fraction at each time step. The PDEs are solved using a finite
    difference method.
    """
    dx, dy, dz = params['dx'], params['dy'], params['dz']
    nu = params['nu']
    Y_f = params['Y_f']
    turb = params['turbulence']
    conservative = params['conservative']
    # Get variables
    u, v, w, T, Y = R
    # Forces
    F_x, F_y, F_z = f((u, v, w), T, Y)
    # Derivatives #
    # First partial derivatives 
    if conservative: # Conservative form for convection        
        pass
    else: # Non-conservative form for convection        
        uux = compute_first_derivative_upwind(u, u, dx, 1) 
        vuy = compute_first_derivative_upwind(v, u, dy, 0)
        wuz = compute_first_derivative_upwind(w, u, dz, 2, periodic=False)
        uvx = compute_first_derivative_upwind(u, v, dx, 1)
        vvy = compute_first_derivative_upwind(v, v, dy, 0)
        wvz = compute_first_derivative_upwind(w, v, dz, 2, periodic=False)
        uwx = compute_first_derivative_upwind(u, w, dx, 1)
        vwy = compute_first_derivative_upwind(v, w, dy, 0)
        wwz = compute_first_derivative_upwind(w, w, dz, 2, periodic=False)
    Tx, Ty, Tz = compute_gradient(T, (dx, dy, dz), (True, True, False))
    # Second partial derivatives, compute Laplacian
    lap_u = compute_laplacian(u, (dx, dy, dz), (True, True, False))
    lap_v = compute_laplacian(v, (dx, dy, dz), (True, True, False))
    lap_w = compute_laplacian(w, (dx, dy, dz), (True, True, False))
    lap_T = compute_laplacian(T, (dx, dy, dz), (True, True, False))
    # Turbulence
    sgs_x = sgs_y = sgs_z = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_z, sgs_T = turbulence((u, v, w), T, params)
    # np.savez(params['save_path'] + 'sgs.npz', sgs_x=sgs_x, sgs_y=sgs_y, sgs_z=sgs_z, sgs_T=sgs_T)
    # PDE RHS
    # Y_D = 1.0
    # a_v = 1.0
    # F_x = - Y_D * a_v
    # Velocity: \nu \nabla^2 \mathb{u} - (\mathbf{u}\cdot\nabla) \mathbf{u} + \mathbf{f}
    # u_ = nu * lap_u - (uux + vuy + wuz) + F_x - sgs_x
    # v_ = nu * lap_v - (uvx + vvy + wvz) + F_y - sgs_y
    # w_ = nu * lap_w - (uwx + vwy + wwz) + F_z - sgs_z
    # Temperature: \dfrac{\partial k(T)}{\partial T}||\nabla T||^2 + k(T)\nabla^2 T - (\mathbf{u}\cdot\nabla T) + S(T, Y) 
    #T_ = kT(T) * (Tx ** 2 + Ty ** 2 + Tz ** 2) + k(T) * lap_T - (u * Tx  + v * Ty + w * Tz) + S(T, Y) - sgs_T 
    # F_z = 1.0
    u_ = nu * lap_u - (uux + vuy + wuz) + F_x - sgs_x
    v_ = nu * lap_v - (uvx + vvy + wvz) + F_y - sgs_y
    w_ = nu * lap_w - (uwx + vwy + wwz) + F_z - sgs_z
    # Temperature: \dfrac{\partial k(T)}{\partial T}||\nabla T||^2 + k(T)\nabla^2 T - (\mathbf{u}\cdot\nabla T) + S(T, Y) 
    #T_ = kT(T) * (Tx ** 2 + Ty ** 2 + Tz ** 2) + k(T) * lap_T - (u * Tx  + v * Ty + w * Tz) #+ S(T, Y)*0 - sgs_T 
    T_ = params['alpha'] * lap_T - (u * Tx + v * Ty + w * Tz) + S(T, Y) - sgs_T
    # Combustion model: -Y_f K(T) H(T) Y
    Y_ = -Y_f * K(T) * H(T) * Y 
    # Boundary conditions
    u_, v_, w_, T_, Y_ = boundary_conditions_3D(u_, v_, w_, T_, Y_, params)
    return np.array([u_, v_, w_, T_, Y_])

def boundary_conditions_2D_fire(u: np.ndarray, v: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply boundary conditions to the input variables.

    Parameters
    ----------
    u : numpy.ndarray (Ny, Nx-1)
        Velocity in the x direction.
    v : numpy.ndarray (Ny, Nx-1)
        Velocity in the y direction.
    T : numpy.ndarray (Ny, Nx-1)
        Temperature.
    Y : np.ndarray (Ny, Nx)
        Mass fraction of fuel.
    params : dict
        Dictionary containing the following parameters:
        - T_inf : float
            Temperature at infinity.
        - bc_on_y : list
            List containing the boundary conditions for each variable.
        - cut_nodes : tuple
            Tuple containing the indices of the cut nodes.
        - dead_nodes : numpy.ndarray
            Array containing the indices of the dead nodes.
        - values_dead_nodes : tuple
            Tuple containing the values of the dead nodes.

    Returns
    -------
    numpy.ndarray (4, Ny, Nx-1)
        Array containing the input variables with the applied boundary conditions.
    """
    T_inf = params['T_inf']
    bc_on_y = params['bc_on_z'] # Boundary conditions (for Dirichlet)
    u_y_min, u_y_max = bc_on_y[0]
    v_y_min, v_y_max = bc_on_y[1]
    T_y_min, T_y_max = bc_on_y[2]
    Y_y_min, Y_y_max = bc_on_y[3]
    cut_nodes = params['cut_nodes']
    cut_nodes_y, cut_nodes_x = cut_nodes # For FD in BC
    dead_nodes = params['dead_nodes']
    u_dn, v_dn, T_dn, Y_dn = params['dead_nodes_values']
    # Boundary conditions on x: Nothing to do because Phi includes them
    # Boundary conditions on y 
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    # Assume Dirichlet boundary conditions
    u_s, v_s, T_s, Y_s, u_n, v_n, T_n, Y_n = u_y_min, v_y_min, T_y_min, Y_y_min, u_y_max, v_y_max, T_inf, Y_y_max
    # Neumann boundary at south. Derivatives using O(dy^2) 
    T_s = (4 * T[1, :] - T[2, :]) / 3 # dT/dy = 0
    Y_s = (4 * Y[1, :] - Y[2, :]) / 3 # dY/dy = 0
    # Neumann boundary at north. Derivatives using O(dy^2)
    u_n = (4 * u[-2, :] - u[-3, :]) / 3 # du/dy = 0
    v_n = (4 * v[-2, :] - v[-3, :]) / 3 # dv/dy = 0
    T_n = (4 * T[-2, :] - T[-3, :]) / 3 # dT/dy = 0
    Y_n = (4 * Y[-2, :] - Y[-3, :]) / 3 # dY/dy = 0
    # Boundary conditions on y=y_min
    u[0] = u_s
    v[0] = v_s
    T[0] = T_s 
    Y[0] = Y_s
    # Boundary conditions on y=y_max
    u[-1] = u_n
    v[-1] = v_n
    T[-1] = T_n
    Y[-1] = Y_n
    # IBM implementation #
    # Boundary at edge nodes
    T_s = (4 * T[cut_nodes_y + 1, cut_nodes_x] - T[cut_nodes_y + 2, cut_nodes_x]) / 3 # Derivative using O(h^2)	
    Y_s = (4 * Y[cut_nodes_y + 1, cut_nodes_x] - Y[cut_nodes_y + 2, cut_nodes_x]) / 3 # Derivative using O(h^2)
    # Boundary on cut nodes
    u[cut_nodes] = u_s
    v[cut_nodes] = v_s
    T[cut_nodes] = T_s
    Y[cut_nodes] = Y_s
    # Dead nodes
    u[dead_nodes] = u_dn
    v[dead_nodes] = v_dn
    T[dead_nodes] = T_dn
    Y[dead_nodes] = Y_dn
    # Return variables with boundary conditions
    return np.array([u, v, T, Y])

def boundary_conditions_2D_cavity(u: np.ndarray, v: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply boundary conditions to the input variables.

    Parameters
    ----------
    u : numpy.ndarray (Ny, Nx-1)
        Velocity in the x direction.
    v : numpy.ndarray (Ny, Nx-1)
        Velocity in the y direction.
    T : numpy.ndarray (Ny, Nx-1)
        Temperature.
    Y : np.ndarray (Ny, Nx)
        Mass fraction of fuel.
    params : dict
        Dictionary containing the following parameters:
        - T_inf : float
            Temperature at infinity.
        - bc_on_y : list
            List containing the boundary conditions for each variable.
        - cut_nodes : tuple
            Tuple containing the indices of the cut nodes.
        - dead_nodes : numpy.ndarray
            Array containing the indices of the dead nodes.
        - values_dead_nodes : tuple
            Tuple containing the values of the dead nodes.

    Returns
    -------
    numpy.ndarray (4, Ny, Nx-1)
        Array containing the input variables with the applied boundary conditions.
    """
    T_inf = params['T_inf']
    bc_on_y = params['bc_on_z'] # Boundary conditions (for Dirichlet)
    u_y_min, u_y_max = bc_on_y[0]
    v_y_min, v_y_max = bc_on_y[1]
    T_y_min, T_y_max = bc_on_y[2]
    Y_y_min, Y_y_max = bc_on_y[3]
    cut_nodes = params['cut_nodes']
    # cut_nodes_y, cut_nodes_x = cut_nodes # For FD in BC
    dead_nodes = params['dead_nodes']
    u_dn, v_dn, T_dn, Y_dn = params['dead_nodes_values']
    # Boundary conditions on x: Nothing to do because Phi includes them
    # Boundary conditions on y 
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    # Assume Dirichlet boundary conditions
    u_s, v_s, T_s, Y_s, u_n, v_n, T_n, Y_n = u_y_min, v_y_min, T_y_min, Y_y_min, u_y_max, v_y_max, T_y_max, Y_y_max
    if params['experiment'] == 'cavity_diff':
        # Neumann boundary at south. Derivatives using O(dy^2) 
        T_s = (4 * T[1, :] - T[2, :]) / 3 # dT/dy = 0
        # Y_s = (4 * Y[1, :] - Y[2, :]) / 3 # dY/dy = 0
        # # Neumann boundary at north. Derivatives using O(dy^2)
        T_n = (4 * T[-2, :] - T[-3, :]) / 3 # dT/dy = 0
        # Y_n = (4 * Y[-2, :] - Y[-3, :]) / 3 # dY/dy = 0
    # Boundary conditions on y=y_min
    u[0] = u_s
    v[0] = v_s
    T[0] = T_s 
    Y[0] = Y_s
    # Boundary conditions on y=y_max
    u[-1] = u_n
    v[-1] = v_n
    T[-1] = T_n
    Y[-1] = Y_n
    # IBM implementation #
    # Boundary on cut nodes
    # print(u_s.shape, u[cut_nodes].shape)
    # print(cut_nodes)
    # u[cut_nodes] = u_s
    # v[cut_nodes] = v_s
    # T[cut_nodes] = T_s
    # Y[cut_nodes] = Y_s
    # # Dead nodes
    u[dead_nodes] = u_dn
    v[dead_nodes] = v_dn
    T[dead_nodes] = T_dn
    Y[dead_nodes] = Y_dn
    # Return variables with boundary conditions
    return np.array([u, v, T, Y])

def boundary_conditions_2D_periodic(u: np.ndarray, v: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply boundary conditions to the input variables.

    Parameters
    ----------
    u : numpy.ndarray (Ny, Nx-1)
        Velocity in the x direction.
    v : numpy.ndarray (Ny, Nx-1)
        Velocity in the y direction.
    T : numpy.ndarray (Ny, Nx-1)
        Temperature.
    Y : np.ndarray (Ny, Nx)
        Mass fraction of fuel.
    params : dict
        Dictionary containing the following parameters:
        - T_inf : float
            Temperature at infinity.
        - bc_on_y : list
            List containing the boundary conditions for each variable.
        - cut_nodes : tuple
            Tuple containing the indices of the cut nodes.
        - dead_nodes : numpy.ndarray
            Array containing the indices of the dead nodes.
        - values_dead_nodes : tuple
            Tuple containing the values of the dead nodes.

    Returns
    -------
    numpy.ndarray (4, Ny, Nx-1)
        Array containing the input variables with the applied boundary conditions.
    """
    # cut_nodes = params['cut_nodes']
    dead_nodes = params['dead_nodes']
    u_dn, v_dn, T_dn, Y_dn = params['dead_nodes_values']
    # Boundary conditions on x: Nothing to do because Phi includes them
    # IBM implementation #
    # Boundary on cut nodes
    # print(u_s.shape, u[cut_nodes].shape)
    # print(cut_nodes)
    # u[cut_nodes] = u_s
    # v[cut_nodes] = v_s
    # T[cut_nodes] = T_s
    # Y[cut_nodes] = Y_s
    # # Dead nodes
    u[dead_nodes] = u_dn
    v[dead_nodes] = v_dn
    T[dead_nodes] = T_dn
    Y[dead_nodes] = Y_dn
    # Return variables with boundary conditions
    return np.array([u, v, T, Y])

def boundary_conditions_2D(u: np.ndarray, v: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict):
    experiment = params['experiment']
    if experiment == 'fire':
        return boundary_conditions_2D_fire(u, v, T, Y, params)
    elif experiment == 'cavity' or experiment == 'cavity_diff':
        return boundary_conditions_2D_cavity(u, v, T, Y, params)
    elif experiment == 'cylinder':
        return boundary_conditions_2D_periodic(u, v, T, Y, params)

def boundary_conditions_3D(u: np.ndarray, v: np.ndarray, w: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply boundary conditions to the input variables.

    Parameters
    ----------
    u : numpy.ndarray (Ny-1, Nx-1, Nz)
        Velocity in the x direction.
    v : numpy.ndarray (Ny-1, Nx-1, Nz)
        Velocity in the y direction.
    w : numpy.ndarray (Ny-1, Nx-1, Nz)
        Velocity in the z direction.
    T : numpy.ndarray (Ny-1, Nx-1, Nz)
        Temperature.
    Y : np.ndarray (Ny-1, Nx-1, Nz)
        Mass fraction of fuel.
    params : dict
        Dictionary containing the following parameters:
        - T_inf : float
            Temperature at infinity.
        - bc_on_z : list
            List containing the boundary conditions for each variable.
        - cut_nodes : tuple
            Tuple containing the indices of the cut nodes.
        - dead_nodes : numpy.ndarray
            Array containing the indices of the dead nodes.
        - values_dead_nodes : tuple
            Tuple containing the values of the dead nodes.

    Returns
    -------
    numpy.ndarray (5, Ny-1, Nx-1, Nz)
        Array containing the input variables with the applied boundary conditions.
    """
    T_inf = params['T_inf']
    bc_on_z = params['bc_on_z'] # Boundary conditions (for Dirichlet)
    u_z_min, u_z_max = bc_on_z[0]
    v_z_min, v_z_max = bc_on_z[1]
    w_z_min, w_z_max = bc_on_z[2]
    T_z_min, T_z_max = bc_on_z[3]
    Y_z_min, Y_z_max = bc_on_z[4]
    cut_nodes = params['cut_nodes']
    cut_nodes_y, cut_nodes_x, cut_nodes_z = cut_nodes # For FD in BC
    dead_nodes = params['dead_nodes']
    u_dn, v_dn, w_dn, T_dn, Y_dn = params['dead_nodes_values']
    # Boundary conditions on x: Nothing to do because Phi includes them
    # Boundary conditions on y 
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    # Assume Dirichlet boundary conditions
    u_s, v_s, w_s, T_s, Y_s, u_n, v_n, w_n, T_n, Y_n = u_z_min, v_z_min, w_z_min, T_z_min, Y_z_min, u_z_max, v_z_max, w_z_max, T_inf, Y_z_max
    # Neumann boundary at south. Derivatives using O(dz^2) 
    T_s = (4 * T[:, :, 1] - T[:, :, 2]) / 3 # dT/dz = 0
    Y_s = (4 * Y[:, :, 1] - Y[:, :, 2]) / 3 # dY/dz = 0
    # Neumann boundary at north. Derivatives using O(dz^2)
    u_n = (4 * u[:, :, -2] - u[:, :, -3]) / 3 # du/dz = 0
    v_n = (4 * v[:, :, -2] - v[:, :, -3]) / 3 # dv/dz = 0
    w_n = (4 * w[:, :, -2] - w[:, :, -3]) / 3 # dw/dz = 0
    T_n = (4 * T[:, :, -2] - T[:, :, -3]) / 3 # dT/dz = 0
    Y_n = (4 * Y[:, :, -2] - Y[:, :, -3]) / 3 # dY/dz = 0
    # Boundary conditions on z=z_min
    u[:,:,0] = u_s
    v[:,:,0] = v_s
    w[:,:,0] = w_s
    T[:,:,0] = T_s 
    Y[:,:,0] = Y_s
    # Boundary conditions on z=z_max
    u[:,:,-1] = u_n
    v[:,:,-1] = v_n
    w[:,:,-1] = w_n
    T[:,:,-1] = T_n
    Y[:,:,-1] = Y_n
    # IBM implementation #
    # # Boundary at edge nodes
    # T_s = (4 * T[cut_nodes_y, cut_nodes_x, cut_nodes_z + 1] - T[cut_nodes_y, cut_nodes_x, cut_nodes_z + 2]) / 3 # Derivative using O(h^2)	
    # Y_s = (4 * Y[cut_nodes_y, cut_nodes_x, cut_nodes_z + 1] - Y[cut_nodes_y, cut_nodes_x, cut_nodes_z + 2]) / 3 # Derivative using O(h^2)
    # # Boundary on cut nodes
    # # u[cut_nodes] = u_s
    # print(T_s.shape)
    # print(u[cut_nodes].shape, u_s.shape)
    # u[cut_nodes_y, cut_nodes_x, cut_nodes_z] = u_s
    # v[cut_nodes] = v_s
    # w[cut_nodes] = w_s
    # T[cut_nodes] = T_s
    # Y[cut_nodes] = Y_s
    # Dead nodes
    # u[dead_nodes] = u_dn
    # v[dead_nodes] = v_dn
    # w[dead_nodes] = w_dn
    # T[dead_nodes] = T_dn
    # Y[dead_nodes] = Y_dn
    # Return variables with boundary conditions
    return np.array([u, v, w, T, Y])

def solve_pde_2D(r_0: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves a partial differential equation (PDE) using the specified method.

    Parameters
    ----------
    r_0 : numpy.ndarray (4, Ny, Nx-1)
        Initial conditions for the PDE.
    params : dict
        Dictionary containing the parameters for the PDE solver. The dictionary
        should contain the following keys:
        - 'Nx': Number of grid points in the x direction.
        - 'Ny': Number of grid points in the y direction.
        - 'Nt': Number of time steps.
        - 'dx': Grid spacing in the x direction.
        - 'dy': Grid spacing in the y direction.
        - 'dt': Time step size.
        - 'NT': Number of time steps to save.
        - 't': Array of time values.
        - 'method': Method to use for solving the PDE. Must be one of 'euler' or 'RK4'.
        - 'save_path': Path to save the log file.

    Returns
    -------
    tuple
        A tuple containing the solution to the PDE and the corresponding pressure field.
    """
    Nx, Ny, Nt = params['Nx'], params['Ny'], params['Nt']
    dx, dy, dt = params['dx'], params['dy'], params['dt']
    NT = params['NT']
    t = params['t']
    method = params['method']
    methods = {'euler': euler, 'RK2': RK2, 'RK4': RK4}
    log_file = open(params['save_path'] + "log.txt", "w")
    solver_time_start = time.time()
    Nx -= 1
    if params['experiment'] == 'cylinder':
        Ny -= 1
    if NT == 1: # Save all time steps
        # Approximation
        z = np.zeros((Nt+1, r_0.shape[0], Ny, Nx)) 
        p = np.zeros((Nt+1, Ny, Nx))
        z[0] = r_0
        for n in range(Nt):
            # Simulation 
            step_time_start = time.time()            
            # z[n+1], p[n+1] = solve_tn(t[n], z[n], dt, Phi_2D, boundary_conditions_2D, methods[method], params)
            z[n+1], p[n+1] = solve_tn(t[n], z[n], p[n], dt, Phi_2D, boundary_conditions_2D, methods[method], params)
            step_time_end = time.time()
            elapsed_time = (step_time_end - step_time_start)
            # Print log
            CFL = dt * (np.max(np.abs(z[n+1, 0])) / dx + np.max(np.abs(z[n+1, 1])) / dy)
            T_min, T_max = np.min(z[n+1, 2]), np.max(z[n+1, 2])
            Y_min, Y_max = np.min(z[n+1, 3]), np.max(z[n+1, 3])
            # Show/print log
            log_time_step(log_file, n+1, t[n+1], CFL, T_min, T_max, Y_min, Y_max, elapsed_time)
    else: # Save every NT steps
        # Approximation
        z = np.zeros((Nt // NT + 1, r_0.shape[0], Ny, Nx)) 
        p  = np.zeros((Nt // NT + 1, Ny, Nx))
        z[0] = r_0
        z_tmp = z[0].copy()
        p_tmp = p[0].copy()
        for n in range(Nt):
            # Simulation 
            step_time_start = time.time()
            # z_tmp, p_tmp = solve_tn(t[n], z_tmp, dt, Phi_2D, boundary_conditions_2D, methods[method], params)
            z_tmp, p_tmp = solve_tn(t[n], z_tmp, p_tmp, dt, Phi_2D, boundary_conditions_2D, methods[method], params, ((n+1) % NT == 0 or n == (Nt - 1)))
            step_time_end = time.time()
            step_elapsed_time = (step_time_end - step_time_start)
            if (n+1) % NT == 0 or n == (Nt - 1): # Save every NT steps and last step
                z[n // NT + 1], p[n // NT + 1] = z_tmp, p_tmp
                # Print log
                CFL = dt * (np.max(np.abs(z_tmp[0])) / dx + np.max(np.abs(z_tmp[1])) / dy)  # Compute CFL
                T_min, T_max = np.min(z_tmp[2]), np.max(z_tmp[2])
                Y_min, Y_max = np.min(z_tmp[3]), np.max(z_tmp[3]) 
                log_time_step(log_file, n+1, t[n+1], CFL, T_min, T_max, Y_min, Y_max, step_elapsed_time)
    solver_time_end = time.time()
    solver_time = (solver_time_end - solver_time_start)
    print("\nSolver time: ", str(timedelta(seconds=round(solver_time))), "\n")
    print("\nSolver time: ", str(timedelta(seconds=round(solver_time))), "\n", file=log_file)
    # Close log file
    log_file.close()
    return data_post_processing(z, p, params)

def solve_pde_3D(r_0: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves a partial differential equation (PDE) using the specified method.

    Parameters
    ----------
    r_0 : numpy.ndarray (5, Ny, Nx-1)
        Initial conditions for the PDE.
    params : dict
        Dictionary containing the parameters for the PDE solver. The dictionary
        should contain the following keys:
        - 'Nx': Number of grid points in the x direction.
        - 'Ny': Number of grid points in the y direction.
        - 'Nz': Number of grid points in the z direction.
        - 'Nt': Number of time steps.
        - 'dx': Grid spacing in the x direction.
        - 'dy': Grid spacing in the y direction.
        - 'dz': Grid spacing in the z direction.
        - 'dt': Time step size.
        - 'NT': Number of time steps to save.
        - 't': Array of time values.
        - 'method': Method to use for solving the PDE. Must be one of 'euler' or 'RK4'.
        - 'save_path': Path to save the log file.

    Returns
    -------
    tuple
        A tuple containing the solution to the PDE and the corresponding pressure field.
    """
    Nx, Ny, Nz, Nt = params['Nx'], params['Ny'], params['Nz'], params['Nt']
    dx, dy, dz, dt = params['dx'], params['dy'], params['dz'], params['dt']
    NT = params['NT']
    t = params['t']
    method = params['method']
    methods = {'euler': euler, 'RK2': RK2, 'RK4': RK4}
    log_file = open(params['save_path'] + "log.txt", "w")
    solver_time_start = time.time()
    if NT == 1: # Save all time steps
        # Approximation
        z = np.zeros((Nt+1, r_0.shape[0], Ny - 1, Nx - 1, Nz), dtype=np.float64) 
        p = np.zeros((Nt+1, Ny - 1, Nx - 1, Nz))
        z[0] = r_0
        for n in range(Nt):
            # Simulation 
            step_time_start = time.time()
            z[n+1], p[n+1] = solve_tn(t[n], z[n], p[n], dt, Phi_3D, boundary_conditions_3D, methods[method], params)
            step_time_end = time.time()
            elapsed_time = (step_time_end - step_time_start)
            # Print log
            CFL = dt * (np.max(np.abs(z[n+1, 0])) / dx + np.max(np.abs(z[n+1, 1])) / dy + np.max(np.abs(z[n+1, 2])) / dz)  # Compute CFL
            T_min, T_max = np.min(z[n+1, 3]), np.max(z[n+1, 3])
            Y_min, Y_max = np.min(z[n+1, 4]), np.max(z[n+1, 4])
            log_time_step(log_file, n+1, t[n+1], CFL, T_min, T_max, Y_min, Y_max, elapsed_time)
    else: # Save every NT steps
        # Approximation
        z = np.zeros((Nt // NT + 1, r_0.shape[0], Ny - 1, Nx - 1, Nz)) 
        p  = np.zeros((Nt // NT + 1, Ny - 1, Nx - 1, Nz))
        z[0] = r_0
        z_tmp = z[0].copy()
        p_tmp = p[0].copy()
        for n in range(Nt):
            # Simulation 
            step_time_start = time.time()
            z_tmp, p_tmp = solve_tn(t[n], z_tmp, p_tmp, dt, Phi_3D, boundary_conditions_3D, methods[method], params)
            step_time_end = time.time()
            step_elapsed_time = (step_time_end - step_time_start)
            if (n+1) % NT == 0 or n == (Nt - 1): # Save every NT steps and last step
                z[n // NT + 1], p[n // NT + 1] = z_tmp, p_tmp
                # Print log
                CFL = dt * (np.max(np.abs(z_tmp[0])) / dx + np.max(np.abs(z_tmp[1])) / dy + np.max(np.abs(z_tmp[2])) / dz)  # Compute CFL
                T_min, T_max = np.min(z_tmp[3]), np.max(z_tmp[3])
                Y_min, Y_max = np.min(z_tmp[4]), np.max(z_tmp[4]) 
                log_time_step(log_file, n+1, t[n+1], CFL, T_min, T_max, Y_min, Y_max, step_elapsed_time)
    solver_time_end = time.time()
    solver_time = (solver_time_end - solver_time_start)
    print("\nSolver time: ", str(timedelta(seconds=round(solver_time))), "\n")
    print("\nSolver time: ", str(timedelta(seconds=round(solver_time))), "\n", file=log_file)
    # Close log file
    log_file.close()
    return data_post_processing(z, p, params)