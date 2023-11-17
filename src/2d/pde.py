import time
import numpy as np
from datetime import timedelta
from derivatives import compute_gradient, compute_laplacian, compute_first_derivative_upwind, compute_first_derivative
from poisson import solve_pressure
from turbulence import turbulence
from utils import f, S, k, kT, K, H#, Km, hv, source, sink, #, Yft, S_T, AT
from plots import plot_2D

OUTPUT_LOG = "Time step: {:=6d}, Simulation time: {:.2f} s"

def grad_pressure(p: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the gradient of pressure.

    .. math::
        \nabla p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}\right)

    Parameters
    ----------
    p : numpy.ndarray (Ny, Nx-1)
        Pressure field.
    params : dict
        Dictionary containing the interval sizes `dx` and `dy`.

    Returns
    -------
    numpy.ndarray (2, Ny, Nx-1)
        Gradient of pressure.
    """
    # Get interval size
    dx, dy = params['dx'], params['dy']
    # Compute grad(p)
    grad_p = compute_gradient(p, dx, dy, (False, True))
    return grad_p

def Phi(t: float, C: np.ndarray, params: dict) -> np.ndarray:
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
    C : array_like (4, Ny, Nx-1)
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
    u, v, T, Y = C
    # Forces
    F_x, F_y = f((u, v), T, Y)
    # Derivatives #
    # First partial derivatives 
    if conservative: # Conservative form for convection        
        uux = compute_first_derivative(u * u, dx, 1, (False, True)) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
        vuy = compute_first_derivative(u * v, dy, 0, (False, True)) # (u_{i, j+1} * v_{i, j+1} - u_{i, j-1} * v_{i, j-1}) / (2 * dy)
        uvx = compute_first_derivative(v * u, dx, 1, (False, True)) # (v_{i+1, j} * u_{i+1, j} - v_{i-1, j} * u_{i-1, j}) / (2 * dx)
        vvy = compute_first_derivative(v * v, dy, 0, (False, True)) # (v_{i, j+1}^2 - v_{i, j-1}^2) / (2 * dy)
    else: # Non-conservative form for convection        
        uux = compute_first_derivative_upwind(u, u, dx, 1) 
        vuy = compute_first_derivative_upwind(v, u, dy, 0, periodic=False)
        uvx = compute_first_derivative_upwind(u, v, dx, 1)
        vvy = compute_first_derivative_upwind(v, v, dy, 0, periodic=False)
    Tx, Ty = compute_gradient(T, dx, dy, (False, True))
    # Second partial derivatives, compute Laplacian
    lap_u = compute_laplacian(u, dx, dy, (False, True))
    lap_v = compute_laplacian(v, dx, dy, (False, True))
    lap_T = compute_laplacian(T, dx, dy, (False, True))
    # Turbulence
    sgs_x = sgs_y = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_T = turbulence(u, v, T, params)
    # PDE RHS
    # Velocity: \nu \nabla^2 \mathb{u} - (\mathbf{u}\cdot\nabla) \mathbf{u} + \mathbf{f}
    # u_ = nu * lap_u + F_x - sgs_x - (uux + vuy) 
    # v_ = nu * lap_v + F_y - sgs_y - (uvx + vvy)
    u_ = nu * lap_u - (uux + vuy) + F_x - sgs_x 
    v_ = nu * lap_v - (uvx + vvy) + F_y - sgs_y 
    # u_, v_ = u_1, u_2
    # u1 = np.sum([nu * lap_u, - uux, - vuy, F_x, - sgs_x], axis=0)
    # v1 = np.sum([nu * lap_v, - uvx, - vvy, F_y, - sgs_y], axis=0)
    # u2 = np.sum([nu * lap_u, F_x, - sgs_x, - uux, - vuy], axis=0)
    # v2 = np.sum([nu * lap_v, F_y, - sgs_y, - uvx, - vvy], axis=0)
    # print("u", np.linalg.norm(u_1 - u1), np.linalg.norm(u_2 - u2))
    # print("v", np.linalg.norm(v_1 - v1), np.linalg.norm(v_2 - v2))
    # Temperature: \dfrac{\partial k(T)}{\partial T}||\nabla T||^2 + k(T)\nabla^2 T - (\mathbf{u}\cdot\nabla T) + S(T, Y) 
    T_ = kT(T) * (Tx ** 2 + Ty ** 2) + k(T) * lap_T - (u * Tx  + v * Ty) + S(T, Y) - sgs_T 
    # Combustion model: -Y_f K(T) H(T) Y
    Y_ = -Y_f * K(T) * H(T) * Y 
    # Boundary conditions
    u_, v_, T_, Y_ = boundary_conditions(u_, v_, T_, Y_, params)
    return np.array([u_, v_, T_, Y_])

def boundary_conditions(u: np.ndarray, v: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
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
    bc_on_y = params['bc_on_y'] # Boundary conditions (for Dirichlet)
    u_y_min, u_y_max = bc_on_y[0]
    v_y_min, v_y_max = bc_on_y[1]
    T_y_min, T_y_max = bc_on_y[2]
    Y_y_min, Y_y_max = bc_on_y[3]
    cut_nodes = params['cut_nodes']
    cut_nodes_y, cut_nodes_x = cut_nodes # For FD in BC
    dead_nodes = params['dead_nodes']
    u_dn, v_dn, T_dn, Y_dn = params['values_dead_nodes']
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

def solve_tn(t_n: float, y_n: np.ndarray, dt: float, method: callable, params: dict) -> tuple[np.ndarray, np.ndarray]:
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
    rho = params['rho']
    T_mask = params['T_mask']
    t_source = params['t_source']
    T_source = params['T_source']
    bound = params['bound']
    T_min, T_max = params['T_min'], params['T_max']
    Y_min, Y_max = params['Y_min'], params['Y_max']
    # Solve time step 
    y_np1 = method(t_n, y_n, dt, params)
    Ut, Vt = y_np1[:2].copy()
    # Solve Pressure problem
    p = solve_pressure(Ut, Vt, params)
    grad_p = grad_pressure(p, params)
    # Velocity correction (Chorin's projection method)
    y_np1[:2] = y_np1[:2] - dt / rho * grad_p
    Ut, Vt, Tt, Yt = y_np1.copy()
    # Update boundary conditions
    y_np1 = boundary_conditions(Ut, Vt, Tt, Yt, params)
    if bound:
        # Bound mass fraction
        y_np1[3, y_np1[3] < Y_min] = Y_min 
        y_np1[3, y_np1[3] > Y_max] = Y_max 
        # Bound temperature
        y_np1[2, y_np1[2] < T_min] = T_min
        y_np1[2, y_np1[2] > T_max] = T_max
    # Add temperature source if needed (permanent source up to t_source)
    if t_n <= t_source:
        y_np1[2, T_mask] = T_source[T_mask]
    return y_np1, p

def euler(t_n: float, y_n: np.ndarray, dt: float, params: dict) -> np.ndarray:
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

def RK4(t_n: float, y_n: np.ndarray, dt: float, params: dict) -> np.ndarray:
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
    y_np1 = y_n + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_np1

def data_post_processing(z: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Post-processes the input data by concatenating the last column to simulate periodic boundary in x.

    Parameters
    ----------
    z : numpy.ndarray (Nt, 4, Ny, Nx)
        Variables u, v, T, Y, without periodic BC.
    p : numpy.ndarray (Nt, Ny, Nx)
        Pressure without periodic BC.

    Returns
    -------
    tuple of numpy.ndarray
        Tuple containing the post-processed data arrays u, v, T, Y, and p.
        Each array has shape (Nt, Ny, Nx+1).

    """
    # Get data
    u, v, T, Y = z[:, 0], z[:, 1], z[:, 2], z[:, 3]
    # Concatenate last column (periodic boundary in x)
    u = np.concatenate((u, u[:, :, 0].reshape(u.shape[0], u.shape[1], 1)), axis=2)
    v = np.concatenate((v, v[:, :, 0].reshape(v.shape[0], v.shape[1], 1)), axis=2)
    T = np.concatenate((T, T[:, :, 0].reshape(T.shape[0], T.shape[1], 1)), axis=2)
    Y = np.concatenate((Y, Y[:, :, 0].reshape(Y.shape[0], Y.shape[1], 1)), axis=2)
    p = np.concatenate((p, p[:, :, 0].reshape(p.shape[0], p.shape[1], 1)), axis=2)
    return u, v, T, Y, p

def solve_pde(z_0: np.ndarray, params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves a partial differential equation (PDE) using the specified method.

    Parameters
    ----------
    z_0 : numpy.ndarray (4, Ny, Nx-1)
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
    methods = {'euler': euler, 'RK4': RK4}
    log_file = open(params['save_path'] + "log.txt", "w")
    solver_time_start = time.time()
    if NT == 1: # Save all time steps
        # Approximation
        z = np.zeros((Nt, z_0.shape[0], Ny, Nx - 1)) 
        p = np.zeros((Nt, Ny, Nx - 1))
        z[0] = z_0
        for n in range(Nt - 1):
            # Simulation 
            step_time_start = time.time()
            z[n+1], p[n+1] = solve_tn(t[n], z[n], dt, methods[method], params)
            step_time_end = time.time()
            elapsed_time = (step_time_end - step_time_start)
            # Print log
            CFL = dt * (np.max(np.abs(z[n+1, 0])) / dx + np.max(np.abs(z[n+1, 1])) / dy)
            T_min, T_max = np.min(z[n+1, 2]), np.max(z[n+1, 2])
            Y_min, Y_max = np.min(z[n+1, 3]), np.max(z[n+1, 3])
            print(OUTPUT_LOG.format(n, t[n]))
            print("CFL: {:.6f}".format(CFL))
            print("Temperature: Min = {:.2f} K, Max {:.2f} K".format(T_min, T_max))
            print("Fuel: Min = {:.2f}, Max {:.2f}".format(Y_min, Y_max))
            print("Step time: {:.6f} s".format(elapsed_time))
            # Save log to file
            print(OUTPUT_LOG.format(n, t[n]), file=log_file)
            print("CFL: {:.6f}".format(CFL), file=log_file)
            print("Temperature: Min = {:.2f} K, Max {:.2f} K".format(T_min, T_max), file=log_file)
            print("Fuel: Min = {:.2f}, Max {:.2f}".format(Y_min, Y_max), file=log_file)
            print("Step time: {:.6f} s".format(elapsed_time), file=log_file)
    else: # Save every NT steps
        # Approximation
        z = np.zeros((Nt // NT + 1, z_0.shape[0], Ny, Nx - 1)) 
        p  = np.zeros((Nt // NT + 1, Ny, Nx - 1))
        z[0] = z_0
        z_tmp = z[0].copy()
        p_tmp = p[0].copy()
        for n in range(Nt - 1):
            # Simulation 
            step_time_start = time.time()
            z_tmp, p_tmp = solve_tn(t[n], z_tmp, dt, methods[method], params)
            step_time_end = time.time()
            step_elapsed_time = (step_time_end - step_time_start)
            if n % NT == 0 or n == (Nt - 2): # Save every NT steps and last step
                z[n // NT + 1], p[n // NT + 1] = z_tmp, p_tmp
                # Print log
                CFL = dt * (np.max(np.abs(z_tmp[0])) / dx + np.max(np.abs(z_tmp[1])) / dy)  # Compute CFL
                T_min, T_max = np.min(z_tmp[2]), np.max(z_tmp[2])
                Y_min, Y_max = np.min(z_tmp[3]), np.max(z_tmp[3]) 
                print(OUTPUT_LOG.format(n, t[n]))            
                print("CFL: {:.6f}".format(CFL))
                print("Temperature: Min = {:.2f} K, Max {:.2f} K".format(T_min, T_max))
                print("Fuel: Min = {:.2f}, Max {:.2f}".format(Y_min, Y_max))
                print("Step time: {:f} s".format(step_elapsed_time))
                # Print to log file
                print(OUTPUT_LOG.format(n, t[n]), file=log_file)
                print("CFL: {:.6f}".format(CFL), file=log_file)
                print("Temperature: Min = {:.2f} K, Max {:.2f} K".format(T_min, T_max), file=log_file)
                print("Fuel: Min = {:.2f}, Max {:.2f}".format(Y_min, Y_max), file=log_file)
                print("Step time: {:f} s".format(step_elapsed_time), file=log_file)
    solver_time_end = time.time()
    solver_time = (solver_time_end - solver_time_start)
    print("\nSolver time: ", str(timedelta(seconds=round(solver_time))), "\n")
    print("\nSolver time: ", str(timedelta(seconds=round(solver_time))), "\n", file=log_file)
    # Close log file
    log_file.close()
    return data_post_processing(z, p)
