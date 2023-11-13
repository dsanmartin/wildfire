import time
import numpy as np
from derivatives import compute_gradient, compute_laplacian, compute_first_derivative_upwind, compute_first_derivative
from poisson import solve_pressure
from turbulence import turbulence
from utils import K, H, Km, hv, source, sink, kT, kTp #, Yft, S_T, AT
from plots import plot_2D

OUTPUT_LOG = "Time step: {:=6d}, Simulation time: {:.2f} s"

def grad_pressure(p: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the gradient of pressure.

    .. math::
        \nabla p = \left(\frac{\partial p}{\partial x}, \frac{\partial p}{\partial y}\right)

    Parameters
    ----------
    p : numpy.ndarray (Ny, Nx)
        Pressure field.
    params : dict
        Dictionary containing the interval sizes `dx` and `dy`.

    Returns
    -------
    numpy.ndarray (2, Ny, Nx)
        Gradient of pressure.
    """
    # Get interval size
    dx, dy = params['dx'], params['dy']
    # Compute grad(p)
    grad_p = compute_gradient(p, dx, dy, (False, True))
    return grad_p

def Phi(t, C, params):
    dx = params['dx']
    dy = params['dy']
    nu = params['nu']
    rho = params['rho']
    k = params['k']
    #P = params['p']
    C_p = params['C_p']
    F = params['F']
    g = params['g']
    T_inf = params['T_inf']
    A = params['A']
    T_act = params['T_act']
    # E_A = params['E_A']
    # R = params['R']
    H_R = params['H_R']
    h = params['h']
    T_pc = params['T_pc']
    C_D = params['C_D']
    a_v = params['a_v']
    Y_thr = params['Y_thr']
    Y_f = params['Y_f']
    turb = params['turbulence']
    conservative = params['conservative']
    S_top = params['S_top']
    S_bot = params['S_bot']
    T_mask = params['T_mask'] # Temperature fixed source
    T_hot = params['T_hot'] # Temperature fixed source
    T_0 = params['T_0'] # Temperature fixed source
    sigma = params['sigma']
    delta = params['delta']
    debug = params['debug']
    source_filter = params['source_filter']
    radiation = params['radiation']
    include_source = params['include_source']
    sutherland_law = params['sutherland_law']

    # Get variables
    u, v, T, Y = C
    
    # Forces
    F_x, F_y = F # 'External' forces
    g_x, g_y = g # Gravity
    # Drag force
    mod_U = np.sqrt(u ** 2 + v ** 2)
    # Y_mask = (Y > Y_thr).astype(int) # Valid only for solid fuel
    Y_mask = Y * Y_thr # Valid only for solid fuel
    F_d_x = C_D * a_v * mod_U * u * Y_mask
    F_d_y = C_D * a_v * mod_U * v * Y_mask
    
    # All forces
    F_x = F_x - g_x * (T - T_inf) / T - F_d_x 
    F_y = F_y - g_y * (T - T_inf) / T - F_d_y
    # F_x = F_x - g_x * ((T - T_inf) + (T - T_inf) ** 2) / T - F_d_x
    # F_y = F_y - g_y * ((T - T_inf) + (T - T_inf) ** 2) / T - F_d_y
    

    # Derivatives #
    # First derivatives 
    if conservative:
        # Conservative form for convection
        uux = compute_first_derivative(u * u, dx, 1, (False, True)) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
        vuy = compute_first_derivative(u * v, dy, 0, (False, True)) # (u_{i, j+1} * v_{i, j+1} - u_{i, j-1} * v_{i, j-1}) / (2 * dy)
        uvx = compute_first_derivative(v * u, dx, 1, (False, True))# (v_{i+1, j} * u_{i+1, j} - v_{i-1, j} * u_{i-1, j}) / (2 * dx)
        vvy = compute_first_derivative(v * v, dy, 0, (False, True)) # (v_{i, j+1}^2 - v_{i, j-1}^2) / (2 * dy)
    else:
        # Non-conservative form for convection
        uux = compute_first_derivative_upwind(u, u, dx, 1) 
        vuy = compute_first_derivative_upwind(v, u, dy, 0, periodic=False)
        uvx = compute_first_derivative_upwind(u, v, dx, 1)
        vvy = compute_first_derivative_upwind(v, v, dy, 0, periodic=False)
    Tx, Ty = compute_gradient(T, dx, dy, (False, True))

    # Compute Laplacian
    lap_u = compute_laplacian(u, dx, dy, (False, True))
    lap_v = compute_laplacian(v, dx, dy, (False, True))
    lap_T = compute_laplacian(T, dx, dy, (False, True))

    # Turbulence
    sgs_x = sgs_y = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_T = turbulence(u, v, T, params)

    # Temperature source term
    S = 0 # No source
    if include_source:
        S1 = source(T, Y)
        S2 = sink(T)
        if source_filter:
            S1[S1 >= S_top] = S_bot
        S = S1 + S2
        
        #S = S_T(S)

    # S = Y * 200 * S3(T, T_pc) -  h * (T - T_inf) / (rho * C_p)
    # S = 0
    if debug:
        # dt = params['dt']
        # if t % (dt * 100) == 0:
        # print(t, dt * 100)
        print("-" * 30)
        # print("A:", A)
        # print("h:", h)
        # print("S_top:", S_top)
        # print("S_bot:", S_bot)
        # print("K:", np.min(Ke), np.max(Ke))
        # print("S:", np.min(S), np.max(S))
        # print("S1:", np.min(S1), np.max(S1))
        # print("S2:", np.min(S2), np.max(S2))
        # print("Tx:", np.min(Tx), np.max(Tx))
        # print("Ty:", np.min(Ty), np.max(Ty))
        # print("Txx:", np.min(Txx), np.max(Txx))
        # print("Tyy:", np.min(Tyy), np.max(Tyy))
        # lapT = k * (Txx + Tyy)
        # print("lapT", np.min(lapT), np.max(lapT))
        # print("sgs_T", np.min(sgs_T), np.max(sgs_T))
        # print("S - sgs_T", np.min(S - sgs_T), np.max(S - sgs_T))
        print("K(T) * H(T)", np.min(K(T) * H(T)) , np.max(K(T) * H(T)))
        print("-" * 30)

    # PDE
    # Velocity
    U_ = nu * lap_u + F_x - sgs_x - (uux + vuy) 
    V_ = nu * lap_v + F_y - sgs_y - (uvx + vvy)
    # U_ = nu * lap_u + F_x - ((uux + vuy) + sgs_x)
    # V_ = nu * lap_v + F_y - ((uvx + vvy) + sgs_y)
    # Temperature
    # Radiation
    if radiation:
        T_ = 12 * sigma * delta * T ** 2 * (Tx ** 2 + Ty ** 2) + (k + 4 * sigma * delta * T ** 3) * lap_T - (u * Tx  + v * Ty) + S - sgs_T
    elif sutherland_law:
        T_ = kTp(T) * (Tx ** 2 + Ty ** 2) + kT(T) * lap_T - (u * Tx  + v * Ty) + S - sgs_T 
    else:
        T_ = k * lap_T - (u * Tx  + v * Ty) + S - sgs_T 

    # Combustion model
    Y_ = -Y_f * K(T) * H(T) * Y 

    # Boundary conditions
    U_, V_, T_, Y_ = boundary_conditions(U_, V_, T_, Y_, params)

    return np.array([U_, V_, T_, Y_])

def boundary_conditions(u: np.ndarray, v: np.ndarray, T: np.ndarray, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply boundary conditions to the input variables.

    Parameters
    ----------
    u : np.ndarray (Ny, Nx)
        Velocity in the x direction.
    v : np.ndarray (Ny, Nx)
        Velocity in the y direction.
    T : np.ndarray (Ny, Nx)
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
    np.ndarray (Ny, Nx)
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
    y_n : np.ndarray (4, Ny, Nx)
        Array with the current solution.
    dt : float
        Time step.
    method : function
        Function that computes the next solution given the current one.
    params : dict
        Dictionary with the parameters of the PDE system.

    Returns
    -------
    y_np1 : np.ndarray (4, Ny, Nx)
        Array with the solution at the next time step.
    p : np.ndarray (Ny, Nx)
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

def solve_pde(z_0, params):
    Nx, Ny, Nt = params['Nx'], params['Ny'], params['Nt']
    dx, dy, dt = params['dx'], params['dy'], params['dt']
    # print(T_hot)
    NT = params['NT']
    t = params['t']
    method = params['method']
    methods = {
        'euler': euler,
        'RK4': RK4
    }
    if NT == 1:
        # Approximation
        z = np.zeros((Nt, z_0.shape[0], Ny, Nx - 1)) 
        p = np.zeros((Nt, Ny, Nx - 1))
        z[0] = z_0
        for n in range(Nt - 1):
            # Simulation 
            #print("Time step:", n)
            #print("Simulation time:", t[n], " s")
            print(OUTPUT_LOG.format(n, t[n]))
            time_start = time.time()
            # z[n+1], p[n+1] = methods[method](t[n], z[n], dt, params)
            z[n+1], p[n+1] = solve_tn(t[n], z[n], dt, methods[method], params)
            Ut, Vt = z[n+1, :2].copy()
            # if t[n] <= 5:
            #     z[n+1, 2] = T_0 * T_mask + z[n+1, 2] * (1 - T_mask)
            time_end = time.time()
            elapsed_time = (time_end - time_start)
            print("CFL: {:.6f}".format(dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy)))
            print("Temperature: Min = {:.2f} K, Max {:.2f} K".format(np.min(z[n+1, 2]), np.max(z[n+1, 2])))
            print("Fuel: Min = {:.2f}, Max {:.2f}".format(np.min(z[n+1, 3]), np.max(z[n+1, 3])))
            print("Step time: {:.6f} s".format(elapsed_time))
        
    else:
        # Approximation
        z = np.zeros((Nt // NT + 1, z_0.shape[0], Ny, Nx - 1)) 
        p  = np.zeros((Nt // NT + 1, Ny, Nx - 1))
        z[0] = z_0
        z_tmp = z[0].copy()
        p_tmp = p[0].copy()
        for n in range(Nt - 1):
            # Simulation 
            time_start = time.time()
            # z_tmp, p_tmp = methods[method](t[n], z_tmp, dt, params)
            z_tmp, p_tmp = solve_tn(t[n], z_tmp, dt, methods[method], params)
            if n % NT == 0:
                # print("Time step:", n)
                # print("Simulation time:", t[n], " s")
                print(OUTPUT_LOG.format(n, t[n]))
                z[n // NT + 1], p[n // NT + 1] = z_tmp, p_tmp
                time_end = time.time()
                Ut, Vt = z_tmp[:2].copy()
                elapsed_time = (time_end - time_start)
                print("CFL: {:.6f}".format(dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy)))
                print("Temperature: Min = {:.2f} K, Max {:.2f} K".format(np.min(z_tmp[2]), np.max(z_tmp[2])))
                print("Fuel: Min = {:.2f}, Max {:.2f}".format(np.min(z_tmp[3]), np.max(z_tmp[3])))
                # print(np.unique(z_tmp[3]))
                print("Step time: {:f} s".format(elapsed_time))
        # Last approximation
        z[-1] = z_tmp
        p[-1] = p_tmp
        # Last time step
        print(OUTPUT_LOG.format(Nt - 1, t[Nt - 1]))
        time_end = time.time()
        Ut, Vt = z_tmp[:2].copy()
        elapsed_time = (time_end - time_start)
        print("CFL: {:.6f}".format(dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy)))
        print("Step time: {:f} s".format(elapsed_time))

    return data_post_processing(z, p)
