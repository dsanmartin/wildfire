import time
import numpy as np
from derivatives import compute_gradient, compute_laplacian, compute_first_derivative_upwind, compute_first_derivative
from poisson import solve_fftfd
from turbulence import turbulence
from utils import K, H, Km, hv, source, sink, kT, kTp #, Yft, S_T, AT
from plots import plot_2D

OUTPUT_LOG = "Time step: {:=6d}, Simulation time: {:.2f} s"

def grad_pressure(p, **params):
    # Get interval size
    dx = params['dx']
    dy = params['dy']
    # Compute grad(p)
    grad_p = compute_gradient(p, dx, dy)
    return grad_p

# Right hand side of the PDE
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
        uux = compute_first_derivative(u * u, dx, 1) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
        vuy = compute_first_derivative(u * v, dy, 0) # (u_{i, j+1} * v_{i, j+1} - u_{i, j-1} * v_{i, j-1}) / (2 * dy)
        uvx = compute_first_derivative(v * u, dx, 1)# (v_{i+1, j} * u_{i+1, j} - v_{i-1, j} * u_{i-1, j}) / (2 * dx)
        vvy = compute_first_derivative(v * v, dy, 0) # (v_{i, j+1}^2 - v_{i, j-1}^2) / (2 * dy)
    else:
        # Non-conservative form for convection
        uux = compute_first_derivative_upwind(u, u, dx, 1) 
        vuy = compute_first_derivative_upwind(v, u, dy, 0)
        uvx = compute_first_derivative_upwind(u, v, dx, 1)
        vvy = compute_first_derivative_upwind(v, v, dy, 0)
    Tx, Ty = compute_gradient(T, dx, dy)

    # Compute Laplacian
    lap_u = compute_laplacian(u, dx, dy)
    lap_v = compute_laplacian(v, dx, dy)
    lap_T = compute_laplacian(T, dx, dy)

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
    U_ = nu * lap_u - (uux + vuy) + F_x - sgs_x
    V_ = nu * lap_v - (uvx + vvy) + F_y - sgs_y
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

    U_, V_, T_, Y_ = boundary_conditions(U_, V_, T_, Y_, params)

    return np.array([U_, V_, T_, Y_])

def boundary_conditions(u, v, T, Y, params):
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
    # T_0 = params['T_0']
    
    # Boundary conditions on x
    # Nothing to do because RHS includes them
    # Boundary conditions on y 
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    u_s, v_s, T_s, Y_s, u_n, v_n, T_n, Y_n = u_y_min, v_y_min, T_y_min, Y_y_min, u_y_max, v_y_max, T_inf, Y_y_max

    # Boundary at south. Derivative using O(h^2)	
    T_s = (4 * T[1, :] - T[2, :]) / 3 # dT/dy = 0
    Y_s = (4 * Y[1, :] - Y[2, :]) / 3 # dY/dy = 0

    # T_s = T_0[0] # Fixed temperature

    # Boundary at north. Derivative using O(h^2)
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

    # BC at edge nodes
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

    return np.array([u, v, T, Y])

def solve_tn(t_n, y_n, dt, method, params):
    # Get parameters
    rho = params['rho']
    T_mask = params['T_mask']
    t_source = params['t_source']
    T_source = params['T_source']
    truncate = params['truncate']
    T_min, T_max = params['T_min'], params['T_max']
    Y_min, Y_max = params['Y_min'], params['Y_max']
    # Solve time step 
    y_np1 = method(t_n, y_n, dt, params)
    Ut, Vt = y_np1[:2].copy()
    # Solve Pressure problem
    p = solve_fftfd(Ut, Vt, **params).copy()
    grad_p = grad_pressure(p, **params)
    # Correct velocity
    y_np1[:2] = y_np1[:2] - dt / rho * grad_p
    Ut, Vt, Tt, Yt = y_np1.copy()
    # Update boundary conditions
    y_np1 = boundary_conditions(Ut, Vt, Tt, Yt, params)
    if truncate:
        # Remove wrong mass fraction
        y_np1[3, y_np1[3] < Y_min] = Y_min 
        y_np1[3, y_np1[3] > Y_max] = Y_max 
        # Remove wrong temperature
        y_np1[2, y_np1[2] < T_min] = T_min
        y_np1[2, y_np1[2] > T_max] = T_max
    # Add temperature source if needed
    if t_n <= t_source:
        y_np1[2, T_mask] = T_source[T_mask]
    # plot_2D(params['Xm'], params['Ym'], np.abs(tmp - y_np1[2]))
    return y_np1, p
    
# Euler step
def euler(t_n, y_n, dt, params):
    y_np1 = y_n + dt * Phi(t_n, y_n, params)
    return y_np1#, p

# Fourth order Runge-Kutta step
def RK4(t_n, y_n, dt, params):
    k1 = Phi(t_n, y_n, params)
    k2 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k1, params)
    k3 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k2, params)
    k4 = Phi(t_n + dt, y_n + dt * k3, params)
    y_np1 = y_n + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_np1

# Repeat boundary condition due tu periodicity
def data_post_processing(z, p, params):
    Nx, Ny, Nt = params['Nx'], params['Ny'], params['Nt']
    NT = params['NT']
     # Get variables
    U = z[:, 0]
    V = z[:, 1]
    T = z[:, 2]
    Y = z[:, 3]
    # Output arrays
    U_ = np.zeros((z.shape[0], Ny, Nx))
    V_ = np.zeros_like(U_)
    T_ = np.zeros_like(U_)
    Y_ = np.zeros_like(U_)
    P_ = np.zeros_like(U_)
    # Copy last column (periodic boundary in x)
    U_[:, :, :-1] = U
    U_[:, :, -1] = U[:, :, 0]
    V_[:, :, :-1] = V
    V_[:, :, -1] = V[:, :, 0]
    T_[:, :, :-1] = T
    T_[:, :, -1] = T[:, :, 0]
    Y_[:, :, :-1] = Y
    Y_[:, :, -1] = Y[:, :, 0]
    P_[:, :, :-1] = p
    P_[:, :, -1] = p[:, :, 0]
    # U = U_.copy()
    # V = V_.copy()
    # T = T_.copy()
    # Y = Y_.copy()
    # P = P_.copy()
    return U_, V_, T_, Y_, P_


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

    return data_post_processing(z, p, params)
