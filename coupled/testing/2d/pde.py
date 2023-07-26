import numpy as np
from poisson import solve_fftfd
from turbulence import turbulence
from utils import K, S2, S3#, hv#, Yft, S_T, AT
import time

OUTPUT_LOG = "Time step: {:=6d}, Simulation time: {:.2f} s"

def grad_pressure(p, **params):
    dx = params['dx']
    dy = params['dy']
    # cut_nodes_y, cut_nodes_x = kwparams['cut_nodes']
    # dead_nodes = kwparams['dead_nodes']
    px, py = np.zeros_like(p), np.zeros_like(p)
    # Get nodes
    p_ij = np.copy(p) # p_{i, j}
    p_ip1j = np.roll(p,-1, axis=1) # p_{i+1, j}
    p_im1j = np.roll(p, 1, axis=1) # p_{i-1, j}
    p_im2j = np.roll(p, 2, axis=1) # p_{i-2, j}
    p_ijp1 = np.roll(p,-1, axis=0) # p_{i, j+1}
    p_ijm1 = np.roll(p, 1, axis=0) # p_{i, j-1}
    p_ijm2 = np.roll(p, 2, axis=0) # p_{i, j-2}
    
    # Computing derivatives
    # Using central difference O(h^2).
    px = (p_ip1j - p_im1j) / (2 * dx) 
    py = (p_ijp1 - p_ijm1) / (2 * dy)
    # Using backward difference
    # px = (p_ij - p_im1j) / dx
    # py = (p_ij - p_ijm1) / dy
    # Using forward difference
    # px = (p_ip1j - p_ij) / dx
    # py = (p_ijp1 - p_ij) / dy
    # px = (3 * p_ij - 4 * p_im1j + p_im2j) / (2 * dx) # Backward difference. 
    # py = (3 * p_ij - 4 * p_ijm1 + p_ijm2) / (2 * dy) # Backward difference.

    # Boundary conditions correction on y. x is periodic, so no correction needed.
    # O(h) correction for forward difference
    # py[-1] = (p_ij[-1] - p_ij[-2]) / dy # Backward at y=y_max
    # O(h) correction for backward difference
    # py[0] = (p_ij[1] - p_ij[0]) / dy # Forward at y=y_min
    # O(h^2) correction for central difference
    py[0, :] = (-p_ij[2, :] + 4 * p_ij[1, :] - 3 * p_ij[0, :]) / (2 * dy) # Forward at y=y_min
    py[-1,:] = (3 * p_ij[-1, :] - 4 * p_ij[-2, :] + p_ij[-3, :]) / (2 * dy) # Backward at y=y_max

    return np.array([px, py])

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
    T_ign = params['T_ign']
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
    debug = params['debug']

    # Get variables
    u, v, T, Y = C
    
    # Forces
    F_x, F_y = F
    g_x, g_y = g
    # # Drag force
    mod_U = np.sqrt(u ** 2 + v ** 2)
    Y_mask = Y > Y_thr # Valid only for solid fuel mask = Ym <= Y_height
    F_d_x = C_D * a_v * mod_U * u * Y_mask
    F_d_y = C_D * a_v * mod_U * v * Y_mask
    
    # All forces
    F_x = F_x - g_x * (T - T_inf) / T - F_d_x 
    F_y = F_y - g_y * (T - T_inf) / T - F_d_y
    
    # Nodes for finite difference. I will assume periodic boundary for both axis, but then the values will be fixed
    # u 
    u_ij   = u.copy() # u_{i,j}
    u_ip1j = np.roll(u,-1, axis=1) # u_{i+1, j}
    u_ip2j = np.roll(u,-2, axis=1) # u_{i+2, j}
    u_im1j = np.roll(u, 1, axis=1) # u_{i-1, j}
    u_im2j = np.roll(u, 2, axis=1) # u_{i-2, j}
    u_ijp1 = np.roll(u,-1, axis=0) # u_{i, j+1}
    u_ijp2 = np.roll(u,-2, axis=0) # u_{i, j+2}
    u_ijm1 = np.roll(u, 1, axis=0) # u_{i, j-1}
    u_ijm2 = np.roll(u, 2, axis=0) # u_{i, j-2}
    # v
    v_ij   = v.copy() # v_{i, j}
    v_ip1j = np.roll(v,-1, axis=1) # v_{i+1, j}
    v_ip2j = np.roll(v,-2, axis=1) # v_{i+2, j}
    v_im1j = np.roll(v, 1, axis=1) # v_{i-1, j}
    v_im2j = np.roll(v, 2, axis=1) # v_{i-2, j}
    v_ijp1 = np.roll(v,-1, axis=0) # v_{i, j+1}
    v_ijp2 = np.roll(v,-2, axis=0) # v_{i, j+2}
    v_ijm1 = np.roll(v, 1, axis=0) # v_{i, j-1}
    v_ijm2 = np.roll(v, 2, axis=0) # v_{i, j-2}
    # T
    T_ij   = T.copy() # T_{i,j}
    T_ip1j = np.roll(T,-1, axis=1) # T_{i+1, j}
    T_im1j = np.roll(T, 1, axis=1) # T_{i-1, j}
    T_ijp1 = np.roll(T,-1, axis=0) # T_{i, j+1}
    T_ijm1 = np.roll(T, 1, axis=0) # T_{i, j-1}
    
    # Mask for upwind
    u_plu = np.maximum(u_ij, 0)
    u_min = np.minimum(u_ij, 0)
    v_plu = np.maximum(v_ij, 0)
    v_min = np.minimum(v_ij, 0)

    # Derivatives #
    # First derivatives 
    # Forward/backward difference O(h) (for upwind scheme)
    # uxm = (u_ij - u_im1j) / dx
    # uxp = (u_ip1j - u_ij) / dx
    # uym = (u_ij - u_ijm1) / dy
    # uyp = (u_ijp1 - u_ij) / dy
    # vxm = (v_ij - v_im1j) / dx
    # vxp = (v_ip1j - v_ij) / dx
    # vym = (v_ij - v_ijm1) / dy
    # vyp = (v_ijp1 - v_ij) / dy
    # Forward/backward difference O(h^2) (for upwind scheme)
    uxm = (3 * u_ij - 4 * u_im1j + u_im2j) / (2 * dx)
    uxp = (-u_ip2j + 4 * u_ip1j - 3 * u_ij) / (2 * dx)
    uym = (3 * u_ij - 4 * u_ijm1 + u_ijm2) / (2 * dy)
    uyp = (-u_ijp2 + 4 * u_ijp1 - 3 * u_ij) / (2 * dy)
    vxm = (3 * v_ij - 4 * v_im1j + v_im2j) / (2 * dx)
    vxp = (-v_ip2j + 4 * v_ip1j - 3 * v_ij) / (2 * dx)
    vym = (3 * v_ij - 4 * v_ijm1 + v_ijm2) / (2 * dy)
    vyp = (-v_ijp2 + 4 * v_ijp1 - 3 * v_ij) / (2 * dy)

    # Fixed boundary nodes
    # O(h)
    # uym[0, :] = (-u_ij[0, :] + u_ij[1, :])  / dy # Forward at y=y_min
    # uym[-1, :] = (u_ij[-1, :] - u_ij[-2, :]) / dy # Backward at y=y_max
    # uyp[0, :] = (-u_ij[0, :] + u_ij[1, :])  / dy # Forward at y=y_min
    # uyp[-1, :] = (u_ij[-1, :] - u_ij[-2, :]) / dy # Backward at y=y_max
    # vym[0, :] = (-v_ij[0, :] + v_ij[1, :])  / dy # Forward at y=y_min
    # vym[-1, :] = (v_ij[-1, :] - v_ij[-2, :]) / dy # Backward at y=y_max
    # vyp[0, :] = (-v_ij[0, :] + v_ij[1, :])  / dy # Forward at y=y_min
    # vyp[-1, :] = (v_ij[-1, :] - v_ij[-2, :]) / dy # Backward at y=y_max
    # O(h^2)
    uym[0, :] = (-3 * u_ij[0, :] + 4 * u_ij[1, :] - u_ij[2, :]) / (2 * dy) # Forward at y=y_min
    uym[1, :] = (-3 * u_ij[1, :] + 4 * u_ij[2, :] - u_ij[3, :]) / (2 * dy) # Forward at y=y_min+dy
    uyp[-1,:] = (u_ij[-1, :] - 4 * u_ij[-2, :] + 3 * u_ij[-3, :]) / (2 * dy) # Backward at y=y_max
    uyp[-2,:] = (u_ij[-2, :] - 4 * u_ij[-3, :] + 3 * u_ij[-4, :]) / (2 * dy) # Backward at y=y_max-dy
    vym[0, :] = (-3 * v_ij[0, :] + 4 * v_ij[1, :] - v_ij[2, :]) / (2 * dy) # Forward at y=y_min
    vym[1, :] = (-3 * v_ij[1, :] + 4 * v_ij[2, :] - v_ij[3, :]) / (2 * dy) # Forward at y=y_min+dy
    vyp[-1,:] = (v_ij[-1, :] - 4 * v_ij[-2, :] + 3 * v_ij[-3, :]) / (2 * dy) # Backward at y=y_max
    vyp[-2,:] = (v_ij[-2, :] - 4 * v_ij[-3, :] + 3 * v_ij[-4, :]) / (2 * dy) # Backward at y=y_max-dy


    # Finite difference for turbulence
    # O(h)
    # Backward
    # Tx = (T_ij - T_im1j) / dx
    # Ty = (T_ij - T_ijm1) / dy
    # Forward
    # Tx = (T_ip1j - T_ij) / dx
    # Ty = (T_ijp1 - T_ij) / dy 
    # O(h^2)
    # Central difference O(h^2) 
    ux = (u_ip1j - u_im1j) / (2 * dx)
    uy = (u_ijp1 - u_ijm1) / (2 * dy)
    vx = (v_ip1j - v_im1j) / (2 * dx)
    vy = (v_ijp1 - v_ijm1) / (2 * dy)
    Tx = (T_ip1j - T_im1j) / (2 * dx)
    Ty = (T_ijp1 - T_ijm1) / (2 * dy)
    # Fixed boundary nodes
    uy[0, :] = (-3 * u_ij[0, :] + 4 * u_ij[1, :] - u_ij[2, :]) / (2 * dy) # Forward at y=y_min
    uy[-1, :] = (3 * u_ij[-1, :] - 4 * u_ij[-2, :] + u_ij[-3, :]) / (2 * dy) # Backward at y=y_max
    vy[0, :] = (-3 * v_ij[0, :] + 4 * v_ij[1, :] - v_ij[2, :]) / (2 * dy) # Forward at y=y_min
    vy[-1, :] = (3 * v_ij[-1, :] - 4 * v_ij[-2, :] + v_ij[-3, :]) / (2 * dy) # Backward at y=y_max
    Ty[0, :] = (-3 * T_ij[0, :] + 4 * T_ij[1, :] - T_ij[2, :]) / (2 * dy) # Forward at y=y_min
    Ty[-1, :] = (3 * T_ij[-1, :] - 4 * T_ij[-2, :] + T_ij[-3, :]) / (2 * dy) # Backward at y=y_max

    # Second derivatives
    uxx = (u_ip1j - 2 * u_ij + u_im1j) / dx ** 2
    uyy = (u_ijp1 - 2 * u_ij + u_ijm1) / dy ** 2
    vxx = (v_ip1j - 2 * v_ij + v_im1j) / dx ** 2
    vyy = (v_ijp1 - 2 * v_ij + v_ijm1) / dy ** 2
    Txx = (T_ip1j - 2 * T_ij + T_im1j) / dx ** 2
    Tyy = (T_ijp1 - 2 * T_ij + T_ijm1) / dy ** 2
    # Fixed boundary nodes
    uyy[0, :] = (2 * u_ij[0, :] - 5 * u_ij[1, :] + 4 * u_ij[2, :] - u_ij[3, :]) / dy ** 2 # Forward at y=y_min
    uyy[-1, :] = (2 * u_ij[-1, :] - 5 * u_ij[-2, :] + 4 * u_ij[-3, :] - u_ij[-4, :]) / dy ** 2 # Backward at y=y_max
    vyy[0, :] = (2 * v_ij[0, :] - 5 * v_ij[1, :] + 4 * v_ij[2, :] - v_ij[3, :]) / dy ** 2 # Forward at y=y_min
    vyy[-1, :] = (2 * v_ij[-1, :] - 5 * v_ij[-2, :] + 4 * v_ij[-3, :] - v_ij[-4, :]) / dy ** 2 # Backward at y=y_max
    Tyy[0, :] = (2 * T_ij[0, :] - 5 * T_ij[1, :] + 4 * T_ij[2, :] - T_ij[3, :]) / dy ** 2 # Forward at y=y_min
    Tyy[-1, :] = (2 * T_ij[-1, :] - 5 * T_ij[-2, :] + 4 * T_ij[-3, :] - T_ij[-4, :]) / dy ** 2 # Backward at y=y_max

    # Turbulence
    sgs_x = sgs_y = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_T = turbulence(u, v, ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, params)
    
    # Reaction Rate
    # Y_f = 2, A_T=200, 100 (works but the fire ends soon)
    # if t <= 30:
    #     A_T = 250.0
    #     Y_f = 1.1#.25
    # elif t <= 60:
    #     A_T = 100
    #     Y_f = 1.5#2.5
    # elif t <= 90:
    #     A_T = 200
    #     Y_f = 1.2
    # else:
    #     A_T = 100
    #     Y_f = 1
    # Ke = K(T, AT(T, A_T), B) * S3(T, T_ign) # S2(T, 1, 10, T_ign) # 
    Ke = K(T, A, T_act) * S3(T, T_ign) # S2(T, 1, 10, T_ign) # 
    # Ke[Ke > 1] = 1
    # Testing
    #Ke[Ke >= 1000] = 1000
    #T_mask = T >= T_ign
    
    # Dissipation function
    # phi = nu * rho * (2 * (uxx ** 2 + uyy ** 2) + (vx + uy) ** 2)

    # # Temperature source term
    # S = rho * H_R * Y * Ke - h * (T - T_inf)
    # h = hv(mod_U)
    S1 = H_R * Y * Ke / C_p 
    S2 = -h * a_v * (T - T_inf) / (rho * C_p) 
    S1[S1 >= S_top] = S_bot
    S = S1 + S2
    #S = S_T(S)
    
    # S = Y * 200 * S3(T, T_ign) -  h * (T - T_inf) / (rho * C_p)
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
        print("S:", np.min(S), np.max(S))
        print("S1:", np.min(S1), np.max(S1))
        print("S2:", np.min(S2), np.max(S2))
        # print("Tx:", np.min(Tx), np.max(Tx))
        # print("Ty:", np.min(Ty), np.max(Ty))
        # print("Txx:", np.min(Txx), np.max(Txx))
        # print("Tyy:", np.min(Tyy), np.max(Tyy))
        # lapT = k * (Txx + Tyy)
        # print("lapT", np.min(lapT), np.max(lapT))
        print("sgs_T", np.min(sgs_T), np.max(sgs_T))
        print("S - sgs_T", np.min(S - sgs_T), np.max(S - sgs_T))
        print("-" * 30)

    if conservative: # RHS Inside domain (conservative form - using central difference)
        # Conservative form for convection
        uux = (u_ip1j ** 2 - u_im1j ** 2) / (2 * dx) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
        uvy = (u_ijp1 * v_ijp1 - u_ijm1 * v_ijm1) / (2 * dx)
        vux = (v_ip1j * u_ip1j - v_im1j * u_im1j) / (2 * dx)
        vvy = (v_ijp1 ** 2 - v_im1j ** 2) / (2 * dy)
        # PDE
        U_ = nu * (uxx + uyy) - (uux + uvy) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (vux + vvy) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx + v * Ty) + S - sgs_T
    else: # RHS Inside domain (non-conservative form - using upwind!)
        U_ = nu * (uxx + uyy) - (u_plu * uxm + u_min * uxp + v_plu * uym + v_min * uyp) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (u_plu * vxm + u_min * vxp + v_plu * vym + v_min * vyp) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx  + v * Ty) + S - sgs_T 

    if t < 5:
        if T_mask is not None:
            T[T_mask] = T_hot
    
    # Y_f = Yft(t)
    Y_ = -Y * Ke * Y_f 

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
    

    # Boundary conditions on x
    # Nothing to do because Phi includes them
    # Boundary conditions on y (Dirichlet)
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    u_s, v_s, T_s, Y_s, u_n, v_n, T_n, Y_n = u_y_min, v_y_min, T_y_min, Y_y_min, u_y_max, v_y_max, T_inf, Y_y_max

    # Boundary at south. Derivative using O(h^2)	
    T_s = (4 * T[1, :] - T[2, :]) / 3 # dT/dy = 0
    Y_s = (4 * Y[1, :] - Y[2, :]) / 3 # dY/dy = 0

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

    # if T_mask is not None:
    #    T[T_mask] = T_hot

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
    
# Time integration
def euler(t_n, y_n, dt, params):
    rho = params['rho']
    y_np1 = y_n + dt * Phi(t_n, y_n, params)
    Ut, Vt = y_np1[:2].copy()
    p = solve_fftfd(Ut, Vt, **params).copy()
    grad_p = grad_pressure(p, **params)
    y_np1[:2] = y_np1[:2] - dt / rho * grad_p
    Ut, Vt, Tt, Yt = y_np1.copy()
    y_np1 = boundary_conditions(Ut, Vt, Tt, Yt, params)
    return y_np1, p

def RK4(t_n, y_n, dt, params):
    rho = params['rho']
    k1 = Phi(t_n, y_n, params)
    k2 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k1, params)
    k3 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k2, params)
    k4 = Phi(t_n + dt, y_n + dt * k3, params)
    y_np1 = y_n + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    Ut, Vt = y_np1[:2].copy()
    p = solve_fftfd(Ut, Vt, **params).copy()
    grad_p = grad_pressure(p, **params)
    y_np1[:2] = y_np1[:2] - dt / rho * grad_p
    Ut, Vt, Tt, Yt = y_np1.copy()
    y_np1 = boundary_conditions(Ut, Vt, Tt, Yt, params)
    return y_np1, p

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
    # T_mask = params['T_mask']
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
            z[n+1], p[n+1] = methods[method](t[n], z[n], dt, params)
            Ut, Vt = z[n+1, :2].copy()
            time_end = time.time()
            elapsed_time = (time_end - time_start)
            print("CFL: {:.6f}".format(dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy)))
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
            z_tmp, p_tmp = methods[method](t[n], z_tmp, dt, params)
            # if t[n] <= 3:
            #     z_tmp[2][T_mask] = TA
            if n % NT == 0:
                # print("Time step:", n)
                # print("Simulation time:", t[n], " s")
                print(OUTPUT_LOG.format(n, t[n]))
                z[n // NT + 1], p[n // NT + 1] = z_tmp, p_tmp
                time_end = time.time()
                Ut, Vt = z_tmp[:2].copy()
                elapsed_time = (time_end - time_start)
                print("CFL: {:.6f}".format(dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy)))
                print("Step time: {:f} s".format(elapsed_time))
        # Last approximation
        z[-1] = z_tmp
        p[-1] = p_tmp

    return data_post_processing(z, p, params)
