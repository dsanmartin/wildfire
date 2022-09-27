import numpy as np
from poisson import solve_fftfd#, solve_iterative, solve_iterative_ibm, solve_gmres
from turbulence import turbulence
from ibm import building, cylinder
from utils import create_2d_gaussian#multivariate_normal
import time
import sys
import pickle

# np.random.seed(666)

def grad_pressure(p, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    # cut_nodes_y, cut_nodes_x = kwargs['cut_nodes']
    # dead_nodes = kwargs['dead_nodes']
    px, py = np.zeros_like(p), np.zeros_like(p)
    # Get nodes
    p_ij = np.copy(p)
    p_ip1j = np.roll(p, -1, axis=1) # p_{i+1, j}
    p_im1j = np.roll(p, 1, axis=1) # p_{i-1, j}
    p_ijp1 = np.roll(p, -1, axis=0) # p_{i, j+1}
    p_ijm1 = np.roll(p, 1, axis=0) # p_{i, j-1}
    
    # Computing derivatives
    # Using central difference O(h^2).
    px = (p_ip1j - p_im1j) / (2 * dx) 
    # py = (p_ijp1 - p_ijm1) / (2 * dy)
    # Using backward difference
    #py = (p_ij - p_ijm1) / dy
    # Using forward difference
    py = (p_ijp1 - p_ij) / dy

    # Derivatives at boundary, dp/dy at y = y_min and y = y_max
    # Periodic on x, included before
    # Forward/backward difference O(h^2)
    # py[0, 1:-1] = (-3 * p[0, 1:-1] + 4 * p[1, 1:-1] - p[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # # py[cut_nodes_y, 1:-1] = (-3 * p[cut_nodes_y, 1:-1] + 4 * p[cut_nodes_y + 1, 1:-1] - p[cut_nodes_y + 2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # py[-1, 1:-1] = (3 * p[-1, 1:-1] - 4 * p[-2, 1:-1] + p[-3, 1:-1]) / (2 * dy) # Backward at y=y_max
    py[0, :] = (-3 * p[0, :] + 4 * p[1, :] - p[2, :]) / (2 * dy) # Forward at y=y_min
    # py[cut_nodes_y, 1:-1] = (-3 * p[cut_nodes_y, 1:-1] + 4 * p[cut_nodes_y + 1, 1:-1] - p[cut_nodes_y + 2, 1:-1]) / (2 * dy) # Forward at y=y_min
    py[-1, :] = (3 * p[-1, :] - 4 * p[-2, :] + p[-3, :]) / (2 * dy) # Backward at y=y_max
    # Dead nodes
    # px[dead_nodes] = 0
    # py[dead_nodes] = 0

    return np.array([px, py])

def Phi(t, C, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    nu = kwargs['nu']
    rho = kwargs['rho']
    k = kwargs['k']
    #P = kwargs['p']
    F = kwargs['F']
    g = kwargs['g']
    T_inf = kwargs['T_inf']
    A = kwargs['A']
    E_A = kwargs['E_A']
    R = kwargs['R']
    H_R = kwargs['H_R']
    h = kwargs['h']
    T_pc = kwargs['T_pc']
    C_D = kwargs['C_D']
    a_v = kwargs['a_v']
    turb = kwargs['turb']
    conservative = kwargs['conservative']

    # Get variables
    u, v, T = C
    
    # Forces
    F_x, F_y = F
    g_x, g_y = g
    # # Drag force
    mod_U = np.sqrt(u ** 2 + v ** 2)
    # mask = Y > 0.5 # Valid only for solid fuel
    # mask = Ym <= dx
    # F_d_x = rho * C_D * a_v * mod_U * u * mask
    # F_d_y = rho * C_D * a_v * mod_U * v * mask
    
    # All forces
    F_x = F_x - g_x * (T - T_inf) / T#_inf - F_d_x 
    F_y = F_y - g_y * (T - T_inf) / T#_inf - F_d_y
    
    # Nodes for finite difference. I will assume periodic boundary for both axis, but then the values will be fixed
    # u 
    u_ij   = u.copy() # u_{i,j}
    u_ip1j = np.roll(u, -1, axis=1) # u_{i+1, j}
    u_im1j = np.roll(u, 1, axis=1) # u_{i-1, j}
    u_ijp1 = np.roll(u, -1, axis=0) # u_{i, j+1} 
    u_ijm1 = np.roll(u, 1, axis=0) # u_{i, j-1}
    # v
    v_ij   = v.copy() # v_{i, j}
    v_ip1j = np.roll(v, -1, axis=1) # v_{i+1, j}
    v_im1j = np.roll(v, 1, axis=1) # v_{i-1, j}
    v_ijp1 = np.roll(v, -1, axis=0) # v_{i, j+1}
    v_ijm1 = np.roll(v, 1, axis=0) # v_{i, j-1}
    # T
    T_ij   = T.copy() # T_{i,j}
    T_ip1j = np.roll(T, -1, axis=1) # T_{i+1, j}
    T_im1j = np.roll(T, 1, axis=1) # T_{i-1, j}
    T_ijp1 = np.roll(T, -1, axis=0) # T_{i, j+1}
    T_ijm1 = np.roll(T, 1, axis=0) # T_{i, j-1}
    
    # Mask for upwind
    u_plu = np.maximum(u_ij, 0)
    u_min = np.minimum(u_ij, 0)
    v_plu = np.maximum(v_ij, 0)
    v_min = np.minimum(v_ij, 0)

    # Derivatives #
    # First derivatives 
    # Forward/backward difference O(h) (for upwind scheme)
    uxm = (u_ij - u_im1j) / dx
    uxp = (u_ip1j - u_ij) / dx
    uym = (u_ij - u_ijm1) / dy
    uyp = (u_ijp1 - u_ij) / dy
    vxm = (v_ij - v_im1j) / dx
    vxp = (v_ip1j - v_ij) / dx
    vym = (v_ij - v_ijm1) / dy
    vyp = (v_ijp1 - v_ij) / dy
    # Fixed boundary nodes
    # uym[0, 1:-1] = (-u_ij[0, 1:-1] + u_ij[1, 1:-1])  / dy # Forward at y=y_min
    # uym[-1, 1:-1] = (u_ij[-1, 1:-1] - u_ij[-2, 1:-1]) / dy # Backward at y=y_max
    # uyp[0, 1:-1] = (-u_ij[0, 1:-1] + u_ij[1, 1:-1])  / dy # Forward at y=y_min
    # uyp[-1, 1:-1] = (u_ij[-1, 1:-1] - u_ij[-2, 1:-1]) / dy # Backward at y=y_max
    # vym[0, 1:-1] = (-v_ij[0, 1:-1] + v_ij[1, 1:-1])  / dy # Forward at y=y_min
    # vym[-1, 1:-1] = (v_ij[-1, 1:-1] - v_ij[-2, 1:-1]) / dy # Backward at y=y_max
    # vyp[0, 1:-1] = (-v_ij[0, 1:-1] + v_ij[1, 1:-1])  / dy # Forward at y=y_min
    # vyp[-1, 1:-1] = (v_ij[-1, 1:-1] - v_ij[-2, 1:-1]) / dy # Backward at y=y_max
    uym[0, :] = (-u_ij[0, :] + u_ij[1, :])  / dy # Forward at y=y_min
    uym[-1, :] = (u_ij[-1, :] - u_ij[-2, :]) / dy # Backward at y=y_max
    uyp[0, :] = (-u_ij[0, :] + u_ij[1, :])  / dy # Forward at y=y_min
    uyp[-1, :] = (u_ij[-1, :] - u_ij[-2, :]) / dy # Backward at y=y_max
    vym[0, :] = (-v_ij[0, :] + v_ij[1, :])  / dy # Forward at y=y_min
    vym[-1, :] = (v_ij[-1, :] - v_ij[-2, :]) / dy # Backward at y=y_max
    vyp[0, :] = (-v_ij[0, :] + v_ij[1, :])  / dy # Forward at y=y_min
    vyp[-1, :] = (v_ij[-1, :] - v_ij[-2, :]) / dy # Backward at y=y_max
    
    # Central difference O(h^2) (for turbulence)
    ux = (u_ip1j - u_im1j) / (2 * dx)
    uy = (u_ijp1 - u_ijm1) / (2 * dy)
    vx = (v_ip1j - v_im1j) / (2 * dx)
    vy = (v_ijp1 - v_ijm1) / (2 * dy)
    Tx = (T_ip1j - T_im1j) / (2 * dx)
    Ty = (T_ijp1 - T_ijm1) / (2 * dy)
    # Tx = (T_ij - T_im1j) / dx
    # Ty = (T_ij - T_ijm1) / dy
    # Tx = (T_ip1j - T_ij) / dx
    # Ty = (T_ijp1 - T_ij) / dy 
    # Fixed boundary nodes
    # uy[0, 1:-1] = (-3 * u_ij[0, 1:-1] + 4 * u_ij[1, 1:-1] - u_ij[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # uy[-1, 1:-1] = (3 * u_ij[-1, 1:-1] - 4 * u_ij[-2, 1:-1] + u_ij[-3, 1:-1]) / (2 * dy) # Backward at y=y_max
    # vy[0, 1:-1] = (-3 * v_ij[0, 1:-1] + 4 * v_ij[1, 1:-1] - v_ij[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # vy[-1, 1:-1] = (3 * v_ij[-1, 1:-1] - 4 * v_ij[-2, 1:-1] + v_ij[-3, 1:-1]) / (2 * dy) # Backward at y=y_max
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
    # uyy[0, 1:-1] = (2 * u_ij[0, 1:-1] - 5 * u_ij[1, 1:-1] + 4 * u_ij[2, 1:-1] - u_ij[3, 1:-1]) / dy ** 2 # Forward at y=y_min
    # uyy[-1, 1:-1] = (2 * u_ij[-1, 1:-1] - 5 * u_ij[-2, 1:-1] + 4 * u_ij[-3, 1:-1] - u_ij[-4, 1:-1]) / dy ** 2 # Backward at y=y_max
    # vyy[0, 1:-1] = (2 * v_ij[0, 1:-1] - 5 * v_ij[1, 1:-1] + 4 * v_ij[2, 1:-1] - v_ij[3, 1:-1]) / dy ** 2 # Forward at y=y_min
    # vyy[-1, 1:-1] = (2 * v_ij[-1, 1:-1] - 5 * v_ij[-2, 1:-1] + 4 * v_ij[-3, 1:-1] - v_ij[-4, 1:-1]) / dy ** 2 # Backward at y=y_max
    uyy[0, :] = (2 * u_ij[0, :] - 5 * u_ij[1, :] + 4 * u_ij[2, :] - u_ij[3, :]) / dy ** 2 # Forward at y=y_min
    uyy[-1, :] = (2 * u_ij[-1, :] - 5 * u_ij[-2, :] + 4 * u_ij[-3, :] - u_ij[-4, :]) / dy ** 2 # Backward at y=y_max
    vyy[0, :] = (2 * v_ij[0, :] - 5 * v_ij[1, :] + 4 * v_ij[2, :] - v_ij[3, :]) / dy ** 2 # Forward at y=y_min
    vyy[-1, :] = (2 * v_ij[-1, :] - 5 * v_ij[-2, :] + 4 * v_ij[-3, :] - v_ij[-4, :]) / dy ** 2 # Backward at y=y_max
    Tyy[0, :] = (2 * T_ij[0, :] - 5 * T_ij[1, :] + 4 * T_ij[2, :] - T_ij[3, :]) / dy ** 2 # Forward at y=y_min
    Tyy[-1, :] = (2 * T_ij[-1, :] - 5 * T_ij[-2, :] + 4 * T_ij[-3, :] - T_ij[-4, :]) / dy ** 2 # Backward at y=y_max

    # Turbulence
    sgs_x = sgs_y = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_T = turbulence(u, v, ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, kwargs)
    
    # Reaction Rate
    # K = A * np.exp(-E_A / (R * T))
    # K[T < T_pc] = 0 # Only for T > T_pc

    # # Temperature source term
    # S = rho * H_R * Y * K - h * (T - T_inf)
    S = 0

    # # RHS Inside domain (non-conservative form - using upwind!)
    # U_ = nu * (uxx + uyy) - (u_plu * uxm + u_min * uxp + v_plu * uym + v_min * uyp) + F_x - sgs_x
    # V_ = nu * (vxx + vyy) - (u_plu * vxm + u_min * vxp + v_plu * vym + v_min * vyp) + F_y - sgs_y
    # T_ = k * (Txx + Tyy) - (u * Tx  + v * Ty) + S - sgs_T

    # # RHS Inside domain (conservative form - using central difference)
    # # U_ = nu * (uxx + uyy) - (uux + uvy) + F_x - sgs_x
    # # V_ = nu * (vxx + vyy) - (vux + vvy) + F_y - sgs_y
    # # T_ = k * (Txx + Tyy) - (u * Tx + v * Ty) - sgs_T #+ S

    if conservative: # RHS Inside domain (conservative form - using central difference)
        # Conservative form for convection
        uux = (u_ip1j ** 2 - u_im1j ** 2) / (2 * dx) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
        uvy = (u_ijp1 * v_ijp1 - u_ijm1 * v_ijm1) / (2 * dx)
        vux = (v_ip1j * u_ip1j - v_im1j * u_im1j) / (2 * dx)
        vvy = (v_ijp1 ** 2 - v_im1j ** 2) / (2 * dy)
        # PDE
        U_ = nu * (uxx + uyy) - (uux + uvy) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (vux + vvy) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx + v * Ty) + S - sgs_T * 0
    else: # RHS Inside domain (non-conservative form - using upwind!)
        U_ = nu * (uxx + uyy) - (u_plu * uxm + u_min * uxp + v_plu * uym + v_min * uyp) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (u_plu * vxm + u_min * vxp + v_plu * vym + v_min * vyp) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx  + v * Ty) + S - sgs_T 


    U_, V_, T_ = boundary_conditions(U_, V_, T_, kwargs)

    return np.array([U_, V_, T_])

def boundary_conditions(u, v, T, args):
    T_inf = args['T_inf']
    u0 = args['u0']
    v0 = args['v0']
    u_y_min = args['u_y_min']
    u_y_max = args['u_y_max']
    v_y_min = args['v_y_min']
    v_y_max = args['v_y_max']
    T_y_min = args['T_y_min']
    mask = args['mask']
    TA = args['TA']
    # cut_nodes = args['cut_nodes']
    # dead_nodes = args['dead_nodes']

    # Boundary conditions on x
    # Nothing to do because Phi includes them
    # Boundary conditions on y (Dirichlet)
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    u_s, v_s, T_s, u_n, v_n, T_n = u_y_min, v_y_min, T_y_min, u_y_max, v_y_max, T_inf

    # Boundary conditions on y=y_min
    u[0] = u_s
    v[0] = v_s
    T[0] = (4 * T[1, :] - T[2, :]) / 3 # Derivative using O(h^2)	
    # T[0] = T_s

    # Boundary conditions on y=y_max
    u[-1] = u_n#[-2]
    v[-1] = v_n#[-2]
    # T[-1] = T_n#[-2]#T_n
    T[-1] = (4 * T[-2, :] - T[-3, :]) / 3

    #T[(Ym > y_start) & (Ym <= y_start + y_shift) & (Xm <= x_med + x_shift) & (Xm >= x_med - x_shift)] = TA + T_inf
    T[mask] = TA + T_inf

    # 0 at dead nodes
    # u[dead_nodes] = 0
    # v[dead_nodes] = 0
    # T[dead_nodes] = 0
    # Y[dead_nodes] = 0

    # BC at edge nodes
    # cut_nodes_y, cut_nodes_x = cut_nodes
    # T_s = (4 * T[cut_nodes_y + 1, :] - T[cut_nodes_y + 2, :]) / 3 # Derivative using O(h^2)	
    # Y_s = (4 * Y[cut_nodes_y + 1, :] - Y[cut_nodes_y + 2, :]) / 3 # Derivative using O(h^2)

    # Boundary on y
    # u[cut_nodes] = 0#u_s
    # v[cut_nodes] = 0#v_s
    # T[cut_nodes] = T_s
    # Y[cut_nodes] = Y_s

    return np.array([u, v, T])

def euler(t_n, y_n, dt, args):
    rho = args['rho']
    y_np1 = y_n + dt * Phi(t_n, y_n, **args)
    Ut, Vt = y_np1[:2].copy()
    p = solve_fftfd(Ut, Vt, **args).copy()
    grad_p = grad_pressure(p, **args)
    y_np1[:2] = y_np1[:2] - dt / rho * grad_p
    Ut, Vt, Tt = y_np1.copy()
    y_np1 = boundary_conditions(Ut, Vt, Tt, args)
    return y_np1, p

def rk4(t_n, y_n, dt, **args):
    rho = args['rho']
    k1 = Phi(t_n, y_n, **args)
    k2 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k1, **args)
    k3 = Phi(t_n + 0.5 * dt, y_n + 0.5 * dt * k2, **args)
    k4 = Phi(t_n + dt, y_tmp + dt * k3, **args)
    y_tmp = y_n + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    Ut, Vt, Tt = y_tmp.copy()
    p = solve_fftfd(Ut, Vt, **args).copy()
    grad_p = grad_pressure(p, **args)
    y_np1[:2] = y_tmp[:2] - dt / rho * grad_p
    Ut, Vt = y_tmp[:2].copy()
    y_np1 = boundary_conditions(Ut, Vt, Tt, args)
    return y_np1, p

# def algo(t_n, y_n, dt, args):
#     method = args['method']
#     if method == 'Euler':
#         return euler(t_n, y_n, dt, args)
#     elif method == 'RK4':
#         return rk4(t_n, y_n, dt, args)

# def F(t_n, y_n, dt, args):
#     y_np1 = algo(t_n, y_n, dt, args) 
#     fd = (y_np1 - y_n) / dt
#     return fd

# # This is a very instructive implementation of GMRes.
# def GMRes_Ax(Ax, b, x0=np.array([0.0]), m=1000, flag_display=False, threshold=1e-12):
#     n = b.shape[0]#len(b)
#     if len(x0)==1:
#         x0=np.zeros(n)
#     #r0 = b - np.dot(A, x0)
#     r0 = b - Ax(x0)
#     nr0=np.linalg.norm(r0)
#     out_res=np.array(nr0)
#     Q = np.zeros((n,n))
#     H = np.zeros((n,n))
#     Q[:,0] = r0 / nr0
#     flag_break=False
#     for k in np.arange(np.min((m,n))):
#         #y = np.dot(A, Q[:,k])
#         y = Ax(Q[:,k])
#         if flag_display:
#             print('||y||=',np.linalg.norm(y))
#         for j in np.arange(k+1):
#             H[j][k] = np.dot(Q[:,j], y)
#             if flag_display:
#                 print('H[',j,'][',k,']=',H[j][k])
#             y = y - np.dot(H[j][k],Q[:,j])
#             if flag_display:
#                 print('||y||=',np.linalg.norm(y))
#         # All but the last equation are treated equally. Why?
#         if k+1<n:
#             H[k+1][k] = np.linalg.norm(y)
#             if flag_display:
#                 print('H[',k+1,'][',k,']=',H[k+1][k])
#             if (np.abs(H[k+1][k]) > 1e-16):
#                 Q[:,k+1] = y/H[k+1][k]
#             else:
#                 print('flag_break has been activated')
#                 flag_break=True
#             # Do you remember e_1? The canonical vector.
#             e1 = np.zeros((k+1)+1)        
#             e1[0]=1
#             H_tilde=H[0:(k+1)+1,0:k+1]
#         else:
#             H_tilde=H[0:k+1,0:k+1]
#         # Solving the 'SMALL' least square problem. 
#         # This could be improved with Givens rotations!
#         ck = np.linalg.lstsq(H_tilde, nr0*e1, rcond=None)[0] 
#         if k+1<n:
#             x = x0 + np.dot(Q[:,0:(k+1)], ck)
#         else:
#             x = x0 + np.dot(Q, ck)
#         # Why is 'norm_small' equal to 'norm_full'?
#         norm_small=np.linalg.norm(np.dot(H_tilde,ck)-nr0*e1)
#         out_res = np.append(out_res,norm_small)
#         if flag_display:
#             #norm_full=np.linalg.norm(b-np.dot(A,x))
#             norm_full=np.linalg.norm(b-Ax(x))
#             print('..........||b-A\,x_k||=',norm_full)
#             print('..........||H_k\,c_k-nr0*e1||',norm_small);
#         if flag_break:
#             if flag_display: 
#                 print('EXIT: flag_break=True')
#             break
#         if norm_small<threshold:
#             if flag_display:
#                 print('EXIT: norm_small<threshold')
#             break
#     return x, out_res, H_tilde

# # This function computes the 'matrix-vector' product of the matrix we don't have explicitly stored!!
# def compute_matrix_vector_product(v, y_n, t_n, dt, args, eps=1e-10):
#     V = np.reshape(v, y_n.shape)
#     Jv = (F(t_n, y_n + eps * V, dt, args) - F(t_n, y_n, dt, args)) / eps
#     return Jv.flatten()

def main():
    ### MAIN ###
    sim_name = ""
    if len(sys.argv) > 1:
        sim_name = sys.argv[1]
    # Domain
    # x_min, x_max = 0, 1000
    # y_min, y_max = 0, 200
    x_min, x_max = 0, 200
    y_min, y_max = 0, 400
    t_min, t_max = 0, 120
    Nx, Ny, Nt = 256, 256, 10001 # Number of grid points
    # Nx, Ny, Nt = 64, 128, 10001 # Number of grid points
    # Nx, Ny, Nt = 64, 64, 101
    samples = 100 # Samples to store data
    # Arrays
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    t = np.linspace(t_min, t_max, Nt)
    # Meshgrid
    Xm, Ym = np.meshgrid(x[:-1], y)

    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]

    # Parameters
    nu = 1e-4# [m^2/s]  Viscosity
    rho = 1 # [kg/m^3] Density
    k = 1e-3 # 5 [m^2/s] Thermal diffusivity
    T_inf = 293 # [K] Temperature of the environment
    g = (0, -9.81) # [m/s^2] Gravity
    A = 1e9 # [s^{-1}] Pre-exponential factor (Asensio 2002)
    #A = 2.5e3 # [s^{-1}] Pre-exponential factor
    #A = 1e9
    #E_A = 83.68e3 # [J mol^{-1}] Activation energy (Asensio 2002)
    E_A = 20e3 # [cal mol^{-1}] (Asensio 2002)
    T_pc = 300 # [K] Temperature phase change
    H_R = 21.20e6 # [J/kg] Heat energy per unit of mass (wood) https://en.wikipedia.org/wiki/Heat_of_combustion
    H_R = 1e2
    #R = 8.314 # [J mol^{-1} K^{-1}] Universal gas constant
    R = 1.9872 # [cal mol^{-1} K^{-1}] Universal gas constant (Asensio 2002)
    h = 5e-2 # [W m^{-2}] Convection coefficient
    # Turbulence
    C_s = 0.173 # Smagorinsky constant
    # Pr = 1e1 # Prandtl number
    Pr = nu / k # Prandtl number
    # Pr = 1#000
    # Drag force by solid fuel
    C_D = 1 # [1] Drag coefficient "1 or near to unity according to works of Mell and Linn"
    a_v = 1 # [m] contact area per unit volume between the gas and the solid
    # Options
    turb = True
    conser = False

    # Force term
    fx = lambda x, y: x * 0
    fy = lambda x, y: x * 0 
    FF = lambda x, y: (fx(x, y), fy(x, y))

    # Topography
    A = 50
    sx = 10000
    sy = 10000
    x_c = (x_max + x_min) / 2
    y_c = (y_max + y_min) / 2
    top = lambda x, y: A * np.exp(-((x - x_c) ** 2 / sx + (y - y_c) ** 2 / sy))
    topo = lambda x: top(x, y_c)

    # Initial conditions
    # # Power-law
    # u_r = 5
    # y_r = 1 
    # alpha = 1 / 7
    # # u0 = lambda x, y: u_r * ((y - topo(x)) / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2 # Power-law
    # # u0 = lambda x, y: u_r * ((y_max - topo(x)) / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2 # Power-law
    # u0 = lambda x, y: u_r * (y / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * .5 # Power-law

    # Log wind profile
    z0 = 0.05
    d = 0.1
    ua = .1
    kk = 0.41
    u0 = lambda x, y: np.piecewise(y, [y > 0, y == 0], [lambda y: ua / kk * np.log((y - d) / z0), lambda y: y * 0])

    u0 = lambda x, y: x * y * 0
    v0 = lambda x, y: x * y * 0 
    # y_0_v = (y_max + y_min) / 2
    # sy_v = 10000
    # VV = -1e0
    # v0 = lambda x, y: VV * y * (y_max - y) * np.exp(- (y - y_0_v) ** 2 / sy_v)
    # # v0 = lambda x, y: x * 0 - 1

    Y0 = lambda x, y: x * 0 
    TA = 50 # 700 # 500
    # TA = 800

    # Gaussian Temperature
    # x_0_T, y_0_T = (x_max + x_min) / 2, 0#(y_max - y_min) / 2
    # sx_T, sy_T = 30, 10
    # sx_T, sy_T = 100, 20#17, 45
    # Sigma = np.array([[sx_T, 1], [1, sy_T]])
    # T0 = lambda x, y: 0 * TA * np.exp(-((x - x_0_T) ** 2 / (2 * sx_T ** 2) + (y - y_0_T) ** 2 / (2 * sy_T ** 2))) / (2 * np.pi * sx_T * sy_T) + T_inf
    # # T0 = lambda x, y: multivariate_normal(np.array([x, y]), np.array([x_0_T, y_0_T]), Sigma)
    # # T0 = lambda x, y: (((x - x_0_T) ** 2 / sx_T ** 2 + (y - y_0_T) / sy_T ** 2) <= 1) #* TA + T_inf
    # #AA = np.pi / 20
    # #T0 = lambda x, y: T_inf + TA * ((((x - x_0_T) * np.cos(AA) + (y - y_0_T) * np.sin(AA)) ** 2 / sx_T ** 2 + ((x - x_0_T) * np.sin(AA) - (y - y_0_T) * np.cos(AA)) ** 2 / sy_T ** 2 ) <= 1)
    # Sigma = np.array([
    #     [sx_T, 400],
    #     [400, sy_T]
    #     # [1, .8],
    #     # [.8, 1]
    # ]) 
    # mu = np.array([x_0_T, y_0_T])
    # G2D = create_2d_gaussian(mu, Sigma)
    # T0 = lambda x, y: 2e3 * TA * G2D(x, y) + T_inf
    # x_0_T, y_0_T = (x_max - x_min) / 4, (y_max - y_min) / 2 * 0
    # A = 400
    # B = 273
    # theta = np.pi / 180 * 25 # * 0
    # sx, sy = 50, 30
    # Ta = np.cos(theta) ** 2 / (2 * sx ** 2) + np.sin(theta) ** 2 / (2 * sy ** 2)
    # Tb = np.sin(2 * theta) / (4 * sx ** 2) - np.sin(2 * theta) / (4 * sy ** 2)
    # Tc = np.sin(theta) ** 2 / (2 * sx ** 2) + np.cos(theta) ** 2 / (2 * sy ** 2)
    # theta *= 0
    # Tb *= 0
    # Ta, Tc = 1/2000, 1/10
    # T1 = lambda x, y: TA * np.exp(-(Ta * (x - x_0_T) ** 2 + 2 * Tb * (x - x_0_T) * (y - y_0_T) + Tc * (y - y_0_T) ** 2))
    # T0 = lambda x, y: T1(x, y) + T1(x-50, y)*0 + T_inf  #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 20


    # Temperature square
    # T_0 = np.zeros_like(Xm) + T_inf
    # T_0[(Ym <= 10) & (Xm >= 200) & (Xm <= 300)] = TA + T_inf

    # print(np.max(G2D(Xm, Ym)))


    # Gaussian fuel
    x_0_Y, y_0_Y = (x_max - x_min) / 2, 0 
    sx_Y, sy_Y = 100000, 50
    Y0 = lambda x, y: np.exp(-((x - x_0_Y) ** 2 / sx_Y + (y - y_0_Y) ** 2 / sy_Y))

    # # Building
    # x_lims = [x[Nx // 2 - 6], x[Nx // 2 + 6]] #[(x_max + x_min) / 2 - dx * 10, (x_max + x_min) / 2 + dx * 10]
    # y_lims = [0, y[Ny // 4]]
    # cut_nodes, dead_nodes = building(Xm, Ym, x_lims, y_lims, dx, dy)

    p0 = lambda x, y: x * 0 #+ 1e-12

    U_0 = u0(Xm, Ym)
    V_0 = v0(Xm, Ym)
    # T_0 = T0(Xm, Ym) 
    # Y_0 = Y0(Xm, Ym) 


    # Plate
    # T_0[0, (x [:-1]<= 300) & (x[:-1] >= 200)] = TA + T_inf
    x_med = (x_max + x_min) / 2
    y_start = 50
    x_shift = 25
    y_shift = 5
    # T_0[(y <= 5), (x[:-1] <= x_med + shift) & (x[:-1] >= x_med - shift)] = TA + T_inf
    #T_0[(Ym > y_start) & (Ym <= y_start + y_shift) & (Xm <= x_med + x_shift) & (Xm >= x_med - x_shift)] = TA + T_inf

    mask = (Ym > y_start) & (Ym <= y_start + y_shift) & (Xm <= x_med + x_shift) & (Xm >= x_med - x_shift)
    
    # Circle
    # R = 15
    # x_T_0, y_T_0 = (x_min + x_max) / 2, 75
    # # T_0 = TA * (((Xm - x_T_0) ** 2 + (Ym - y_T_0) ** 2 / 4) <= R ** 2) + T_inf
    # mask = ((Xm - x_T_0) ** 2 + (Ym - y_T_0) ** 2 / 4) <= R ** 2
    
    T_0 = T_inf + Xm*0
    T_0[mask] = T_inf + TA

    # Dead nodes
    # U_0[dead_nodes] = 0
    # V_0[dead_nodes] = 0
    # T_0[dead_nodes] = 0import matplotlib.pyplot as plt
    # Y_0[dead_nodes] = 0

    # import matplotlib.pyplot as plt
    # plt.contourf(Xm, Ym, T_0, cmap=plt.cm.jet)
    # plt.colorbar()
    # plt.show()
    # #plt.savefig()
    # return False
    # # print(asd)

    L = ((x_max - x_min) * (y_max - y_min)) ** 0.5
    Re = np.mean(U_0[:,0]) * L / nu
    T_avg = np.mean(T_0[:, Nx // 2])
    beta = 1 / T_avg
    L_v = (y_max - y_min)
    Gr = abs(g[-1]) * beta * (TA + T_inf - T_inf) * L_v ** 3 / nu ** 2 
    Ra = Gr * Pr


    ## INFO ##
    print("Simulation id:", sim_name)
    print("Domain: [%.2f, %.2f] x [%.2f, %.2f] x [%.2f, %.2f]" % (x_min, x_max, y_min, y_max, t_min, t_max))
    print("Grid: Nx: %d, Ny: %d, Nt: %d" % (Nx, Ny, Nt))
    print("dx: %.2f, dy: %.2f, dt: %.2f" % (dx, dy, dt))
    print("nu: %.2e, k: %.2e, T_inf: %.2f, T_hot: %.2f, g: (%.2f, %.2f)" % (nu, k, T_inf, T_inf + TA, g[0], g[1]))
    print("Turbulence: %r" % turb)
    print("Conservative: %r" % conser)
    print("Reynolds: %.2f" %  Re)
    print("Prandtl: %.2f" % Pr)
    print("Grashof: %.2e" % Gr)
    print("Rayleigh: %.2e\n" % Ra)

    # Boundary conditions
    u_y_min = U_0[0]#U_0[cut_nodes]
    u_y_max = U_0[-1]
    v_y_min = V_0[0]
    v_y_max = 0
    p_y_max = 0
    T_y_min = T_0[0]

    y_0 = np.array([U_0, V_0, T_0])
    p_0 = p0(Xm, Ym)

    # Array for approximations
    yy = np.zeros((Nt, y_0.shape[0], Ny, Nx - 1)) 
    P  = np.zeros((Nt, Ny, Nx - 1))

    yy[0] = y_0
    P[0]  = p_0
    F_e = FF(Xm, Ym)

    method = 'Euler'

    methods = {
        'Euler': euler,
        'RK4': rk4,
    }

    args = {
        'x': x,
        'y': y,
        'Y': Ym,
        'dx': dx,
        'dy': dy,
        'dt': dt,
        'nu': nu,
        'rho': rho,
        'k': k,
        'g': g,
        'T_inf': T_inf,
        'A': A,
        'E_A': E_A,
        'F': F_e,
        'u0': U_0,
        'v0': V_0,
        'u_y_min': u_y_min,
        'u_y_max': u_y_max,
        'v_y_min': v_y_min,
        'v_y_max': v_y_max,
        'p_y_max': p_y_max,
        'Nx': Nx,
        'Ny': Ny,
        'Nt': Nt,
        'Pr': Pr,
        'C_s': C_s,
        'T_pc': T_pc,
        'H_R': H_R,
        'R': R,
        'h': h,
        'C_D': C_D,
        'a_v': a_v,
        'turb': turb,
        'conservative': conser,
        # 'cut_nodes': cut_nodes,
        # 'dead_nodes': dead_nodes,
        'Ym': Ym,
        'T_y_min': T_y_min,
        'mask': mask,
        'method': method,
        'TA': TA,
    }

    #Ax = lambda x: compute_matrix_vector_product(x, y_0, 0)


    # y_tmp = yy[0].copy()
    # p_tmp = P[0].copy()

    start = time.time()

    # H = []
    # x_gmres_ = []
    # output_gmres_ = []

    for n in range(Nt - 1):
        # Simulation 
        print("Step:", n)
        yy[n+1], P[n+1] = methods[method](t[n], yy[n], dt, args)
        Ut, Vt, Tt = yy[n+1].copy()
        print("CFL", dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy))
        # For stability we store the Hessemberg matrix for each 
        # print("Computing")
        # Ax = lambda x: compute_matrix_vector_product(x, yy[n], t[n], dt, args)
        # b = np.ones(3 * (Nx-1) * Ny)
        # x0 = np.zeros_like(b)
        # x_gmres, output_gmres, Hessemgber = GMRes_Ax(Ax, b, x0)
        # x_gmres_.append(x_gmres)
        # output_gmres_.append(output_gmres)
        # H.append(Hessemgber)

    # x_gmres_ = np.array(x_gmres_)
    # output_gmres_ = np.array(output_gmres_)
    # H = np.array(H)

    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\nElapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes), seconds))

    # Get velocities
    U = yy[:, 0]
    V = yy[:, 1]
    T = yy[:, 2]
    # Y = yy[:, 3]

    # Copy last column
    U_ = np.zeros((Nt, Ny, Nx))
    V_ = np.zeros((Nt, Ny, Nx))
    T_ = np.zeros((Nt, Ny, Nx))
    # Y_ = np.zeros((Nt // samples + 1, Ny, Nx))
    P_ = np.zeros((Nt, Ny, Nx))
    U_[:, :, :-1] = U
    U_[:, :, -1] = U[:, :, 0]
    V_[:, :, :-1] = V
    V_[:, :, -1] = V[:, :, 0]
    T_[:, :, :-1] = T
    T_[:, :, -1] = T[:, :, 0]
    # Y_[:, :, :-1] = Y
    # Y_[:, :, -1] = Y[:, :, 0]
    P_[:, :, :-1] = P
    P_[:, :, -1] = P[:, :, 0]
    U = U_.copy()
    V = V_.copy()
    T = T_.copy()
    # Y = Y_.copy()
    P = P_.copy()

    # Save approximation
    # turb_str = "turbulence"
    # if not turb:
    #     turb_str = "no_turbulence"

    # cons_str = "conservative"
    # if not conser:
    #     cons_str = "non_conservative"

    # Save approximation
    if sim_name == "":
        output = '../output/temperature_stability.npz'#.format(turb_str, cons_str)
    else:
        output = '../output/{}/temperature_stability.npz'.format(sim_name) 
    np.savez(output, u=U, v=V, T=T, p=P, x=x, y=y, t=t)
    # np.savez(output, yy=yy, x=x, y=y, t=t)

    with open('../output/{}/arguments.pkl'.format(sim_name), 'wb') as f:
        pickle.dump(args, f)
        

if __name__ == "__main__":
    main()
