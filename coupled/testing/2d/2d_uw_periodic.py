import numpy as np
from poisson import fft_solver

f_sgs = lambda z, A: (1 - np.exp(-z / A)) ** (3 / 2) #1 - np.exp(-z / A)

def grad_pressure(p, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    px, py = np.zeros_like(p), np.zeros_like(p)
    # Get nodes
    p_ip1j = np.roll(p, -1, axis=1) # p_{i+1, j}
    p_im1j = np.roll(p, 1, axis=1) # p_{i-1, j}
    p_ijp1 = np.roll(p, -1, axis=0) # p_{i, j+1}
    p_ijm1 = np.roll(p, 1, axis=0) # p_{i, j-1}
    
    # First derivative using central difference O(h^2).
    px = (p_ip1j - p_im1j) / (2 * dx)
    py = (p_ijp1 - p_ijm1) / (2 * dy)

    # Derivatives at boundary, dp/dy at y = y_min and y = y_max
    # Periodic on x, included before
    # Periodic on y, included before
    # py[0, 1:-1] = (-3 * p[0, 1:-1] + 4 * p[1, 1:-1] - p[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # py[-1, 1:-1] = (3 * p[-1, 1:-1] - 4 * p[-2, 1:-1] + p[-3, 1:-1]) / (2 * dy) # Backward at y=y_max

    return np.array([px, py])

def solve_pressure_iterative(u, v, p, n_iter=1000, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    dt = kwargs['dt']
    rho = kwargs['rho']
    p_y_max = kwargs['p_y_max']

    # div(u) 
    # Get nodes for u and v
    u_ip1j = np.roll(u, -1, axis=1) # u_{i+1, j}
    u_im1j = np.roll(u, 1, axis=1) # u_{i-1, j}
    v_ijp1 = np.roll(v, -1, axis=0) # v_{i, j+1}
    v_ijm1 = np.roll(v, 1, axis=0) # v_{i, j-1}

    # First derivative using central difference O(h^2)
    ux = (u_ip1j - u_im1j) / (2 * dx)
    vy = (v_ijp1 - v_ijm1) / (2 * dy)
    b = rho / dt * (ux + vy)
    # Iterative
    for n in range(n_iter):
        pn = p.copy()
        p[1:-1, 1:-1] = (
            dy ** 2 * (pn[1:-1, 2:] + pn[1:-1, :-2]) + 
            dx ** 2 * (pn[2:, 1:-1] + pn[:-2, 1:-1]) - 
            dx ** 2 * dy ** 2 * b[1:-1,1:-1]
        ) / (2 * (dx ** 2 + dy ** 2))

        # Periodic BC Pressure at x = x_min
        p[1:-1, 0] = (
            dy ** 2 * (pn[1:-1,1] + pn[1:-1,-1]) +
            dx ** 2 * (pn[2: , 0] + pn[:-2, 0]) -
            dx ** 2 * dy ** 2 * b[1:-1, 0]
        ) / (2 * (dx ** 2 + dy ** 2)) 

        # Periodic BC Pressure at x = x_max
        p[1:-1, -1] = (
            dy ** 2 * (pn[1:-1,0] + pn[1:-1,-2]) +
            dx ** 2 * (pn[2:, -1] + pn[:-2, -1]) -
            dx ** 2 * dy ** 2 * b[1:-1, -1]
        ) / (2 * (dx ** 2 + dy ** 2)) 
        
        # Boundary conditions
        # dp/dy = 0 at y = y_min
        # Forward difference O(h^2)
        # p[0] = (4 * p[1, :] - p[2, :]) / 3 
        # p=p_y_max conditions at y = y_max
        # p[-1,:] = p_y_max 
        #p[-1] = (4 * p[-2, :] - p[-3, :]) / 3 
        
        # Bye Jacobi!
        if np.linalg.norm(pn.flatten() - p.flatten()) < 1e-10:
            break

    return p

def solve_fft(u, v, **kwargs):
    x = kwargs['x']
    y = kwargs['y']
    dt = kwargs['dt']
    rho = kwargs['rho']
    p_y_max = kwargs['p_y_max']

    # div(u) 
    # Get nodes for u and v
    u_ip1j = np.roll(u, -1, axis=1) # u_{i+1, j}
    u_im1j = np.roll(u, 1, axis=1) # u_{i-1, j}
    v_ijp1 = np.roll(v, -1, axis=0) # v_{i, j+1}
    v_ijm1 = np.roll(v, 1, axis=0) # v_{i, j-1}

    # First derivative using central difference O(h^2)
    ux = (u_ip1j - u_im1j) / (2 * dx)
    vy = (v_ijp1 - v_ijm1) / (2 * dy)
    f = rho / dt * (ux + vy)
    f = np.vstack([f, f[0]])
    f = np.hstack([f, f[:,0].reshape(-1, 1)])
    p = fft_solver(x, y, f)

    return p[:-1, :-1]

def turbulence(u, v, ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, args):
    dx = args['dx']
    dy = args['dy']
    C_s = args['C_s'] 
    Pr = args['Pr']
    y = args['y']
    Delta = (dx * dy) ** (1/2)
    # z_plus = y * u / (v + 1e-16)

    # Turbulence
    S_ij_mod = (2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2) ** (1 / 2) + 1e-16
    # Mixed derivatives
    vxy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
    uyx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dx)
    vyx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
    uxy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2 * dy)
    # Mixed derivatives at boundaries
    # Periodic on x (nothing to do)
    # On y
    # vxy[0, :] = (-3 * vx[0, :] + 4 * vx[1, :] - vx[2, :]) / (2 * dy) # Forward at y=y_min
    # vxy[-1, :] = (3 * vx[-1, :] - 4 * vx[-2, :] + vx[-3, :]) / (2 * dy) # Backward at y=y_max
    # uxy[0, :] = (-3 * ux[0, :] + 4 * ux[1, :] - ux[2, :]) / (2 * dy) # Forward at y=y_min
    # uxy[-1, :] = (3 * ux[-1, :] - 4 * ux[-2, :] + ux[-3, :]) / (2 * dy) # Backward at y=y_max

    # 'psi_x' and 'psi_y'
    psi_x = 4 * (ux * uxx + vy * vyx) + 2 * (uy + vx) * (uyx + vxx) 
    psi_y = 4 * (ux * uxy + vy * vyy) + 2 * (uy + vx) * (uyy + vxy)

    l = C_s * Delta #* f_sgs(z_plus, 25)

    sgs_x = -2 * l ** 2 * ( 
        1 / (2 * S_ij_mod) * psi_x * ux + 0.5 * psi_y * (uy + vx) +
        S_ij_mod * (uxx + 0.5 * (vxy + uyy))
    )
    sgs_y = -2 * l ** 2 * (
        1 / (2 * S_ij_mod) * psi_y * vy + 0.5 * psi_x * (vx + uy) +
        S_ij_mod * (vyy + 0.5 * (uyx + vxx))
    )

    # SGS thermal energy
    sgs_T = -l ** 2 / Pr * (
        1 / (2 * S_ij_mod) * (psi_x * Tx + psi_y * Ty) + S_ij_mod * (Txx + Tyy)
    )

    return np.array([sgs_x, sgs_y, sgs_T])

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
    # Pr = kwargs['Pr']
    # C_s = args['C_s'] 
    turb = kwargs['turb']
    conservative = kwargs['conservative']

    # Get variables
    u, v, T, Y = C
    
    # Forces
    F_x, F_y = F
    g_x, g_y = g
    F_x = F_x - g_x * (T - T_inf) / T 
    F_y = F_y - g_y * (T - T_inf) / T

    # New values (to do)
    Y_ = np.zeros_like(Y)
    
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
    T_ijm1 = np.roll(T, 1, axis=1) # T_{i, j-1}
    
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
    
    # Central difference O(h^2) (for turbulence)
    ux = (u_ip1j - u_im1j) / (2 * dx)
    uy = (u_ijp1 - u_ijm1) / (2 * dy)
    vx = (v_ip1j - v_im1j) / (2 * dx)
    vy = (v_ijp1 - v_ijm1) / (2 * dy)
    Tx = (T_ip1j - T_im1j) / (2 * dx)
    Ty = (T_ijp1 - T_ijm1) / (2 * dy)

    # Conservative form for convection
    uux = (u_ip1j ** 2 - u_im1j ** 2) / (2 * dx) # (u_{i+1, j}^2 - u_{i-1, j}^2) / (2 * dx)
    uvy = (u_ijp1 * v_ijp1 - u_ijm1 * v_ijm1) / (2 * dx)
    vux = (v_ip1j * u_ip1j - v_im1j * u_im1j) / (2 * dx)
    vvy = (v_ijp1 ** 2 - v_im1j ** 2) / (2 * dy)

    # Second derivatives
    uxx = (u_ip1j - 2 * u_ij + u_im1j) / dx ** 2
    uyy = (u_ijp1 - 2 * u_ij + u_ijm1) / dy ** 2
    vxx = (v_ip1j - 2 * v_ij + v_im1j) / dx ** 2
    vyy = (v_ijp1 - 2 * v_ij + v_ijm1) / dy ** 2
    Txx = (T_ip1j - 2 * T_ij + T_im1j) / dx ** 2
    Tyy = (T_ijp1 - 2 * T_ij + T_ijm1) / dy ** 2

    # Turbulence
    sgs_x = sgs_y = sgs_T = 0
    if turb:
        sgs_x, sgs_y, sgs_T = turbulence(u, v, ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, kwargs)

    # Reaction Rate
    # K = A * np.exp(-E_A / (R * T))
    # K[T < T_pc] = 0 # Only for T > T_pc

    # # Temperature source term
    # S = rho * H_R * Y * K - h * (T - T_inf)

    if conservative: # RHS Inside domain (conservative form - using central difference)
        U_ = nu * (uxx + uyy) - (uux + uvy) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (vux + vvy) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx + v * Ty) - sgs_T #+ S
    else: # RHS Inside domain (non-conservative form - using upwind!)
        U_ = nu * (uxx + uyy) - (u_plu * uxm + u_min * uxp + v_plu * uym + v_min * uyp) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (u_plu * vxm + u_min * vxp + v_plu * vym + v_min * vyp) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx  + v * Ty) - sgs_T # + S

    

    # Fuel
    # Y_ = -Y * K
    # Y_[Y_ < 0] = 0 # Fix fuel values

    # U_, V_, T_, Y_ = boundary_conditions(U_, V_, T_, Y_, kwargs)

    return np.array([U_, V_, T_, Y_])

def boundary_conditions(u, v, T, Y, args):
    T_inf = args['T_inf']
    u0 = args['u0']
    v0 = args['v0']
    u_y_min = args['u_y_min']
    u_y_max = args['u_y_max']

    # Boundary conditions on x
    # Nothing to do because Phi includes them
    # Boundary conditions on y (Dirichlet)
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    u_s, v_s, T_s, Y_s, u_n, v_n, T_n, Y_n = u_y_min, 0, T_inf, 0, u_y_max, 0, T_inf, 0
    T_s = (4 * T[1, :] - T[2, :]) / 3 # Derivative using O(h^2)	
    Y_s = (4 * Y[1, :] - Y[2, :]) / 3 # Derivative using O(h^2)

    # Boundary on y
    u[0] = u_s
    u[-1] = u_n
    v[0] = v_s
    v[-1] = v_n
    T[0] = T_s
    T[-1] = T_n
    Y[0] = Y_s
    Y[-1] = Y_n

    return np.array([u, v, T, Y])

### MAIN ###
# Domain
x_min, x_max = 0, 2 * np.pi
y_min, y_max = 0, 2 * np.pi #100
t_min, t_max = 0, 10
Nx, Ny, Nt = 64, 64, 2001 # Number of grid points
samples = 100 # Samples to store data
# Arrays
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)
# Meshgrid
Xm, Ym = np.meshgrid(x, y)

dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
print(dx, dy, dt)

# Parameters
nu = 1e-6
rho = 1
k = 1e-5
T_inf = .5
g = (0, 9.81)
g = (0, 0)
A = 2.5e3 #1e1
E_A = 83.68e3 #2e1 #200 [cal mol^{-1}]
T_pc = 500 # [K] Temperature phase change
H_R = 21.20e6 * 0 # [J/kg] Heat energy per unit of mass (wood) https://en.wikipedia.org/wiki/Heat_of_combustion
R = 8.314 # [J mol^{-1} K^{-1}] Universal gas constant
h = 1e-1 # [W m^{-2}] Convection coefficient
Pr = .1 #nu / k #.1
C_s = 0.173
turb = True
conser = False

# Force term
fx = lambda x, y: x * 0
fy = lambda x, y: x * 0 
F = lambda x, y: (fx(x, y), fy(x, y))

# Initial conditions
u_r = 1
y_r = 1
alpha = 1 / 7
TU = u_r 
x_0_u, y_0_u = (x_max - x_min) / 4, (y_max - y_min) / 2
sx_u, sy_u = .25, .25
# u0 = lambda x, y: u_r * (y / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2 # Power-law
# u0 = lambda x, y: - 1 / y_max / 5 *(y - y_min) * (y - y_max)#u_r * (1 - ((y - (y_max + y_min) / 2) / (y_max - y_min)) ** 2) # Parabolic
u0 = lambda x, y:  TU * np.exp(-((x - x_0_u) ** 2 / sx_u + (y - y_0_u) ** 2 / sy_u)) + .1#+ u_r
v0 = lambda x, y: x * 0 
# v0 = lambda x, y: 2 * np.cos(np.pi * x) * np.sin(np.pi * y) 
Y0 = lambda x, y: x * 0
TA = 1 # 700 # 500

# Gaussian Temperature
x_0_T, y_0_T = (x_max - x_min) / 4, (y_max - y_min) / 2
sx_T, sy_T = .25, .25
T0 = lambda x, y: TA * np.exp(-((x - x_0_T) ** 2 / sx_T + (y - y_0_T) ** 2 / sy_T)) + T_inf

# Gaussian fuel
x_0_Y, y_0_Y = (x_max - x_min) / 2, 0 
sx_Y, sy_Y = 100000, 50
Y0 = lambda x, y: np.exp(-((x - x_0_Y) ** 2 / sx_Y + (y - y_0_Y) ** 2 / sy_Y))

# import matplotlib.pyplot as plt
# plt.quiver(Xm, Ym, u0(Xm, Ym), v0(Xm, Ym))
# plt.contourf(Xm, Ym, u0(Xm, Ym), cmap=plt.cm.jet, alpha=.75)
# plt.colorbar()
# plt.show()
# plt.contourf(Xm, Ym, T0(Xm, Ym), cmap=plt.cm.Oranges)
# plt.colorbar()
# plt.show()
# print(asdasd)

p0 = lambda x, y: x * 0 #+ 1e-12

U_0 = u0(Xm, Ym)
V_0 = v0(Xm, Ym)
T_0 = T0(Xm, Ym)
Y_0 = Y0(Xm, Ym)

# Removing last row and column
U_0 = U_0[:-1, :-1]
V_0 = V_0[:-1, :-1]
Y_0 = Y_0[:-1, :-1]
T_0 = T_0[:-1, :-1]

# Boundary conditions
u_y_min = U_0[0]
u_y_max = U_0[-1]
p_y_max = 0

y_0 = np.array([U_0, V_0, T_0, Y_0])
p_0 = p0(Xm, Ym)
p_0 = p_0[:-1, :-1]

# Array for approximations
yy = np.zeros((Nt // samples + 1, y_0.shape[0], Ny - 1, Nx - 1)) 
P  = np.zeros((Nt // samples + 1, Ny - 1, Nx - 1))
yy[0] = y_0
P[0]  = p_0
F_e = F(Xm, Ym)
tmp = [0, 0]
tmp[0] = F_e[0][:-1, :-1]
tmp[1] = F_e[1][:-1, :-1]
F_e = list(tmp)

args = {
    'x': x,
    'y': y,#Ym[:-1, :-1],
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
    'p_y_max': p_y_max,
    'Nx': Nx,
    'Ny': Ny,
    'Pr': Pr,
    'C_s': C_s,
    'T_pc': T_pc,
    'H_R': H_R,
    'R': R,
    'h': h,
    'turb': turb,
    'conservative': conser,
}

y_tmp = yy[0].copy()
p_tmp = P[0].copy()

met = 1
time_method = ['Euler', 'RK4']
method = time_method[met]

if method == 'Euler': # Solve Euler
    for n in range(Nt - 1):
        #print("Step:", n)
        # U, V, T, Y = y_tmp.copy() 
        y_tmp = y_tmp + dt * Phi(t[n], y_tmp, **args)
        Ut, Vt = y_tmp[:2].copy()
        # print(np.min(Tt), np.max(Tt))
        p_tmp = solve_pressure_iterative(Ut, Vt, p_tmp, **args).copy()
        # p_tmp = solve_pressure(Ut, Vt, **args).copy()
        grad_p = grad_pressure(p_tmp, **args)
        y_tmp[:2] = y_tmp[:2] - dt / rho * grad_p
        Ut, Vt = y_tmp[:2].copy()
        # y_tmp = boundary_conditions(Ut, Vt, Tt, Yt, args)
        # Save samples
        if n % samples == 0:
            print("Step:", n)
            P[n // samples + 1] = p_tmp.copy()
            yy[n // samples + 1] = y_tmp.copy()
            Ut, Vt = y_tmp[:2]
            print("CFL", dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy))

elif method == 'RK4': # Solve RK4
    for n in range(Nt - 1):
        U, V, T, Y = y_tmp.copy() 
        k1 = Phi(t[n], y_tmp, **args)
        k2 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k1, **args)
        k3 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k2, **args)
        k4 = Phi(t[n] + dt, y_tmp + dt * k3, **args)
        y_tmp = y_tmp + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        Ut, Vt, Tt, Yt = y_tmp.copy()
        #p_tmp = solve_pressure_iterative(Ut, Vt, p_tmp, **args).copy()
        p_tmp = solve_fft(Ut, Vt, **args)
        grad_p = grad_pressure(p_tmp, **args)
        y_tmp[:2] = y_tmp[:2] - dt / rho * grad_p
        Ut, Vt = y_tmp[:2].copy()
        # U_, V_, T_, Y_ = boundary_conditions(Ut, Vt, Tt, Yt, args)
        # y_tmp[0] = U_
        # y_tmp[1] = V_
        # y_tmp[2] = T_
        # y_tmp[3] = Y_
        if n % samples == 0:
            print("Step:", n)
            P[n // samples + 1] = p_tmp
            yy[n // samples + 1] = y_tmp
            Ut, Vt, Wt = y_tmp[:3]
            print("CFL", dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy))

# Last approximation
yy[-1] = y_tmp

# Get velocities
U = yy[:, 0]
V = yy[:, 1]
T = yy[:, 2]
Y = yy[:, 3]

# Copy last column
U_ = np.zeros((Nt // samples + 1, Ny, Nx))
V_ = np.zeros((Nt // samples + 1, Ny, Nx))
T_ = np.zeros((Nt // samples + 1, Ny, Nx))
Y_ = np.zeros((Nt // samples + 1, Ny, Nx))
P_ = np.zeros((Nt // samples + 1, Ny, Nx))
U_[:, :-1,:-1] = U
U_[:, :-1, -1] = U[:, :, 0]
U_[:, -1,:] = U_[:, 0, :]
V_[:, :-1,:-1] = V
V_[:, :-1, -1] = V[:, :, 0]
V_[:, -1,:] = V_[:, 0, :]
T_[:, :-1,:-1] = T
T_[:, :-1, -1] = T[:, :, 0]
T_[:, -1,:] = T_[:, 0, :]
Y_[:, :-1,:-1] = Y
Y_[:, :-1, -1] = Y[:, :, 0]
Y_[:, -1,:] = Y_[:, 0, :]
P_[:, :-1,:-1] = P
P_[:, :-1, -1] = P[:, :, 0]
P_[:, -1,:] = P_[:, 0, :]
U = U_.copy()
V = V_.copy()
T = T_.copy()
Y = Y_.copy()
P = P_.copy()

# Save approximation
turb_str = "turbulence"
if not turb:
    turb_str = "no_turbulence"

np.savez('../output/2d_uw_periodic_test_{}.npz'.format(turb_str), U=U, V=V, T=T, Y=Y, P=P, x=x, y=y, t=t[::samples])