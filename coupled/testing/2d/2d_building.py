import numpy as np
from poisson import solve_fftfd, solve_iterative, solve_iterative_ibm, solve_gmres
from turbulence import turbulence
from ibm import building, cylinder

def grad_pressure(p, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    cut_nodes_y, cut_nodes_x = kwargs['cut_nodes']
    dead_nodes = kwargs['dead_nodes']
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
    u, v, T, Y = C
    
    # Forces
    F_x, F_y = F
    # g_x, g_y = g
    # # Drag force
    # mod_U = np.sqrt(u ** 2 + v ** 2)
    # mask = Y > 0.5 # Valid only for solid fuel
    # F_d_x = rho * C_D * a_v * mod_U * u * mask
    # F_d_y = rho * C_D * a_v * mod_U * v * mask
    
    # # All forces
    # F_x = F_x - g_x * (T - T_inf) / T - F_d_x 
    # F_y = F_y - g_y * (T - T_inf) / T - F_d_y

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
    # Fixed boundary nodes
    # uy[0, 1:-1] = (-3 * u_ij[0, 1:-1] + 4 * u_ij[1, 1:-1] - u_ij[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # uy[-1, 1:-1] = (3 * u_ij[-1, 1:-1] - 4 * u_ij[-2, 1:-1] + u_ij[-3, 1:-1]) / (2 * dy) # Backward at y=y_max
    # vy[0, 1:-1] = (-3 * v_ij[0, 1:-1] + 4 * v_ij[1, 1:-1] - v_ij[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # vy[-1, 1:-1] = (3 * v_ij[-1, 1:-1] - 4 * v_ij[-2, 1:-1] + v_ij[-3, 1:-1]) / (2 * dy) # Backward at y=y_max
    uy[0, :] = (-3 * u_ij[0, :] + 4 * u_ij[1, :] - u_ij[2, :]) / (2 * dy) # Forward at y=y_min
    uy[-1, :] = (3 * u_ij[-1, :] - 4 * u_ij[-2, :] + u_ij[-3, :]) / (2 * dy) # Backward at y=y_max
    vy[0, :] = (-3 * v_ij[0, :] + 4 * v_ij[1, :] - v_ij[2, :]) / (2 * dy) # Forward at y=y_min
    vy[-1, :] = (3 * v_ij[-1, :] - 4 * v_ij[-2, :] + v_ij[-3, :]) / (2 * dy) # Backward at y=y_max

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
        T_ = k * (Txx + Tyy) - (u * Tx + v * Ty) + S - sgs_T
    else: # RHS Inside domain (non-conservative form - using upwind!)
        U_ = nu * (uxx + uyy) - (u_plu * uxm + u_min * uxp + v_plu * uym + v_min * uyp) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (u_plu * vxm + u_min * vxp + v_plu * vym + v_min * vyp) + F_y - sgs_y
        T_ = k * (Txx + Tyy) - (u * Tx  + v * Ty) + S - sgs_T 

    # Fuel
    Y_ = -Y #* K

    U_, V_, T_, Y_ = boundary_conditions(U_, V_, T_, Y_, kwargs)

    return np.array([U_, V_, T_, Y_])

def boundary_conditions(u, v, T, Y, args):
    T_inf = args['T_inf']
    u0 = args['u0']
    v0 = args['v0']
    u_y_min = args['u_y_min']
    u_y_max = args['u_y_max']
    v_y_min = args['v_y_min']
    v_y_max = args['v_y_max']
    cut_nodes = args['cut_nodes']
    dead_nodes = args['dead_nodes']

    # Boundary conditions on x
    # Nothing to do because Phi includes them
    # Boundary conditions on y (Dirichlet)
    # u = u_y_min, v = 0, dT/dy = 0 at y = y_min
    # u = u_y_max, v = 0, T=T_inf at y = y_max
    u_s, v_s, T_s, Y_s, u_n, v_n, T_n, Y_n = u_y_min, v_y_min, T_inf, 0, u_y_max, v_y_max, T_inf, 0

    # Boundary conditions on y=y_min
    u[0] = u_s
    v[0] = v_s

    # Boundary conditions on y=y_max
    u[-1] = u_n
    v[-1] = v_n
    T[-1] = T_n
    Y[-1] = Y_n

    # 0 at dead nodes
    u[dead_nodes] = 0
    v[dead_nodes] = 0
    T[dead_nodes] = 0
    Y[dead_nodes] = 0

    # BC at edge nodes
    # cut_nodes_y, cut_nodes_x = cut_nodes
    # T_s = (4 * T[cut_nodes_y + 1, :] - T[cut_nodes_y + 2, :]) / 3 # Derivative using O(h^2)	
    # Y_s = (4 * Y[cut_nodes_y + 1, :] - Y[cut_nodes_y + 2, :]) / 3 # Derivative using O(h^2)

    # Boundary on y
    u[cut_nodes] = 0#u_s
    v[cut_nodes] = 0#v_s
    # T[cut_nodes] = T_s
    # Y[cut_nodes] = Y_s

    return np.array([u, v, T, Y])

### MAIN ###
# Domain
x_min, x_max = 0, 600
y_min, y_max = 0, 100
t_min, t_max = 0, 100
Nx, Ny, Nt = 256, 128, 2001 # Number of grid points
samples = 100 # Samples to store data
# Arrays
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)
# Meshgrid
Xm, Ym = np.meshgrid(x[:-1], y)

dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
print(dx, dy, dt)

# Parameters
nu = 1e-6 # [m^2/s]  Viscosity
rho = 1 # [kg/m^3] Density
k = 1e-1 # 5 [m^2/s] Thermal diffusivity
T_inf = 273 # [K] Temperature of the environment
g = (0, 9.81) # [m/s^2] Gravity
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
Pr = .1 # Prandtl number
Pr = nu / k # Prandtl number
# Drag force by solid fuel
C_D = 1 # [1] Drag coefficient "1 or near to unity according to works of Mell and Linn"
a_v = 1 # [m] contact area per unit volume between the gas and the solid
# Options
turb = True
conser = False

# Force term
fx = lambda x, y: x * 0
fy = lambda x, y: x * 0 
F = lambda x, y: (fx(x, y), fy(x, y))

# Topography
A = 50
sx = 10000
sy = 10000
x_c = (x_max + x_min) / 2
y_c = (y_max + y_min) / 2
top = lambda x, y: A * np.exp(-((x - x_c) ** 2 / sx + (y - y_c) ** 2 / sy))
topo = lambda x: top(x, y_c)

# Initial conditions
u_r = 5
y_r = 1 
alpha = 1 / 7
# u0 = lambda x, y: u_r * ((y - topo(x)) / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2 # Power-law
# u0 = lambda x, y: u_r * ((y_max - topo(x)) / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2 # Power-law
u0 = lambda x, y: u_r * (y / y_r) ** alpha #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2 # Power-law
#u0 = lambda x, y: u_r + x * y * 0
v0 = lambda x, y: x * 0 #+ np.random.rand(Xm.shape[0], Xm.shape[1]) * 2
# v0 = lambda x, y: 2 * np.cos(np.pi * x) * np.sin(np.pi * y) 
Y0 = lambda x, y: x * 0 
TA = 0 #400 # 700 # 500

# Gaussian Temperature
x_0_T, y_0_T = (x_max - x_min) / 4, 0#(y_max - y_min) / 4
sx_T, sy_T = 500, 200
T0 = lambda x, y: TA * np.exp(-((x - x_0_T) ** 2 / sx_T + (y - y_0_T) ** 2 / sy_T)) + T_inf

# Gaussian fuel
x_0_Y, y_0_Y = (x_max - x_min) / 2, 0 
sx_Y, sy_Y = 100000, 50
Y0 = lambda x, y: np.exp(-((x - x_0_Y) ** 2 / sx_Y + (y - y_0_Y) ** 2 / sy_Y))

# Building
x_lims = [x[Nx // 2 - 6], x[Nx // 2 + 6]] #[(x_max + x_min) / 2 - dx * 10, (x_max + x_min) / 2 + dx * 10]
y_lims = [0, y[Ny // 4]]
cut_nodes, dead_nodes = building(Xm, Ym, x_lims, y_lims, dx, dy)

# import matplotlib.pyplot as plt
# A = np.zeros_like(Xm)
# B = np.zeros_like(Xm)
# A[cut_nodes] = 1
# B[dead_nodes] = 1
# plt.contourf(Xm, Ym, A)
# # plt.contourf(Xm, Ym, B)
# # plt.imshow(A, origin='lower', extent=[x_min, x_max-dx, y_min, y_max])
# plt.colorbar()
# # plt.contourf(Xm, Ym, cut_nodes, origin='lower')
# plt.show()
# print(asd)


# Domo
# x_0 = (x_max + x_min) / 2
# y_0 = 0
# R = 20
# cut_nodes, dead_nodes = cylinder(Xm, Ym, x_0, y_0, R, dx, dy)

# print(cut_nodes)

# import matplotlib.pyplot as plt
# A = np.zeros_like(Xm)
# A[cut_nodes] = 1
# A[dead_nodes] = 1
# plt.contourf(Xm, Ym, A)
# # plt.imshow(A, origin='lower', extent=[x_min, x_max-dx, y_min, y_max])
# plt.colorbar()
# # plt.contourf(Xm, Ym, cut_nodes, origin='lower')
# plt.show()
# # print(asd)

# import matplotlib.pyplot as plt
# plt.scatter(cut_nodes[1], cut_nodes[0], c='r')
# plt.show()
# # plt.contourf(Xm, Ym, u0(Xm, Ym), cmap=plt.cm.jet)
# # plt.colorbar()
# # plt.show()
# # plt.contourf(Xm, Ym, Y0(Xm, Ym), cmap=plt.cm.Oranges)
# # plt.colorbar()
# # plt.show()
# print(asdasd)

p0 = lambda x, y: x * 0 #+ 1e-12

U_0 = u0(Xm, Ym)
V_0 = v0(Xm, Ym)
T_0 = T0(Xm, Ym) #* 0 + 1
Y_0 = Y0(Xm, Ym) #* 0 + 1

# Dead nodes
U_0[dead_nodes] = 0
V_0[dead_nodes] = 0
T_0[dead_nodes] = 0
Y_0[dead_nodes] = 0

# import matplotlib.pyplot as plt
# plt.contourf(Xm, Ym, U_0, cmap=plt.cm.jet)
# plt.colorbar()
# plt.show()
# print(asd)

# Boundary conditions
u_y_min = U_0[0]#U_0[cut_nodes]
u_y_max = U_0[-1]
v_y_min = V_0[0]
v_y_max = 0
p_y_max = 0

y_0 = np.array([U_0, V_0, T_0, Y_0])
p_0 = p0(Xm, Ym)

# Array for approximations
yy = np.zeros((Nt // samples + 1, y_0.shape[0], Ny, Nx - 1)) 
P  = np.zeros((Nt // samples + 1, Ny, Nx - 1))

yyd = np.zeros((Nt // samples + 1, 2, Ny, Nx - 1)) 
yyd[0] = y_0[:2]

yy[0] = y_0
P[0]  = p_0
F_e = F(Xm, Ym)

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
    'cut_nodes': cut_nodes,
    'dead_nodes': dead_nodes,
    'Ym': Ym,
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
        # p_tmp = solve_iterative(Ut, Vt, p_tmp, **args).copy()
        # p_tmp = solve_pressure(Ut, Vt, **args).copy()
        p_tmp = solve_fftfd(Ut, Vt, **args).copy()
        grad_p = grad_pressure(p_tmp, **args)
        y_tmp[:2] = y_tmp[:2] - dt / rho * grad_p
        Ut, Vt, Tt, Yt = y_tmp.copy()
        y_tmp = boundary_conditions(Ut, Vt, Tt, Yt, args)
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
        Ud, Vd = Ut.copy(), Vt.copy()
        #p_tmp = solve_pressure_iterative(Ut, Vt, p_tmp, **args).copy()
        p_tmp = solve_fftfd(Ut, Vt, **args).copy()
        # p_tmp = solve_iterative_ibm(Ut, Vt, p_tmp, **args).copy()
        # p_tmp = solve_gmres(Ut, Vt, p_tmp, **args)
        grad_p = grad_pressure(p_tmp, **args)
        y_tmp[:2] = y_tmp[:2] - dt / rho * grad_p
        Ut, Vt = y_tmp[:2].copy()
        U_, V_, T_, Y_ = boundary_conditions(Ut, Vt, Tt, Yt, args)
        # U_, V_, T_, Y_ = y_tmp.copy()
        y_tmp[0] = U_
        y_tmp[1] = V_
        y_tmp[2] = T_
        y_tmp[3] = Y_
        if n % samples == 0:
            print("Step:", n)
            P[n // samples + 1] = p_tmp
            yy[n // samples + 1] = y_tmp
            Ut, Vt = y_tmp[:2]
            print("CFL", dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy))
            yyd[n // samples + 1, 0] = Ud
            yyd[n // samples + 1, 1] = Vd 

# Last approximation
yy[-1] = y_tmp

# Get velocities
U = yy[:, 0]
V = yy[:, 1]
T = yy[:, 2]
Y = yy[:, 3]
Ud = yyd[:, 0]
Vd = yyd[:, 1]

# Copy last column
U_ = np.zeros((Nt // samples + 1, Ny, Nx))
V_ = np.zeros((Nt // samples + 1, Ny, Nx))
T_ = np.zeros((Nt // samples + 1, Ny, Nx))
Y_ = np.zeros((Nt // samples + 1, Ny, Nx))
P_ = np.zeros((Nt // samples + 1, Ny, Nx))
Ud_ = np.zeros((Nt // samples + 1, Ny, Nx))
Vd_ = np.zeros((Nt // samples + 1, Ny, Nx))
U_[:, :, :-1] = U
U_[:, :, -1] = U[:, :, 0]
V_[:, :, :-1] = V
V_[:, :, -1] = V[:, :, 0]
T_[:, :, :-1] = T
T_[:, :, -1] = T[:, :, 0]
Y_[:, :, :-1] = Y
Y_[:, :, -1] = Y[:, :, 0]
P_[:, :, :-1] = P
P_[:, :, -1] = P[:, :, 0]
Ud_[:, :, :-1] = Ud
Ud_[:, :, -1] = Ud[:, :, 0]
Vd_[:, :, :-1] = Vd
Vd_[:, :, -1] = Vd[:, :, 0]
U = U_.copy()
V = V_.copy()
T = T_.copy()
Y = Y_.copy()
P = P_.copy()
Ud = Ud_.copy()
Vd = Vd_.copy()

# Save approximation
turb_str = "turbulence"
if not turb:
    turb_str = "no_turbulence"

cons_str = "conservative"
if not conser:
    cons_str = "non_conservative"

# Save approximation
np.savez('../output/2d_building_{}_{}.npz'.format(turb_str, cons_str), U=U, V=V, T=T, Y=Y, P=P, x=x, y=y, t=t[::samples], Ud=Ud, Vd=Vd)