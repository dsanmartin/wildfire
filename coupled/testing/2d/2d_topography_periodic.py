import numpy as np
from poisson import solve_fft, solve_iterative
from turbulence import periodic_turbulence_2d
from ibm import topography, boundary

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
    # Return grad p
    return np.array([px, py])

def Phi(t, C, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    nu = kwargs['nu']
    rho = kwargs['rho']
    F = kwargs['F']
    turb = kwargs['turb']
    conservative = kwargs['conservative']
    cut_nodes = kwargs['cut_nodes']
    dead_nodes = kwargs['dead_nodes']

    # Get variables
    u, v, = C
    
    # Forces
    F_x, F_y = F

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

    # Turbulence
    sgs_x = sgs_y = 0
    if turb:
        sgs_x, sgs_y = periodic_turbulence_2d(u, v, ux, uy, vx, vy, uxx, uyy, vxx, vyy, kwargs)

    if conservative: # RHS Inside domain (conservative form - using central difference)
        U_ = nu * (uxx + uyy) - (uux + uvy) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (vux + vvy) + F_y - sgs_y
    else: # RHS Inside domain (non-conservative form - using upwind!)
        U_ = nu * (uxx + uyy) - (u_plu * uxm + u_min * uxp + v_plu * uym + v_min * uyp) + F_x - sgs_x
        V_ = nu * (vxx + vyy) - (u_plu * vxm + u_min * vxp + v_plu * vym + v_min * vyp) + F_y - sgs_y

    # IBM
    U_, V_ = boundary(U_, V_, cut_nodes, dead_nodes)

    return np.array([U_, V_])

### MAIN ###
# Domain
x_min, x_max = 0, 2 * np.pi
y_min, y_max = 0, 2 * np.pi #100
t_min, t_max = 0, 20
Nx, Ny, Nt = 128, 128, 1001 # Number of grid points
samples = 100 # Samples to store data
# Arrays
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)
# Meshgrid (without boundary)
Xm, Ym = np.meshgrid(x[:-1], y[:-1])

dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
print(dx, dy, dt)

# Parameters
nu = 1e-8 # Viscosity
rho = 1
C_s = 0.173
turb = True
conser = False

# Gaussian hill
A = 1
sx = 1 / 2
sy = 1 / 2
x_c= (x_max + x_min) / 2
y_c = (y_max + y_min) / 2
top = lambda x, y: A * np.exp(-((x - x_c) **2 / sx + (y - y_c) ** 2 / sy))
topo = lambda x: top(x, y_c)

# import matplotlib.pyplot as plt
# plt.plot(x, topo(x))
# plt.ylim([y_min, y_max])
# plt.show()

# Force term
fx = lambda x, y: x * 0
fy = lambda x, y: x * 0 
F = lambda x, y: (fx(x, y), fy(x, y))

# Initial conditions
u0 = lambda x, y: (x + y) * 0 + 1
v0 = lambda x, y: x * 0 
p0 = lambda x, y: x * 0 

# IBM for topography
cut_nodes, dead_nodes = topography(Xm, Ym, topo, dx, dy)

# print(cut_nodes)
# print(dead_nodes)


# import matplotlib.pyplot as plt
# A = np.zeros_like(Xm)
# A[cut_nodes[0], cut_nodes[1]] = 1
# A[dead_nodes[0], dead_nodes[1]] = 1
# plt.contourf(Xm, Ym, A)
# plt.colorbar()
# plt.show()
# print(asd)

U_0 = u0(Xm, Ym)
V_0 = v0(Xm, Ym)

# Cylinder
U_0, V_0 = boundary(U_0, V_0, cut_nodes, dead_nodes)

y_0 = np.array([U_0, V_0])
p_0 = p0(Xm, Ym)

# Array for approximations
yy = np.zeros((Nt // samples + 1, y_0.shape[0], Ny - 1, Nx - 1)) 
P  = np.zeros((Nt // samples + 1, Ny - 1, Nx - 1))
yy[0] = y_0
P[0]  = p_0
F_e = F(Xm, Ym)

args = {
    'x': x,
    'y': y,#Ym[:-1, :-1],
    'dx': dx,
    'dy': dy,
    'dt': dt,
    'nu': nu,
    'rho': rho,
    'F': F_e,
    'u0': U_0,
    'v0': V_0,
    'Nx': Nx,
    'Ny': Ny,
    'C_s': C_s,
    'turb': turb,
    'conservative': conser,
    'cut_nodes': cut_nodes,
    'dead_nodes': dead_nodes,
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
        Ut, Vt = y_tmp.copy()
        # print(np.min(Tt), np.max(Tt))
        p_tmp = solve_iterative(Ut, Vt, p_tmp, **args).copy()
        # p_tmp = solve_pressure(Ut, Vt, **args).copy()
        grad_p = grad_pressure(p_tmp, **args)
        y_tmp = y_tmp - dt / rho * grad_p
        #Ut, Vt = y_tmp[:2].copy()
        # IBM
        y_tmp[0, cut_nodes[0], cut_nodes[1]] = 0
        y_tmp[1, cut_nodes[0], cut_nodes[1]] = 0
        y_tmp[0, dead_nodes[0], dead_nodes[1]] = 0
        y_tmp[1, dead_nodes[0], dead_nodes[1]] = 0
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
        k1 = Phi(t[n], y_tmp, **args)
        k2 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k1, **args)
        k3 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k2, **args)
        k4 = Phi(t[n] + dt, y_tmp + dt * k3, **args)
        y_tmp = y_tmp + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
        Ut, Vt = y_tmp.copy()
        #p_tmp = solve_pressure_iterative(Ut, Vt, p_tmp, **args).copy()
        p_tmp = solve_fft(Ut, Vt, **args)
        grad_p = grad_pressure(p_tmp, **args)
        y_tmp = y_tmp - dt / rho * grad_p
        Ut, Vt = y_tmp.copy()
        # IBM
        y_tmp[:2] = boundary(Ut, Vt, cut_nodes, dead_nodes)
        # y_tmp[0, cut_nodes[0], cut_nodes[1]] = 0
        # y_tmp[1, cut_nodes[0], cut_nodes[1]] = 0
        # y_tmp[0, dead_nodes[0], dead_nodes[1]] = 0
        # y_tmp[1, dead_nodes[0], dead_nodes[1]] = 0
        # U_, V_, T_, Y_ = boundary_conditions(Ut, Vt, Tt, Yt, args)
        # y_tmp[0] = U_
        # y_tmp[1] = V_
        # y_tmp[2] = T_
        # y_tmp[3] = Y_
        if n % samples == 0:
            print("Step:", n)
            P[n // samples + 1] = p_tmp
            yy[n // samples + 1] = y_tmp
            Ut, Vt = y_tmp
            print("CFL", dt * (np.max(np.abs(Ut)) / dx + np.max(np.abs(Vt)) / dy))

# Last approximation
yy[-1] = y_tmp

# Get velocities
U = yy[:, 0]
V = yy[:, 1]

# Copy last column
U_ = np.zeros((Nt // samples + 1, Ny, Nx))
V_ = np.zeros((Nt // samples + 1, Ny, Nx))
P_ = np.zeros((Nt // samples + 1, Ny, Nx))
U_[:, :-1,:-1] = U
U_[:, :-1, -1] = U[:, :, 0]
U_[:, -1,:] = U_[:, 0, :]
V_[:, :-1,:-1] = V
V_[:, :-1, -1] = V[:, :, 0]
V_[:, -1,:] = V_[:, 0, :]
P_[:, :-1,:-1] = P
P_[:, :-1, -1] = P[:, :, 0]
P_[:, -1,:] = P_[:, 0, :]
U = U_.copy()
V = V_.copy()
P = P_.copy()

# Save approximation
turb_str = "turbulence"
if not turb:
    turb_str = "no_turbulence"

np.savez('../output/2d_topo_{}.npz'.format(turb_str), U=U, V=V, P=P, x=x, y=y, t=t[::samples])