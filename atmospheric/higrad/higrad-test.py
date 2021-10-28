import numpy as np
import matplotlib.pyplot as plt
from utils import h, hx, hy, G, Gx, Gy, Gz, G13, G23, G13z, G23z, zc, zz

def Phi(t, A, **kwargs):
    """
    The function to be solved.
    """
    # Parameters
    x = kwargs['x']
    y = kwargs['y']
    z = kwargs['z']
    Ge = kwargs['G']
    G13e = kwargs['G13']
    G23e = kwargs['G23']
    G13ze = kwargs['G13z']
    G23ze = kwargs['G23z']
    Gxe = kwargs['Gx']
    Gye = kwargs['Gy']
    Gze = kwargs['Gz']
    f = kwargs['f']
    fh = kwargs['fh']
    ue = kwargs['ue']
    ve = kwargs['ve']
    we = kwargs['we']
    g = kwargs['g']
    C_v = kwargs['C_v']
    C_p = kwargs['C_p']
    R_d = kwargs['R_d']
    p_o = kwargs['p_o']
    rho_e = kwargs['rho_e']
    p_e = kwargs['p_e']
    px_e = kwargs['px_e']
    py_e = kwargs['py_e']
    pz_e = kwargs['pz_e']
    dx = x[0, 1, 0] - x[0, 0, 0]
    dy = y[1, 0, 0] - y[0, 0, 0]
    dz = z[0, 0, 1] - z[0, 0, 0]

    # Get the values 
    Uh, Vh, Wh, Th, Rh = A
    R = Rh / Ge
    U = Uh / R / Ge
    V = Vh / R / Ge
    W = Wh / R / Ge
    T = Th / R / Ge

    # G function
    G = Ge[1:-1, 1:-1, 1:-1]
    Gx = Gxe[1:-1, 1:-1, 1:-1]
    Gy = Gye[1:-1, 1:-1, 1:-1]
    G13 = G13e[1:-1, 1:-1, 1:-1]
    G23 = G23e[1:-1, 1:-1, 1:-1]
    G13z = G13ze[1:-1, 1:-1, 1:-1]
    G23z = G23ze[1:-1, 1:-1, 1:-1]

    # Temporal arrays
    Uf = np.zeros_like(U)
    Vf = np.zeros_like(V)
    Wf = np.zeros_like(W)
    Tf = np.zeros_like(T)
    Rf = np.zeros_like(R)

    # Omega
    omega = G13e * U + G23e * V +  W / Ge

    # u velocities
    UGUh = U * Uh
    VGUh = V * Uh
    OGUh = omega * Uh
    # Derivatives
    UGUhx = (UGUh[1:-1, 2:, 1:-1] - UGUh[1:-1, :-2, 1:-1]) / (2 * dx)
    VGUhy = (VGUh[2:, 1:-1, 1:-1] - VGUh[:-2, 1:-1, 1:-1]) / (2 * dy)
    OGUhz = (OGUh[1:-1, 1:-1, 2:] - OGUh[1:-1, 1:-1, :-2]) / (2 * dz)
    divU = UGUhx + VGUhy + OGUhz

    # v velocities
    UGVh = U * Vh
    VGVh = V * Vh
    OGVh = omega * Vh
    # Derivatives
    UGVhx = (UGVh[1:-1, 2:, 1:-1] - UGVh[1:-1, :-2, 1:-1]) / (2 * dx)
    VGVhy = (VGVh[2:, 1:-1, 1:-1] - VGVh[:-2, 1:-1, 1:-1]) / (2 * dy)
    OGVhz = (OGVh[1:-1, 1:-1, 2:] - OGVh[1:-1, 1:-1, :-2]) / (2 * dz)
    divV = UGVhx + VGVhy + OGVhz

    # w velocities
    UGWh = U * Wh
    VGWh = V * Wh
    OGWh = omega * Wh
    # Derivatives
    UGWhx = (UGWh[1:-1, 2:, 1:-1] - UGWh[1:-1, :-2, 1:-1]) / (2 * dx)
    VGWhy = (VGWh[2:, 1:-1, 1:-1] - VGWh[:-2, 1:-1, 1:-1]) / (2 * dy)
    OGWhz = (OGWh[1:-1, 1:-1, 2:] - OGWh[1:-1, 1:-1, :-2]) / (2 * dz)
    divW = UGWhx + VGWhy + OGWhz

    # Temperature
    UGTh = U * Th
    VGTh = V * Th  
    OGTh = omega * Th
    # Derivatives
    UGThx = (UGTh[1:-1, 2:, 1:-1] - UGTh[1:-1, :-2, 1:-1]) / (2 * dx)
    VGThy = (VGTh[2:, 1:-1, 1:-1] - VGTh[:-2, 1:-1, 1:-1]) / (2 * dy)
    OGThz = (OGTh[1:-1, 1:-1, 2:] - OGTh[1:-1, 1:-1, :-2]) / (2 * dz)
    divT = UGThx + VGThy + OGThz

    # Density
    UGR = U * Rh
    VGR = V * Rh
    OGR = omega * Rh
    # Derivatives
    UGRx = (UGR[1:-1, 2:, 1:-1] - UGR[1:-1, :-2, 1:-1]) / (2 * dx)
    VGRy = (VGR[2:, 1:-1, 1:-1] - VGR[:-2, 1:-1, 1:-1]) / (2 * dy)
    OGRz = (OGR[1:-1, 1:-1, 2:] - OGR[1:-1, 1:-1, :-2]) / (2 * dz)
    divR = UGRx + VGRy + OGRz

    # Pressure
    P = (R_d * R * T) ** (C_v / C_p) / (p_o ** (R_d / C_v))
    Pp = P - p_e # p'

    # Forces
    px = (Pp[1:-1, 2:, 1:-1] - Pp[1:-1, :-2, 1:-1]) / 2 / dx
    py = (Pp[2:, 1:-1, 1:-1] - Pp[:-2, 1:-1, 1:-1]) / 2 / dy
    pz = (Pp[1:-1, 1:-1, 2:] - Pp[1:-1, 1:-1, :-2]) / 2 / dz
    pxp = px - px_e[1:-1, 1:-1, 1:-1]
    pyp = py - py_e[1:-1, 1:-1, 1:-1]
    pzp = pz - pz_e[1:-1, 1:-1, 1:-1]
    rhop = R - rho_e # Rho'
    Rup = -pxp - G13e[1:-1, 1:-1, 1:-1] * pzp + f * R[1:-1, 1:-1, 1:-1] * (V[1:-1, 1:-1, 1:-1]  -  ve) - fh * R[1:-1, 1:-1, 1:-1] *(W[1:-1, 1:-1, 1:-1] - we)
    Rvp = -pyp - G23e[1:-1, 1:-1, 1:-1] * pzp - f * R[1:-1, 1:-1, 1:-1] * (U[1:-1, 1:-1, 1:-1] - ue)
    Rwp = -pzp / Ge[1:-1, 1:-1, 1:-1] - rhop[1:-1, 1:-1, 1:-1] * g + fh * R[1:-1, 1:-1, 1:-1] * (U[1:-1, 1:-1, 1:-1] - ue)
    
    # Computation of RHS
    Uf[1:-1, 1:-1, 1:-1] = -divU + G * Rup
    Vf[1:-1, 1:-1, 1:-1] = -divV + G * Rvp
    Wf[1:-1, 1:-1, 1:-1] = -divW + G * Rwp
    Tf[1:-1, 1:-1, 1:-1] = -divT + 100
    Rf[1:-1, 1:-1, 1:-1] = -divR

    # Boundary conditions
    # At x=x_min
    Uf[:, 0, :] = ue 
    Vf[:, 0, :] = ve
    Wf[:, 0, :] = we
    # At x=x_max
    Uf[:, -1, :] = ue
    Vf[:, -1, :] = ve
    Wf[:, -1, :] = we
    # At y=y_min
    Uf[:, :, 0] = ue
    Vf[0, :, :] = ve 
    Wf[0, :, :] = we
    # At y=y_max
    Uf[-1:, :, :] = ue
    Vf[-1, :, :] = ve
    Wf[-1, :, :] = we
    # At z=z_min
    Uf[:, :, 0] = 0
    Vf[:, :, 0] = 0
    Wf[:, :, 0] =  0
    # At z=z_max 
    Uf[:, :, -1] = ue
    Vf[:, :, -1] = ve
    Wf[:, :, -1] = we

    return np.array([Uf, Vf, Wf, Tf, Rf])


### MAIN ###
# Domain
x_min, x_max = 0, 400 
y_min, y_max = 0, 400 
z_min, z_max = 0, 125
t_min, t_max = 0, 2
Nx, Ny, Nz, Nt = 41, 41, 21, 501 # 
## OK ##
# t_min, t_max = 0, 1
# Nx, Ny, Nz, Nt = 51, 51, 31, 1001 # 
# t_min, t_max = 0, .5
# Nx, Ny, Nz, Nt = 41, 41, 21, 501 # 
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
z = np.linspace(z_min, z_max, Nz)
t = np.linspace(t_min, t_max, Nt)
X, Y, Z = np.meshgrid(x, y, z)
dx, dy, dz, dt = x[1] - x[0], y[1] - y[0], z[1] - z[0], t[1] - t[0]
print(dx, dy, dz, dt)

# Parameters
C_p = 1004 # Specific heat of dry air at constant pressure [J kg^{−1} K^{−1}]
C_v = 717 # Specific heat of dry air at constant volume [J kg^{−1} K^{−1}]
R_d = 287 # Gas constant for dry air [J K^{-1} kg^{-1}]
p_o = 100000 # Base-state pressure [N m^{-2}]
H = z_max - z_min # Model depth [m]
# Balanced enviornment velocities
ue = 2 # [m s^{-1}]
ve = 0 # [m s^{-1}]
we = 0 # [m s^{-1}]
g = 9.80665 # [m s^{-2}]
phi = np.pi # 
Omega = 7.2921e-5 # Earth rotation rate [rad s^{-1}]
f  = 2 * Omega * np.sin(phi) # [rad s^{-1}]
fh = 2 * Omega * np.cos(phi) # [rad s^{-1}]
f = fh = 0
T_0 = 300 # Initial temperature of dry gas [K]
# Z correction
#Z = zc(X, Y, Z)
# Pressure
p = lambda x, y, z: p_o * np.exp(-g / (T_0 * R_d) * ((H - h(x, y)) * z / H + h(x, y))) # Environmental pressure [N m^{-2}]
p_0 = p(X, Y, Z) # Initial pressure [N m^{-2}]
# Pressure derivatives
px = lambda x, y, z: -p_o * g * (H - z) / (R_d * T_0 * H) * np.exp(-g / (R_d * T_0) * ((H - h(x, y)) * z / H + h(x, y))) * hx(x, y) # [N m^{-2}]
py = lambda x, y, z: -p_o * g * (H - z) / (R_d * T_0 * H) * np.exp(-g / (R_d * T_0) * ((H - h(x, y)) * z / H + h(x, y))) * hy(x, y) 
pz = lambda x, y, z: -p_o * g * (H - h(x, y)) / (R_d * T_0 * H) * np.exp(-g / (R_d * T_0) * ((H - h(x, y)) * z / H + h(x, y))) 
# Density
L = 0.0065 # Temperature lapse [K/m]
rho_0 = p_o / (R_d * T_0) # Initial density [kg m^{-3}]
#rho = lambda z: rho_0 * (1 - L * z / T_0) ** (g / (R_d * L) - 1) # Environmental density [kg m^{-3}]
rho = lambda x, y, z: rho_0 * (1 - L / T_0 * ((H - h(x, y)) / H * z + h(x, y))) ** (g / (R_d * L) - 1) # Environmental density [kg m^{-3}]
# Initial condition functions
u0 = lambda x, y, z: x * 0 + 2 # [m s^{-1}]
v0 = lambda x, y, z: x * 0 # [m s^{-1}]
w0 = lambda x, y, z: x * 0 # [m s^{-1}]
theta0 = lambda x, y, z: x * 0 + T_0 #* (p_o / p_0) ** (R_d / C_v) # Potential temperature [K]
rho0 = lambda x, y, z: x * 0 + rho_0 #1.18 # Density [kg m^{-3}]

# Evaluation of G
Ge = G(X, Y, Z)
G13e = G13(X, Y, Z)
G23e = G23(X, Y, Z)
G13ze = G13z(X, Y, Z)
G23ze = G23z(X, Y, Z)
Gxe = Gx(X, Y, Z)
Gye = Gy(X, Y, Z)
Gze = Gz(X, Y, Z)
# Evaluation of density and pressure
rhoe = rho(X, Y, Z) * 0 + 1
pe = p(X, Y, Z) * 0 + 1
pxe = px(X, Y, Z) * 0
pye = py(X, Y, Z) * 0
pze = pz(X, Y, Z) * 0

# Initial condition
U0 = u0(X, Y, Z)
V0 = v0(X, Y, Z)
W0 = w0(X, Y, Z)
T0 = theta0(X, Y, Z)
R0 = rho0(X, Y, Z)
#y0 = np.array([U0, V0, W0, T0, R0])
y0 = np.array([U0 * R0, V0 * R0, W0 * R0, T0 * R0, R0]) 

# Array for approximations
samples = 100
yy = np.zeros((Nt // samples + 1, 5, Ny, Nx, Nz))
# yy = np.zeros((Nt, 5, Ny, Nx, Nz))
yy[0] = y0 * Ge

# Arguments for RHS
args = {
    'C_p': C_p,
    'C_v': C_v,
    'R_d': R_d,
    'p_o': p_o,
    'H': H,
    'ue': ue,
    've': ve,
    'we': we,
    'g': g,
    'phi': phi,
    'Omega': Omega,
    'f': f,
    'fh': fh,
    'G': Ge,
    'rho_e': rhoe,
    'p_e': pe,
    'px_e': pxe,
    'py_e': pye,
    'pz_e': pze,
    'G13': G13e,
    'G23': G23e,
    'G13z': G13ze,
    'G23z': G23ze,
    'Gx': Gxe,
    'Gy': Gye,
    'Gz': Gze,
    'x': X,
    'y': Y,
    'z': Z,
    'dt': dt,
}

# Solve
y_tmp = yy[0]
for n in range(Nt-1):
    y_tmp = y_tmp + dt * Phi(t[n], y_tmp, **args)
    if n % samples == 0:
        print("Step:", n)
        yy[n // samples + 1] = y_tmp
        Ut = y_tmp[0] / y_tmp[-1]
        Vt = y_tmp[1] / y_tmp[-1]
        Wt = y_tmp[2] / y_tmp[-1]
        print("CFL", dt * (np.max(Ut) / dx + np.max(Vt) / dy + np.max(Wt) / dz))
yy[-1] = y_tmp

# # Solve
# y_tmp = yy[0]
# for n in range(Nt-1):
#     #print("Step:", n)
#     k1 = Phi(t[n], y_tmp, **args)
#     k2 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k1, **args)
#     k3 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k2, **args)
#     k4 = Phi(t[n] + dt, y_tmp + dt * k3, **args)
#     y_tmp = y_tmp + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
#     if n % samples == 0:
#         print("Step:", n)
#         yy[n // samples + 1] = y_tmp
#         Ut = y_tmp[0] / y_tmp[-1]
#         Vt = y_tmp[1] / y_tmp[-1]
#         Wt = y_tmp[2] / y_tmp[-1]
#         print("CFL", dt * (np.max(Ut) / dx + np.max(Vt) / dy + np.max(Wt) / dz))
# yy[-1] = y_tmp

# Get velocities
R = yy[:, 4] / Ge
U = yy[:, 0] / R / Ge
V = yy[:, 1] / R / Ge
W = yy[:, 2] / R / Ge
T = yy[:, 3] / R / Ge
P = (R_d * R * T) ** (C_v / C_p) / (p_o ** (R_d / C_v))

# Save 
np.savez('output/higrad-test.npz', U=U, V=V, W=W, T=T, R=R, P=P, X=X, Y=Y, Z=Z, t=t[::samples])