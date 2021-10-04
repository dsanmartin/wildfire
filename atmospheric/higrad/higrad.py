from zipfile import PyZipFile
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
    U, V, W, T, R = A

    # Temporal arrays
    Uf = np.zeros_like(U)
    Vf = np.zeros_like(V)
    Wf = np.zeros_like(W)
    Tf = np.zeros_like(T)
    Rf = np.zeros_like(R)

    # Derivatives #
    # Velocities
    Ux = (U[1:-1, 2:, 1:-1] - U[1:-1, :-2, 1:-1]) / 2 / dx
    Uy = (U[2:, 1:-1, 1:-1] - U[:-2, 1:-1, 1:-1]) / 2 / dy
    Uz = (U[1:-1, 1:-1, 2:] - U[1:-1, 1:-1, :-2]) / 2 / dz
    Vx = (V[1:-1, 2:, 1:-1] - V[1:-1, :-2, 1:-1]) / 2 / dx
    Vy = (V[2:, 1:-1, 1:-1] - V[:-2, 1:-1, 1:-1]) / 2 / dy
    Vz = (V[1:-1, 1:-1, 2:] - V[1:-1, 1:-1, :-2]) / 2 / dz
    Wx = (W[1:-1, 2:, 1:-1] - W[1:-1, :-2, 1:-1]) / 2 / dx
    Wy = (W[2:, 1:-1, 1:-1] - W[:-2, 1:-1, 1:-1]) / 2 / dy
    Wz = (W[1:-1, 1:-1, 2:] - W[1:-1, 1:-1, :-2]) / 2 / dz

    # Temperature
    Tx = (T[1:-1, 2:, 1:-1] - T[1:-1, :-2, 1:-1]) / 2 / dx
    Ty = (T[2:, 1:-1, 1:-1] - T[:-2, 1:-1, 1:-1]) / 2 / dy
    Tz = (T[1:-1, 1:-1, 2:] - T[1:-1, 1:-1, :-2]) / 2 / dz
    # Rho
    Rx = (R[1:-1, 2:, 1:-1] - R[1:-1, :-2, 1:-1]) / 2 / dx
    Ry = (R[2:, 1:-1, 1:-1] - R[:-2, 1:-1, 1:-1]) / 2 / dy
    Rz = (R[1:-1, 1:-1, 2:] - R[1:-1, 1:-1, :-2]) / 2 / dz

    # G function
    G = Ge[1:-1, 1:-1, 1:-1]
    Gx = Gxe[1:-1, 1:-1, 1:-1]
    Gy = Gye[1:-1, 1:-1, 1:-1]
    G13 = G13e[1:-1, 1:-1, 1:-1]
    G23 = G23e[1:-1, 1:-1, 1:-1]
    G13z = G13ze[1:-1, 1:-1, 1:-1]
    G23z = G23ze[1:-1, 1:-1, 1:-1]

    # omega
    omega = G13e * U + G23e * V +  W / Ge
    omegaz = G13z * U[1:-1, 1:-1, 1:-1] + G13 * Uz + G23z * V[1:-1, 1:-1, 1:-1]  + G23 * Vz + Wz / G

    # # Other omega
    # G13ez = (G13e[1:-1, 1:-1, 2:] - G13e[1:-1, 1:-1, :-2]) / 2 / dz
    # G23ez = (G23e[1:-1, 1:-1, 2:] - G23e[1:-1, 1:-1, :-2]) / 2 / dz
    # omegaz = G13ez * U[1:-1, 1:-1, 1:-1] + G13e[1:-1, 1:-1, 1:-1] * Uz + \
    #     G23ez * V[1:-1, 1:-1, 1:-1] + G23e[1:-1, 1:-1, 1:-1] * Vz + Wz / Ge[1:-1, 1:-1, 1:-1]

    # div(phi V), phi={G\rho u, G\rho v, G\rho w, G\rho\theta, G\rho}
    # div(V), V=(u, v, \omega)
    divVel = Ux + Vy + omegaz
    # phi = G\rho u
    phiu = R[1:-1, 1:-1, 1:-1] * U[1:-1, 1:-1, 1:-1]
    phiux = Gx * phiu / G + (Rx * U[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Ux)
    phiuy = Gy * phiu / G + (Ry * U[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Uy)
    phiuz = Rz * U[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Uz
    # phi = G\rho v
    phiv = R[1:-1, 1:-1, 1:-1] * V[1:-1, 1:-1, 1:-1]
    phivx = Gx * phiv / G + (Rx * V[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Vx)
    phivy = Gy * phiv / G + (Ry * V[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Vy)
    phivz = Rz * V[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Vz
    # phi = G\rho w
    phiw = R[1:-1, 1:-1, 1:-1] * W[1:-1, 1:-1, 1:-1]
    phiwx = Gx * phiw / G + (Rx * W[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Wx)
    phiwy = Gy * phiw / G + (Ry * W[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Wy)
    phiwz = Rz * W[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Wz
    # phi = G\rho\theta
    phitheta = R[1:-1, 1:-1, 1:-1] * T[1:-1, 1:-1, 1:-1]
    phithetax = Gx * phitheta / G + (Rx * T[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Tx)
    phithetay = Gy * phitheta / G + (Ry * T[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Ty)
    phithetaz = Rz * T[1:-1, 1:-1, 1:-1] + R[1:-1, 1:-1, 1:-1] * Tz
    # phi = G\rho
    phir = R[1:-1, 1:-1, 1:-1]
    phirx = Gx / G + Rx
    phiry = Gy / G + Ry
    phirz = Rz
    # Divergences
    divU = phiux * U[1:-1, 1:-1, 1:-1] + phiuy * V[1:-1, 1:-1, 1:-1] + phiuz * omega[1:-1, 1:-1, 1:-1] + phiu * divVel
    divV = phivx * U[1:-1, 1:-1, 1:-1] + phivy * V[1:-1, 1:-1, 1:-1] + phivz * omega[1:-1, 1:-1, 1:-1] + phiv * divVel
    divW = phiwx * U[1:-1, 1:-1, 1:-1] + phiwy * V[1:-1, 1:-1, 1:-1] + phiwz * omega[1:-1, 1:-1, 1:-1] + phiw * divVel
    divT = phithetax * U[1:-1, 1:-1, 1:-1] + phithetay * V[1:-1, 1:-1, 1:-1] + phithetaz * omega[1:-1, 1:-1, 1:-1] + phitheta * divVel
    divR = phirx * U[1:-1, 1:-1, 1:-1] + phiry * V[1:-1, 1:-1, 1:-1] + phirz * omega[1:-1, 1:-1, 1:-1] + phir * divVel

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
    Rup = -pxp - G13e[1:-1, 1:-1, 1:-1] * pzp + f * R[1:-1, 1:-1, 1:-1] * (V[1:-1, 1:-1, 1:-1] - ve) - fh * R[1:-1, 1:-1, 1:-1] * (W[1:-1, 1:-1, 1:-1] - we)
    Rvp = -pyp - G23e[1:-1, 1:-1, 1:-1] * pzp - f * R[1:-1, 1:-1, 1:-1] * (U[1:-1, 1:-1, 1:-1] - ue) 
    Rwp = -pzp / Ge[1:-1, 1:-1, 1:-1] - rhop[1:-1, 1:-1, 1:-1] * g + fh * R[1:-1, 1:-1, 1:-1] * (U[1:-1, 1:-1, 1:-1] - ue)
    
    # Computation of RHS
    Uf[1:-1, 1:-1, 1:-1] = -divU + Rup
    Vf[1:-1, 1:-1, 1:-1] = -divV + Rvp
    Wf[1:-1, 1:-1, 1:-1] = -divW + Rwp
    Tf[1:-1, 1:-1, 1:-1] = -divT
    Rf[1:-1, 1:-1, 1:-1] = -divR
    Uf[1:-1, 1:-1, 1:-1] /= R[1:-1, 1:-1, 1:-1]
    Vf[1:-1, 1:-1, 1:-1] /= R[1:-1, 1:-1, 1:-1]
    Wf[1:-1, 1:-1, 1:-1] /= R[1:-1, 1:-1, 1:-1]
    Tf[1:-1, 1:-1, 1:-1] /= R[1:-1, 1:-1, 1:-1]

    # Boundary conditions
    # Fixed in U and W
    Uf[:, 0, :] = ue 
    Uf[:, -1, :] = ue 
    Vf[0, :, :] = ve
    Vf[-1, :, :] = ve
    Wf[:, :, 0] =  we #Ge[:,:,0] * (G13e[:,:,0] * U[:,:,0] + G23e[:,:,0] * V[:,:,0])
    Wf[:, :, -1] = we #Ge[:,:,-1] * (G13e[:,:,-1] * U[:,:,-1] + G23e[:,:,-1] * V[:,:,-1])

    # # Periodic on y = y_min, y = y_max
    # # y = y_min
    # Ux = (U[0, 2:, 1:-1] - U[0, :-2, 1:-1]) / 2 / dx
    # Vx = (V[0, 2:, 1:-1] - V[0, :-2, 1:-1]) / 2 / dx
    # Vy = (V[1, 1:-1, 1:-1] - V[-1, 1:-1, 1:-1]) / 2 / dy
    # Vz = (V[0, 1:-1, 2:] - V[0, 1:-1, :-2]) / 2 / dz
    # # Gx = (Ge[0, 2:, 1:-1] - Ge[0, :-2, 1:-1]) / 2 / dx
    # # Gy = (Ge[1, 1:-1, 1:-1] - Ge[-1, 1:-1, 1:-1]) / 2 / dy
    # # Gz = (Ge[0, 1:-1, 2:] - Ge[0, 1:-1, :-2]) / 2 / dz
    # Gx = Gxe[0, 1:-1, 1:-1]
    # Gy = Gye[0, 1:-1, 1:-1]
    # Gz = Gze[0, 1:-1, 1:-1]
    # Rx = (R[0, 2:, 1:-1] - R[0, :-2, 1:-1]) / 2 / dx
    # Ry = (R[1, 1:-1, 1:-1] - R[-1, 1:-1, 1:-1]) / 2 / dy
    # Rz = (R[0, 1:-1, 2:] - R[0, 1:-1, :-2]) / 2 / dz
    # omegaz = (omega[0, 1:-1, 2:] - omega[0, 1:-1, :-2]) / 2 / dz
    # divVel = Ux + Vy + omegaz
    # # U = U[0, 1:-1, 1:-1]
    # # V = V[0, 1:-1, 1:-1]
    # # R = R[0, 1:-1, 1:-1]
    # #Ge = Ge[0, 1:-1, 1:-1]
    # #G23e = G23e[0, 1:-1, 1:-1]
    # #omega = omega[0, 1:-1, 1:-1]
    # phivx = Gx * R[0, 1:-1, 1:-1] * V[0, 1:-1, 1:-1] + Ge[0, 1:-1, 1:-1] * (Rx * V[0, 1:-1, 1:-1] + R[0, 1:-1, 1:-1] * Vx)
    # phivy = Gy * R[0, 1:-1, 1:-1] * V[0, 1:-1, 1:-1] + Ge[0, 1:-1, 1:-1] * (Ry * V[0, 1:-1, 1:-1] + R[0, 1:-1, 1:-1] * Vy)
    # phivz = Gz * R[0, 1:-1, 1:-1] * V[0, 1:-1, 1:-1] + Ge[0, 1:-1, 1:-1] * (Rz * V[0, 1:-1, 1:-1] + R[0, 1:-1, 1:-1] * Vz)
    # phiv = Ge[0, 1:-1, 1:-1] * R[0, 1:-1, 1:-1] * V[0, 1:-1, 1:-1]
    # divV = phivx * U[0, 1:-1, 1:-1] + phivy * V[0, 1:-1, 1:-1] + phivz * omega[0, 1:-1, 1:-1] + phiv * divVel
    # py = (Pp[1, 1:-1, 1:-1] - Pp[-1, 1:-1, 1:-1]) / 2 / dy
    # pz = (Pp[0, 1:-1, 2:] - Pp[0, 1:-1, 1:-1]) / 2 / dz
    # Rvp = -py - G23e[0, 1:-1, 1:-1] * pz - f * R[0, 1:-1, 1:-1] * (U[0, 1:-1, 1:-1] - ue) 
    # Vf[0, 1:-1, 1:-1] = -divV + Ge[0, 1:-1, 1:-1] * Rvp
    # Vf[0, 1:-1, 1:-1] /= Ge[0, 1:-1, 1:-1]
    # # y = y_max
    # Ux = (U[-1, 2:, 1:-1] - U[-1, :-2, 1:-1]) / 2 / dx
    # Vx = (V[-1, 2:, 1:-1] - V[-1, :-2, 1:-1]) / 2 / dx
    # Vy = (V[0, 1:-1, 1:-1] - V[-2, 1:-1, 1:-1]) / 2 / dy
    # Vz = (V[-1, 1:-1, 2:] - V[-1, 1:-1, :-2]) / 2 / dz
    # # Gx = (Ge[-1, 2:, 1:-1] - Ge[-1, :-2, 1:-1]) / 2 / dx
    # # Gy = (Ge[0, 1:-1, 1:-1] - Ge[-2, 1:-1, 1:-1]) / 2 / dy
    # # Gz = (Ge[-1, 1:-1, 2:] - Ge[-1, 1:-1, :-2]) / 2 / dz
    # Gx = Gxe[-1, 1:-1, 1:-1]
    # Gy = Gye[-1, 1:-1, 1:-1]
    # Gz = Gze[-1, 1:-1, 1:-1]
    # Rx = (R[-1, 2:, 1:-1] - R[-1, :-2, 1:-1]) / 2 / dx
    # Ry = (R[0, 1:-1, 1:-1] - R[-2, 1:-1, 1:-1]) / 2 / dy
    # Rz = (R[-1, 1:-1, 2:] - R[-1, 1:-1, :-2]) / 2 / dz
    # omegaz = (omega[-1, 1:-1, 2:] - omega[-1, 1:-1, :-2]) / 2 / dz
    # divVel = Ux + Vy + omegaz
    # # U = U[-1, 1:-1, 1:-1]
    # # V = V[-1, 1:-1, 1:-1]
    # # R = R[-1, 1:-1, 1:-1]
    # #Ge = Ge[-1, 1:-1, 1:-1]
    # # G23e = G23e[-1, 1:-1, 1:-1]
    # # omega = omega[-1, 1:-1, 1:-1]
    # phivx = Gx * R[-1, 1:-1, 1:-1] * V[-1, 1:-1, 1:-1] + Ge[-1, 1:-1, 1:-1] * (Rx * V[-1, 1:-1, 1:-1] + R[-1, 1:-1, 1:-1] * Vx)
    # phivy = Gy * R[-1, 1:-1, 1:-1] * V[-1, 1:-1, 1:-1] + Ge[-1, 1:-1, 1:-1] * (Ry * V[-1, 1:-1, 1:-1] + R[-1, 1:-1, 1:-1] * Vy)
    # phivz = Gz * R[-1, 1:-1, 1:-1] * V[-1, 1:-1, 1:-1] + Ge[-1, 1:-1, 1:-1] * (Rz * V[-1, 1:-1, 1:-1] + R[-1, 1:-1, 1:-1] * Vz)
    # divV = phivx * U[-1, 1:-1, 1:-1] + phivy * V[-1, 1:-1, 1:-1] + phivz * omega[-1, 1:-1, 1:-1] + phiv * divVel
    # py = (Pp[0, 1:-1, 1:-1] - Pp[-2, 1:-1, 1:-1]) / 2 / dy
    # pz = (Pp[-1, 1:-1, 2:] - Pp[-1, 1:-1, 1:-1]) / 2 / dz
    # Rvp = -py - G23e[-1, 1:-1, 1:-1] * pz - f * R[-1, 1:-1, 1:-1] * (U[-1, 1:-1, 1:-1] - ue) 
    # Vf[-1, 1:-1, 1:-1] = -divV + Ge[-1, 1:-1, 1:-1] * Rvp
    # Vf[-1, 1:-1, 1:-1] /= Ge[-1, 1:-1, 1:-1]

    return np.array([Uf, Vf, Wf, Tf, Rf])


### MAIN ###
# Domain
x_min, x_max = 0, 400 
y_min, y_max = 0, 400 
z_min, z_max = 0, 125
t_min, t_max = 0, 1
Nx, Ny, Nz, Nt = 41, 41, 21, 1001 # 
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
T_0 = 300 # Initial temperature of dry gas [K]
# Z correction
#Z = zc(X, Y, Z)
# Pressure
p = lambda x, y, z: p_o * np.exp(-g / (T_0 * R_d) * ((H - h(x, y)) * z / H + h(x, y))) # Environmental pressure [N m^{-2}]
p_0 = p(X, Y, Z) # Initial pressure [N m^{-2}]
# Pressure derivatives
px = lambda x, y, z: -p_0 * g * (H - z) / (R_d * T_0 * H) * np.exp(-g / (R_d * T_0) * ((H - h(x, y)) * z / H + h(x, y))) * hx(x, y) # [N m^{-2}]
py = lambda x, y, z: -p_0 * g * (H - z) / (R_d * T_0 * H) * np.exp(-g / (R_d * T_0) * ((H - h(x, y)) * z / H + h(x, y))) * hy(x, y) 
pz = lambda x, y, z: -p_0 * g * (H - z) / (R_d * T_0 * H) * np.exp(-g / (R_d * T_0) * ((H - h(x, y)) * z / H + h(x, y))) 
# Density
L = 0.0065 # Temperature lapse [K/m]
rho_0 = p_o / (R_d * T_0) # Initial density [kg m^{-3}]
#rho = lambda z: rho_0 * (1 - L * z / T_0) ** (g / (R_d * L) - 1) # Environmental density [kg m^{-3}]
rho = lambda x, y, z: rho_0 * (1 - L / T_0 * ((H - h(x, y)) / H * z + h(x, y))) ** (g / (R_d * L) - 1) # Environmental density [kg m^{-3}]
# Initial condition functions
u0 = lambda x, y, z: x * 0 + 5 # [m s^{-1}]
v0 = lambda x, y, z: x * 0 # [m s^{-1}]
w0 = lambda x, y, z: x * 0 # [m s^{-1}]
theta0 = lambda x, y, z: x * 0 + T_0 * (p_o / p_0) ** (R_d / C_v) # Potential temperature [K]
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
rhoe = rho(X, Y, Z)
pe = p(X, Y, Z)
pxe = px(X, Y, Z)
pye = py(X, Y, Z)
pze = pz(X, Y, Z)

# Initial condition
U0 = u0(X, Y, Z)
V0 = v0(X, Y, Z)
W0 = w0(X, Y, Z)
T0 = theta0(X, Y, Z)
R0 = rho0(X, Y, Z)
y0 = np.array([U0, V0, W0, T0, R0])

# Array for approximations
samples = 100
yy = np.zeros((Nt // samples + 1, 5, Ny, Nx, Nz))
yy[0] = y0

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


# for n in range(Nt):
#     Un = yy[n, :(Nx + 1) * (Ny + 1)].reshape(Ny + 1, Nx + 1)
#     Vn = yy[n, (Nx + 1) * (Ny + 1):].reshape(Ny + 1, Nx + 1)
#     B = build_up_b(rho, dt, Un, Vn, dx, dy)  
#     P[n+1] = pressure_poisson(P[n], dx, dy, B)
#     yy[n+1] = yy[n] + dt * Phi(t[n], yy[n].copy(), x=x, y=y, nu=nu, rho=rho, p=P[n+1])  

# for n in range(Nt-1):
#     print("Step:", n)
#     k1 = Phi(t[n], yy[n], **args)
#     k2 = Phi(t[n] + 0.5 * dt, yy[n] + 0.5 * dt * k1, **args)
#     k3 = Phi(t[n] + 0.5 * dt, yy[n] + 0.5 * dt * k2, **args)
#     k4 = Phi(t[n] + dt, yy[n] + dt * k3, **args)
#     yy[n+1] = yy[n] + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)

# Solve
y_tmp = yy[0]
for n in range(Nt-1):
    #print("Step:", n)
    k1 = Phi(t[n], y_tmp, **args)
    k2 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k1, **args)
    k3 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k2, **args)
    k4 = Phi(t[n] + dt, y_tmp + dt * k3, **args)
    y_tmp = y_tmp + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    if n % samples == 0:
        print("Step:", n)
        yy[n // samples + 1] = y_tmp
        Ut = y_tmp[0]
        Vt = y_tmp[1]
        Wt = y_tmp[2]
        print("CFL", dt * (np.max(Ut) / dx + np.max(Vt) / dy + np.max(Wt) / dz))
yy[-1] = y_tmp

# Get velocities
U = yy[:, 0]
V = yy[:, 1]
W = yy[:, 2]
T = yy[:, 3]
R = yy[:, 4]
P = (R_d * R * T) ** (C_v / C_p) / (p_o ** (R_d / C_v))

# Save 
np.savez('output/higrad.npz', U=U, V=V, W=W, T=T, R=R, P=P, X=X, Y=Y, Z=Z, t=t[::samples])