import numpy as np

SIGMA = 5.670374e-8

def Phi(t, A, **kwargs):
    
    # Parameters
    x = kwargs['x']
    y = kwargs['y']
    if len(x.shape) == 2:
        dx = x[0, 1] - x[0, 0]
        dy = y[1, 0] - y[0, 0]
    else:
        z = kwargs['z']
        dx = x[0, 1, 0] - x[0, 0, 0]
        dy = y[1, 0, 0] - y[0, 0, 0]
        dz = z[0, 0, 1] - z[0, 0, 0]

    N_f = kwargs['N_f']
    F = kwargs['F']
    c_pf = kwargs['c_pf']
    c_pw = kwargs['c_pw']
    Q_rad_s = kwargs['Q_rad_s']
    Q_rad_g = kwargs['Q_rad_g']
    h = kwargs['h']
    a_v = kwargs['a_v']
    # Tg = kwargs['Tg']
    H_w = kwargs['H_w']
    T_vap = kwargs['T_vap']
    Theta = kwargs['Theta']
    H_f = kwargs['H_f']
    T_pyr = kwargs['T_pyr']
    nu_T = kwargs['nu_T']
    C_D = kwargs['C_D']
    c_p = kwargs['c_p']
    c_v = kwargs['c_v']
    R_d = kwargs['R_d']
    p_o = kwargs['p_o']

    # RHS
    Rf, Rw, Ts, Rg, RgU, RgV, RgT, Ro = A
    U = RgU / Rg
    V = RgV / Rg
    Th = RgT / Rg
    Rf_ = np.zeros_like(Rf)
    Rw_ = np.zeros_like(Rw)
    Ts_ = np.zeros_like(Ts)
    Rg_ = np.zeros_like(Rg)
    Ro_ = np.zeros_like(Ro)
    U_  = np.zeros_like(U)
    V_  = np.zeros_like(V)
    Th_ = np.zeros_like(Th)

    p = (R_d * RgT) ** (c_v / c_p) / p_o ** (R_d / c_v)
    Tg = Th / (p_o / p) ** (R_d / c_p)

    # A = 1e-5
    # eps = .5
    # Q_rad_s = A * eps * SIGMA * Ts ** 4
    # Q_rad_g = A * eps * SIGMA * Tg ** 4
    # Q_rad_g = Q_rad_g[1:-1, 1:-1]

    # Solid phase #
    Ff = F(Rf, Ro, Ts)
    Fw = F(Rw, Ro, Ts)
    Rf_ = -N_f * Ff
    Rw_ = -Fw
    Ts_ = Q_rad_s + h * a_v * (Tg - Ts) \
        - Fw * (H_w + c_pw * T_vap) \
        + Ff * (Theta * H_f - c_pf * T_pyr * N_f)
    Ts_ /= (c_pf * Rf + c_pw * Rw)

    # Gas mass #
    # \div (\rho_g \mathbf{u})
    # Version 1 (direct)
    RgUx = (RgU[1:-1, 2:] - RgU[1:-1, :-2]) / (2 * dx)
    RgVy = (RgV[2:, 1:-1] - RgV[:-2, 1:-1]) / (2 * dy)
    divRgUV = RgUx + RgVy
    # Version 2 (expanded)
    # Rgx = (Rg[1:-1, 2:] - Rg[1:-1, :-2]) / (2 * dx)
    # Rgy = (Rg[2:, 1:-1] - Rg[:-2, 1:-1]) / (2 * dy)
    # Ux = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dx)
    # Vy = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dy)
    # divRgUV = Rgx * U[1:-1, 1:-1] + Rg[1:-1, 1:-1] * Ux + Rgy * V[1:-1, 1:-1] + Rg[1:-1, 1:-1] * Vy
    Rg_[1:-1, 1:-1] = N_f * Ff[1:-1, 1:-1] + Fw[1:-1, 1:-1] - divRgUV
    # Gas momentum #
    # U_ = np.copy(RgU) 
    # V_ = np.copy(RgV) 
    RgUU = RgU * U
    RgUV = RgU * V
    RgVU = RgV * U
    RgVV = RgV * V
    RgUUx = (RgUU[1:-1, 2:] - RgUU[1:-1, :-2]) / (2 * dx)
    RgUVy = (RgUV[2:, 1:-1] - RgUV[:-2, 1:-1]) / (2 * dy)
    RgVUx = (RgVU[1:-1, 2:] - RgVU[1:-1, :-2]) / (2 * dx)
    RgVVy = (RgVV[2:, 1:-1] - RgVV[:-2, 1:-1]) / (2 * dy)
    divRgU = RgUUx + RgUVy
    divRgV = RgVUx + RgVVy
    normU = np.sqrt(U**2 + V**2)
    gx = 0
    gy = 0
    Ru = 0
    Rv = 0
    px = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
    py = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
    U_[1:-1, 1:-1] = -divRgU - px + Rg[1:-1, 1:-1] * gx - Ru - Rg[1:-1, 1:-1] * C_D * a_v * normU[1:-1, 1:-1] * U[1:-1, 1:-1]
    V_[1:-1, 1:-1] = -divRgV - py + Rg[1:-1, 1:-1] * gy - Rv - Rg[1:-1, 1:-1] * C_D * a_v * normU[1:-1, 1:-1] * V[1:-1, 1:-1]

    # Temperature #
    # \div(\rho_g\theta)
    # Version 1 (direct)
    RgTU = RgT * U
    RgTV = RgT * V
    RgTUx = (RgTU[1:-1, 2:] - RgTU[1:-1, :-2]) / (2 * dx)
    RgTVy = (RgTV[2:, 1:-1] - RgTV[:-2, 1:-1]) / (2 * dy)
    divRgTUV = RgTUx + RgTVy
    # \div(\rho_g\grad\theta)
    # Version 1 (direct)
    Thx = np.zeros_like(Th)
    Thy = np.zeros_like(Th)
    Thx[1:-1, 1:-1] = (Th[1:-1, 2:] - Th[1:-1, :-2]) / (2 * dx)
    Thy[1:-1, 1:-1] = (Th[2:, 1:-1] - Th[:-2, 1:-1]) / (2 * dy)
    RNTx = Rg * nu_T * Thx
    RNTy = Rg * nu_T * Thy
    RNTxx = (RNTx[1:-1, 2:] - RNTx[1:-1, :-2]) / (2 * dx)
    RNTyy = (RNTy[2:, 1:-1] - RNTy[:-2, 1:-1]) / (2 * dy)
    divGradTh = RNTxx + RNTyy
    Th_[1:-1, 1:-1] = -divRgTUV  \
        - divGradTh \
        + Th[1:-1, 1:-1] / (c_p * Tg[1:-1, 1:-1]) * (h * a_v * (Ts[1:-1, 1:-1] - Tg[1:-1, 1:-1]) * Q_rad_g + (1 - Theta) * Ff[1:-1, 1:-1] * H_f)

    # Ro mass #
    NoFf = N_o * Ff
    # \div(\rho \mathbf{u})  
    # Version 1 (direct)
    RoU = Ro * U
    RoV = Ro * V
    RoUx = (RoU[1:-1, 2:] - RoU[1:-1, :-2]) / (2 * dx)
    RoVy = (RoV[2:, 1:-1] - RoV[:-2, 1:-1]) / (2 * dy)
    divRoUV = RoUx + RoVy
    # Version 2 (expanded)
    # Rox = (Ro[1:-1, 2:] - Ro[1:-1, :-2]) / (2 * dx)
    # Roy = (Ro[2:, 1:-1] - Ro[:-2, 1:-1]) / (2 * dy)
    # Ux = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dx)
    # Vy = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dy)
    # divRoUV = Rox * U[1:-1, 1:-1] + Ro[1:-1, 1:-1] * Ux + Roy * V[1:-1, 1:-1] + Ro[1:-1, 1:-1] * Vy
    # \div (\rho_g\nu_T \nabla(\rho_o/\rho_g))
    # Version 1 (direct)
    Ro_Rg = Ro / Rg
    RoRgx = (Ro_Rg[1:-1, 2:] - Ro_Rg[1:-1, :-2]) / (2 * dx)
    RoRgy = (Ro_Rg[2:, 1:-1] - Ro_Rg[:-2, 1:-1]) / (2 * dy)
    dRgRox = Rg[1:-1, 1:-1] * nu_T * RoRgx
    dRgRoy = Rg[1:-1, 1:-1] * nu_T * RoRgy
    divRgRox = np.zeros_like(dRgRox)
    divRgRoy = np.zeros_like(dRgRoy)
    divRgRox[1:-1, 1:-1] = (dRgRox[1:-1, 2:] - dRgRox[1:-1, :-2]) / (2 * dx)
    divRgRoy[1:-1, 1:-1] = (dRgRoy[2:, 1:-1] - dRgRoy[:-2, 1:-1]) / (2 * dy)
    divRgRo = divRgRox + divRgRoy
    # Version 2 (expanded)
    # Rgx = (Rg[1:-1, 2:] - Rg[1:-1, :-2]) / (2 * dx)
    # Rgy = (Rg[2:, 1:-1] - Rg[:-2, 1:-1]) / (2 * dy)
    # Rox = (Ro[1:-1, 2:] - Ro[1:-1, :-2]) / (2 * dx)
    # Roy = (Ro[2:, 1:-1] - Ro[:-2, 1:-1]) / (2 * dy)
    # RoRgx = (Rox * Rg[1:-1, 1:-1] - Rgx * Ro[1:-1, 1:-1]) / Rg[1:-1, 1:-1] ** 2
    # RoRgy = (Roy * Rg[1:-1, 1:-1] - Rgy * Ro[1:-1, 1:-1]) / Rg[1:-1, 1:-1] ** 2
    # Roxx = (Ro[1:-1, 2:] - 2 * Ro[1:-1, 1:-1] + Ro[1:-1, :-2]) / (dx ** 2)
    # Royy = (Ro[2:, 1:-1] - 2 * Ro[1:-1, 1:-1] + Ro[:-2, 1:-1]) / (dy ** 2)
    # Rgxx = (Rg[1:-1, 2:] - 2 * Rg[1:-1, 1:-1] + Rg[1:-1, :-2]) / (dx ** 2)
    # Rgyy = (Rg[2:, 1:-1] - 2 * Rg[1:-1, 1:-1] + Rg[:-2, 1:-1]) / (dy ** 2)
    # tmpx = Roxx * Rg[1:-1, 1:-1] + Rgx * Rox - (Rox * Rgx + Rg[1:-1, 1:-1] * Rgxx)
    # tmpy = Royy * Rg[1:-1, 1:-1] + Rgy * Roy - (Roy * Rgy + Rg[1:-1, 1:-1] * Rgyy)
    # a = (Rox * Rg[1:-1, 1:-1] - Rgx * Ro[1:-1, 1:-1])
    # b = (Roy * Rg[1:-1, 1:-1] - Rgy * Ro[1:-1, 1:-1])
    # DX = (tmpx * Rg[1:-1, 1:-1] ** 2 - 2 * Rgx * a)
    # DY = (tmpy * Rg[1:-1, 1:-1] ** 2 - 2 * Rgy * b)
    # DDX = Rgx * RoRgx + Rg[1:-1, 1:-1] * DX
    # DDY = Rgy * RoRgy + Rg[1:-1, 1:-1] * DY
    # divRgRo = nu_T * (DDX + DDY)

    Ro_[1:-1, 1:-1] = divRgRo - divRoUV - NoFf[1:-1, 1:-1]

    return np.array([Rf_, Rw_, Ts_, Rg_, U_, V_, Th_, Ro_])


### MAIN ###
# Domain
x_min, x_max = 0, 160#400 
y_min, y_max = 0, 160#400 
z_min, z_max = 0, 120
t_min, t_max = 0, 2
Nx, Ny, Nz, Nt = 41, 41, 21, 501 # 
# Arrays
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
z = np.linspace(z_min, z_max, Nz)
t = np.linspace(t_min, t_max, Nt)
# Meshgrid
# 2D
X, Y = np.meshgrid(x, y)
# 3D
#X, Y, Z = np.meshgrid(x, y, z)
dx, dy, dz, dt = x[1] - x[0], y[1] - y[0], z[1] - z[0], t[1] - t[0]
print(dx, dy, dz, dt)

# Parameters
N_f = 0.4552 # Fuel stochiometric coefficient [1] (Linn, 1997)
N_o = 0.5448 # Oxygen stochiometric coefficient [1] (Linn, 1997)
c_F = 0.07 # Combustion parameter [1] (Linn, 1997)
rho_ref = 1 # Reference density [kg/m^3] (Linn, 1997)
sigma = .001 # Turbulent intensity ? .001 - .1 - 0.01
s = .1 # Scale of smallest fuel element [m] (Linn, 1997)
rho_o = 1.3311 # 1.42902 [kg/m^3] This is variable
c_pf = 2380 # Specific heat of wood [J/kg/K] https://theengineeringmindset.com/specific-heat-capacity-of-materials/
c_pw = 4184 # Specific heat of water [J/kg/K] https://en.wikipedia.org/wiki/Specific_heat_capacity
#Q_rad = 1e6 # Thermal radiation heat flux  [J/m^2/s] ? This is variable
Q_rad_s = 1e2
Q_rad_g = 1e2
h = 3000 # Convective heat exchange coefficient [J/m^2/K/s] https://www.engineeringtoolbox.com/convective-heat-transfer-d_430.html
a_v = 1 # The contact area per unit volume between the gas and the solid [m] ?
Tg = 300 # Temperature of the combined gases [K] ?
H_w = 2.259e6 # 2257e3 Heat energy per unit of mass (water) [J/kg] https://www.khanacademy.org/science/biology/water-acids-and-bases/water-as-a-solid-liquid-and-gas/a/specific-heat-heat-of-vaporization-and-freezing-of-water
T_vap = 373.15 #  Temperature at which the liquid water evaporates [K] https://www.usgs.gov/special-topic/water-science-school/science/evaporation-and-water-cycle?qt-science_center_objects=0#qt-science_center_objects
Theta = .5 #  ? [1]
H_f = 21.20e6 # Heat energy per unit of mass (wood) [J/kg] https://en.wikipedia.org/wiki/Heat_of_combustion
H_f = 8914e3
T_pyr = 773.15 # The temperature at which the solid fuel begins to pyrolyse [K] https://www.sciencedirect.com/science/article/pii/B978012815497700004X
nu_T = 1e-1 # Turbulent Viscosity [m^2/s] 
C_D = .5 # ? Coefficient of drag
R_d = 287 # Gas constant for dry air [J/kg/K] 
c_p = 1004 # Specific heat of air at constant pressure [J/kg/K]
c_v = 717 # Specific heat of air at constant volumex [J/kg/K]
p_o = 1e5 # Base state pressure [N m^2]
rho_g = 1.225 # Density of air [kg/m^3]
# Psi density function
b = 200
m = np.sqrt(2)
Psi = lambda T: m * T + b


# Lambda
lamb = lambda rho_f: rho_f * rho_o / (rho_f / N_f + rho_o / N_o) ** 2

# Universal reaction rate
F = lambda rho_f, rho_o, T: c_F * rho_f * rho_o * sigma * Psi(T) * lamb(rho_f) / rho_ref / s ** 2

# Initial conditions
# Solid phase
T = 200 # [K]
Rf0 = lambda x, y: 1 + x * 0
Rw0 = lambda x, y: 1 + y * 0
x_0, y_0 = (x_max - x_min) / 2, (y_max - y_min) / 2
ss = 500 # 1000
Ts0 = lambda x, y: T * np.exp(-((x - x_0) ** 2 + (y - y_0) ** 2) / ss) + 300
# Gas phase
p = p_o + 1
Rg0 = lambda x, y: rho_g + x * 0
U0 = lambda x, y: 0 * x + 3
V0 = lambda x, y: 0 * y + 0
Th0 = lambda x, y: Tg * (p_o / p) ** (R_d / c_p) + x * 0 
Ro0 = lambda x, y: rho_o + x * 0

# Initial condition
Rf_0 = Rf0(X, Y)
Rw_0 = Rw0(X, Y)
Ts_0 = Ts0(X, Y)
Rg_0 = Rg0(X, Y)
U_0 = U0(X, Y) 
V_0 = V0(X, Y)
Th_0 = Th0(X, Y)
Ro_0 = Ro0(X, Y)
y0 = np.array([Rf_0, Rw_0, Ts_0, Rg_0, U_0 * Rg_0, V_0 * Rg_0, Th_0 * Rg_0, Ro_0])
# y0 = np.array([Rf_0, Rw_0, Ts_0, Rg_0, U_0, V_0, Ro_0])

print(np.min(Th_0), np.max(Th_0))

# Array for approximations
samples = 100
yy = np.zeros((Nt // samples + 1, len(y0), Ny, Nx))
# yy = np.zeros((Nt, 5, Ny, Nx, Nz))
yy[0] = y0

args = {
    'x': X,
    'y': Y,
    'N_f': N_f,
    'N_o': N_o,
    'c_pf': c_pf,
    'c_pw': c_pw,
    'Q_rad_s': Q_rad_s,
    'Q_rad_g': Q_rad_g,
    'h': h,
    'a_v': a_v,
    'Tg': Tg,
    'H_w': H_w,
    'T_vap': T_vap,
    'Theta': Theta,
    'H_f': H_f,
    'T_pyr': T_pyr,
    'F': F,
    'nu_T': nu_T,
    'C_D': C_D,
    'c_p': c_p,
    'c_v': c_v,
    'p_o': p_o,
    'R_d': R_d,
}

# Solve Euler
y_tmp = yy[0]
for n in range(Nt-1):
    #print("Step:", n)
    y_tmp = y_tmp + dt * Phi(t[n], y_tmp, **args)
    if n % samples == 0:
        print("Step:", n)
        yy[n // samples + 1] = y_tmp
yy[-1] = y_tmp

# # Solve RK4
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
# yy[-1] = y_tmp


# Get velocities
Rf = yy[:, 0]
Rw = yy[:, 1]
Ts = yy[:, 2]
Rg = yy[:, 3]
U = yy[:, 4] / Rg
V = yy[:, 5] / Rg
Tg = yy[:, 6] / Rg
Ro = yy[:, 7]

# Save 
np.savez('output/higrad-firetec.npz', Rf=Rf, Rw=Rw, Ts=Ts, Rg=Rg, U=U, V=V, Tg=Tg, Ro=Ro, X=X, Y=Y, t=t[::samples])