import numpy as np
from arguments import T_act, A, H_R, h, k # Parameters from command line
from parameters import T_pc, T_inf, n_arrhenius, h_rad, C_p, rho, a_v, S_T_0, S_k_0, S_k, sigma, delta, sutherland_law, radiation # Default parameters

# Gaussian
G = lambda x, y, x0, y0, sx, sy, A: A * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2)) 

# Arrhenius-like equation 
K = lambda T: A * np.exp(-T_act / T) # K = lambda T, A, T_act: A * np.exp(-T_act / T) 
Km = lambda T: K(T) * T ** n_arrhenius
# Step function and mofiied step function
H = lambda T: T > T_pc
CV = lambda x, T_pc: 2 * x / T_pc - 1
Sg = lambda x, x0, k: 1 / (1 + np.exp(-2 * k * (x - x0)))
HS2 = lambda x, x0, k: .5 * (1 + np.tanh(k * (x - x0)))#
HS3 = lambda x, x0, k, T_pc: Sg(CV(x, T_pc), x0, k)
# Source/sink
Q_rad = lambda T: h_rad * (T ** 4 - T_inf ** 4) / (rho * C_p)
source = lambda T, Y: H_R * Y * K(T) * H(T) / C_p #+ Q_rad(T)
sink = lambda T: -h * a_v * (T - T_inf) / (rho * C_p) 

# Diffusion
sutherland = lambda T: S_k_0 * (T / S_T_0) ** 1.5 * (S_T_0 + S_k) / (T + S_k) / (rho * C_p)
sutherland_T = lambda T: 1.5 * S_k_0 * (S_T_0 + S_k) / T ** 1.5 * (T ** .5 * (T + S_k) - T ** 1.5) / (T + S_k) ** 2 / (rho * C_p)
stefan_radiation = lambda T: 4 * sigma * delta * T ** 3 / (rho * C_p)
stefan_radiation_T = lambda T: 12 * sigma * delta * T ** 2 / (rho * C_p)
# Conduction
# if sutherland_law:
#     kc = lambda T: sutherland(T)
# else:
#     kc = lambda T: k + T * 0
kc = lambda T: sutherland_law * sutherland(T) 
kr = lambda T: radiation * stefan_radiation(T) 
# Radiation
# if radiation:
#     kr = lambda T: 1 # Radiative
kT = lambda T: kc(T) + kr(T) + (sutherland_law == radiation == False) * k # Total
kTT = lambda T: sutherland_law * sutherland_T(T) + radiation * stefan_radiation_T(T) 
# # A parameter nodel, 
# if A < 0:
#     # AT = lambda T: np.exp(B_tilde) * T ** (A_alpha) # A(T) = B * T ** A_alpha
#     # AT = lambda T: A_T * C_p / H_R / np.exp(-B / T)
#     AT = lambda T, A_T: A_T * C_p / H_R / np.exp(-T_act / T)
#     AT = lambda T, A_T: C_p / H_R * (A_T  + h / (rho * C_p) * (T - T_inf)) / np.exp(-T_act / T)
# else:
#     AT = lambda T: A # Constant
# # Convective heat transfer coefficient
if h < 0:
    hv = lambda v: np.piecewise(v, [v < 2, v >= 2], [
            lambda v: 0 * v, # No information
            lambda v: 12.12 - 1.16 * v + 11.6 * v ** 0.5 # Wind chill factor
    ])
else:
    hv = lambda v: h # Constant 17-18 W/m^2/K?
# # Fuel consumption coefficient
# if Y_f < 0:
#     Yft = lambda t: np.piecewise(t, [t <= 20, t > 20], [lambda t: 2, lambda t: 3])
# else:
#     Yft = lambda t: Y_f
# Source/sink bounds
S_tilde = lambda S, S_top, S_bot, Sx: np.piecewise(S, [S <= S_top, S > S_top, (S > S_top) & (S > Sx)], [
    lambda S: S,
    lambda S: (S_bot - S_top) / (Sx - S_top) * (S - S_top) + S_top,
    lambda S: S_bot
])
# S_tilde = lambda S, S_top, S_bot, Sx: np.piecewise(S, [S <= S_top, S > S_top], [
#     lambda S: S,
#     lambda S: (S_bot - S_top) / (Sx - S_top) * (S - S_top) + S_top
# ])
# S_T = lambda S: S_tilde(S, S_top, S_bot, Sx)
# Sutherland's law
kT = lambda T: S_k_0 * (T / S_T_0) ** 1.5 * (S_T_0 + S_k) / (T + S_k) / (rho * C_p)
kTp = lambda T: 1.5 * S_k_0 * (S_T_0 + S_k) / T ** 1.5 * (T ** .5 * (T + S_k) - T ** 1.5) / (T + S_k) ** 2 / (rho * C_p)


def domain(x_min, x_max, y_min, y_max, t_min, t_max, Nx, Ny, Nt):
    # 1d arrays
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    t = np.linspace(t_min, t_max, Nt)
    # Meshgrid
    Xm, Ym = np.meshgrid(x[:-1], y)
    # Interval size
    dx, dy, dt = x[1] - x[0], y[1] - y[0], t[1] - t[0]
    return x, y, t, Xm, Ym, dx, dy, dt

# For temperature initial condition
def create_plate(x_start, x_end, y_start, y_end):
    plate = lambda x, y: ((y >= y_start) & (y <= y_end) & (x <= x_end) & (x >= x_start)).astype(int)
    return plate

def create_half_gaussian(x_center, width, height):
    half_gaussian = lambda x, y: G(x, y, x_center, 0, width, height, 1)
    return half_gaussian

def multivariate_normal(x, nu, Sigma):
    return np.exp(-0.5 * (x - nu) @ np.linalg.inv(Sigma) @ (x - nu)) / ((2 * np.pi) ** x.shape[0] * np.linalg.det(Sigma)) ** 0.5


def create_2d_gaussian(mu, Sigma):
    Sinv = np.linalg.inv(Sigma)
    a, b, c, d = Sinv[0, 0], Sinv[0, 1], Sinv[1, 0], Sinv[1, 1]
    nx, ny = mu
    return lambda x, y: np.exp(-0.5 * ((x - nx) * (a * (x - nx) + b * (y - ny)) + (y - ny) * (c * (x - nx) + d * (y - ny)))) / ((2 * np.pi) ** 2 * np.linalg.det(Sigma)) ** 0.5


def non_dimensional_numbers(parameters):
    x_min, x_max = parameters['x'][0], parameters['x'][-1]
    y_min, y_max = parameters['y'][0], parameters['y'][-1]
    t_min, t_max = parameters['t'][0], parameters['t'][-1]
    nu, Pr = parameters['nu'], parameters['Pr']
    rho, C_p, h = parameters['rho'], parameters['C_p'], parameters['h']
    A = parameters['A']
    g = parameters['g']
    U_0, V_0, T_0 = parameters['u0'], parameters['v0'], parameters['T0']
    L_v = (y_max - y_min)
    L = ((x_max - x_min) * L_v) ** 0.5
    dT = (T_0.max() - T_0.min()) 
    speed = np.sqrt(U_0 ** 2 + V_0 ** 2)
    U = np.max(speed)
    T = np.max(T_0)
    T_avg = np.max(T_0)
    alpha = 1 / T_avg
    # Reynolds
    Re = U * L / nu
    # Grashof
    Gr = abs(g[-1]) * alpha * dT * L_v ** 3 / nu ** 2 
    # Rayleigh
    Ra = Gr * Pr
    # Strouhal
    Sr = A * L / U
    # Stefan 
    Ste = C_p * dT / H_R
    # Stanton
    St = h * L / (rho * C_p * U)
    # Zeldovich
    Ze = T_act  * dT /  T ** 2
    # Return
    return Re, Gr, Ra, Sr, Ste, St, Ze
