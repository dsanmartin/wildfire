import numpy as np
from parameters import h, Y_f, A, C_p, H_R, T_act, A_T, rho, T_inf#, S_top, S_bot, Sx #  B_tilde, A_alpha

# Gaussian
G = lambda x, y, x0, y0, sx, sy, A: A * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2)) 

# Arrhenius-like equation 
K = lambda T, A, B: A * np.exp(-B / T) 
CV = lambda x, T_pc: 2 * x / T_pc - 1
Sg = lambda x, x0, k: 1 / (1 + np.exp(-2 * k * (x - x0)))
S1 = lambda x, x0, k: .5 * (1 + np.tanh(k * (x - x0)))#
S2 = lambda x, x0, k, T_pc: Sg(CV(x, T_pc), x0, k)
S3 = lambda x, T_pc: x > T_pc
# # A parameter nodel, 
# if A < 0:
#     # AT = lambda T: np.exp(B_tilde) * T ** (A_alpha) # A(T) = B * T ** A_alpha
#     # AT = lambda T: A_T * C_p / H_R / np.exp(-B / T)
#     AT = lambda T, A_T: A_T * C_p / H_R / np.exp(-T_act / T)
#     AT = lambda T, A_T: C_p / H_R * (A_T  + h / (rho * C_p) * (T - T_inf)) / np.exp(-T_act / T)
# else:
#     AT = lambda T: A # Constant
# # Convective heat transfer coefficient
# if h < 0:
#     hv = lambda v: np.piecewise(v, [v < 2, v >= 2], [
#             lambda v: 0 * v, # No information
#             lambda v: 12.12 - 1.16 * v + 11.6 * v ** 0.5 # Wind chill factor
#     ])
# else:
#     hv = lambda v: h # Constant 17-18 W/m^2/K?
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
    plate = lambda x, y: (y >= y_start) & (y <= y_end) & (x <= x_end) & (x >= x_start)
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