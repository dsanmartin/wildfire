import numpy as np

# Gaussian
G = lambda x, y, x0, y0, sx, sy, A: A * np.exp(-((x - x0) ** 2 / sx ** 2 + (y - y0) ** 2 / sy ** 2)) 

# Arrhenius-like equation 
K = lambda T, A, B: A * np.exp(-B / T) 
CV = lambda x, T_pc: 2 * x / T_pc - 1
Sg = lambda x, x0, k: 1 / (1 + np.exp(-2 * k * (x - x0)))
S1 = lambda x, x0, k: .5 * (1 + np.tanh(k * (x - x0)))#
S2 = lambda x, x0, k, T_pc: Sg(CV(x, T_pc), x0, k)
S3 = lambda x, T_pc: x > T_pc

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