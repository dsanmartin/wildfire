import numpy as np
import scipy.linalg as spla

# Poisson solver using FFT in x direction and FD in y direction.
def fftfd_solver(x, y, f, p_top):
    Nx, Ny = x.shape[0], y.shape[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = f[:-1, :-1] # Remove boundary
    kx = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    # For any domain
    kx = 2 * np.pi * kx / x[-1]
    F_k = np.fft.fft(F, axis=1)
    P_k = np.zeros_like(F_k)
    Dyv = np.zeros(Ny - 1)
    Dyv[1] = 1
    Dyv[-1] = 1
    P_kNy = np.fft.fft(np.ones(Nx - 1) * p_top)
    for i in range(Nx-1):
        Dyv[0] = -2 - (kx[i] * dy) ** 2
        Dy = spla.circulant(Dyv) / dy ** 2  
        # Fix boundary conditions
        Dy[0, 0] = - 1.5 * dy 
        Dy[0, 1] = 2 * dy
        Dy[0, 2] = - 0.5 * dy
        Dy[0, -1] = 0
        Dy[-1, 0] = 0
        F_k[0, i] = 0
        F_k[-1, i] -=  P_kNy[i] / dy ** 2
        P_k[:, i] = np.linalg.solve(Dy, F_k[:, i])
    P_FFTFD = np.real(np.fft.ifft(P_k, axis=1))
    P_FFTFD = np.vstack([P_FFTFD, np.ones(Nx - 1) * p_top])
    P_FFTFD = np.hstack([P_FFTFD, P_FFTFD[:, 0].reshape(-1, 1)])
    return P_FFTFD

### Periodic solvers ###

# Poisson solver using 2D FFT.
def fft_solver(x, y, f):
    Nx, Ny = x.shape[0], y.shape[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = f[:-1, :-1] # Remove boundary
    kx = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    ky = np.fft.fftfreq(Ny - 1) * (Ny - 1)
    # For any domain
    kx = 2 * np.pi * kx / x[-1]
    ky = 2 * np.pi * ky / y[-1]
    kx[0] = ky[0] = 1e-16 # To avoid zero division
    F_hat = np.fft.fft2(F)
    Kx, Ky = np.meshgrid(kx, ky)
    tmp = - F_hat / (Kx ** 2 + Ky ** 2) 
    tmp[0, 0] = 0 # Fix kx,ky = (0, 0)
    P_a = np.real(np.fft.ifft2(tmp))
    P_a = np.vstack([P_a, P_a[0]])
    P_a = np.hstack([P_a, P_a[:, 0].reshape(-1, 1)])
    return P_a

# Poisson solver using Jacobi
def solve_iterative(u, v, p, tol=1e-10, n_iter=1000, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    dt = kwargs['dt']
    rho = kwargs['rho']

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
        # Interior nodes
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
        # Periodic BC pressure at y = y_min
        p[0, 1:-1] = (
            dy ** 2 * (pn[0, 2:] + pn[0, :-2]) +
            dx ** 2 * (pn[1, 1:-1] + pn[-1, 1:-1]) -
            dx ** 2 * dy ** 2 * b[0, 1:-1]
        ) / (2 * (dx ** 2 + dy ** 2))
        # Periodic BC pressure at y = y_max
        p[-1, 1:-1] = (
            dy ** 2 * (pn[-1, 2:] + pn[-1, :-2]) +
            dx ** 2 * (pn[0, 1:-1] + pn[-2, 1:-1]) -
            dx ** 2 * dy ** 2 * b[-1, 1:-1]
        ) / (2 * (dx ** 2 + dy ** 2))
        # If achieved convergence, break
        if np.linal.norm(p - pn) / np.linalg.norm(p) < tol:
            break
    # Return pressure
    return p

def solve_fft(u, v, **kwargs):
    x = kwargs['x']
    y = kwargs['y']
    dx = kwargs['dx']
    dy = kwargs['dy']
    dt = kwargs['dt']
    rho = kwargs['rho']

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