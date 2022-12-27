import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
import time

### Solver for non-all-axis periodic boundary ###

# Poisson solver using FFT in x direction and FD in y direction.
def fftfd(x, y, f, p_top):
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

def solve_fftfd(u, v, **kwargs):
    x = kwargs['x']
    y = kwargs['y']
    dx = kwargs['dx']
    dy = kwargs['dy']
    dt = kwargs['dt']
    rho = kwargs['rho']
    p_y_min, p_y_max = kwargs['bc_on_y'][4]

    # div(u) 
    # Get nodes for u and v
    u_ij = u.copy()
    u_ip1j = np.roll(u, -1, axis=1) # u_{i+1, j}
    u_im1j = np.roll(u, 1, axis=1) # u_{i-1, j}
    v_ij = v.copy()
    v_ijp1 = np.roll(v, -1, axis=0) # v_{i, j+1}
    v_ijm1 = np.roll(v, 1, axis=0) # v_{i, j-1}

    # First derivative using central difference O(h^2)
    ux = (u_ip1j - u_im1j) / (2 * dx)
    # vy = (v_ijp1 - v_ijm1) / (2 * dy) # Central difference
    # vy = (v_ijp1 - v_ij) / dy # Forward difference
    vy = (v_ij - v_ijm1) / dy # Backward difference
    # Fixed boundary conditions on y
    # vy[0, 1:-1] = (-3 * v[0, 1:-1] + 4 * v[1, 1:-1] - v[2, 1:-1]) / (2 * dy) # Forward at y=y_min
    # vy[-1, 1:-1] = (3 * v[-1, 1:-1] - 4 * v[-2, 1:-1] + v[-3, 1:-1]) / (2 * dy) # Backward at y=y_max
    vy[0] = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * dy) # Forward at y=y_min
    vy[-1] = (3 * v[-1] - 4 * v[-2] + v[-3]) / (2 * dy) # Backward at y=y_max
    f = rho / dt * (ux + vy)
    f = np.hstack([f, f[:,0].reshape(-1, 1)])
    p = fftfd(x, y, f, p_y_max)

    return p[:, :-1]

def solve_iterative(u, v, p, tol=1e-10, n_iter=1000, **kwargs):
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
        p[0] = (4 * p[1, :] - p[2, :]) / 3 
        # p=p_y_max conditions at y = y_max
        p[-1,:] = p_y_max 
        # If achieved convergence, break
        if np.linal.norm(p - pn) / np.linalg.norm(p) < tol:
            break
    # Return pressure approximation
    return p

def solve_iterative_ibm(u, v, p, tol=1e-10, n_iter=10000, **kwargs):
    dx = kwargs['dx']
    dy = kwargs['dy']
    dt = kwargs['dt']
    rho = kwargs['rho']
    p_y_max = kwargs['p_y_max']
    cut_nodes_y, cut_nodes_x = kwargs['cut_nodes']
    dead_nodes = kwargs['dead_nodes']

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

    # Check error
    f = b.copy()
    f[dead_nodes] = 0
    f[cut_nodes_y, :] = 0 
    M = np.ones_like(f)
    M[dead_nodes] = 0
    M = M[:-1]
    f = f[:-1]
    f[-1] -= p_y_max / dy ** 2
    M = M.astype(bool)
    F = flatten_tilde(f, M)
    Nx, Ny = kwargs['Nx'], kwargs['Ny']
    Dxv = np.zeros(Nx - 1)
    Dxv[0] = -2
    Dxv[1] = 1
    Dxv[-1] = 1
    Dx = spla.circulant(Dxv) / dx ** 2
    Dyv = np.zeros(Ny - 1)
    Dyv[0] = -2
    Dyv[1] = 1
    Dyv[-1] = 1
    Dy = spla.circulant(Dyv) / dy ** 2

    # Iterative
    for n in range(n_iter):
        pn = p.copy()
        # Inside domain
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

        # Set dead nodes to zero
        p[dead_nodes] = 0
        
        # Boundary conditions
        # dp/dy = 0 at y = y_min
        # Forward difference O(h^2)
        #p[0] = (4 * p[1, :] - p[2, :]) / 3 
        p[cut_nodes_y, cut_nodes_x] = (4 * p[cut_nodes_y + 1, cut_nodes_x] - p[cut_nodes_y + 2, cut_nodes_x]) / 3 
        
        # p=p_y_max conditions at y = y_max
        p[-1] = p_y_max 
        # If achieved convergence, break
        # if np.linalg.norm(p - pn) / np.linalg.norm(p) < tol:
        #     break
        if np.linalg.norm(compute_matrix_vector_product(flatten_tilde(p[:-1], M), Dx, Dy, M, kwargs) - F) / np.linalg.norm(F) < tol:
            break
    print("Residuo Iterative:", np.linalg.norm(compute_matrix_vector_product(flatten_tilde(p[:-1], M), Dx, Dy, M, kwargs) - F) / np.linalg.norm(F))
    # Return pressure approximation
    return p

# Using GMRES
def flatten_tilde(A, M):
    return A[M]

def reshape_tilde(b, M, C):
    C[M] = b
    return C
    
def compute_matrix_vector_product(x, Dx, Dy, M, args):
    Nx, Ny = Dx.shape[0], Dy.shape[0]
    dy = args['dy']
    cut_nodes_y, cut_nodes_x = args['cut_nodes']
    X = reshape_tilde(x, M, np.zeros((Ny, Nx)))
    B = Dy.dot(X) + X.dot(Dx)
    # B[0, :] = (X[1, :] - X[0, :]) / dy 
    B[cut_nodes_y, :] = (X[cut_nodes_y + 1, :] - X[cut_nodes_y, :]) / dy
    w = flatten_tilde(B, M)
    return w

def solve_gmres(u, v, p, tol=1e-10, n_iter=1000, **kwargs):
    rho = kwargs['rho']
    p_y_max = kwargs['p_y_max']
    Nx, Ny = kwargs['Nx'], kwargs['Ny']
    dx, dy, dt = kwargs['dx'], kwargs['dy'], kwargs['dt']
    dead_nodes = kwargs['dead_nodes']
    cut_nodes_y, cut_nodes_x = kwargs['cut_nodes']
    Dxv = np.zeros(Nx - 1)
    Dxv[0] = -2
    Dxv[1] = 1
    Dxv[-1] = 1
    Dx = spla.circulant(Dxv) / dx ** 2
    Dyv = np.zeros(Ny - 1)
    Dyv[0] = -2
    Dyv[1] = 1
    Dyv[-1] = 1
    Dy = spla.circulant(Dyv) 
    # BC 
    Dy[0, -1] = 0
    Dy[-1, 0] = 0
    Dy = Dy / dy ** 2
    # F
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
    f[dead_nodes] = 0
    f[cut_nodes_y, :] = 0 
    M = np.ones_like(f)
    M[dead_nodes] = 0
    M = M[:-1]
    f = f[:-1]
    p = p[:-1]
    f[-1] -= p_y_max / dy ** 2
    M = M.astype(bool)
    F = flatten_tilde(f, M)
    p0 = flatten_tilde(p, M)
    Av = lambda v: compute_matrix_vector_product(v, Dx, Dy, M, kwargs)
    afun = spspla.LinearOperator(shape=(F.shape[0], F.shape[0]), matvec=Av)
    start = time.time()
    P_a, _ = spspla.gmres(afun, F, x0=p0, tol=tol, maxiter=n_iter)
    print("Residuo GMRES: ", np.linalg.norm(compute_matrix_vector_product(P_a, Dx, Dy, M, kwargs) - F) / np.linalg.norm(F))
    end = time.time()
    # print("Time:", end - start)
    P = reshape_tilde(P_a, M, np.zeros_like(f))
    P = np.vstack([P, np.ones(Nx-1) * p_y_max])
    return P

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
def solve_iterative_periodic(u, v, p, tol=1e-10, n_iter=1000, **kwargs):
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