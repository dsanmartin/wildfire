import numpy as np
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
from scipy.sparse.linalg import LinearOperator, gmres, spsolve
from scipy import sparse
import time
from numba import jit  
import matplotlib.pyplot as plt

def TDMA(a, b, c, d):
    N = d.shape[0]
    w = np.zeros(N-1, dtype=np.complex128)
    g = np.zeros(N, dtype=np.complex128)
    p = np.zeros(N, dtype=np.complex128)
    
    w[0] = c[0]/b[0]
    g[0] = d[0]/b[0]

    for i in range(1, N-1):
        w[i] = c[i]/(b[i] - a[i-1]*w[i-1])
    for i in range(1, N):
        g[i] = (d[i] - a[i-1]*g[i-1])/(b[i] - a[i-1]*w[i-1])
    p[N-1] = g[N-1]
    for i in range(N-1, 0, -1):
        p[i-1] = g[i-1] - w[i-1]*p[i]
    return p

@jit
def tridiag_solver(A, f):
    a, b, c = A # Get diagonals
    N = b.shape[0]
    v = np.zeros(N, dtype=np.complex128)
    l = np.zeros(N-1, dtype=np.complex128)
    y = np.zeros(N, dtype=np.complex128)
    u = np.zeros(N, dtype=np.complex128)
    # Determine L, U
    v[0] = b[0]
    for k in range(1, N):
        l[k-1] = a[k-1] / v[k-1]
        v[k] = b[k] - l[k-1] * c[k-1]
    # Solve Ly = f
    y[0] = f[0]
    for k in range(1, N):
        y[k] = f[k] - l[k-1] * y[k-1]
    # Solve Uu = y
    u[-1] = y[-1] / v[-1]
    #for k in range(N-1, -1, -1):
    for k in range(-1, -N, -1):
        u[k-1] = (y[k-1] - c[k] * u[k]) / v[k-1]
    return u

@jit
def mv(a, b, c, v):
    N = b.shape[0]
    out = np.zeros(N, dtype=np.complex128)
    # First element and last element
    out[0] = b[0] * v[0] + c[0] * v[0]
    out[-1] = a[-1] * v[-2] + b[-1] * v[-1]
    # Interior elements
    for i in range(1, N - 1):
        out[i] = a[i] * v[i-1] + b[i] * v[i] + c[i] * v[i+1]
    return out


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
        D2y = spla.circulant(Dyv) 
        # Fix boundary conditions
        D2y[0, 0] = - 1.5 * dy 
        D2y[0, 1] = 2 * dy
        D2y[0, 2] = - 0.5 * dy
        D2y[0, -1] = 0
        D2y[-1, 0] = 0
        D2y /= dy ** 2  
        F_k[0, i] = 0
        F_k[-1, i] -=  P_kNy[i] / dy ** 2
        P_k[:, i] = np.linalg.solve(D2y, F_k[:, i])
    P_FFTFD = np.real(np.fft.ifft(P_k, axis=1))
    P_FFTFD = np.vstack([P_FFTFD, np.ones(Nx - 1) * p_top])
    P_FFTFD = np.hstack([P_FFTFD, P_FFTFD[:, 0].reshape(-1, 1)])
    return P_FFTFD

def fftfd_tridiag(x, y, f, p_top):
    Nx, Ny = x.shape[0], y.shape[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    F = f[:-1, :-1] # Remove boundary
    r = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    # For any domain
    kx = 2 * np.pi * r * dy / x[-1]
    F_k = np.fft.fft(F, axis=1)
    P_k = np.zeros_like(F_k)
    P_kNy = np.fft.fft(np.ones(Nx - 1) * p_top)
    for i in range(Nx-1):
        gamma_r = - 2 - kx[i] ** 2
        F_k[0, i] = 0 + 0.5 * dy * F_k[1, i] # dp/dy = 0
        F_k[-1, i] -=  P_kNy[i] / dy ** 2
        a = np.ones(Ny-2) / dy ** 2
        c = np.ones(Ny-2) / dy ** 2
        c[0] = (2 + 0.5 * gamma_r) / dy
        #c[-1] = 0
        b = np.ones(Ny - 1) * gamma_r / dy ** 2
        b[0] = -1 / dy
        # Ax = LinearOperator((Ny-1, Nx-1), matvec=lambda x: mv(a, b, c, x))
        # P_k[:, i], exitCode = gmres(Ax, F_k[:, i], tol=1e-10)
        # print(exitCode)
        P_k[:, i] = tridiag_solver((a, b, c), F_k[:, i])
    P_FFTFD = np.real(np.fft.ifft(P_k, axis=1))
    P_FFTFD = np.vstack([P_FFTFD, np.ones(Nx - 1) * p_top])
    P_FFTFD = np.hstack([P_FFTFD, P_FFTFD[:, 0].reshape(-1, 1)])
    return P_FFTFD

def fftfd_sparse(x, y, f, p_top):
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
        D2y = spla.circulant(Dyv) 
        # Fix boundary conditions
        # D2y[0, 0] = - 1.5 * dy 
        # D2y[0, 1] = 2 * dy
        # D2y[0, 2] = - 0.5 * dy
        # D2y[0, -1] = 0
        # D2y[-1, 0] = 0
        # D2y /= dy ** 2
        # F_k[0, i] = 0
        # F_k[-1, i] -=  P_kNy[i] / dy ** 2  
        gamma_r = - 2 - kx[i] ** 2
        F_k[0, i] = 0 #+ 0.5 * dy * F_k[1, i] # dp/dy = 0
        F_k[-1, i] -=  P_kNy[i] / dy ** 2
        a = np.ones(Ny - 2) / dy ** 2
        c = np.ones(Ny - 2) / dy ** 2
        c[0] = (2 + 0.5 * gamma_r) / dy
        #c[-1] = 0
        b = np.ones(Ny - 1) * gamma_r
        b[0] = -1 / dy
        D2y = sparse.csc_matrix(D2y)
        P_k[:, i] = spsolve(D2y, F_k[:, i])
    P_FFTFD = np.real(np.fft.ifft(P_k, axis=1))
    P_FFTFD = np.vstack([P_FFTFD, np.ones(Nx - 1) * p_top])
    P_FFTFD = np.hstack([P_FFTFD, P_FFTFD[:, 0].reshape(-1, 1)])
    return P_FFTFD

# Poisson solver using FFT in x direction and FD in y direction.
def fftfd_debug(x, y, f, p_top):
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
    cond = []
    eigv = []
    for i in range(Nx-1):
        Dyv[0] = -2 - (kx[i] * dy) ** 2
        D2y = spla.circulant(Dyv) 
        # Fix boundary conditions
        D2y[0, 0] = - 1.5 #* dy 
        D2y[0, 1] = 2 #* dy
        D2y[0, 2] = - 0.5 #* dy
        D2y[0, -1] = 0
        D2y[-1, 0] = 0
        #D2y /= dy ** 2  
        F_k[0, i] = 0
        F_k[-1, i] -=  P_kNy[i] / dy ** 2
        F_k[1:, i] *= dy ** 2 
        F_k[0, i] *= dy 
        P_k[:, i] = np.linalg.solve(D2y, F_k[:, i])
        asd = np.linalg.cond(D2y) 
        qwe = np.min(np.absolute(np.linalg.eigvals(D2y)))
        cond.append(asd)
        eigv.append(qwe)
    cond = np.array(cond)
    eigv = np.array(eigv)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(cond.shape[0]), cond, 'bo', label=r'$cond(D^{(2)}_y)$')
    plt.title(r"Condition number of $D^{(2)}_y$ for each system")
    plt.legend()
    plt.xlabel("# System")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.semilogy(np.arange(eigv.shape[0]), eigv, 'bo', label=r'$\min(\|\lambda\|)$')
    plt.title(r"Min absolute eigenvalue of $D^{(2)}_y$ for each system")
    plt.legend()
    plt.xlabel("# System")
    plt.grid(True)
    plt.show()
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

    # Compute div(u) = ux + uy 
    # Get nodes for u and v
    # u
    u_ij = u.copy() # u_{i, j} 
    u_ip1j = np.roll(u,-1, axis=1) # u_{i+1, j}
    u_ip2j = np.roll(u,-2, axis=1) # u_{i+2, j}
    u_im1j = np.roll(u, 1, axis=1) # u_{i-1, j}
    u_im2j = np.roll(u, 2, axis=1) # u_{i-2, j}
    # v
    v_ij = v.copy() # v_{i, j}
    v_ijp1 = np.roll(v,-1, axis=0) # v_{i, j+1}
    v_ijp2 = np.roll(v,-2, axis=0) # v_{i, j+2}
    v_ijm1 = np.roll(v, 1, axis=0) # v_{i, j-1}    
    v_ijm2 = np.roll(v, 2, axis=0) # v_{i, j-2}

    # First derivative 
    # Forward/backward difference O(h)
    # ux = (u_ip1j - u_ij) / dx # Forward difference
    # vy = (v_ijp1 - v_ij) / dy # Forward difference
    # ux = (u_ij - u_im1j) / dx # Backward difference. 
    # vy = (v_ij - v_ijm1) / dy # Backward difference. This work but is O(h)
    # Using central difference O(h^2)
    ux = (u_ip1j - u_im1j) / (2 * dx)
    vy = (v_ijp1 - v_ijm1) / (2 * dy) 
    # Forward/backward difference O(h^2)
    # ux = (-u_ip2j + 4 * u_ip1j - 3 * u_ij) / (2 * dx) # Forward difference.
    # vy = (-v_ijp2 + 4 * v_ijp1 - 3 * v_ij) / (2 * dy) # Forward difference. 
    # ux = (3 * u_ij - 4 * u_im1j + u_im2j) / (2 * dx) # Backward difference. 
    # vy = (3 * v_ij - 4 * v_ijm1 + v_ijm2) / (2 * dy) # Backward difference. This doesn't work
    # Boundary conditions correction on y. x is periodic, so no correction needed.
    # O(h) correction for O(h)-forward difference
    # vy[-1] = (v_ij[-1] - v_ij[-2]) / dy # Backward at y=y_max
    # O(h^2) correction for O(h)/O(h^2)-backward or central difference
    vy[0] = (-v_ij[2] + 4 * v_ij[1] - 3 * v_ij[0]) / (2 * dy) # Forward at y=y_min
    # vy[1] = (-v_ij[3] + 4 * v_ij[2] - 3 * v_ij[1]) / (2 * dy) # Forward at y=y_min+dy
    # O(h^2) correction for O(h^2)-forward difference
    vy[-1] = (3 * v_ij[-1] - 4 * v_ij[-2] + v_ij[-3]) / (2 * dy) # Backward at y=y_max
    # vy[-2] = (3 * v_ij[-2] - 4 * v_ij[-3] + v_ij[-4]) / (2 * dy) # Backward at y=y_max-dy
    # Corrections for O(h^2)-backward difference at y=y_min+dy 
    # vy[1] = (v_ij[2] - v_ij[0]) / (2 * dy) # O(h^2)
    # vy[1] = (-v_ij[3] + 6 * v_ij[2] - 3 * v_ij[1] - 2 * v_ij[0]) / (6 * dy) # O(h^3)
    # vy[1] = (v_ij[4] - 6 * v_ij[3] + 18 * v_ij[2] - 10 * v_ij[1] - 3 * v_ij[0]) / (12 * dy) # O(h^4)
    # vy[1] = (-3 * v_ij[5] + 20 * v_ij[4] - 60 * v_ij[3] + 120 * v_ij[2] - 65 * v_ij[1] - 12 * v_ij[0]) / (60 * dy) # O(h^5)
    # RHS of Poisson equation
    f = rho / dt * (ux + vy)
    # Periodic boundary condition on x
    f = np.hstack([f, f[:,0].reshape(-1, 1)]) 
    # Solve Poisson equation
    # p = fftfd(x, y, f, p_y_max)
    p = fftfd_tridiag(x, y, f, p_y_max)
    # Return without right boundary condition on x
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