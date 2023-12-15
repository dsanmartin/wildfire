import numpy as np
from utils import gamma
from numba import jit, njit, prange
from derivatives import compute_first_derivative_half_step
from multiprocessing import Pool
import itertools

@jit(nopython=True)
def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal linear system Ax = f using the Thomas algorithm.
    Numba is used to speed up the computation.

    Parameters
    ----------
    a: np.ndarray (N-1,)
        Lower diagonal of the matrix A.
    b: np.ndarray (N,)
        Main diagonal of the matrix A.
    c: np.ndarray (N-1,)
        Upper diagonal of the matrix A.
    f : np.ndarray (N,)
        Right-hand side of the linear system.

    Returns
    -------
    np.ndarray
        Solution of the linear system.

    Notes
    -----
    The Thomas algorithm is a specialized algorithm for solving tridiagonal
    linear systems. It is more efficient than general-purpose algorithms such
    as Gaussian elimination, especially for large systems.
    """
    # a, b, c = A # Get diagonals
    N = b.shape[0]
    v = np.zeros(N, dtype=np.complex128)
    l = np.zeros(N - 1, dtype=np.complex128)
    y = np.zeros(f.shape, dtype=np.complex128)
    u = np.zeros(f.shape, dtype=np.complex128)
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

@jit(nopython=True)
def get_LU_thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple:
    """
    Compute the LU decomposition of a tridiagonal matrix A using the Thomas algorithm.
    Numba is used to speed up the computation.

    Parameters
    ----------
    a: np.ndarray (N-1,)
        Lower diagonal of the matrix A.
    b: np.ndarray (N,)
        Main diagonal of the matrix A.
    c: np.ndarray (N-1,)
        Upper diagonal of the matrix A.

    Returns
    -------
    tuple
        Tuple containing the lower and upper diagonal of the LU decomposition.
    """
    N = b.shape[0]
    u = np.zeros(N, dtype=np.complex128)
    l = np.zeros(N - 1, dtype=np.complex128)
    # Determine L, U
    u[0] = b[0]
    for i in range(1, N):
        l[i-1] = a[i-1] / u[i-1]
        u[i] = b[i] - l[i-1] * c[i-1]
    return l, u

@jit(nopython=True)
def solve_Ly_thomas(l: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Solve a linear system Ly = f using the Thomas algorithm.
    Numba is used to speed up the computation.

    Parameters
    ----------
    l: np.ndarray (N-1,)
        Lower diagonal of the matrix L.
    f : np.ndarray (N,)
        Right-hand side of the linear system.

    Returns
    -------
    np.ndarray
        Solution of the linear system.
    """
    N = f.shape[0]
    y = np.zeros(f.shape, dtype=np.complex128)
    # Solve Ly = f
    y[0] = f[0]
    for k in prange(1, N):
        y[k] = f[k] - l[k-1] * y[k-1]
    return y

@jit(nopython=True)
def solve_triangular_systems_thomas(l: np.ndarray, u: np.ndarray, c: np.ndarray, f: np.ndarray) -> np.ndarray:
    N = f.shape[0]
    x = np.zeros(f.shape, dtype=np.complex128)
    y = np.zeros(f.shape, dtype=np.complex128)
    # Solve Ly = f
    y[0] = f[0]
    for i in range(1, N):
        y[i] = f[i] - l[i-1] * y[i-1]
    # Solve Ux = y
    x[-1] = y[-1] / u[-1]
    #for k in range(N-1, -1, -1):
    for i in range(-1, -N, -1):
        x[i-1] = (y[i-1] - c[i] * x[i]) / u[i-1]
    return x

@jit(nopython=True, parallel=True)
def numba_process_loop(kx, ky, Nx, Ny, Nz, dz, F_k, P_kNz, P_k, solver):
    for r in prange(Nx - 1):
        for s in prange(Ny - 1):
            # Compute gamma
            gamma_rs = - 2 - kx[r] ** 2 - ky[s] ** 2
            # Create RHS of system
            F_k[s, r, 0] = 0 + 0.5 * dz * F_k[s, r, 1] # dp/dy = 0
            # Substract coefficient of top boundary condition
            F_k[s, r, -1] -=  P_kNz[s, r] / dz ** 2 
            # Create A in the system. Only keep diagonals of matrix
            a = np.ones(Nz - 2) / dz ** 2
            b = np.ones(Nz - 1) * gamma_rs / dz ** 2
            c = np.ones(Nz - 2) / dz ** 2
            # Fix first coefficients
            c[0] = (2 + 0.5 * gamma_rs) / dz
            b[0] = -1 / dz
            # Solve system A P_k = F_k
            P_k[s, r, :] = solver(a, b, c, F_k[s, r, :])

def fftfd(f: np.ndarray, params: dict, solver: callable = thomas_algorithm) -> np.ndarray:
    """
    Compute the 2D Poisson equation using the FFT-FD method.
    FFT for x-direction and central differences for y-direction.

    Parameters
    ----------
    f : array_like
        Input array of shape (Nx, Ny) containing the right-hand side of the Poisson equation.
    params : dict
        Dictionary containing the parameters of the problem:
        - Nx : int
            Number of intervals in the x direction.
        - Ny : int
            Number of intervals in the y direction.
        - dx : float
            Spacing between grid points in the x direction.
        - dy : float
            Spacing between grid points in the y direction.
        - x : array_like
            Array of shape (Nx,) containing the x coordinates of the grid points.
        - bc_on_y : list
            List containing the boundary conditions on the y axis:
            [bc_left, bc_right, bc_bottom, bc_top].
        - p_top : float
            Pressure value at the top boundary.
    solver : function, optional
        Function to solve the tridiagonal system. Default is thomas_algorithm.

    Returns
    -------
    ndarray (Ny, Nx)
        Solution of the Poisson equation.
    """
    Nx, Ny = params['Nx'], params['Ny'] # Number of intervals
    _, dy = params['dx'], params['dy'] # Space step
    x_max = params['x'][-1] # Max x value
    _, p_top = params['bc_on_z'][4] # Pressure top boundary condition
    # F = f[:-1, :-1] # Remove boundary
    F = f[:-1, :] # Remove last row
    r = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    # Scale frequencies
    kx = 2 * np.pi * r * dy / x_max
    # Compute FFT in x direction (column-wise)
    F_k = np.fft.fft(F, axis=1)
    # To store pressure in Fourier space
    P_k = np.zeros_like(F_k)
    # Compute FFT in the last row (top boundary condition)
    P_kNy = np.fft.fft(np.ones(Nx - 1) * p_top)
    # Solve the system for each gamma
    for k in range(Nx - 1):
        # Compute gamma
        gamma_r = - 2 - kx[k] ** 2 
        # Create RHS of system
        F_k[0, k] = 0 + 0.5 * dy * F_k[1, k] # dp/dy = 0
        # Substract coefficient of top boundary condition
        F_k[-1, k] -=  P_kNy[k] / dy ** 2 
        # Create A in the system. Only keep diagonals of matrix
        a = np.ones(Ny-2) / dy ** 2
        b = np.ones(Ny - 1) * gamma_r / dy ** 2
        c = np.ones(Ny-2) / dy ** 2
        # Fix first coefficients
        c[0] = (2 + 0.5 * gamma_r) / dy
        b[0] = -1 / dy
        # Solve system A P_k = F_k
        # A = (a, b, c) # Create tuple of diagonals
        P_k[:, k] = solver(a, b, c, F_k[:, k])
    # Compute IFFT in x direction (column-wise) to restore pressure
    p = np.real(np.fft.ifft(P_k, axis=1))
    # Add top boundary condition
    p = np.vstack([p, np.ones(Nx - 1) * p_top])
    return p

def fftfd_3D(f: np.ndarray, params: dict, solver: callable = thomas_algorithm) -> np.ndarray:
    """
    Compute the 3D Poisson equation using the FFT2D-FD method.
    FFT2D for x-direction, y-direction and central differences for z-direction.

    Parameters
    ----------
    f : array_like
        Input array of shape (Ny, Nx, Nz) containing the right-hand side of the Poisson equation.
    params : dict
        Dictionary containing the parameters of the problem:
        - Nx : int
            Number of intervals in the x direction.
        - Ny : int
            Number of intervals in the y direction.
        - Nz : int
            Number of intervals in the z direction.
        - dx : float
            Spacing between grid points in the x direction.
        - dy : float
            Spacing between grid points in the y direction.
        - dz : float
            Spacing between grid points in the z direction.
        - x : array_like
            Array of shape (Nx,) containing the x coordinates of the grid points.
        - y : array_like
            Array of shape (Ny,) containing the y coordinates of the grid points.
        - bc_on_z : list
            List containing the boundary conditions on the z axis
        - p_top : float
            Pressure value at the top boundary.
    solver : function, optional
        Function to solve the tridiagonal system. Default is thomas_algorithm.

    Returns
    -------
    ndarray (Ny, Nx)
        Solution of the Poisson equation.
    """
    Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz'] # Number of intervals
    dz = params['dz']# Space step
    x_max = params['x'][-1] # Max x value
    y_max = params['y'][-1] # Max y value
    _, p_top = params['bc_on_z'][5] # Pressure top boundary condition
    # F = f[:-1, :-1] # Remove boundary
    F = f[:, :, :-1] # Remove last slice
    rr = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    ss = np.fft.fftfreq(Ny - 1) * (Ny - 1)
    # Scale frequencies
    kx = 2 * np.pi * rr * dz / x_max
    ky = 2 * np.pi * ss * dz / y_max
    # Compute FFT in x direction (column-wise)
    F_k = np.fft.fft2(F, axes=(0, 1))
    # To store pressure in Fourier space
    P_k = np.zeros_like(F_k)
    # Compute FFT in the last row (top boundary condition)
    P_kNz = np.fft.fft2(p_top, axes=(0, 1))
    # Solve the system for each gamma
    for r in range(Nx - 1):
        for s in range(Ny - 1):
            # Compute gamma
            gamma_rs = - 2 - kx[r] ** 2 - ky[s] ** 2
            # Create RHS of system
            F_k[s, r, 0] = 0 + 0.5 * dz * F_k[s, r, 1] # dp/dy = 0
            # Substract coefficient of top boundary condition
            F_k[s, r, -1] -=  P_kNz[s, r] / dz ** 2 
            # Create A in the system. Only keep diagonals of matrix
            a = np.ones(Nz - 2) / dz ** 2
            b = np.ones(Nz - 1) * gamma_rs / dz ** 2
            c = np.ones(Nz - 2) / dz ** 2
            # Fix first coefficients
            c[0] = (2 + 0.5 * gamma_rs) / dz
            b[0] = -1 / dz
            # Solve system A P_k = F_k
            P_k[s, r, :] = solver(a, b, c, F_k[s, r, :])
    # numba_process_loop(kx, ky, Nx, Ny, Nz, dz, F_k, P_kNz, P_k, solver)
    # Compute IFFT in x direction (column-wise) to restore pressure
    p = np.real(np.fft.ifft2(P_k, axes=(0, 1)))
    # Add top boundary condition
    p = np.concatenate([p, np.expand_dims(p_top, axis=2)], axis=2)
    return p

def parallel_process(gamma, r, s, Nz, dz, F_k, P_kNz):
    # Create A in the system. Only keep diagonals of matrix
    a = np.ones(Nz - 2) / dz ** 2
    b = np.ones(Nz - 1) * gamma / dz ** 2
    c = np.ones(Nz - 2) / dz ** 2
    # Fix first coefficients
    c[0] = (2 + 0.5 * gamma) / dz
    b[0] = -1 / dz
    l, u = get_LU_thomas(a, b, c)
    # Create RHS of system
    F_k[s, r, 0] = 0 + 0.5 * dz * F_k[s, r, 1] # dp/dy = 0
    # Substract coefficient of top boundary condition
    F_k[s, r, -1] -=  P_kNz[s, r] / dz ** 2 
    sol = solve_triangular_systems_thomas(l, u, c, F_k[s, r, :])
    return sol, r, s

@jit(nopython=False)
def parallel_process_numba(gammas, rs, ss, Nz, dz, F_k, P_kNz, P_k):
    for i in prange(len(gammas)):
        gamma = gammas[i]
        # Create A in the system. Only keep diagonals of matrix
        a = np.ones(Nz - 2) / dz ** 2
        b = np.ones(Nz - 1) * gamma / dz ** 2
        c = np.ones(Nz - 2) / dz ** 2
        # Fix first coefficients
        c[0] = (2 + 0.5 * gamma) / dz
        b[0] = -1 / dz
        l, u = get_LU_thomas(a, b, c)
        # r = rs[i]
        # s = ss[i]
        for r, s in zip(rs[i], ss[i]):
            # Create RHS of system
            F_k[s, r, 0] = 0 + 0.5 * dz * F_k[s, r, 1] # dp/dy = 0
            # Substract coefficient of top boundary condition
            F_k[s, r, -1] -=  P_kNz[s, r] / dz ** 2 
            P_k[s, r, :] = solve_triangular_systems_thomas(l, u, c, F_k[s, r, :])

def fftfd_3D_parallel(f: np.ndarray, params: dict, solver: callable = thomas_algorithm) -> np.ndarray:
    """
    Compute the 3D Poisson equation using the FFT2D-FD method.
    FFT2D for x-direction, y-direction and central differences for z-direction.

    Parameters
    ----------
    f : array_like
        Input array of shape (Ny, Nx, Nz) containing the right-hand side of the Poisson equation.
    params : dict
        Dictionary containing the parameters of the problem:
        - Nx : int
            Number of intervals in the x direction.
        - Ny : int
            Number of intervals in the y direction.
        - Nz : int
            Number of intervals in the z direction.
        - dx : float
            Spacing between grid points in the x direction.
        - dy : float
            Spacing between grid points in the y direction.
        - dz : float
            Spacing between grid points in the z direction.
        - x : array_like
            Array of shape (Nx,) containing the x coordinates of the grid points.
        - y : array_like
            Array of shape (Ny,) containing the y coordinates of the grid points.
        - bc_on_z : list
            List containing the boundary conditions on the z axis
        - p_top : float
            Pressure value at the top boundary.
    solver : function, optional
        Function to solve the tridiagonal system. Default is thomas_algorithm.

    Returns
    -------
    ndarray (Ny, Nx)
        Solution of the Poisson equation.
    """
    Nz = params['Nz'] # Number of intervals
    dz = params['dz']# Space step
    indices = params['indices']
    gammas = indices[0]
    rr = indices[1]
    ss = indices[2]
    _, p_top = params['bc_on_z'][5] # Pressure top boundary condition
    F = f[:, :, :-1] # Remove last slice
    # Compute FFT in x direction (column-wise)
    F_k = np.fft.fft2(F, axes=(0, 1))
    # To store pressure in Fourier space
    P_k = np.zeros_like(F_k)
    # Compute FFT in the last row (top boundary condition)
    P_kNz = np.fft.fft2(p_top, axes=(0, 1))
    # Solve the system for each gamma
    # Sequential
    for i in range(len(gammas)):
        gamma = gammas[i]
        # Create A in the system. Only keep diagonals of matrix
        a = np.ones(Nz - 2) / dz ** 2
        b = np.ones(Nz - 1) * gamma / dz ** 2
        c = np.ones(Nz - 2) / dz ** 2
        # Fix first coefficients
        c[0] = (2 + 0.5 * gamma) / dz
        b[0] = -1 / dz
        l, u = get_LU_thomas(a, b, c)
        # for r, s in zip(rr[i], ss[i]):
        r = rr[i]
        s = ss[i]
        # Create RHS of system
        F_k[s, r, 0] = 0 + 0.5 * dz * F_k[s, r, 1] # dp/dy = 0
        # Substract coefficient of top boundary condition
        F_k[s, r, -1] -=  P_kNz[s, r] / dz ** 2 
        P_k[s, r, :] = solve_triangular_systems_thomas(l, u, c, F_k[s, r, :])
    # Parallel with pool
    # with Pool() as pool:
    #     results = pool.starmap(parallel_process, zip(gammas, rr, ss, itertools.repeat(Nz), itertools.repeat(dz), itertools.repeat(F_k), itertools.repeat(P_kNz)))
    # for sol, r, s in results:
    #     P_k[s, r, :] = sol
    # Parallel with numba
    # parallel_process_numba(np.array(gammas), np.array(rr, dtype=object), np.array(ss, dtype=object), Nz, dz, F_k, P_kNz, P_k)
    p = np.real(np.fft.ifft2(P_k, axes=(0, 1)))
    # Add top boundary condition
    p = np.concatenate([p, np.expand_dims(p_top, axis=2)], axis=2)
    return p

def solve_pressure_2D(u: np.ndarray, v: np.ndarray, params: dict) -> np.ndarray:
    """
    Solves the pressure Poisson equation for a given temporal velocity field.
    Used to correct the velocity field to satisfy the continuity equation.

    .. math::
        \nabla_h^2 p = rho / dt \nabla_h \cdot \mathbf{u^*}

    Parameters
    ----------
    u : np.ndarray (Ny, Nx)
        The x-component of the velocity field.
    v : np.ndarray (Ny, Nx)
        The y-component of the velocity field.
    params : dict
        A dictionary containing the simulation parameters. It must contain the following keys:
        - rho : float
            The fluid density.
        - dx : float
            The grid spacing in the x-direction.
        - dy : float
            The grid spacing in the y-direction.
        - dt : float
            The time step.

    Returns
    -------
    np.ndarray
        The pressure field.

    """
    rho = params['rho']
    dx, dy, dt = params['dx'], params['dy'], params['dt']
    # Compute ux and vy using half step to avoid odd-even decoupling
    ux = compute_first_derivative_half_step(u, dx, 1) 
    vy = compute_first_derivative_half_step(v, dy, 0, False)
    # Compute f
    f = rho / dt * (ux + vy)
    # Solve using FFT-FD
    p = fftfd(f, params)
    # p = fftfd_parallel(f, params)
    return p

def solve_pressure_3D(u: np.ndarray, v: np.ndarray, w: np.ndarray, params: dict) -> np.ndarray:
    """
    Solves the pressure Poisson equation for a given temporal velocity field.
    Used to correct the velocity field to satisfy the continuity equation.

    .. math::
        \nabla_h^2 p = rho / dt \nabla_h \cdot \mathbf{u^*}

    Parameters
    ----------
    u : np.ndarray (Ny, Nx, Nz)
        The x-component of the velocity field.
    v : np.ndarray (Ny, Nx, Nz)
        The y-component of the velocity field.
    w : np.ndarray (Ny, Nx, Nz)
        The z-component of the velocity field.
    params : dict
        A dictionary containing the simulation parameters. It must contain the following keys:
        - rho : float
            The fluid density.
        - dx : float
            The grid spacing in the x-direction.
        - dy : float
            The grid spacing in the y-direction.
        - dz : float
            The grid spacing in the z-direction.
        - dt : float
            The time step.

    Returns
    -------
    np.ndarray
        The pressure field.

    """
    rho = params['rho']
    dx, dy, dz, dt = params['dx'], params['dy'], params['dz'], params['dt']
    # Compute ux, vy and wz using half step to avoid odd-even decoupling
    ux = compute_first_derivative_half_step(u, dx, 1) 
    vy = compute_first_derivative_half_step(v, dy, 0)
    wz = compute_first_derivative_half_step(w, dz, 2, periodic=False)
    # Compute f
    f = rho / dt * (ux + vy + wz)
    # Solve using FFT-FD
    # p = fftfd_3D(f, params)
    p = fftfd_3D_parallel(f, params)
    return p

def solve_pressure(U, params):
    ndims = len(U)
    if ndims == 2:
        u, v = U
        p = solve_pressure_2D(u, v, params)
    elif ndims == 3:
        u, v, w = U
        p = solve_pressure_3D(u, v, w, params)
    return p

def pre_computation(Nx: int, Ny: int, dz: float, x_max: float, y_max: float):
    r = np.fft.fftfreq(Nx - 1) * (Nx - 1)
    s = np.fft.fftfreq(Ny - 1) * (Ny - 1)
    # Scale frequencies
    kx = 2 * np.pi * r * dz / x_max
    ky = 2 * np.pi * s * dz / y_max
    Kx, Ky = np.meshgrid(kx, ky)
    G = - 2 - Kx ** 2 - Ky ** 2
    diff_G = np.unique(G)
    # indices = {}
    # indices = []
    gammas = []
    rs = []
    ss = []
    for gamma in diff_G:
        r, s = np.where(G == gamma)
        # if gamma not in indices:
        # indices[gamma] = {
        #     'r': r,
        #     's': s                                                                                                                                                  ,
        # }
        # indices.append([gamma, r, s])
        gammas.append(gamma)
        rs.append(r)
        ss.append(s)
    return (gammas, rs, ss)
