import numpy as np
from numba import jit
from derivatives import compute_first_derivative_half_step

@jit
def thomas_algorithm(A: tuple[np.ndarray], f: np.ndarray) -> np.ndarray:
    """
    Solve a tridiagonal linear system Ax = f using the Thomas algorithm.
    Numba is used to speed up the computation.

    Parameters
    ----------
    A : tuple of np.ndarrays
        Tuple containing the three diagonals of the tridiagonal matrix A.
        The first ndarray contains the lower diagonal, the second ndarray
        contains the main diagonal, and the third ndarray contains the upper
        diagonal.
    f : np.ndarray (Ny, Nx)
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
    a, b, c = A # Get diagonals
    N = b.shape[0]
    v = np.zeros(N, dtype=np.complex128)
    l = np.zeros(N - 1, dtype=np.complex128)
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

def fftfd(f: np.ndarray, params: dict, solver=thomas_algorithm) -> np.ndarray:
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
    _, p_top = params['bc_on_y'][4] # Pressure top boundary condition
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
        A = (a, b, c) # Create tuple of diagonals
        P_k[:, k] = solver(A, F_k[:, k])
    # Compute IFFT in x direction (column-wise) to restore pressure
    p = np.real(np.fft.ifft(P_k, axis=1))
    # Add top boundary condition
    p = np.vstack([p, np.ones(Nx - 1) * p_top])
    return p

def solve_pressure(u: np.ndarray, v: np.ndarray, params: dict) -> np.ndarray:
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
    vy = compute_first_derivative_half_step(v, dy, 0)
    # Compute f
    f = rho / dt * (ux + vy)
    # Solve using FFT-FD
    p = fftfd(f, params)
    return p
