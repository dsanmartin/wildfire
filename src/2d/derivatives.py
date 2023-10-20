import numpy as np

### Derivatives for PDE solver ###
def compute_first_derivative(phi: np.ndarray, h: float, axis: int) -> np.ndarray:
    """
    Compute the first derivative of a 2D scalar field along a given axis.

    .. math::
        \frac{\partial \phi}{\partial h} = \frac{\phi_{i+1} - \phi_{i-1}}{2 h}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx)
        2D scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for y, 1 for x.

    Returns
    -------
    numpy.ndarray
        First derivative of `phi` along `axis`.

    Notes
    -----
    This function uses a second-order central difference scheme to compute the
    derivative in the general case. For the boundary points along the `axis=0`
    or y direction, a second-order forward/backward difference scheme is used.

    """
    # General case 
    dphi_h = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2 * h)
    if axis == 0: # Fix boundary in y - O(dy^2)
        dphi_h[0, :] = (-3 * phi[0, :] + 4 * phi[1, :] - phi[2, :]) / (2 * h) # Forward
        dphi_h[-1,:] = (3 * phi[-1, :] - 4 * phi[-2,:] + phi[-3,:]) / (2 * h) # Backward
    return dphi_h

def compute_gradient(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the gradient of a 2D scalar field.

    .. math::
        \nabla \phi = \left( \frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y} \right)

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx)
        Scalar field to compute the derivatives of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.

    Returns
    -------
    numpy.ndarray
        Gradient of `phi`.
    """
    dphi_x = compute_first_derivative(phi, dx, axis=1) # dphi/dx
    dphi_y = compute_first_derivative(phi, dy, axis=0) # dphi/dy
    gradient = np.array([dphi_x, dphi_y])
    return gradient

def compute_curl(vphi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the curl of a 2D vector field.

    .. math::
        \nabla \times \phi = \frac{\partial \mathbf{phi}_y}{\partial x} - \frac{\partial \mathbf{phi}_x}{\partial y}

    Parameters
    ----------
    vphi : numpy.ndarray (2, Ny, Nx)
        Vector field to compute the curl of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.

    Returns
    -------
    numpy.ndarray (Ny, Nx)
        Curl of `vphi`.
    """
    vphix = vphi[0]
    vphiy = vphi[1]
    dphiy_x = compute_first_derivative(vphiy, dx, axis=1) # dphiy/dx
    dphix_y = compute_first_derivative(vphix, dy, axis=0) # dphix/dy
    curl = dphiy_x - dphix_y
    return curl

def compute_divergence(vphi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the divergence of a 2D vector field.

    .. math::
        \nabla \cdot \phi = \frac{\partial \mathbf{phi}_x}{\partial x} + \frac{\partial \mathbf{phi}_y}{\partial y}

    Parameters
    ----------
    vphi : numpy.ndarray (2, Ny, Nx)
        Vector field to compute the divergence of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.

    Returns
    -------
    numpy.ndarray (Ny, Nx)
        Divergence of `vphi`.
    """
    vphix = vphi[0]
    vphiy = vphi[1]
    dphix_x = compute_first_derivative(vphix, dx, axis=1) # dphix/dx
    dphiy_y = compute_first_derivative(vphiy, dy, axis=0) # dphiy/dy
    divergence = dphix_x + dphiy_y
    return divergence

### Derivatives for plotting ###
# The difference es the number of axis. For plotting we have an extra axis for time!
def compute_first_derivative_plots(phi: np.ndarray, h: float, axis: int) -> np.ndarray:
    """
    Compute the first derivative of a 2D scalar field along a given axis. This is the same
    as compute_first_derivative, but it is used for plotting purposes.

    Parameters
    ----------
    phi : numpy.ndarray (Nt, Ny, Nx)
        2D scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for t, 1 for y, and 2 for x.

    Returns
    -------
    numpy.ndarray
        First derivative of `phi` along `axis`.
    """
    # General case 
    dphi_h = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2 * h)
    if axis == 1: # Fix boundary in y - O(dy^2)
        dphi_h[:, 0, :] = (-3 * phi[:, 0, :] + 4 * phi[:, 1, :] - phi[:, 2, :]) / (2 * h) # Forward
        dphi_h[:, -1,:] = (3 * phi[:, -1, :] - 4 * phi[:, -2,:] + phi[:, -3,:]) / (2 * h) # Backward
    return dphi_h

def compute_gradient_plots(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the gradient of a 2D scalar field. This is the same as compute_gradient,
    but it is used for plotting purposes.

    Parameters
    ----------
    phi : numpy.ndarray (Nt, Ny, Nx)
        Scalar field to compute the derivatives of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.

    Returns
    -------
    numpy.ndarray (2, Nt, Ny, Nx)
        Gradient of `phi`.
    """
    dphi_x = compute_first_derivative_plots(phi, dx, axis=2) # dphi/dx
    dphi_y = compute_first_derivative_plots(phi, dy, axis=1) # dphi/dy
    gradient = np.array([dphi_x, dphi_y])
    return gradient

def compute_curl_plots(vphi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the curl of a 2D vector field. This is the same as compute_curl,
    but it is used for plotting purposes.

    Parameters
    ----------
    vphi : numpy.ndarray (2, Nt, Ny, Nx)
        Vector field to compute the curl of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.

    Returns
    -------
    numpy.ndarray (Nt, Ny, Nx)
        Curl of `vphi`.
    """
    vphix = vphi[0]
    vphiy = vphi[1]
    dphiy_x = compute_first_derivative_plots(vphiy, dx, axis=2) # dphiy/dx
    dphix_y = compute_first_derivative_plots(vphix, dy, axis=1) # dphix/dy
    curl = dphiy_x - dphix_y
    return curl

def compute_divergence_plots(vphi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute the divergence of a 2D vector field. This is the same as compute_divergence,
    but it is used for plotting purposes.

    Parameters
    ----------
    vphi : numpy.ndarray (2, Nt, Ny, Nx)
        Vector field to compute the divergence of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.

    Returns
    -------
    numpy.ndarray (Nt, Ny, Nx)
        Divergence of `vphi`.
    """
    vphix = vphi[0]
    vphiy = vphi[1]
    dphix_x = compute_first_derivative_plots(vphix, dx, axis=2) # dphix/dx
    dphiy_y = compute_first_derivative_plots(vphiy, dy, axis=1) # dphiy/dy
    divergence = dphix_x + dphiy_y
    return divergence