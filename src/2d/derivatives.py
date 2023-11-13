import numpy as np

### Derivatives for PDE solver ###
def compute_first_derivative(phi: np.ndarray, h: float, axis: int, periodic: bool = True, type: str = 'central') -> np.ndarray:
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
    periodic : bool
        True if the domain is periodic, False otherwise.
    type : str
        Type of difference scheme. 'forward' for forward difference, 'backward'
        for backward difference and 'central' for central difference. Default is 'central'.

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
    # Get nodes
    phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    # Compute derivative
    if type == 'forward':
        phi_h = (phi_ip1 - phi) / h # Forward difference
        if periodic == False:
            if axis == 0: # Fix boundary in y - O(dy)
                phi_h[-1,:] = (phi[-1,:] - phi[-2,:]) / h # Backward
            elif axis == 1: # Fix boundary in x - O(dx)
                phi_h[:,-1] = (phi[:,-1] - phi[:,-2]) / h
    elif type == 'backward':
        phi_h = (phi - phi_im1) / h
        if periodic == False:
            if axis == 0: # Fix boundary in y - O(dy)
                phi_h[0, :] = (phi[1, :] - phi[0, :]) / h
            elif axis == 1: # Fix boundary in x - O(dx)
                phi_h[:, 0] = (phi[:, 1] - phi[:, 0]) / h
    elif type == 'central':
        phi_h = (phi_ip1 - phi_im1) / (2 * h) # Central difference
        if periodic == False:
            if axis == 0: # Fix boundary in y - O(dy^2)
                phi_h[0, :] = (-3 * phi[0, :] + 4 * phi[1, :] - phi[2, :]) / (2 * h) # Forward
                phi_h[-1,:] = (3 * phi[-1, :] - 4 * phi[-2,:] + phi[-3,:]) / (2 * h) # Backward
            elif axis == 1: # Fix boundary in x - O(dx^2)
                phi_h[:, 0] = (-3 * phi[:, 0] + 4 * phi[:, 1] - phi[:, 2]) / (2 * h)
                phi_h[:,-1] = (3 * phi[:,-1] - 4 * phi[:,-2] + phi[:,-3]) / (2 * h)
    return phi_h

def compute_first_derivative_upwind(a: np.ndarray, phi: np.ndarray, h: float, axis: int, order: int = 2, periodic: bool = True) -> np.ndarray:
    """
    Compute the first derivative of a 2D scalar field along a given axis.

    .. math::
        \frac{\partial \phi}{\partial h} = \frac{\phi_{i+1} - \phi_{i}}{h}

    Parameters
    ----------
    a: numpy.ndarray (Ny, Nx)
        2D scalar field.
    phi : numpy.ndarray (Ny, Nx)
        2D scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for y, 1 for x.
    order : int
        Order of the difference scheme. 1 for first-order, 2 for second-order.
    periodic : bool
        True if the domain is periodic, False otherwise. Default is True.

    Returns
    -------
    numpy.ndarray
        First derivative of `phi` along `axis`.

    Notes
    -----
    This function uses a first-order forward difference scheme to compute the
    derivative in the general case. For the boundary points along the `axis=0`
    or y direction, a first-order forward/backward difference scheme is used.

    """
    # Mask for upwind scheme
    a_plu = np.maximum(a, 0)
    a_min = np.minimum(a, 0)
    # Get nodes
    phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    if order == 1: # First order
        phi_hm = compute_first_derivative(phi, h, axis, periodic=periodic, type='backward') # Backward
        phi_hp = compute_first_derivative(phi, h, axis, periodic=periodic, type='forward') # Forward
    elif order == 2: # Second order
        phi_ip2 = np.roll(phi,-2, axis=axis) # phi_{i+2}
        phi_im2 = np.roll(phi, 2, axis=axis) # phi_{i-2}
        phi_hm = (3 * phi - 4 * phi_im1 + phi_im2) / (2 * h) # Backward
        phi_hp = (-phi_ip2 + 4 * phi_ip1 - 3 * phi) / (2 * h) # Forward
        if periodic == False: 
            if axis == 0:# Fix boundary in y - O(dy^2)
                phi_hm[0, :] = (-3 * phi[0, :] + 4 * phi[1, :] - phi[2, :]) / (2 * h) # Forward
                phi_hm[1, :] = (-3 * phi[1, :] + 4 * phi[2, :] - phi[3, :]) / (2 * h) # Forward
                phi_hp[-1,:] = (phi[-1, :] - 4 * phi[-2,:] + 3 * phi[-3,:]) / (2 * h) # Backward
                phi_hp[-2,:] = (phi[-2, :] - 4 * phi[-3,:] + 3 * phi[-4,:]) / (2 * h) # Backward
            elif axis == 1: # Fix boundary in x - O(dx^2)
                phi_hm[:, 0] = (-3 * phi[:, 0] + 4 * phi[:, 1] - phi[:, 2]) / (2 * h)
                phi_hm[:, 1] = (-3 * phi[:, 1] + 4 * phi[:, 2] - phi[:, 3]) / (2 * h)
                phi_hp[:,-1] = (phi[:,-1] - 4 * phi[:,-2] + 3 * phi[:,-3]) / (2 * h)
                phi_hp[:,-2] = (phi[:,-2] - 4 * phi[:,-3] + 3 * phi[:,-4]) / (2 * h)
    # Upwind scheme
    phi_h = a_plu * phi_hm + a_min * phi_hp
    return phi_h

def compute_first_derivative_half_step(phi: np.ndarray, h: float, axis: int, periodic: bool = True) -> np.ndarray:
    """
    Computes the derivative of a 2D array `phi` along a given `axis` using a central difference scheme.
    The derivative is computed at half-integer positions using the values of `phi` at integer positions.
    
    .. math::
        \frac{\partial \phi}{\partial h} = \frac{\phi_{i+1/2} - \phi_{i-1/2}}{h}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx)
        The 2D array to compute the derivative of.
    h : float
        The grid spacing.
    axis : int
        The axis along which to compute the derivative.
    periodic : bool
        True if the domain is periodic, False otherwise. Default is True.
    
    Returns
    -------
    numpy.ndarray
        The derivative of `phi` along the specified `axis` at half-integer positions.
    """
    phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    phi_iphj = 0.5 * (phi_ip1 + phi) # phi_{i+1/2}
    phi_imhj = 0.5 * (phi_im1 + phi) # phi_{i-1/2}
    phi_h = (phi_iphj - phi_imhj) / h # Central difference
    if periodic == False: 
        if axis == 0:# Fix boundary in y - O(dy^2)
            phi_h[0] = (-phi[2] + 4 * phi[1] - 3 * phi[0]) / (2 * h)
            phi_h[-1] = (3 * phi[-1] - 4 * phi[-2] + phi[-3]) / (2 * h)
        if axis == 1:
            phi_h[:, 0] = (-phi[:, 2] + 4 * phi[:, 1] - 3 * phi[:, 0]) / (2 * h)
            phi_h[:,-1] = (3 * phi[:,-1] - 4 * phi[:,-2] + phi[:,-3]) / (2 * h)
    return phi_h

def compute_second_derivative(phi: np.ndarray, h: float, axis: int, periodic: bool = True) -> np.ndarray:
    """
    Compute the second derivative of a 2D scalar field along a given axis.

    .. math::
        \frac{\partial^2 \phi}{\partial h^2} = \frac{\phi_{i+1} - 2 \phi_i + \phi_{i-1}}{h^2}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx)
        2D scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for y, 1 for x.
    periodic : bool
        True if the domain is periodic, False otherwise. Default is True.

    Returns
    -------
    numpy.ndarray
        Second derivative of `phi` along `axis`.

    Notes
    -----
    This function uses a second-order central difference scheme to compute the
    derivative in the general case. For the boundary points along the `axis=0`
    or y direction, a second-order forward/backward difference scheme is used.

    """
    # Get nodes
    phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    # Second derivative
    phi_hh = (phi_ip1 - 2 * phi + phi_im1) / h ** 2
    if periodic == False: 
        if axis == 0: # Fix boundary in y - O(dy^2)
            phi_hh[0, :] = (2 * phi[0, :] - 5 * phi[1, :] + 4 * phi[2, :] - phi[3, :]) / h ** 2 # Forward
            phi_hh[-1,:] = (2 * phi[-1, :] - 5 * phi[-2,:] + 4 * phi[-3,:] - phi[-4,:]) / h ** 2 # Backward
        elif axis == 1:
            phi_hh[:, 0] = (2 * phi[:, 0] - 5 * phi[:, 1] + 4 * phi[:, 2] - phi[:, 3]) / h ** 2
            phi_hh[:,-1] = (2 * phi[:,-1] - 5 * phi[:,-2] + 4 * phi[:,-3] - phi[:,-4]) / h ** 2
    return phi_hh

def compute_gradient(phi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
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
    periodic_axes : tuple
        Axes along which the domain is periodic. True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Gradient of `phi`.
    """
    dphi_x = compute_first_derivative(phi, dx, axis=1, periodic=periodic_axes[1]) # dphi/dx
    dphi_y = compute_first_derivative(phi, dy, axis=0, periodic=periodic_axes[0]) # dphi/dy
    gradient = np.array([dphi_x, dphi_y])
    return gradient

def compute_laplacian(phi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the Laplacian of a 2D scalar field.

    .. math::
        \nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx)
        Scalar field to compute the derivatives of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    periodic_axes : tuple
        Axes along which the domain is periodic. True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Laplacian of `phi`.
    """
    phi_xx = compute_second_derivative(phi, dx, axis=1, periodic=periodic_axes[1]) # d^2phi/dx^2
    phi_yy = compute_second_derivative(phi, dy, axis=0, periodic=periodic_axes[0]) # d^2phi/dy^2
    laplacian = phi_xx + phi_yy
    return laplacian

def compute_curl(vphi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
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
    periodic_axes : tuple
        Axes along which the domain is periodic. True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (Ny, Nx)
        Curl of `vphi`.
    """
    vphix, vphiy = vphi
    dphiy_x = compute_first_derivative(vphiy, dx, axis=1, periodic=periodic_axes[1]) # dphiy/dx
    dphix_y = compute_first_derivative(vphix, dy, axis=0, periodic=periodic_axes[0]) # dphix/dy
    curl = dphiy_x - dphix_y
    return curl

def compute_divergence(vphi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
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
    periodic_axes : tuple
        Axes along which the domain is periodic. True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (Ny, Nx)
        Divergence of `vphi`.
    """
    vphix, vphiy = vphi
    dphix_x = compute_first_derivative(vphix, dx, axis=1, periodic=periodic_axes[1]) # dphix/dx
    dphiy_y = compute_first_derivative(vphiy, dy, axis=0, periodic=periodic_axes[0]) # dphiy/dy
    divergence = dphix_x + dphiy_y
    return divergence

### Derivatives for plotting ###
# The difference is the number of axes. For plotting we have an extra axis for time!
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