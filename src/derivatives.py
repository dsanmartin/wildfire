import numpy as np
from numba import jit

# @jit(nopython=True)
def get_nodes(phi: np.ndarray, n: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    # phi_ipn = np.zeros_like(phi)
    # phi_imn = np.zeros_like(phi)
    # if axis == 0:
    #     phi_ipn[:-1] = phi[1:]
    #     phi_ipn[-1] = phi[0]
    #     phi_imn[1:] = phi[:-1]
    #     phi_imn[0] = phi[-1]
    #     if n == 2:
    #         phi_ipn[:-2] = phi[2:]
    #         phi_ipn[-2:] = phi[:2]
    #         phi_imn[2:] = phi[:-2]
    #         phi_imn[:2] = phi[-2:]
    # elif axis == 1:
    #     phi_ipn[:,:-1] = phi[:,1:]
    #     phi_ipn[:,-1] = phi[:,0]
    #     phi_imn[:,1:] = phi[:,:-1]
    #     phi_imn[:,0] = phi[:,-1]
    #     if n == 2:
    #         phi_ipn[:,:-2] = phi[:,2:]
    #         phi_ipn[:,-2:] = phi[:,:2]
    #         phi_imn[:,2:] = phi[:,:-2]
    #         phi_imn[:,:2] = phi[:,-2:]
    # elif axis == 2:
    #     phi_ipn[:,:,:-1] = phi[:,:,1:]
    #     phi_ipn[:,:,-1] = phi[:,:,0]
    #     phi_imn[:,:,1:] = phi[:,:,:-1]
    #     phi_imn[:,:,0] = phi[:,:,-1]
    #     if n == 2:
    #         phi_ipn[:,:,:-2] = phi[:,:,2:]
    #         phi_ipn[:,:,-2:] = phi[:,:,:2]
    #         phi_imn[:,:,2:] = phi[:,:,:-2]
    #         phi_imn[:,:,:2] = phi[:,:,-2:]
    phi_ipn = np.roll(phi,-n, axis=axis)
    phi_imn = np.roll(phi, n, axis=axis)
    return phi_imn, phi_ipn

### Derivatives for PDE solver ###
# @jit(nopython=True)
def compute_first_derivative(phi: np.ndarray, h: float, axis: int, periodic: bool = True, type: str = 'central') -> np.ndarray:
    """
    Compute the first derivative of a scalar field along a given axis.

    .. math::
        \frac{\partial \phi}{\partial h} = \frac{\phi_{i+1} - \phi_{i-1}}{2 h}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for y, 1 for x, and 2 for z.
    periodic : bool
        True if the domain is periodic, False otherwise.
    type : str
        Type of difference scheme. 'forward' for forward difference, 'backward'
        for backward difference and 'central' for central difference. Default is 'central'.

    Returns
    -------
    numpy.ndarray
        First derivative of `phi` along `axis`.
    """
    # Get nodes
    # if roll:
    # phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    # phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    # else: # Using numpy slicing
    # if axis == 0:
    #     phi_ip1 = np.concatenate((phi[1:], phi[0]), axis=0)
    #     phi_im1 = np.concatenate((phi[-1], phi[:-1]), axis=0)
    # elif axis == 1:
    #     phi_ip1 = np.concatenate((phi[:,1:], phi[:,0]), axis=1)
    #     phi_im1 = np.concatenate((phi[:,-1], phi[:,:-1]), axis=1)
    # elif axis == 2:
    #     phi_ip1 = np.concatenate((phi[:,:,1:], phi[:,:,0]), axis=2)
    #     phi_im1 = np.concatenate((phi[:,:,-1], phi[:,:,:-1]), axis=2)
    phi_im1, phi_ip1 = get_nodes(phi, 1, axis)
    # Compute derivative
    if type == 'forward':
        phi_h = (phi_ip1 - phi) / h # Forward difference
        if periodic == False: # Fix boundary using backward difference in the last node - O(h)
            if axis == 0: # Fix boundary in y
                phi_h[-1,:] = (phi[-1,:] - phi[-2,:]) / h
            elif axis == 1: # Fix boundary in x
                phi_h[:,-1] = (phi[:,-1] - phi[:,-2]) / h
            elif axis == 2: # Fix boundary in z
                phi_h[:,:,-1] = (phi[:,:,-1] - phi[:,:,-2]) / h
    elif type == 'backward':
        phi_h = (phi - phi_im1) / h # Backward difference
        if periodic == False: # Fix boundary using forward difference in the first node - O(h)
            if axis == 0: # Fix boundary in y
                phi_h[0,:] = (phi[1, :] - phi[0, :]) / h
            elif axis == 1: # Fix boundary in x 
                phi_h[:,0] = (phi[:, 1] - phi[:, 0]) / h
            elif axis == 2:
                phi_h[:,:,0] = (phi[:, :, 1] - phi[:, :, 0]) / h
    elif type == 'central':
        phi_h = (phi_ip1 - phi_im1) / (2 * h) # Central difference
        if periodic == False:
            if axis == 0: # Fix boundary in y - O(dy^2)
                phi_h[0, :] = (-3 * phi[0, :] + 4 * phi[1, :] - phi[2, :]) / (2 * h) # Forward
                phi_h[-1,:] = (3 * phi[-1, :] - 4 * phi[-2,:] + phi[-3,:]) / (2 * h) # Backward
            elif axis == 1: # Fix boundary in x - O(dx^2)
                phi_h[:, 0] = (-3 * phi[:, 0] + 4 * phi[:, 1] - phi[:, 2]) / (2 * h)
                phi_h[:,-1] = (3 * phi[:,-1] - 4 * phi[:,-2] + phi[:,-3]) / (2 * h)
            elif axis == 2:
                phi_h[:,:,0] = (-3 * phi[:, :, 0] + 4 * phi[:, :, 1] - phi[:, :, 2]) / (2 * h)
                phi_h[:,:,-1] = (3 * phi[:,:,-1] - 4 * phi[:,:,-2] + phi[:,:,-3]) / (2 * h)
    return phi_h

# @jit(nopython=True)
def compute_first_derivative_upwind(a: np.ndarray, phi: np.ndarray, h: float, axis: int, order: int = 2, periodic: bool = True) -> np.ndarray:
    """
    Compute the first derivative of a scalar field along a given axis.

    .. math::
        \frac{\partial \phi}{\partial h} = \frac{\phi_{i+1} - \phi_{i}}{h}

    Parameters
    ----------
    a: numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Scalar field.
    phi : numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for y, 1 for x, and 2 for z.
    order : int
        Order of the difference scheme. 1 for first-order, 2 for second-order.
    periodic : bool
        True if the domain is periodic, False otherwise. Default is True.

    Returns
    -------
    numpy.ndarray
        First derivative of `phi` along `axis`.
    """
    # Mask for upwind scheme
    a_plu = np.maximum(a, 0)
    a_min = np.minimum(a, 0)
    # Get nodes
    # phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    # phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    phi_im1, phi_ip1 = get_nodes(phi, 1, axis)
    if order == 1: # First order
        phi_hm = compute_first_derivative(phi, h, axis, periodic=periodic, type='backward') # Backward
        phi_hp = compute_first_derivative(phi, h, axis, periodic=periodic, type='forward') # Forward
    elif order == 2: # Second order
        # phi_ip2 = np.roll(phi,-2, axis=axis) # phi_{i+2}
        # phi_im2 = np.roll(phi, 2, axis=axis) # phi_{i-2}
        phi_im2, phi_ip2 = get_nodes(phi, 2, axis)
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
            elif axis == 2:
                phi_hm[:,:,0] = (-3 * phi[:, :, 0] + 4 * phi[:, :, 1] - phi[:, :, 2]) / (2 * h)
                phi_hm[:,:,1] = (-3 * phi[:, :, 1] + 4 * phi[:, :, 2] - phi[:, :, 3]) / (2 * h)
                phi_hp[:,:,-1] = (phi[:,:,-1] - 4 * phi[:,:,-2] + 3 * phi[:,:,-3]) / (2 * h)
                phi_hp[:,:,-2] = (phi[:,:,-2] - 4 * phi[:,:,-3] + 3 * phi[:,:,-4]) / (2 * h)
    # Upwind scheme
    phi_h = a_plu * phi_hm + a_min * phi_hp
    return phi_h

# @jit(nopython=True)
def compute_first_derivative_half_step(phi: np.ndarray, h: float, axis: int, periodic: bool = True) -> np.ndarray:
    """
    Computes the derivative of a scalar field `phi` along a given `axis` using a central difference scheme.
    The derivative is computed at half-integer positions using the values of `phi` at integer positions.
    
    .. math::
        \frac{\partial \phi}{\partial h} = \frac{\phi_{i+1/2} - \phi_{i-1/2}}{h}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Scalar field.
    h : float
        The grid spacing.
    axis : int
        The axis along which to compute the derivative. 0 for y, 1 for x, and 2 for z.
    periodic : bool
        True if the domain is periodic, False otherwise. Default is True.
    
    Returns
    -------
    numpy.ndarray
        The derivative of `phi` along the specified `axis` at half-integer positions.
    """
    # phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    # phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    phi_im1, phi_ip1 = get_nodes(phi, 1, axis)
    phi_iphj = 0.5 * (phi_ip1 + phi) # phi_{i+1/2}
    phi_imhj = 0.5 * (phi_im1 + phi) # phi_{i-1/2}
    phi_h = (phi_iphj - phi_imhj) / h # Central difference
    if periodic == False: 
        if axis == 0:# Fix boundary in y - O(dy^2)
            phi_h[0 ,:] = (-phi[2, :] + 4 * phi[1, :] - 3 * phi[0, :]) / (2 * h)
            phi_h[-1,:] = (3 * phi[-1, :] - 4 * phi[-2, :] + phi[-3, :]) / (2 * h)
        elif axis == 1:
            phi_h[:, 0] = (-phi[:, 2] + 4 * phi[:, 1] - 3 * phi[:, 0]) / (2 * h)
            phi_h[:,-1] = (3 * phi[:,-1] - 4 * phi[:,-2] + phi[:,-3]) / (2 * h)
        elif axis == 2:
            phi_h[:,:, 0] = (-phi[:, :, 2] + 4 * phi[:, :, 1] - 3 * phi[:, :, 0]) / (2 * h)
            phi_h[:,:,-1] = (3 * phi[:,:,-1] - 4 * phi[:,:,-2] + phi[:,:,-3]) / (2 * h)
    return phi_h

# @jit(nopython=True)
def compute_second_derivative(phi: np.ndarray, h: float, axis: int, periodic: bool = True) -> np.ndarray:
    """
    Compute the second derivative of a scalar field along a given axis.

    .. math::
        \frac{\partial^2 \phi}{\partial h^2} = \frac{\phi_{i+1} - 2 \phi_i + \phi_{i-1}}{h^2}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        A scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for y, 1 for x, and 2 for z.
    periodic : bool
        True if the domain is periodic, False otherwise. Default is True.

    Returns
    -------
    numpy.ndarray
        Second derivative of `phi` along `axis`.
    """
    # Get nodes
    # phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    # phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    phi_im1, phi_ip1 = get_nodes(phi, 1, axis)
    # Second derivative
    phi_hh = (phi_ip1 - 2 * phi + phi_im1) / h ** 2
    if periodic == False: # Fix boundary using second-order forward/backward difference - O(h^2)
        if axis == 0: # Fix boundary in y
            phi_hh[0, :] = (2 * phi[0, :] - 5 * phi[1, :] + 4 * phi[2, :] - phi[3, :]) / h ** 2 # Forward
            phi_hh[-1,:] = (2 * phi[-1,:] - 5 * phi[-2,:] + 4 * phi[-3,:] - phi[-4,:]) / h ** 2 # Backward
        elif axis == 1: # Fix boundary in x
            phi_hh[:, 0] = (2 * phi[:, 0] - 5 * phi[:, 1] + 4 * phi[:, 2] - phi[:, 3]) / h ** 2
            phi_hh[:,-1] = (2 * phi[:,-1] - 5 * phi[:,-2] + 4 * phi[:,-3] - phi[:,-4]) / h ** 2
        elif axis == 2: # Fix boundary in z
            phi_hh[:,:, 0] = (2 * phi[:, :,0] - 5 * phi[:, :, 1] + 4 * phi[:, :, 2] - phi[:, :, 3]) / h ** 2
            phi_hh[:,:,-1] = (2 * phi[:,:,-1] - 5 * phi[:,:,-2] + 4 * phi[:,:,-3] - phi[:,:,-4]) / h ** 2
    return phi_hh

# @jit(nopython=True)
def compute_gradient_2D(phi: np.ndarray, dx: float, dy: float, periodic_axes: tuple[bool, bool]) -> np.ndarray:
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
    periodic_axes : tuple (bool, bool)
        Axes along which the domain is periodic, (x-direction, y-direction). 
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Gradient of `phi`.
    """
    # Compute derivatives
    dphi_x = compute_first_derivative(phi, dx, axis=1, periodic=periodic_axes[0]) # dphi/dx
    dphi_y = compute_first_derivative(phi, dy, axis=0, periodic=periodic_axes[1]) # dphi/dy
    gradient = np.array([dphi_x, dphi_y]) # 2D in space case
    return gradient

# @jit(nopython=True)
def compute_gradient_3D(phi: np.ndarray, dx: float, dy: float, dz: float, periodic_axes: tuple[bool, bool, bool]) -> tuple:
    """
    Compute the gradient of a 3D scalar field.

    .. math::
        \nabla \phi = \left( \frac{\partial \phi}{\partial x}, \frac{\partial \phi}{\partial y}, \frac{\partial \phi}{\partial z} \right)

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx, Nz)
        Scalar field to compute the derivatives of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dz : float
        Spacing between grid points in the z direction.
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction, z-direction). 
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Gradient of `phi`.
    """
    # Compute derivatives
    dphi_x = compute_first_derivative(phi, dx, axis=1, periodic=periodic_axes[0]) # dphi/dx
    dphi_y = compute_first_derivative(phi, dy, axis=0, periodic=periodic_axes[1]) # dphi/dy
    dphi_z = compute_first_derivative(phi, dz, axis=2, periodic=periodic_axes[2]) # dphi/dz
    #gradient = np.array([dphi_x, dphi_y, dphi_z]) # 3D in space case
    # return gradient
    return dphi_x, dphi_y, dphi_z

# @jit(nopython=True)
def compute_gradient(phi: np.ndarray, hs: tuple, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the gradient of a 2D or 3D scalar field.

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Scalar field to compute the derivatives of.
    hs : tuple
        Spacing between grid points in the x, y and/or z directions, (dx, dy) or (dx, dy, dz).
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction) or (x-direction, y-direction, z-direction). 
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Gradient of `phi`.
    """
    # Number of dims
    ndims = phi.ndim
    # Compute derivatives
    if ndims == 2:
        return compute_gradient_2D(phi, hs[0], hs[1], periodic_axes)
    elif ndims == 3:
        return compute_gradient_3D(phi, hs[0], hs[1], hs[2], periodic_axes)

# @jit(nopython=True)
def compute_laplacian_2D(phi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the Laplacian of a scalar field.

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
        Axes along which the domain is periodic, (x-direction, y-direction). 
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Laplacian of `phi`.
    """
    # Compute derivatives
    phi_xx = compute_second_derivative(phi, dx, axis=1, periodic=periodic_axes[0]) # d^2phi/dx^2
    phi_yy = compute_second_derivative(phi, dy, axis=0, periodic=periodic_axes[1]) # d^2phi/dy^2
    laplacian = phi_xx + phi_yy
    return laplacian

# @jit(nopython=True)
def compute_laplacian_3D(phi: np.ndarray, dx: float, dy: float, dz: float, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the Laplacian of a 3D scalar field.

    .. math::
        \nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2} + \frac{\partial^2 \phi}{\partial y^2} + \frac{\partial^2 \phi}{\partial z^2}

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx, Nz)
        Scalar field to compute the derivatives of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dz : float
        Spacing between grid points in the z direction.
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction, z-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Laplacian of `phi`.
    """
    # Compute derivatives
    phi_xx = compute_second_derivative(phi, dx, axis=1, periodic=periodic_axes[0]) # d^2phi/dx^2
    phi_yy = compute_second_derivative(phi, dy, axis=0, periodic=periodic_axes[1]) # d^2phi/dy^2
    phi_zz = compute_second_derivative(phi, dz, axis=2, periodic=periodic_axes[2]) # d^2phi/dz^2
    laplacian = phi_xx + phi_yy + phi_zz
    return laplacian

# @jit(nopython=True)
def compute_laplacian(phi: np.ndarray, hs: tuple, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the Laplacian of a 2D or 3D scalar field.

    Parameters
    ----------
    phi : numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Scalar field to compute the derivatives of.
    hs : tuple
        Spacing between grid points in the x, y and/or z directions. (dx, dy) or (dx, dy, dz).
    periodic_axes : tuple
        Axes along which the domain is periodic. (x-direction, y-direction) or (x-direction, y-direction, z-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray
        Laplacian of `phi`.
    """
    # Number of dims
    ndims = phi.ndim
    # Compute derivatives
    if ndims == 2:
        return compute_laplacian_2D(phi, hs[0], hs[1], periodic_axes)
    elif ndims == 3:
        return compute_laplacian_3D(phi, hs[0], hs[1], hs[2], periodic_axes)

def compute_curl_2D(vphi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
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
        Axes along which the domain is periodic, (x-direction, y-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (Ny, Nx)
        Curl of `vphi`.
    """
    vphix, vphiy = vphi
    dphiy_x = compute_first_derivative(vphiy, dx, axis=1, periodic=periodic_axes[0]) # dphiy/dx
    dphix_y = compute_first_derivative(vphix, dy, axis=0, periodic=periodic_axes[1]) # dphix/dy
    curl = dphiy_x - dphix_y
    return curl

def compute_curl_3D(vphi: np.ndarray, dx: float, dy: float, dz: float, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the curl of a 3D vector field.

    .. math::
        \nabla \times \phi = \left( \frac{\partial \mathbf{phi}_z}{\partial y} - \frac{\partial \mathbf{phi}_y}{\partial z}, 
            \frac{\partial \mathbf{phi}_x}{\partial z} - \frac{\partial \mathbf{phi}_z}{\partial x}, 
            \frac{\partial \mathbf{phi}_y}{\partial x} - \frac{\partial \mathbf{phi}_x}{\partial y} \right)

    Parameters
    ----------
    vphi : numpy.ndarray (2, Ny, Nx) or (2, Ny, Nx, Nz)
        Vector field to compute the curl of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dz : float
        Spacing between grid points in the z direction.
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction, z-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (3, Ny, Nx, Nz)
        Curl of `vphi`.
    """
    vphix, vphiy, vphiz = vphi
    
    dphiz_x = compute_first_derivative(vphiz, dx, axis=1, periodic=periodic_axes[0]) # dphiz/dx
    dphiy_x = compute_first_derivative(vphiy, dx, axis=1, periodic=periodic_axes[0]) # dphiy/dx
    dphix_y = compute_first_derivative(vphix, dy, axis=0, periodic=periodic_axes[1]) # dphix/dy
    dphiz_y = compute_first_derivative(vphiz, dy, axis=0, periodic=periodic_axes[1]) # dphiz/dy
    dphix_z = compute_first_derivative(vphix, dz, axis=2, periodic=periodic_axes[2]) # dphix/dz
    dphiy_z = compute_first_derivative(vphiy, dz, axis=2, periodic=periodic_axes[2]) # dphiy/dz
    curl = np.array([dphiz_y - dphiy_z, dphix_z - dphiz_x, dphiy_x - dphix_y])
    return curl

def compute_curl(vphi: np.ndarray, hs: tuple, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the curl of a 2D or 3D vector field.

    Parameters
    ----------
    vphi : numpy.ndarray (2, Ny, Nx) or (2, Ny, Nx, Nz)
        Vector field to compute the curl of.
    hs : tuple
        Spacing between grid points in the x, y and/or z directions. (dx, dy) or (dx, dy, dz).
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction) or (x-direction, y-direction, z-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (3, Ny, Nx) or (3, Ny, Nx, Nz)
        Curl of `vphi`.
    """
    # Number of dims
    ndims = vphi.ndim
    # Compute derivatives
    if ndims == 3:
        return compute_curl_2D(vphi, hs[0], hs[1], periodic_axes)
    elif ndims == 4:
        return compute_curl_3D(vphi, hs[0], hs[1], hs[2], periodic_axes)

def compute_divergence_2D(vphi: np.ndarray, dx: float, dy: float, periodic_axes: tuple) -> np.ndarray:
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
        Axes along which the domain is periodic, (x-direction, y-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (Ny, Nx)
        Divergence of `vphi`.
    """
    vphix, vphiy = vphi
    dphix_x = compute_first_derivative(vphix, dx, axis=1, periodic=periodic_axes[0]) # dphix/dx
    dphiy_y = compute_first_derivative(vphiy, dy, axis=0, periodic=periodic_axes[1]) # dphiy/dy
    divergence = dphix_x + dphiy_y
    return divergence

def compute_divergence_3D(vphi: np.ndarray, dx: float, dy: float, dz: float, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the divergence of a 3D vector field.

    .. math::
        \nabla \cdot \phi = \frac{\partial \mathbf{phi}_x}{\partial x} + \frac{\partial \mathbf{phi}_y}{\partial y} + \frac{\partial \mathbf{phi}_z}{\partial z}

    Parameters
    ----------
    vphi : numpy.ndarray (2, Ny, Nx, Nz)
        Vector field to compute the divergence of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dz : float
        Spacing between grid points in the z direction.
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction, z-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (Ny, Nx, Nz)
        Divergence of `vphi`.
    """
    vphix, vphiy, vphiz = vphi
    dphix_x = compute_first_derivative(vphix, dx, axis=1, periodic=periodic_axes[0]) # dphix/dx
    dphiy_y = compute_first_derivative(vphiy, dy, axis=0, periodic=periodic_axes[1]) # dphiy/dy
    dphiz_z = compute_first_derivative(vphiz, dz, axis=2, periodic=periodic_axes[2]) # dphiz/dz
    divergence = dphix_x + dphiy_y + dphiz_z
    return divergence

def compute_divergence(vphi: np.ndarray, hs: tuple, periodic_axes: tuple) -> np.ndarray:
    """
    Compute the divergence of a 2D or 3D vector field.

    Parameters
    ----------
    vphi : numpy.ndarray (2, Ny, Nx) or (2, Ny, Nx, Nz)
        Vector field to compute the divergence of.
    hs : tuple
        Spacing between grid points in the x, y and/or z directions. (dx, dy) or (dx, dy, dz).
    periodic_axes : tuple
        Axes along which the domain is periodic, (x-direction, y-direction) or (x-direction, y-direction, z-direction).
        True in the position of periodic axes, False otherwise.

    Returns
    -------
    numpy.ndarray (Ny, Nx) or (Ny, Nx, Nz)
        Divergence of `vphi`.
    """
    # Number of dims
    ndims = vphi.ndim
    # Compute derivatives
    if ndims == 3:
        return compute_divergence_2D(vphi, hs[0], hs[1], periodic_axes)
    elif ndims == 4:
        return compute_divergence_3D(vphi, hs[0], hs[1], hs[2], periodic_axes)

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

def compute_first_derivative_plots_3D(phi: np.ndarray, h: float, axis: int) -> np.ndarray:
    """
    Compute the first derivative of a 3D scalar field along a given axis. This is the same
    as compute_first_derivative, but it is used for plotting purposes.

    Parameters
    ----------
    phi : numpy.ndarray (Nt, Ny, Nx, Nz)
        Scalar field.
    h : float
        Spacing between grid points.
    axis : int
        Axis along which to compute the derivative. 0 for t, 1 for y, 2 for x, and 3 for z.

    Returns
    -------
    numpy.ndarray
        First derivative of `phi` along `axis`.
    """
    # General case 
    phi_ip1 = np.roll(phi,-1, axis=axis) # phi_{i+1}
    phi_im1 = np.roll(phi, 1, axis=axis) # phi_{i-1}
    phi_h = (phi_ip1 - phi_im1) / (2 * h) # Central difference
    # dphi_h = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2 * h)
    if axis == 3: # Fix boundary in z
        phi_h[:,:,:, 0] = (-3 * phi[:,:,:, 0] + 4 * phi[:,:,:, 1] - phi[:,:,:, 2]) / (2 * h) # Forward
        phi_h[:,:,:,-1] = (3 * phi[:,:,:, -1] - 4 * phi[:,:,:, -2] + phi[:,:,:, -3]) / (2 * h) # Backward
    return phi_h

def compute_gradient_plots_3D(phi: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Compute the gradient of a 3D scalar field. This is the same as compute_gradient,
    but it is used for plotting purposes.

    Parameters
    ----------
    phi : numpy.ndarray (Nt, Ny, Nx)
        Scalar field to compute the derivatives of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dz : float
        Spacing between grid points in the z direction.

    Returns
    -------
    numpy.ndarray (2, Nt, Ny, Nx, Nz)
        Gradient of `phi`.
    """
    dphi_x = compute_first_derivative_plots_3D(phi, dx, axis=2) # dphi/dx
    dphi_y = compute_first_derivative_plots_3D(phi, dy, axis=1) # dphi/dy
    dphi_z = compute_first_derivative_plots_3D(phi, dz, axis=3) # dphi/dz
    gradient = np.array([dphi_x, dphi_y, dphi_z])
    return gradient

def compute_curl_plots_3D(vphi: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Compute the curl of a 3D vector field. This is the same as compute_curl,
    but it is used for plotting purposes.

    Parameters
    ----------
    vphi : numpy.ndarray (2, Nt, Ny, Nx, Nz)
        Vector field to compute the curl of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dz : float
        Spacing between grid points in the z direction.

    Returns
    -------
    numpy.ndarray (3, Nt, Ny, Nx, Nz)
        Curl of `vphi`.
    """
    vphix, vphiy, vphiz = vphi
    dphiz_y = compute_first_derivative_plots_3D(vphiz, dy, axis=1)# dphiz/dy
    dphiy_z = compute_first_derivative_plots_3D(vphiy, dz, axis=3) # dphiy/dz
    dphix_z = compute_first_derivative_plots_3D(vphix, dz, axis=3) # dphix/dz
    dphiz_x = compute_first_derivative_plots_3D(vphiz, dx, axis=2) # dphiz/dx
    dphiy_x = compute_first_derivative_plots_3D(vphiy, dx, axis=2) # dphiy/dx
    dphix_y = compute_first_derivative_plots_3D(vphix, dy, axis=1) # dphix/dy
    curl = np.array([dphiz_y - dphiy_z, dphix_z - dphiz_x, dphiy_x - dphix_y])
    return curl

def compute_divergence_plots_3D(vphi: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """
    Compute the divergence of a 3D vector field. This is the same as compute_divergence,
    but it is used for plotting purposes.

    Parameters
    ----------
    vphi : numpy.ndarray (2, Nt, Ny, Nx, Nz)
        Vector field to compute the divergence of.
    dx : float
        Spacing between grid points in the x direction.
    dy : float
        Spacing between grid points in the y direction.
    dx : float
        Spacing between grid points in the z direction.

    Returns
    -------
    numpy.ndarray (Nt, Ny, Nx, Nz)
        Divergence of `vphi`.
    """
    vphix, vphiy, vphiz = vphi
    dphix_x = compute_first_derivative_plots_3D(vphix, dx, axis=2) # dphix/dx
    dphiy_y = compute_first_derivative_plots_3D(vphiy, dy, axis=1) # dphiy/dy
    dphiz_z = compute_first_derivative_plots_3D(vphiz, dz, axis=3) # dphiz/dz
    divergence = dphix_x + dphiy_y + dphiz_z
    return divergence