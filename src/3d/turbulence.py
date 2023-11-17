import numpy as np
from derivatives import compute_first_derivative, compute_second_derivative, compute_gradient

# Wall damping functions
f_w1 = lambda z, u_tau, nu: 1 - np.exp(-z * u_tau / 25 / nu)
f_w2 = lambda z, u_tau, nu: (1 - np.exp(-(z * u_tau / 25 / nu) ** 3)) ** 0.5

def turbulence(u: np.ndarray, v: np.ndarray, w: np.ndarray, T: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute the subgrid-scale (SGS) stresses and thermal energy for a turbulent flow.

    Parameters
    ----------
    u : np.ndarray
        Array of x-velocity values.
    v : np.ndarray
        Array of y-velocity values.
    w : np.ndarray
        Array of z-velocity values.
    T : np.ndarray
        Array of temperature values.
    params : dict
        Dictionary containing the following parameters:
        - dx : float
            Grid spacing in the x-direction.
        - dy : float
            Grid spacing in the y-direction.
        - dz : float
            Grid spacing in the z-direction.
        - C_s : float
            Constant used in the wall damping function.
        - Pr : float
            Prandtl number.
        - rho : float
            Density of the fluid.
        - Zm : float
            Mixing length.
        - nu : float
            Kinematic viscosity.

    Returns
    -------
    np.ndarray
        Array containing the x-component of SGS stresses, y-component of SGS stresses, and SGS thermal energy.

    """
    dx, dy, dz = params['dx'], params['dy'], params['dz']
    C_s = params['C_s'] 
    Pr = params['Pr']
    rho = params['rho']
    Zm = params['Zm']
    nu = params['nu']
    Delta = (dx * dy * dz) ** (1 / 3)

    # Compute derivatives #
    # First derivatives
    ux, uy, uz = compute_gradient(u, dx, dy, dz, (True, True, False))
    vx, vy, vz = compute_gradient(v, dx, dy, dz, (True, True, False))
    wx, wy, wz = compute_gradient(w, dx, dy, dz, (True, True, False))
    Tx, Ty, Tz = compute_gradient(T, dx, dy, dz, (True, True, False))
    # Second derivatives
    uxx = compute_second_derivative(u, dx, 1)
    uyy = compute_second_derivative(u, dy, 0)
    uzz = compute_second_derivative(u, dz, 2, False)
    vxx = compute_second_derivative(v, dx, 1)
    vyy = compute_second_derivative(v, dy, 0)
    vzz = compute_second_derivative(v, dz, 2, False)
    wxx = compute_second_derivative(w, dx, 1)
    wyy = compute_second_derivative(w, dy, 0)
    wzz = compute_second_derivative(w, dz, 2, False)
    Txx = compute_second_derivative(T, dx, 1)
    Tyy = compute_second_derivative(T, dy, 0)
    Tzz = compute_second_derivative(T, dz, 2, False)
    # Mixed derivatives
    uyx = compute_first_derivative(uy, dx, 1)
    uzx = compute_first_derivative(uz, dx, 1)
    vyx = compute_first_derivative(vy, dx, 1)
    vzx = compute_first_derivative(vz, dx, 1)
    wyx = compute_first_derivative(wy, dx, 1)
    wzx = compute_first_derivative(wz, dx, 1)
    uxy = compute_first_derivative(ux, dy, 0)
    uzy = compute_first_derivative(uz, dy, 0)
    vxy = compute_first_derivative(vx, dy, 0)
    vzy = compute_first_derivative(vz, dy, 0)
    wxy = compute_first_derivative(wx, dy, 0)
    wzy = compute_first_derivative(wz, dy, 0)
    uxz = compute_first_derivative(ux, dz, 2, False)
    uyz = compute_first_derivative(uy, dz, 2, False)
    vxz = compute_first_derivative(vx, dz, 2, False)
    vyz = compute_first_derivative(vy, dz, 2, False)
    wxz = compute_first_derivative(wx, dz, 2, False)
    wyz = compute_first_derivative(wy, dz, 2, False)
    

    # |S|
    mod_S = (2 * (ux ** 2 + vy ** 2 + wz ** 2) + (uz + wx) ** 2 + (uy + vx) ** 2 + (wy + vz) ** 2) ** (1 / 2) + 1e-16

    # 'psi_x', 'psi_y' and 'psi_z'
    psi_x = 4 * (ux * uxx + vy * vyx + wz * wzx) + 2 * (
        (uz + wx) * (uzx + wxx) +
        (uy + vx) * (uyx + vxx) +
        (wy + vz) * (wyx + vzx)
    )
    psi_y = 4 * (ux * uxy + vy * vyy + wz * wzy) + 2 * ( 
        (uz + wx) * (uzy + wxy) +
        (uy + vx) * (uyy + vxy) + 
        (wy + vz) * (wyy + vzy) 
    )
    psi_z = 4 * (ux * uxz + vy * vyz + wz * wzz) + 2 * (
        (uz + wx) * (uzz + wxz) +
        (uy + vx) * (uyz + vxz) +
        (wy + vz) * (wyz + vzz)
    )

    # Wall damping function
    tau_p = ((nu * 0.5 * (uz + wx)[0]) ** 2 + (nu * 0.5 * (vz + wy)[0]) ** 2) ** 0.5
    u_tau = (tau_p) ** 0.5
    l = C_s * Delta 

    # Damping stuff
    fw = f_w1(Zm, u_tau, nu)
    fwx = compute_first_derivative(fw, dx, 1)
    fwy = compute_first_derivative(fw, dy, 0)
    fwz = compute_first_derivative(fw, dz, 2, False)

    # Intermediary terms
    # with damping
    sgs_x_damp = 2 * mod_S * fw * (fwx * ux + 0.5 * fwy * (vx + uy) + 0.5 * fwz * (wx + uz))
    sgs_y_damp = 2 * mod_S * fw * (fwy * vy + 0.5 * fwx * (uy + vx) + 0.5 * fwz * (wy + vz))
    sgs_z_damp = 2 * mod_S * fw * (fwz * wz + 0.5 * fwx * (wx + uz) + 0.5 * fwy * (wy + vz))
    # without damping
    sgs_x_no_damp = 1 / (2 * mod_S) * (psi_x * ux + 0.5 * psi_y * (uy + vx) + 0.5 * psi_z * (wx + uz)) + mod_S * (uxx + 0.5 * (vxy + uyy) + 0.5 * (wxz + uzz))
    sgs_y_no_damp = 1 / (2 * mod_S) * (psi_y * vy + 0.5 * psi_x * (vx + uy) + 0.5 * psi_z * (wy + vz)) + mod_S * (vyy + 0.5 * (uyx + vxx) + 0.5 * (wyz + vzz))
    sgs_z_no_damp = 1 / (2 * mod_S) * (psi_z * wz + 0.5 * psi_x * (wx + uz) + 0.5 * psi_y * (wy + vz)) + mod_S * (wzz + 0.5 * (uzx + wxx) + 0.5 * (vzx + wyx))
    # SGS stresses
    sgs_x = -2 * l ** 2 * (sgs_x_no_damp * fw ** 2 + sgs_x_damp) # x-component of SGS stresses
    sgs_y = -2 * l ** 2 * (sgs_y_no_damp * fw ** 2 + sgs_y_damp) # y-component of SGS stresses
    sgs_z = -2 * l ** 2 * (sgs_z_no_damp * fw ** 2 + sgs_z_damp) # z-component of SGS stresses
    # SGS thermal energy
    sgs_T_no_damp = 1 / (2 * mod_S) * (psi_x * Tx  + psi_y * Ty + psi_z * Tz) + mod_S * (Txx + Tyy + Tzz) # No damping terms
    sgs_T_damp = 2 * fw * mod_S * (fwx * Tx + fwy * Ty + fwz * Tz) # Damping terms
    sgs_T = -l ** 2 / Pr * (sgs_T_no_damp * fw ** 2 + sgs_T_damp) # SGS thermal energy

    return np.array([sgs_x, sgs_y, sgs_z, sgs_T])