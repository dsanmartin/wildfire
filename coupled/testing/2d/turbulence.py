import numpy as np

f_w1 = lambda z, u_tau, nu: 1 - np.exp(-z * u_tau / 25 / nu)
f_w2 = lambda z, u_tau, nu: (1 - np.exp(-(z * u_tau / 25 / nu) ** 3)) ** 0.5

def turbulence1(ux, uy, vx, vy, uxx, uyy, vxx, vyy, args):
    dx = args['dx']
    dy = args['dy']
    C_s = args['C_s'] 
    Delta = (dx * dy) ** (1/2)

    # Turbulence
    S_ij_mod = (2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2) ** (1 / 2)
    nu_sgs = (C_s * Delta) ** 2 * S_ij_mod
    # For velocity
    # Mixed derivatives
    vxy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
    uyx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dx)
    # Mixed derivatives at boundaries
    # Periodic on x
    # On y
    vxy[0, :] = (-3 * vx[0, :] + 4 * vx[1, :] - vx[2, :]) / (2 * dy) # Forward at y=y_min
    vxy[-1, :] = (3 * vx[-1, :] - 4 * vx[-2, :] + vx[-3, :]) / (2 * dy) # Backward at y=y_max
    tau_x = uxx + 0.5 * (vxy + uyy)
    tau_y = vyy + 0.5 * (uyx + vxx)
    sgs_x = -2 * nu_sgs * tau_x 
    sgs_y = -2 * nu_sgs * tau_y 
    nu_sgs_x = (np.roll(nu_sgs, -1, axis=1) - np.roll(nu_sgs, 1, axis=1)) / (2 * dx)
    nu_sgs_y = (np.roll(nu_sgs, -1, axis=0) - np.roll(nu_sgs, 1, axis=0)) / (2 * dy)
    sgs_x = -2 * (nu_sgs_x * ux + 0.5 * nu_sgs_y * (uy + vx) + nu_sgs * (uxx + 0.5 * (vxy + uyy)))
    sgs_y = -2 * (nu_sgs_y * vy + 0.5 * nu_sgs_x * (vx + uy) + nu_sgs * (vyy + 0.5 * (uyx + vxx)))

    return np.array([sgs_x, sgs_y])

def turbulence2(ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, args):
    dx = args['dx']
    dy = args['dy']
    C_s = args['C_s'] 
    Pr = args['Pr']
    Delta = (dx * dy) ** (1/2)

    # Turbulence
    S_ij_mod = (2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2) ** (1 / 2)
    nu_sgs = (C_s * Delta) ** 2 * S_ij_mod
    # Mixed derivatives
    vxy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
    uyx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dx)
    vyx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
    uxy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2 * dy)
    # Mixed derivatives at boundaries
    # Periodic on x (nothing to do)
    # On y
    vxy[0, :] = (-3 * vx[0, :] + 4 * vx[1, :] - vx[2, :]) / (2 * dy) # Forward at y=y_min
    vxy[-1, :] = (3 * vx[-1, :] - 4 * vx[-2, :] + vx[-3, :]) / (2 * dy) # Backward at y=y_max
    uxy[0, :] = (-3 * ux[0, :] + 4 * ux[1, :] - ux[2, :]) / (2 * dy) # Forward at y=y_min
    uxy[-1, :] = (3 * ux[-1, :] - 4 * ux[-2, :] + ux[-3, :]) / (2 * dy) # Backward at y=y_max

    # 'psi_x' and 'psi_y'
    psi_x = 4 * uxx + 4 * vyx + 2 * (uy + vx) * (uyx + vxx) 
    psi_y = 4 * vyy + 4 * uxy + 2 * (uy + vx) * (uyy + vxy)

    sgs_x = -2 * (C_s * Delta) ** 2 * ( 
        1 / (2 * S_ij_mod) * psi_x * ux + 0.5 * psi_y * (uy + vx) +
        S_ij_mod * (uxx + 0.5 * (vxy + uyy))
    )
    sgs_y = -2 * (C_s * Delta) ** 2 * (
        1 / (2 * S_ij_mod) * psi_y * vy + 0.5 * psi_x * (vx + uy) +
        S_ij_mod * (vyy + 0.5 * (uyx + vxx))
    )
    # For temperature
    # sgsT_x = -nu_sgs / Pr * Txx
    # sgsT_y = -nu_sgs / Pr * Tyy
    sgs_T = -nu_sgs / Pr * (Txx + Tyy)

    return np.array([sgs_x, sgs_y, sgs_T])

def turbulence(u, v, ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, args):
    dx = args['dx']
    dy = args['dy']
    C_s = args['C_s'] 
    Pr = args['Pr']
    rho = args['rho']
    Ym = args['Ym']
    nu = args['nu']
    Delta = (dx * dy) ** (1/2)

    # Turbulence
    S_ij_mod = (2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2) ** (1 / 2) + 1e-16
    # Mixed derivatives
    vxy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
    uyx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dx)
    vyx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
    uxy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2 * dy)
    # Mixed derivatives at boundaries
    # Periodic on x (nothing to do)
    # On y
    vxy[0, :] = (-3 * vx[0, :] + 4 * vx[1, :] - vx[2, :]) / (2 * dy) # Forward at y=y_min
    vxy[-1, :] = (3 * vx[-1, :] - 4 * vx[-2, :] + vx[-3, :]) / (2 * dy) # Backward at y=y_max
    uxy[0, :] = (-3 * ux[0, :] + 4 * ux[1, :] - ux[2, :]) / (2 * dy) # Forward at y=y_min
    uxy[-1, :] = (3 * ux[-1, :] - 4 * ux[-2, :] + ux[-3, :]) / (2 * dy) # Backward at y=y_max

    # 'psi_x' and 'psi_y'
    psi_x = 4 * (ux * uxx + vy * vyx) + 2 * (uy + vx) * (uyx + vxx) 
    psi_y = 4 * (ux * uxy + vy * vyy) + 2 * (uy + vx) * (uyy + vxy)

    # Wall damping function
    #tau_w = 1e-1
    #u_tau = (tau_w / rho) ** 0.5
    tau_p = ((0.5 * nu * (uy + vx)[0]) ** 2) ** 0.5 
    u_tau = (tau_p) ** 0.5
    fw = f_w1(Ym, u_tau, nu)
    l = C_s * Delta #* fw

    sgs_x = -2 * l ** 2 * ( 
        1 / (2 * S_ij_mod) * (psi_x * ux + 0.5 * psi_y * (uy + vx)) +
        S_ij_mod * (uxx + 0.5 * (vxy + uyy))
    )
    sgs_y = -2 * l ** 2 * (
        1 / (2 * S_ij_mod) * (psi_y * vy + 0.5 * psi_x * (vx + uy)) +
        S_ij_mod * (vyy + 0.5 * (uyx + vxx))
    )

    # SGS thermal energy
    sgs_T = -l ** 2 / Pr * (
        1 / (2 * S_ij_mod) * (psi_x * Tx  + psi_y * Ty) + S_ij_mod * (Txx + Tyy)
    )

    return np.array([sgs_x, sgs_y, sgs_T])

def periodic_turbulence_2d(u, v, ux, uy, vx, vy, uxx, uyy, vxx, vyy, args):
    dx = args['dx']
    dy = args['dy']
    C_s = args['C_s'] 
    Delta = (dx * dy) ** (1/2)

    # Turbulence
    S_ij_mod = (2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2) ** (1 / 2) + 1e-16

    # Mixed derivatives
    vxy = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dy)
    uyx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2 * dx)
    vyx = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dx)
    uxy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2 * dy)

    # 'psi_x' and 'psi_y'
    psi_x = 4 * (ux * uxx + vy * vyx) + 2 * (uy + vx) * (uyx + vxx) 
    psi_y = 4 * (ux * uxy + vy * vyy) + 2 * (uy + vx) * (uyy + vxy)

    l = C_s * Delta #* f_sgs(z_plus, 25)

    sgs_x = -2 * l ** 2 * ( 
        1 / (2 * S_ij_mod) * psi_x * ux + 0.5 * psi_y * (uy + vx) +
        S_ij_mod * (uxx + 0.5 * (vxy + uyy))
    )
    sgs_y = -2 * l ** 2 * (
        1 / (2 * S_ij_mod) * psi_y * vy + 0.5 * psi_x * (vx + uy) +
        S_ij_mod * (vyy + 0.5 * (uyx + vxx))
    )

    return np.array([sgs_x, sgs_y])