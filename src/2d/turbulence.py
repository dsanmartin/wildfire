import numpy as np
from derivatives import compute_first_derivative, compute_second_derivative, compute_gradient

f_w1 = lambda z, u_tau, nu: 1 - np.exp(-z * u_tau / 25 / nu)
f_w2 = lambda z, u_tau, nu: (1 - np.exp(-(z * u_tau / 25 / nu) ** 3)) ** 0.5

# def turbulence(u, v, ux, uy, vx, vy, Tx, Ty, uxx, uyy, vxx, vyy, Txx, Tyy, args):
def turbulence(u, v, T, args):
    dx = args['dx']
    dy = args['dy']
    C_s = args['C_s'] 
    Pr = args['Pr']
    rho = args['rho']
    Ym = args['Ym']
    nu = args['nu']
    Delta = (dx * dy) ** (1/2)

    # Compute derivatives #
    # First derivatives
    ux, uy = compute_gradient(u, dx, dy)
    vx, vy = compute_gradient(v, dx, dy)
    Tx, Ty = compute_gradient(T, dx, dy)
    # Second derivatives
    uxx = compute_second_derivative(u, dx, 1)
    uyy = compute_second_derivative(u, dy, 0)
    vxx = compute_second_derivative(v, dx, 1)
    vyy = compute_second_derivative(v, dy, 0)
    Txx = compute_second_derivative(T, dx, 1)
    Tyy = compute_second_derivative(T, dy, 0)
    # Mixed derivatives
    vxy = compute_first_derivative(vx, dy, 0)
    uyx = compute_first_derivative(uy, dx, 1)
    vyx = compute_first_derivative(vy, dx, 1)
    uxy = compute_first_derivative(ux, dy, 0)

    # Turbulence
    S_ij_mod = (2 * (ux ** 2 + vy ** 2) + (uy + vx) ** 2) ** (1 / 2) + 1e-16

    # 'psi_x' and 'psi_y'
    psi_x = 4 * (ux * uxx + vy * vyx) + 2 * (uy + vx) * (uyx + vxx) 
    psi_y = 4 * (ux * uxy + vy * vyy) + 2 * (uy + vx) * (uyy + vxy)

    # Wall damping function
    #tau_w = 1e-1
    #u_tau = (tau_w / rho) ** 0.5
    tau_p = ((0.5 * nu * (uy + vx)[0]) ** 2) ** 0.5 
    u_tau = (tau_p) ** 0.5
    l = C_s * Delta 

    # Damping stuff
    fw = f_w1(Ym, u_tau, nu)
    fwx = compute_first_derivative(fw, dx, 1)
    fwy = compute_first_derivative(fw, dy, 0)

    sgs_x_damp = 2 * S_ij_mod * fw * (fwx * ux + 0.5 * fwy * (vx + uy))
    sgs_y_damp = 2 * S_ij_mod * fw * (0.5 * fwx * (uy + vx) + fwy * vy)

    sgs_x_no_damp = 1 / (2 * S_ij_mod) * (psi_x * ux + 0.5 * psi_y * (uy + vx)) + S_ij_mod * (uxx + 0.5 * (vxy + uyy)) 
    sgs_y_no_damp = 1 / (2 * S_ij_mod) * (psi_y * vy + 0.5 * psi_x * (vx + uy)) + S_ij_mod * (vyy + 0.5 * (uyx + vxx))

    sgs_x = -2 * l ** 2 * (sgs_x_no_damp * fw ** 2 + sgs_x_damp)
    sgs_y = -2 * l ** 2 * (sgs_y_no_damp * fw ** 2 + sgs_y_damp)

    # SGS thermal energy
    sgs_T_no_damp = 1 / (2 * S_ij_mod) * (psi_x * Tx  + psi_y * Ty) + S_ij_mod * (Txx + Tyy)
    sgs_T_damp = 2 * fw * S_ij_mod * (fwx * Tx + fwy * Ty)
    sgs_T = -l ** 2 / Pr * (sgs_T_no_damp * fw ** 2 + sgs_T_damp)

    return np.array([sgs_x, sgs_y, sgs_T])
