import numpy as np
import scipy.linalg as spla


def fftfd_solver(x, y, f, p_top):
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