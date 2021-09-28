#%%
import numpy as np
import matplotlib.pyplot as plt
import utils
from time_solver import *
import time

def boundaryConditions(U, B):
    Ub = np.copy(U)
    Bb = np.copy(B)
    Ny, Nx = Ub.shape
    # Only Dirichlet: 
    # Temperature
    Ub[ 0,:] = np.zeros(Nx)
    Ub[-1,:] = np.zeros(Nx)
    Ub[:, 0] = np.zeros(Ny)
    Ub[:,-1] = np.zeros(Ny)
    # Fuel
    Bb[0 ,:] = np.zeros(Nx)
    Bb[-1,:] = np.zeros(Nx)
    Bb[:, 0] = np.zeros(Ny)
    Bb[:,-1] = np.zeros(Ny)

    return Ub, Bb

def RHS(t, r, **kwargs):
    # Parameters 
    X, Y = kwargs['x'], kwargs['y']
    x, y = X[0], Y[:, 0] 
    V   = kwargs['V']
    kap = kwargs['kap']
    eps = kwargs['eps']
    upc = kwargs['upc']
    alp = kwargs['alp']
    f = kwargs['f']
    g = kwargs['g']
    K = kwargs['K']
    Ku = kwargs['Ku']
    Nx = x.shape[0]
    Ny = y.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Vector field evaluation
    V1, V2 = V(x, y, t)
    
    # Recover u and b from y
    # U = np.copy(r[:Ny * Nx].reshape((Ny, Nx), order='F'))
    # B = np.copy(r[Ny * Nx:].reshape((Ny, Nx), order='F'))
    U = np.copy(r[:Ny * Nx].reshape((Ny, Nx)))
    B = np.copy(r[Ny * Nx:].reshape((Ny, Nx)))

    # Compute derivatives #
    Ux = np.zeros_like(U)
    Uy = np.zeros_like(U)
    Uxx = np.zeros_like(U)
    Uyy = np.zeros_like(U)
    # First derivatives
    Ux[1:-1, 1:-1] = (U[1:-1, 1:-1] - U[1:-1, :-2]) / dx
    Uy[1:-1, 1:-1] = (U[1:-1, 1:-1] - U[:-2, 1:-1]) / dy
    # Vx = (V[1:-1, 1:-1] - V[1:-1, :-2]) / dx
    # Vy = (V[1:-1, 1:-1] - V[:-2, 1:-1]) / dy
    # Second derivatives
    Uxx[1:-1, 1:-1] = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dx / dx
    Uyy[1:-1, 1:-1] = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dy / dy
    # Vxx = (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1, :-2]) / dx / dx
    # Vyy = (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / dy / dy
        
    # Laplacian of u
    lapU = Uxx + Uyy

    # Compute diffusion
    #diffusion = Ku(U) * (Ux ** 2 + Uy ** 2) + K(U) * lapU
    diffusion = kap * lapU 
    
    convection = Ux * V1 + Uy * V2 # v \cdot grad u.    
    reaction = f(U, B) # eval fuel
    
    # Compute RHS
    Uf = diffusion - convection + reaction 
    Bf = g(U, B)
    
    # Add boundary conditions
    Uf, Bf = boundaryConditions(Uf, Bf)

    # Build y = [vec(u), vec(\beta)]^T and return
    #return np.r_[Uf.flatten('F'), Bf.flatten('F')] 
    return np.r_[Uf.flatten(), Bf.flatten()] 

#%%
### PARAMETERS ###
# Model parameters #
kap = 1e-1
eps = 3e-1
upc = 3
alp = 1e-3
q = 1
x_min, x_max = 0, 90
y_min, y_max = 0, 90
t_min, t_max = 0, 10

# # Asensio parameters
# Tinf = 300
# Tpc = 550
# eps = 3e-2
# upc = (Tpc - Tinf) / (eps * Tinf)
# x_min, x_max = 0, 300
# y_min, y_max = 0, 300
# t_min, t_max = 0, 0.1625

# Re-define PDE funtions with parameters #
s = lambda u: utils.H(u, upc) #if sf == 'step' else sigmoid(u)
ff = lambda u, b: utils.f(u, b, eps, alp, s)
gg = lambda u, b: utils.g(u, b, eps, q, s)
KK = lambda u: utils.K(u, kap, eps)
KKu = lambda u: utils.Ku(u, kap, eps)

# Numerical #
# Space
Nx = 128
Ny = 128
# Time
Nt = 1000

# # A&F
# dx = dy = 1.875
# dt = 2.5e-7
# dx = dy = 7.5
# dt = 1e-6
# dx = dy = 0.46875
# dt = 6.25e-8
# Nx = int(x_max / dx) + 1
# Ny = int(y_max / dy) + 1
# Nt = int(t_max / dt) + 1

# Domain #
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
t = np.linspace(t_min, t_max, Nt)
X, Y = np.meshgrid(x, y)

# Initial conditions
u0 = lambda x, y: 6 * utils.G(x - 20, y - 20, 20)
b0 = lambda x, y: x * 0 + 1
w1 = lambda x, y, t: np.cos(np.pi/4 + x * 0)
w2 = lambda x, y, t: np.sin(np.pi/4 + x * 0)


# # Real asensio & Ferragut
# u0 = lambda x, y: 30 * G(x - 50, y - 50, 100)
# b0 = lambda x, y: b0_af2002(x, y)
# w1 = lambda x, y, t: 300 + x * 0
# w2 = lambda x, y, t: 300 + x * 0

# Wind effect
V = lambda x, y, t: (w1(x, y, t), w2(x, y, t))

print("upc = ", upc)
print("Nx = ", Nx)
print("Ny = ", Ny)
print("Nt = ", Nt)
print("dx =", x[1] - x[0])
print("dy =", y[1] - y[0])
print("dt =", t[1] - t[0])

# Parameters #
params = {'x': X, 'y': Y, 'V': V, 'kap': kap, 'eps': eps, 'upc': upc, 'alp': alp, 'f': ff, 'g': gg, 'K': KK, 'Ku': KKu}

#y0 = np.r_[u0(X, Y).flatten('F'), b0(X, Y).flatten('F')]
y0 = np.r_[u0(X, Y).flatten(), b0(X, Y).flatten()]

F = lambda t, y: RHS(t, y, **params)

# Solve #
time_start = time.time()
# R = IVP(t, y0, F, 'RK45')
R = RK4(t, y0, F)

time_end = time.time()
solve_time = time_end - time_start
print("Time: ", solve_time, "[s]")

#R = sol.y.T
# U = R[:, :Nx*Ny].reshape((Nt, Ny, Nx), order='F')
# B = R[:, Nx*Ny:].reshape((Nt, Ny, Nx), order='F')
U = R[:, :Nx*Ny].reshape((Nt, Ny, Nx))
B = R[:, Nx*Ny:].reshape((Nt, Ny, Nx))

#%%
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.contourf(X, Y, U[-1], cmap=plt.cm.jet)
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.contourf(X, Y, B[-1], cmap=plt.cm.viridis)
# plt.colorbar()
# plt.tight_layout()
# plt.show()
# %%
