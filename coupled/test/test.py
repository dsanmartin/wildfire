import numpy as np

def Phi(t, A, **kwargs):
    
    # Parameters
    x = kwargs['x']
    y = kwargs['y']
    F = kwargs['F']
    S = kwargs['S']
    c_1 = kwargs['c_1']
    c_2 = kwargs['c_2']
    c_3 = kwargs['c_3']
    d_1 = kwargs['d_1']
    d_2 = kwargs['d_2']
    d_3 = kwargs['d_3']


    U, V, R, T = A
    Uf_ = np.zeros_like(U)
    Vf_ = np.zeros_like(V)
    Rf_ = np.zeros_like(R)
    Tf_ = np.zeros_like(T)

    Ux = (U[1:-1, 2:] - U[1:-1, :-2]) / (2 * dx)
    Uy = (U[2:, 1:-1] - U[:-2, 1:-1]) / (2 * dy)
    Uxx = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (dx ** 2)
    Uyy = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / (dy ** 2)
    Vx = (V[1:-1, 2:] - V[1:-1, :-2]) / (2 * dx)
    Vy = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2 * dy)
    Vxx = (V[1:-1, 2:] - 2 * V[1:-1, 1:-1] + V[1:-1, :-2]) / (dx ** 2)
    Vyy = (V[2:, 1:-1] - 2 * V[1:-1, 1:-1] + V[:-2, 1:-1]) / (dy ** 2)
    Rx = (R[1:-1, 2:] - R[1:-1, :-2]) / (2 * dx)
    Ry = (R[2:, 1:-1] - R[:-2, 1:-1]) / (2 * dy)
    Rxx = (R[1:-1, 2:] - 2 * R[1:-1, 1:-1] + R[1:-1, :-2]) / (dx ** 2)
    Ryy = (R[2:, 1:-1] - 2 * R[1:-1, 1:-1] + R[:-2, 1:-1]) / (dy ** 2)
    Tx = (T[1:-1, 2:] - T[1:-1, :-2]) / (2 * dx)
    Ty = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2 * dy)
    Txx = (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (dx ** 2)
    Tyy = (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dy ** 2)

    fx, fy = F(x, y)
    fx[1:-1, 1:-1] -= 2 * Tx #[1:-1, 1:-1]
    fy[1:-1, 1:-1] -= 2 * Ty #[1:-1, 1:-1]
    s = S(R, T)
    Uf_[1:-1, 1:-1] = -c_1 * (U[1:-1, 1:-1] * Ux + V[1:-1, 1:-1] * Uy) + d_1 * (Uxx + Uyy) + fx[1:-1, 1:-1]
    Vf_[1:-1, 1:-1] = -c_1 * (U[1:-1, 1:-1] * Vx + V[1:-1, 1:-1] * Vy) + d_1 * (Vxx + Vyy) + fy[1:-1, 1:-1]
    Rf_[1:-1, 1:-1] = -c_2 * (U[1:-1, 1:-1] * Rx + V[1:-1, 1:-1] * Ry) + d_2 * (Rxx + Ryy) - s[1:-1, 1:-1]
    Tf_[1:-1, 1:-1] = -c_3 * (U[1:-1, 1:-1] * Tx + V[1:-1, 1:-1] * Ty) + d_3 * (Txx + Tyy) + s[1:-1, 1:-1]


    return np.array([Uf_, Vf_, Rf_, Tf_])

### MAIN ###
# Domain
x_min, x_max = 0, 1
y_min, y_max = 0, 1 
z_min, z_max = 0, 1
t_min, t_max = 0, 1
Nx, Ny, Nz, Nt = 51, 51, 21, 501 # 
# Arrays
x = np.linspace(x_min, x_max, Nx)
y = np.linspace(y_min, y_max, Ny)
z = np.linspace(z_min, z_max, Nz)
t = np.linspace(t_min, t_max, Nt)
# Meshgrid
# 2D
X, Y = np.meshgrid(x, y)
# 3D
#X, Y, Z = np.meshgrid(x, y, z)
dx, dy, dz, dt = x[1] - x[0], y[1] - y[0], z[1] - z[0], t[1] - t[0]
print(dx, dy, dz, dt)

u0 = lambda x, y: x * 0 + 5
v0 = lambda x, y: x * 0 - 5 
R0 = lambda x, y: x * 0 + 1
A = .5
s = .001
x0 = y0 = 0.5
T0 = lambda x, y: A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / s)

U_0 = u0(X, Y)
V_0 = v0(X, Y)
R_0 = R0(X, Y)
T_0 = T0(X, Y)

fx = lambda x, y: x * 0 + 1
fy = lambda x, y: x * 0
F = lambda x, y: (fx(x, y), fy(x, y)) 
S = lambda r, t: r + t

y0 = np.array([U_0, V_0, R_0, T_0])

# Array for approximations
samples = 100
yy = np.zeros((Nt // samples + 1, 4, Ny, Nx))
# yy = np.zeros((Nt, 5, Ny, Nx, Nz))
yy[0] = y0

c_1 = .1
c_2 = .01
c_3 = .01
d_1 = .01
d_2 = .01
d_3 = 1e-4

args = {
    'x': X,
    'y': Y,
    'F': F,
    'S': S,
    'c_1': c_1,
    'c_2': c_2,
    'c_3': c_3,
    'd_1': d_1,
    'd_2': d_2,
    'd_3': d_3,
}

# Solve
y_tmp = yy[0]
for n in range(Nt-1):
    #print("Step:", n)
    k1 = Phi(t[n], y_tmp, **args)
    k2 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k1, **args)
    k3 = Phi(t[n] + 0.5 * dt, y_tmp + 0.5 * dt * k2, **args)
    k4 = Phi(t[n] + dt, y_tmp + dt * k3, **args)
    y_tmp = y_tmp + (1/6) * dt * (k1 + 2 * k2 + 2 * k3 + k4)
    if n % samples == 0:
        print("Step:", n)
        yy[n // samples + 1] = y_tmp
yy[-1] = y_tmp


# Get velocities
Uf = yy[:, 0]
Vf = yy[:, 1]
Rf = yy[:, 2]
Tf = yy[:, 3]

# Save 
np.savez('output/test.npz', U=Uf, V=Vf, R=Rf, T=Tf, X=X, Y=Y, t=t[::samples])