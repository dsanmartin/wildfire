import numpy as np
import matplotlib.pyplot as plt
from utils import h, hx, hy, G, Gx, Gy, Gz, G13, G23, zc

# 2D plot
def genericPlot(x, y, u, v, h, r, t, p, axis_x='x', axis_y='y'):
    # Plot setup
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 7))
    # Set limits
    axs[0,0].set_xlim(np.min(x), np.max(x))
    axs[0,0].set_ylim(np.min(y), np.max(y))
    # Plot hill
    if axis_x == 'x' and axis_y == 'y':
        axs[0,0].contour(x, y, h)
    else:
        axs[0,0].plot(x[:,0], h, 'k-')
    # Plot vector field
    axs[0,0].quiver(x, y, u, v)
    #axs[0,0].streamplot(x[:,0], z[-1,:], u.T, w.T)
    # Magnitude
    mag = axs[0,0].contourf(x, y, np.sqrt(u**2 + v**2), alpha=0.5, cmap=plt.cm.viridis)
    # Pressure 
    pres = axs[0,1].contourf(x, y, p, alpha=0.5, cmap=plt.cm.viridis) 
    # Temperature
    temp = axs[1,0].contourf(x, y, t, alpha=0.5, cmap=plt.cm.jet) 
    # Density
    dens = axs[1,1].contourf(x, y, r, alpha=0.5, cmap=plt.cm.GnBu) 
    # Colorbars
    fig.colorbar(mag, ax=axs[0,0])
    fig.colorbar(pres, ax=axs[0,1])
    fig.colorbar(temp, ax=axs[1,0])
    fig.colorbar(dens, ax=axs[1,1])
    # Titles
    titles = [["Velocities", "Pressure"], ["Temperature", "Density"]]
    #axis 
    for i in range(2):
        for j in range(2):
            axs[i,j].set_xlabel(axis_x)
            axs[i,j].set_ylabel(axis_y)
            axs[i,j].set_title(titles[i][j])
    plt.tight_layout()
    plt.show()

def plotXY(x, y, u, v, h, r, t, p):
    genericPlot(x, y, u, v, h, r, t, p, axis_x='x', axis_y='y')

def plotXZ(x, z, u, w, h, r, t, p):
    genericPlot(x, z, u, w, h, r, t, p, axis_x='x', axis_y='z')

def plotYZ(y, z, v, w, h, r, t, p):
    genericPlot(y, z, v, w, h, r, t, p, axis_x='y', axis_y='z')

# h = lambda x, y: 20 * np.exp(-((x - 200) ** 2 + (y - 200) ** 2) / 5000)
# hx = lambda x, y: -2 / 5000 * (x - 200) * h(x, y)
# hy = lambda x, y: -2 / 5000 * (y - 200) * h(x, y)
# # h = lambda x, y: 0 * x + 0 * y
# # hx = lambda x, y: 0 * x + 0 * y
# # hy = lambda x, y: 0 * x + 0 * y
# H = 125
# zc = lambda x, y, z: z * (H - h(x, y)) / H + h(x, y)
# # Jacobian
# Hh = lambda x, y: H / (H - h(x, y)) ** 2
# #G = lambda x, y, z: (np.abs((Hh(x, y) * (z - H) * hx(x, y)) ** 2 + Hh(x, y) ** 2 + Hh(x, y) * (z - H) * hx(x, y))) ** (-.5)
# G = lambda x, y, z: (H - h(x, y)) / H
# G13 = lambda x, y, z: Hh(x, y) * (z - H) * hx(x, y)
# G23 = lambda x, y, z: Hh(x, y) * (z - H) * hy(x, y)


filename = 'output/higrad.npz'
#filename = "output/higrad_t_1_51x51x31x1001.npz"
#filename = "higrad_firetec_t_0.4.npz"
#filename = "higrad_firetec_t_0.3.npz" # Este está ok
data = np.load(filename)

X = data['X']
Y = data['Y']
Z = data['Z']
U = data['U']
V = data['V']
W = data['W']
R = data['R']
T = data['T']
P = data['P']
t = data['t']

# Plot
n = 70 # 83
i = U.shape[2] // 2 + 1
j = U.shape[1] // 2 + 1
k = 5#20#U.shape[3] // 4 
N = -1

Ge = G(X, Y, Z)
G13e = G13(X, Y, Z)
G23e = G23(X, Y, Z)

plot_ = 'xz'

#for n in [0]:
#for n in range(0, 201, 40): # last_ok.npz / last_periodic.npz
#for n in range(0, len(t), 5): # Vector implementation
for n in range(0, len(t)): # Last samples
    Ul = U[n]
    Vl = V[n]
    Wl = W[n]
    Rl = R[n]
    Tl = T[n]
    Pl = P[n]
    #Ol = G13e * Ul + G23e * Vl +  Wl / Ge
    print("t:", t[n])
    # Plot XY
    if plot_ == 'xz':
        xx = X[j,:,:]
        yy = Y[j,:, :]
        zz = Z[j,:, :]
        zz = zc(xx, yy, zz)
        uu = Ul[j,:,:]
        ww = Wl[j,:,:]
        tt = Tl[j,:,:]
        rr = Rl[j,:,:]
        pp = Pl[j,:,:]
        hh = h(X[j,:,k], Y[j-1,i-1,k])
        plotXZ(xx, zz, uu, ww, hh, rr, tt, pp)
    elif plot_ == 'yz':
        xx = X[:,i,:]
        yy = Y[:,i,:]
        zz = Z[:,i,:]
        zz = zc(xx, yy, zz)
        vv = Vl[:,i,:]
        ww = Wl[:,i,:]
        tt = Tl[:,i,:]
        rr = Rl[:,i,:]
        pp = Pl[:,i,:]
        hh = h(X[j-1,i-1,k], Y[:,i,k])
        plotYZ(yy, zz, vv, ww, hh, rr, tt, pp)
    elif plot_ == 'xy':
        xx = X[:,:,k]
        yy = Y[:,:,k]
        zz = Z[:,:,k]
        zz = zc(xx, yy, zz)
        uu = Ul[:,:,k]
        vv = Vl[:,:,k]
        ww = Wl[:,:,k]
        tt = Tl[:,:,k]
        rr = Rl[:,:,k]
        pp = Pl[:,:,k]
        hh = h(X[:,:,k], Y[:,:,k])
        plotXY(xx, yy, uu, vv, hh, rr, tt, pp)