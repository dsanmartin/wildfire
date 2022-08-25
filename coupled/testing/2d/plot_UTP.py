import numpy as np
import matplotlib.pyplot as plt
import sys

imshow = False
streamplot = True

filename = sys.argv[1]
# data = np.load('output/3.npz')

data = np.load(filename)

U = data['U']
V = data['V']
# U = data['Ud']
# V = data['Vd']
P = data['P']
T = data['T']
x = data['x']
y = data['y']
t = data['t']

dx = x[1] - x[0]
dy = y[1] - y[0]
dt = t[1] - t[0]

x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]
t_min, t_max = t[0], t[-1]

print("x_min:", x_min, "x_max:", x_max, "dx:", dx)
print("y_min:", y_min, "y_max:", y_max, "dy:", dy)
print("t_min:", t_min, "t_max:", t_max, "dt:", dt)

x, y = np.meshgrid(x, y)

modU = np.sqrt(U**2 + V**2)

ux = (np.roll(U, -1, axis=2) - np.roll(U, 1, axis=2)) / (2 * dx)
uy = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2 * dy)
vx = (np.roll(V, -1, axis=2) - np.roll(V, 1, axis=2)) / (2 * dx)
vy = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2 * dy)

px = (np.roll(P, -1, axis=2) - np.roll(P, 1, axis=2)) / (2 * dx)
py = (np.roll(P, -1, axis=1) - np.roll(P, 1, axis=1)) / (2 * dy)

divU =  ux + vy
curlU = vx - uy

axis = [r'$x$', r'$y$']

# Plot
for n in range(0, t.shape[0]):
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 8))

    if imshow:
        extent_ = [x_min, x_max, y_min, y_max]
        p1 = axes[0].imshow(U[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p2 = axes[1].imshow(V[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p3 = axes[2].imshow(modU[n], origin="lower", alpha=.5, cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p4 = axes[3].imshow(T[n], origin="lower", cmap=plt.cm.jet, extent=extent_, interpolation='none', aspect='auto')#, vmin=np.min(T), vmax=np.max(T))
        p5 = axes[4].imshow(P[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
    else:
        levels_1 = np.linspace(np.min(modU), np.max(modU), 11)
        levels_2 = np.linspace(np.min(divU), np.max(divU), 21)
        levels_3 = np.linspace(np.min(curlU), np.max(curlU), 21)
        p1 = axes[0].contourf(x, y, U[n], cmap=plt.cm.viridis)#, vmin=np.min(U), vmax=np.max(U))
        p2 = axes[1].contourf(x, y, V[n], cmap=plt.cm.viridis)#, vmin=np.min(V), vmax=np.max(V)) # levels=levels_2,
        p3 = axes[2].contourf(x, y, modU[n], cmap=plt.cm.viridis)#, vmin=np.min(modU), vmax=np.max(modU))
        p4 = axes[3].contourf(x, y, T[n], cmap=plt.cm.jet)#, vmin=np.min(T), vmax=np.max(T)) #levels=levels_3,
        p5 = axes[4].contourf(x, y, P[n], cmap=plt.cm.viridis, alpha=.5)#, vmin=np.min(P), vmax=np.max(P))

    if streamplot:
        axes[2].streamplot(x, y, U[n], V[n], density=1.2, linewidth=.35, arrowsize=.3, color='k')
        axes[4].streamplot(x, y, px[n], py[n], density=1, linewidth=.35, arrowsize=.3, color='k')
    else:
        axes[2].quiver(x[::2,::2], y[::2,::2], U[n,::2,::2], V[n,::2,::2])
        axes[4].quiver(x[::2,::2], y[::2,::2], px[n,::2,::2], py[n,::2,::2])
    
    # Titles
    axes[0].title.set_text('$u$')
    axes[1].title.set_text('$v$')
    axes[2].title.set_text(r'$\mathbf{u}, ||\mathbf{u}||_2$')
    axes[3].title.set_text(r'$T$')
    axes[4].title.set_text(r'$p, \nabla p$')

    # Colorbars
    fig.colorbar(p1, ax=axes[0])
    fig.colorbar(p2, ax=axes[1])
    fig.colorbar(p3, ax=axes[2])
    fig.colorbar(p4, ax=axes[3])
    fig.colorbar(p5, ax=axes[4])

    # Axis
    axes[-1].set_xlabel(axis[0])
    axes[-1].set_xlim(x_min, x_max)
    for i in range(len(axes)):
        axes[i].set_ylabel(axis[1])
        axes[i].set_ylim(y_min, y_max)

    fig.tight_layout()
    plt.show()