import numpy as np
import matplotlib.pyplot as plt
import sys

temperature = True
imshow = True
streamplot = True

filename = sys.argv[1]
# data = np.load('output/3.npz')

data = np.load(filename)

U = data['U']
V = data['V']
# U = data['Ud']
# V = data['Vd']
P = data['P']
x = data['x']
y = data['y']
t = data['t']

dx = x[1] - x[0]
dy = y[1] - y[0]

x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]

print("x_min:", x_min, "x_max:", x_max, "dx: ", dx)
print("y_min:", y_min, "y_max:", y_max, "dy: ", dy)

x, y = np.meshgrid(x, y)

modU = np.sqrt(U**2 + V**2)

ux = (np.roll(U, -1, axis=2) - np.roll(U, 1, axis=2)) / (2 * dx)
uy = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2 * dy)
vx = (np.roll(V, -1, axis=2) - np.roll(V, 1, axis=2)) / (2 * dx)
vy = (np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)) / (2 * dy)

divU =  ux + vy
curlU = vx - uy

axis = [r'$x$', r'$y$']

# Plot
for n in range(0, t.shape[0], 1):
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 8))

    # First plot u and ||u||

    if imshow:
        extent_ = [x_min, x_max, y_min, y_max]
        p1 = axes[0].imshow(U[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p2 = axes[1].imshow(V[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p3 = axes[2].imshow(modU[n], origin="lower", alpha=.5, cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p4 = axes[3].imshow(divU[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
        p5 = axes[4].imshow(curlU[n], origin="lower", cmap=plt.cm.viridis, extent=extent_, interpolation='none', aspect='auto')
    else:
        levels_1 = np.linspace(np.min(modU), np.max(modU), 11)
        levels_2 = np.linspace(np.min(divU), np.max(divU), 21)
        levels_3 = np.linspace(np.min(curlU), np.max(curlU), 21)
        p1 = axes[0].contourf(x, y, modU[n], cmap=plt.cm.viridis, alpha=.65, levels=levels_1, vmin=np.min(modU), vmax=np.max(modU))
        p2 = axes[1].contourf(x, y, divU[n], cmap=plt.cm.viridis, vmin=np.min(divU), vmax=np.max(divU)) # levels=levels_2,
        p3 = axes[2].contourf(x, y, curlU[n], cmap=plt.cm.viridis, levels=levels_3, vmin=np.min(curlU), vmax=np.max(curlU))
        #plt.plot([500, 200], [0, 200], 'k-')
        # plt.plot([(x_max + x_min) / 2, (x_max + x_min) / 2], [0, y_max], 'k-')


    if streamplot:
        axes[2].streamplot(x, y, U[n], V[n], density=1.2, linewidth=.5, arrowsize=.3, color='k')
    else:
        axes[2].quiver(x[::2,::2], y[::2,::2], U[n,::2,::2], V[n,::2,::2])
    
    # Titles
    axes[0].title.set_text('$u$')
    axes[1].title.set_text('$v$')
    axes[2].title.set_text(r'$\mathbf{u}, ||\mathbf{u}||_2$')
    axes[3].title.set_text(r'$\nabla\cdot\mathbf{u}$')
    axes[4].title.set_text(r'$\nabla\times\mathbf{u}$')

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