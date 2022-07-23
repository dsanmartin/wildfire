import numpy as np
import matplotlib.pyplot as plt
import sys

temperature = True
filename = sys.argv[1]
# data = np.load('output/3.npz')

data = np.load(filename)

U = data['U']
V = data['V']
T = data['T']
Y = data['Y']
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

streamplot = True

# Plot
for n in range(0, t.shape[0], 2):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(9, 3))
    axes[0].set_ylabel(r'$z$')
    axes[0].set_ylim(y_min, y_max)
    for i in range(len(axes)):
        axes[i].set_xlabel(r'$x$')
        axes[i].set_xlim(x_min, x_max)
    # First plot u and ||u||
    levels = np.linspace(np.min(modU), np.max(modU), 11)
    p1 = axes[0].contourf(x, y, modU[n], levels=levels, cmap=plt.cm.viridis, alpha=.65, vmin=np.min(modU), vmax=np.max(modU))
    #plt.plot([500, 200], [0, 200], 'k-')
    # plt.plot([(x_max + x_min) / 2, (x_max + x_min) / 2], [0, y_max], 'k-')
    fig.colorbar(p1, ax=axes[0])

    if streamplot:
        axes[0].streamplot(x, y, U[n], V[n], density=1.2, linewidth=.5, arrowsize=.3, color='k')
    else:
        axes[0].quiver(x[::2,::2], y[::2,::2], U[n,::2,::2], V[n,::2,::2])
    
    # plt.streamplot(x, y, U[n], V[n])
    axes[0].title.set_text(r'$\mathbf{u}, ||\mathbf{u}||_2$')
    
    # Second plot T or P
    # levels = np.linspace(np.min(divU), np.max(divU), 21)
    p2 = axes[1].contourf(x, y, divU[n], cmap=plt.cm.viridis, vmin=np.min(divU), vmax=np.max(divU)) # levels=levels,
    axes[1].title.set_text(r'$\nabla\cdot\mathbf{u}$')
    fig.colorbar(p2, ax=axes[1])

    # Third plot Y - div(u)
    # p3 = axes[2].contourf(x, y, divU[n], cmap=plt.cm.viridis)#, vmin=np.min(P), vmax=np.max(P))
    # axes[2].title.set_text(r'$\nabla\cdot\mathbf{u}$')
    levels = np.linspace(np.min(curlU), np.max(curlU), 21)
    p3 = axes[2].contourf(x, y, curlU[n], levels=levels, cmap=plt.cm.viridis, vmin=np.min(curlU), vmax=np.max(curlU))
    axes[2].title.set_text(r'$\nabla\times\mathbf{u}$')

    fig.colorbar(p3, ax=axes[2])
    fig.tight_layout()
    plt.show()