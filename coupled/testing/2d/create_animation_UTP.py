import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio

temperature = True
filename = sys.argv[1]
gif_name = filename.replace('npz', 'gif')
video_name = filename.replace('npz', 'mp4')

video = True

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

x, y = np.meshgrid(x, y)

modU = np.sqrt(U**2 + V**2)

divU = (np.roll(U, -1, axis=1) - np.roll(U, 1, axis=1)) / (2 * dx) + (np.roll(V, -1, axis=2) - np.roll(V, 1, axis=2)) / (2 * dy)

filenames = []

# Plot
for n in range(t.shape[0]):
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4), dpi=200)
    axes[0].set_ylabel(r'$z$')
    axes[0].set_ylim(y_min, y_max)
    for i in range(len(axes)):
        axes[i].set_xlabel(r'$x$')
        axes[i].set_xlim(x_min, x_max)
    # First plot u and ||u||
    #levels = np.linspace(np.min(modU), np.max(modU), 11)
    p1 = axes[0].contourf(x, y, modU[n], cmap=plt.cm.viridis, alpha=.75, vmin=np.min(modU), vmax=np.max(modU)) #levels=levels, 
    #plt.plot([500, 200], [0, 200], 'k-')
    # plt.plot([(x_max + x_min) / 2, (x_max + x_min) / 2], [0, y_max], 'k-')
    fig.colorbar(p1, ax=axes[0])
    #axes[0].quiver(x[::2,::2], y[::2,::2], U[n,::2,::2], V[n,::2,::2])
    axes[0].streamplot(x, y, U[n], V[n], density=1.2, linewidth=.5, arrowsize=.3, color='k')
    axes[0].title.set_text(r'$\mathbf{u}, ||\mathbf{u}||_2$')
    
    # Second plot T or P
    # levels = np.linspace(np.min(T), np.max(T), 31)
    # levels = None
    # p2 = axes[1].contourf(x, y, T[n], cmap=plt.cm.jet, vmin=np.min(T), vmax=np.max(T))
    p2 = axes[1].contourf(x, y, T[n], cmap=plt.cm.jet)#, vmin=np.min(T), vmax=np.max(T))
    # plt.imshow(T[n], cmap=plt.cm.jet, origin="lower")
    #plt.plot([500, 500], [0, 200], 'k-')
    # plt.plot([(x_max + x_min) / 2, (x_max + x_min) / 2], [0, y_max], 'k-')
    axes[1].title.set_text(r'$T$')
    fig.colorbar(p2, ax=axes[1])

    # Third plot Y - div(u)
    # p3 = axes[2].contourf(x, y, divU[n], cmap=plt.cm.viridis)#, vmin=np.min(P), vmax=np.max(P))
    # axes[2].title.set_text(r'$\nabla\cdot\mathbf{u}$')
    # levels = np.linspace(np.min(Y), np.max(Y), 21)
    # p3 = axes[2].contourf(x, y, Y[n], levels=levels, cmap=plt.cm.Oranges, vmin=np.min(Y), vmax=np.max(Y))
    # plt.title(r'$Y$')
    # fig.colorbar(p3, ax=axes[2])
    p3 = axes[2].contourf(x, y, P[n], cmap=plt.cm.viridis)#, vmin=np.min(P), vmax=np.max(P))
    axes[2].title.set_text(r'$p$')
    fig.colorbar(p3, ax=axes[2])


    fig.tight_layout()
    name = f'{n}.png'
    filenames.append(name)
    plt.savefig(name)
    plt.close()

# build gif
# with imageio.get_writer(gif_name, mode='I') as writer:
if video:
    io_writer = imageio.get_writer(video_name)
else:
    io_writer = imageio.get_writer(gif_name, mode='I')

with io_writer as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)