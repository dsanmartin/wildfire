import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

imshow = False
streamplot = True
celsius = False

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

# lim = 4001
# U = U[:lim]
# V = V[:lim]
# T = T[:lim]
# P = P[:lim]
# t = t[:lim]

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

Tx = (np.roll(T, -1, axis=2) - np.roll(T, 1, axis=2)) / (2 * dx)
Ty = (np.roll(T, -1, axis=1) - np.roll(T, 1, axis=1)) / (2 * dy)

divU =  ux + vy
curlU = vx - uy

if celsius:
    T -= 273.15

T_min, T_max = np.min(T), np.max(T)

axis = [r'$x$', r'$y$']

qs = 4 # Quiver sample

# Plot
for n in range(0, t.shape[0], 10):
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 8))
    # fig.suptitle('Simulation at ' + r'$t=%.1f$' % (t[n]))
    # fig.subplots_adjust(top=0.88)

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
        levels_4 = np.linspace(np.min(T), np.max(T), 11)
        # colormap = plt.cm.jet #or any other colormap
        normalize = matplotlib.colors.Normalize(vmin=np.min(T), vmax=np.max(T))
        p1 = axes[0].contourf(x, y, U[n], cmap=plt.cm.viridis)#, vmin=np.min(U), vmax=np.max(U))
        p2 = axes[1].contourf(x, y, V[n], cmap=plt.cm.viridis)#, vmin=np.min(V), vmax=np.max(V)) # levels=levels_2,
        p3 = axes[2].contourf(x, y, modU[n], cmap=plt.cm.viridis)#, vmin=np.min(modU), vmax=np.max(modU))
        p4 = axes[3].contourf(x, y, T[n], cmap=plt.cm.jet)#, norm=normalize)#, vmin=np.min(T), vmax=np.max(T)) #levels=levels_3,
        p5 = axes[4].contourf(x, y, P[n], cmap=plt.cm.viridis, alpha=.5)#, vmin=np.min(P), vmax=np.max(P))

    if streamplot:
        axes[2].streamplot(x, y, U[n], V[n], density=1.2, linewidth=.35, arrowsize=.3, color='k')
        # axes[3].streamplot(x, y, Tx[n], Ty[n], density=1.2, linewidth=.35, arrowsize=.3, color='k')
        axes[4].streamplot(x, y, px[n], py[n], density=1, linewidth=.35, arrowsize=.3, color='k')
    else:
        axes[2].quiver(x[::qs,::qs], y[::qs,::qs], U[n,::qs,::qs], V[n,::qs,::qs])
        axes[4].quiver(x[::qs,::qs], y[::qs,::qs], px[n,::qs,::qs], py[n,::qs,::qs])
    
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

    # # Temperature
    # pos = cbar.ax.get_position()
    # ax1 = cbar.ax
    # ax1.set_aspect('auto')

    # # create a second axis and specify ticks based on the relation between the first axis and second aces
    # ax2 = ax1.twinx()
    # ax2.set_ylim([T_min, T_max])
    # newlabel = [300,325,350,375,400,425,450] # labels of the ticklabels: the position in the new axis
    # k2degc = lambda t: t-273.15 # convert function: from Kelvin to Degree Celsius
    # newpos   = [k2degc(t) for t in newlabel]   # position of the ticklabels in the old axis
    # ax2.set_yticks(newpos)
    # ax2.set_yticklabels(newlabel)

    # # resize the colorbar
    # pos.x0 += 0.10
    # pos.x1 += 0.10

    # # arrange and adjust the position of each axis, ticks, and ticklabels
    # ax1.set_position(pos)
    # ax2.set_position(pos)
    # ax1.yaxis.set_ticks_position('right') # set the position of the first axis to right
    # ax1.yaxis.set_label_position('right') # set the position of the fitst axis to right
    # ax1.set_ylabel(u'Temperature [\u2103]')
    # ax2.yaxis.set_ticks_position('left') # set the position of the second axis to right
    # ax2.yaxis.set_label_position('left') # set the position of the second axis to right
    # # ax2.spines['left'].set_position(('outward', 50)) # adjust the position of the second axis
    # ax2.set_ylabel('Temperature [K]')

    # Axis
    axes[-1].set_xlabel(axis[0])
    axes[-1].set_xlim(x_min, x_max)
    for i in range(len(axes)):
        axes[i].set_ylabel(axis[1])
        axes[i].set_ylim(y_min, y_max)

    fig.tight_layout()
    plt.show()