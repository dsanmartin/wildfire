import matplotlib.pyplot as plt

def plot_1D(x, y):
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

def plot_2D(x, y, z, cmap=plt.cm.jet):
    plt.figure(figsize=(12, 6))
    plt.contourf(x, y, z, cmap=cmap)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_scalar_field(fig, ax, x, y, z, cmap, z_bounds, ticks, title, label, alpha=1):
    z_min, z_max = z_bounds
    ax.contourf(x, y, z,cmap=cmap, vmin=z_min, vmax=z_max, alpha=alpha)
    # pi = axes[i].imshow(T[n],cmap=plt.cm.jet, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', vmin=np.min(T), vmax=np.max(T))
    if title is not None:
        ax.set_title(title)
    if label is not None:
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(z)
        m.set_clim(z_min, z_max)
        fig.colorbar(m, ax=ax, ticks=ticks, label=label)
    return None

def plot_vector_field(ax, x, y, u, v, streamplot=True, qs=1, density=1.2, linewidth=.5, arrowsize=.3, color='k'):
    if streamplot: 
        ax.streamplot(x, y, u, v, density=density, linewidth=linewidth, arrowsize=arrowsize, color=color)
    else: 
        ax.quiver(x[::qs,::qs], y[::qs,::qs], u[::qs,::qs], v[::qs,::qs])


def plot_ic(x, y, u, v, s, T, Y, plot_lims=None):
    # fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
    # u_plot = axes[0, 0].contourf(x, y, u, cmap=plt.cm.viridis)
    # v_plot = axes[0, 1].contourf(x, y, v, cmap=plt.cm.viridis)
    # T_plot = axes[1, 0].contourf(x, y, T, cmap=plt.cm.jet)
    # Y_plot = axes[1, 1].contourf(x, y, Y, cmap=plt.cm.Oranges)

    # axes[0, 0].set_title(r"Velocity component $u$")
    # axes[0, 1].set_title(r"Velocity component $v$")
    # axes[1, 0].set_title(r"Temperature $T$")
    # axes[1, 1].set_title(r"Fuel $Y$")

    # fig.colorbar(u_plot, ax=axes[0, 0], label=r'm s$^{-1}$')
    # fig.colorbar(v_plot, ax=axes[0, 1], label=r'm s$^{-1}$')
    # fig.colorbar(T_plot, ax=axes[1, 0], label=r'K')
    # fig.colorbar(Y_plot, ax=axes[1, 1], label=r'%')

    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 8))
    s_plot = axes[0].contourf(x, y, s, cmap=plt.cm.viridis)
    axes[0].streamplot(x, y, u, v, density=1.2, linewidth=.5, arrowsize=.3, color='k')
    u_plot = axes[1].contourf(x, y, u, cmap=plt.cm.viridis)
    v_plot = axes[2].contourf(x, y, v, cmap=plt.cm.viridis)
    T_plot = axes[3].contourf(x, y, T, cmap=plt.cm.jet)
    Y_plot = axes[4].contourf(x, y, Y, cmap=plt.cm.Oranges)

    axes[0].set_title(r"Speed $\|\mathbf{u}\|$")
    axes[1].set_title(r"Velocity component $u$")
    axes[2].set_title(r"Velocity component $v$")
    axes[3].set_title(r"Temperature $T$")
    axes[4].set_title(r"Fuel $Y$")

    fig.colorbar(s_plot, ax=axes[0], label=r'm s$^{-1}$')
    fig.colorbar(u_plot, ax=axes[1], label=r'm s$^{-1}$')
    fig.colorbar(v_plot, ax=axes[2], label=r'm s$^{-1}$')
    fig.colorbar(T_plot, ax=axes[3], label=r'K')
    fig.colorbar(Y_plot, ax=axes[4], label=r'%')

    axes[-1].set_xlabel(r"x")
    for i in range(len(axes)):
        axes[i].set_ylabel(r"z")

    if plot_lims is not None:
        for i in range(len(axes)):
            axes[i].set_xlim(plot_lims[0])
            axes[i].set_ylim(plot_lims[1])

    fig.tight_layout()
    # plt.gca().set_aspect('equal')
    plt.show()


def plot_grid_ibm(x, y, topo, cut_nodes, dead_nodes):
    # AA = np.zeros_like(Xm)
    # AA[cut_nodes[0], cut_nodes[1]] = 1
    # AA[dead_nodes[0], dead_nodes[1]] = -1
    plt.plot(x, topo)
    plt.scatter(x[dead_nodes[1]], y[dead_nodes[0]], marker='x', c='red')
    plt.scatter(x[cut_nodes[1]], y[cut_nodes[0]], marker='o', c='blue')
    for i in range(x.shape[0]):
        plt.plot([x[i], x[i]], [y[0], y[-1]], 'k-', linewidth=0.2)
    for j in range(y.shape[0]):
        plt.plot([x[0], x[-1]], [y[j], y[j]], 'k-', linewidth=0.2)
    #plt.contourf(Xm, Ym, AA)
    #plt.imshow(AA, origin="lower", interpolation=None)
    # plt.colorbar()
    plt.show()