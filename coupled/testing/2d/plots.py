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

def plot_ic(x, y, u, v, T, Y):
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

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 8))
    u_plot = axes[0].contourf(x, y, u, cmap=plt.cm.viridis)
    v_plot = axes[1].contourf(x, y, v, cmap=plt.cm.viridis)
    T_plot = axes[2].contourf(x, y, T, cmap=plt.cm.jet)
    Y_plot = axes[3].contourf(x, y, Y, cmap=plt.cm.Oranges)

    axes[0].set_title(r"Velocity component $u$")
    axes[1].set_title(r"Velocity component $v$")
    axes[2].set_title(r"Temperature $T$")
    axes[3].set_title(r"Fuel $Y$")

    fig.colorbar(u_plot, ax=axes[0], label=r'm s$^{-1}$')
    fig.colorbar(v_plot, ax=axes[1], label=r'm s$^{-1}$')
    fig.colorbar(T_plot, ax=axes[2], label=r'K')
    fig.colorbar(Y_plot, ax=axes[3], label=r'%')

    axes[-1].set_xlabel(r"x")
    for i in range(len(axes)):
        axes[i].set_ylabel(r"z")

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