import matplotlib.pyplot as plt

def plot_2d(x, y, z, cmap=plt.cm.jet):
    plt.contourf(x, y, z, cmap=cmap)
    plt.colorbar()
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