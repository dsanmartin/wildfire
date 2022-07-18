import numpy as np
import matplotlib.pyplot as plt

#def generic_plot(x, y, Rf, Rw, Ts, Rg, U, V, Ro, axis_x='x', axis_y='y'):
def generic_plot(x, y, Rf, Rw, Ts, Rg, U, V, Tg, Ro, axis_x='x', axis_y='y'):
    # Plot setup
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(14, 7))
    # Set limits
    axs[0, 0].set_xlim(np.min(x), np.max(x))
    axs[0, 0].set_ylim(np.min(y), np.max(y))
    # Temperature
    tm = axs[0, 0].contourf(x, y, Ts, alpha=0.5, cmap=plt.cm.jet)
    
    # \rho_f 
    rf = axs[0, 1].contourf(x, y, Rf, alpha=0.5, cmap=plt.cm.Oranges) 
    # \rho_w
    rw = axs[0, 2].contourf(x, y, Rw, alpha=0.5, cmap=plt.cm.Blues)
    # u, v
    #uv = axs[1, 0].contourf(x, y, np.sqrt(U ** 2 + V** 2), alpha=0.5, cmap=plt.cm.jet)
    uv = axs[1, 0].contourf(x, y, Tg, alpha=0.5, cmap=plt.cm.jet)
    axs[1, 0].quiver(x, y, U, V)
    # \rho_g
    rg = axs[1, 1].contourf(x, y, Rg, alpha=0.5, cmap=plt.cm.Greens)
    # \rho_o
    ro = axs[1, 2].contourf(x, y, Ro, alpha=0.5, cmap=plt.cm.gray)
    # Colorbars
    fig.colorbar(tm, ax=axs[0, 0])
    fig.colorbar(rf, ax=axs[0, 1])
    fig.colorbar(rw, ax=axs[0, 2])
    fig.colorbar(uv, ax=axs[1, 0])
    fig.colorbar(rg, ax=axs[1, 1])
    fig.colorbar(ro, ax=axs[1, 2])
    # Titles
    titles = [
        [r"$T_{s}$", r"$\rho_{f}$", r"$\rho_{w}$"],
        [r"$T_{g}$", r"$\rho_{g}$", r"$\rho_{o}$"]
    ]
    #axis 
    for i in range(2):
        axs[i, 0].set_ylabel(axis_y)
        for j in range(3):
            axs[1, j].set_xlabel(axis_x)
            axs[i, j].set_title(titles[i][j])
    plt.tight_layout()
    plt.show()

def plot_scalar(x, y, z, vmin=0, vmax=1):
    plt.contourf(x, y, z, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

filename = 'output/higrad-firetec.npz'
data = np.load(filename)

X = data['X']
Y = data['Y']
Rf = data['Rf']
Rw = data['Rw']
Ts = data['Ts']
Rg = data['Rg']
Ro = data['Ro']
Tg = data['Tg']
U = data['U']
V = data['V']
t = data['t']

# ts_min, ts_max = np.min(Ts), np.max(Ts)
# rf_min, rf_max = np.min(Rf), np.max(Rf)
# rw_min, rw_max = np.min(Rw), np.max(Rw)

for n in range(0, len(t)):
    #generic_plot(X, Y, Rf[n], Rw[n], Ts[n], Rg[n], U[n], V[n], Ro[n], axis_x='x', axis_y='y')
    generic_plot(X, Y, Rf[n], Rw[n], Ts[n], Rg[n], U[n], V[n], Tg[n], Ro[n], axis_x='x', axis_y='y')
    # generic_plot(X, Y, Rf[n], Rw[n], Ts[n], Rg[n], U, V, Ro[n], axis_x='x', axis_y='y')

