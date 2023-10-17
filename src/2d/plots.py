import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import shutil

if shutil.which('latex'):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        'font.serif': 'Computer Modern',
        #"font.sans-serif": "Helvetica",
        "font.size": 16,
    })
else:
    print("LaTeX not installed! Using default font.")

black_cmap = ListedColormap(['black'])
vvariable = r'$z$ (m)'
hvariable = r'$x$ (m)'
streamplot = True
qs = 1

def get_variable(data: dict, variable: str, tn: int = None) -> tuple:
    """
    Extracts a variable from a data array and returns it along with its minimum and maximum values.

    Parameters
    ----------
    data : dict
        A dictionary with arrays of data.
    variable : str
        The name of the variable to extract.
    tn : int, optional
        The number of time steps to extract. If None, all time steps are extracted. Default is None.

    Returns
    -------
    tuple
        A tuple containing the extracted variable, its minimum value, and its maximum value.

    """
    phi = data[variable]
    phi = phi[:tn] if tn is not None else phi
    phi_min, phi_max = phi.min(), phi.max()
    return phi, phi_min, phi_max

def compute_first_derivative(phi, delta, axis):
    dphi = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2 * delta)
    if axis == 1: # Fix boundary in y - O(dy^2)
        dphi[:, 0, :] = (-3 * phi[:, 0, :] + 4 * phi[:, 1, :] - phi[:, 2, :]) / (2 * delta)
        dphi[:, -1,:] = (3 * phi[:, -1, :] - 4 * phi[:, -2,:] + phi[:, -3,:]) / (2 * delta)
    return dphi

def compute_derivatives(phi, dx, dy):
    dphi_x = compute_first_derivative(phi, dx, axis=2) # dphi/dx
    dphi_y = compute_first_derivative(phi, dy, axis=1) # dphi/dy
    return dphi_x, dphi_y

def load_data_for_plots(data_path, parameters_path, plots, tn=None):
    data = np.load(data_path)
    with open(parameters_path, 'rb') as fp:
        parameters = pickle.load(fp)
    # Domain
    x, y, t = data['x'], data['y'], data['t']
    # x_min, x_max = x[0], x[-1]
    # y_min, y_max = y[0], y[-1]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x, y = np.meshgrid(x, y)
    t = t[:tn] if tn is not None else t
    # Add each variable to the data_plots dictionary
    data_plots = {}
    if 'u' in plots:
        u, u_min, u_max = get_variable(data, 'u', tn)
        data_plots['u'] = {
            'data': u,
            'bounds': [u_min - 1, u_max + 1],
            'ticks': np.round(np.linspace(u_min, u_max, 5), 1)
        }
    if 'v' in plots:
        v, v_min, v_max = get_variable(data, 'v', tn)
        data_plots['v'] = {
            'data': v,
            'bounds': [v_min - 1, v_max + 1],
            'ticks': np.round(np.linspace(v_min, v_max, 5), 1)
        }
    if 'T' in plots:
        T, T_min, T_max = get_variable(data, 'T', tn)
        data_plots['T'] = {
            'data': T,
            'bounds': [T_min - 100, T_max + 100],
            'ticks': np.array(np.ceil(np.linspace(T_min, T_max, 5, dtype=int) / 100.0) * 100, dtype=int)
        }
    if 'Y' in plots:
        Y, Y_min, Y_max = get_variable(data, 'Y', tn)
        dead_nodes = parameters['dead_nodes']
        terrain = np.zeros_like(Y[0])
        terrain[:] = np.nan
        terrain[dead_nodes] = 1
        data_plots['Y'] = {
            'data': [Y, terrain],
            'bounds': [Y_min - 0.01, Y_max + 0.01],
            'ticks': np.linspace(Y_min, Y_max, 5)
        }
    if 'p' in plots:
        p, p_min, p_max = get_variable(data, 'p', tn)
        gradP = compute_derivatives(p, dx, dy)
        data_plots['p'] = {
            'data': [p, gradP],
            'bounds': [p_min - 1, p_max + 1],
            'ticks': np.round(np.linspace(p_min, p_max, 5), 1)
        }
    # Add vector field computations
    if 'modU' in plots:
        u, _, _ = get_variable(data, 'u', tn)
        v, _, _ = get_variable(data, 'v', tn)
        modU = np.sqrt(u ** 2 + v ** 2)
        modU_min, modU_max = modU.min(), modU.max()
        data_plots['modU'] = {
            'data': [modU, u, v],
            'bounds': [modU_min - 1, modU_max + 1],
            'ticks': np.round(np.linspace(modU_min, modU_max, 5), 1)
        }
    if 'divU' in plots:
        u, _, _ = get_variable(data, 'u', tn)
        v, _, _ = get_variable(data, 'v', tn)
        ux = compute_first_derivative(u, dx, axis=2)
        vy = compute_first_derivative(v, dy, axis=1)
        divU = ux + vy
        divU_min, divU_max = divU.min(), divU.max()
        data_plots['divU'] = {
            'data': divU,
            'bounds': [divU_min - 1, divU_max + 1],
            'ticks': np.round(np.linspace(divU_min, divU_max, 5), 1)
        }
    if 'curlU' in plots:
        u, _, _ = get_variable(data, 'u', tn)
        v, _, _ = get_variable(data, 'v', tn)
        vx = compute_first_derivative(v, dx, axis=2)
        uy = compute_first_derivative(u, dy, axis=1)
        curlU = vx - uy
        curlU_min, curlU_max = curlU.min(), curlU.max()
        data_plots['curlU'] = {
            'data': curlU,
            'bounds': [curlU_min - 1, curlU_max + 1],
            'ticks': np.round(np.linspace(curlU_min, curlU_max, 5), 1)
        }
    return x, y, t, data_plots

def plot_scalar_field(fig, ax, x, y, z, cmap, z_bounds, ticks, title, label, contour=True, alpha=1):
    z_min, z_max = z_bounds
    if contour:
        ax.contourf(x, y, z,cmap=cmap, vmin=z_min, vmax=z_max, alpha=alpha)
    else:
        ax.imshow(z, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', vmin=z_min, vmax=z_max, alpha=alpha)
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
    return None

# Plot one time step
def plot(n, t, x, y, plots, plot_lims, title=True, filename=None, dpi=200):
    n_plots = len(plots)
    x_min, x_max, y_min, y_max = plot_lims
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(12, n_plots * 2), dpi=dpi)
    axes[-1].set_xlabel(hvariable)
    axes[-1].set_xlim(x_min, x_max)
    for i in range(len(axes)):
        axes[i].set_ylabel(vvariable)
        axes[i].set_ylim(y_min, y_max)

    if title:
        fig.suptitle(r'Simulation at $t=%.1f$ s' % (t[n]))
        fig.subplots_adjust(top=0.88)

    i = 0

    if "u" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, plots['u']['data'][n], plt.cm.viridis, 
            plots['u']['bounds'], plots['u']['ticks'], r'Velocity component  $u$', r'm s$^{-1}$'
        )
        i += 1

    if "v" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, plots['v']['data'][n], plt.cm.viridis, 
            plots['v']['bounds'], plots['v']['ticks'], r'Velocity component  $v$', r'm s$^{-1}$'
        )
        i += 1

    if "modU" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, plots['modU']['data'][0][n], plt.cm.viridis, 
            plots['modU']['bounds'], plots['modU']['ticks'], r'Velocity $\mathbf{u}$, Speed $||\mathbf{u}||_2$', r'm s$^{-1}$', .8
        )
        plot_vector_field(
            axes[i], x, y, plots['modU']['data'][1][n], plots['modU']['data'][2][n], streamplot=streamplot, qs=qs
        )
        i += 1

    if "divU" in plots:
        # Plot divergence
        plot_scalar_field(
            fig, axes[i], x, y, plots['divU']['data'][n], plt.cm.viridis, 
            plots['divU']['bounds'], plots['divU']['ticks'], r'Divergence $\nabla\cdot\mathbf{u}$', r's$^{-1}$'
        )
        i += 1

    if "curlU" in plots:
        # Plot curl
        plot_scalar_field(
            fig, axes[i], x, y, plots['curlU']['data'][n], plt.cm.viridis, 
            plots['curlU']['bounds'], plots['curlU']['ticks'], r'Vorticity $\nabla\times\mathbf{u}$', r's$^{-1}$'
        )
        i += 1

    if "T" in plots:
        # Plot temperature
        plot_scalar_field(
            fig, axes[i], x, y, plots['T']['data'][n], plt.cm.jet, 
            plots['T']['bounds'], plots['T']['ticks'], r'Temperature $T$', r'K'
        )
        i += 1
    
    if "Y" in plots:
        # Plot fuel
        plot_scalar_field(
            fig, axes[i], x, y, plots['Y']['data'][0][n], plt.cm.Oranges, 
            plots['Y']['bounds'], plots['Y']['ticks'], r'Fuel $Y$', r'\%'
        )
        # Plot terrain
        plot_scalar_field(
            fig, axes[i], x, y, plots['Y']['data'][1], black_cmap, (0, 1), None, None, None
        )
        i += 1

    if "p" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, plots['p']['data'][0][n], plt.cm.viridis, 
            plots['p']['bounds'], plots['p']['ticks'], r'Pressure $p$', r"kg m$^{-1}s^{-2}$", .8
        )
        plot_vector_field(
            axes[i], x, y, plots['p']['data'][1][0][n], plots['p']['data'][1][1][n], streamplot=streamplot, qs=qs
        )
        i += 1
    
    fig.tight_layout()
    
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()

    return None

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