import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from derivatives import compute_gradient_plots, compute_curl_plots, compute_divergence_plots

# Check if LaTeX is installed
if shutil.which('latex'):
    # Use LaTeX for rendering
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        'font.serif': 'Computer Modern',
        #"font.sans-serif": "Helvetica",
        "font.size": 16,
    })
else:
    print("LaTeX not installed! Using default font.")

black_cmap = ListedColormap(['black']) # For plotting terrain
vvariable = r'$z$ (m)' # Vertical variable
hvariable = r'$x$ (m)' # Horizontal variable
U_comp = ['modU', 'divU', 'curlU'] # Computation over velocity field

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
    phi = data[variable] # Extract variable
    phi = phi[:tn] if tn is not None else phi # Extract time steps
    phi_min, phi_max = phi.min(), phi.max() # Compute min and max
    return phi, phi_min, phi_max # Return variable, min, and max

def load_data_for_plots(data_path: str, parameters_path: str, plots: list, tn: int = None) -> tuple:
    """
    Load data from a given path and return arrays with the domain and a dictionary with the data to be plotted.

    Parameters
    ----------
    data_path : str
        Path to the data file.
    parameters_path : str
        Path to the parameters file.
    plots : list
        List of variables to be plotted.
    tn : int, optional
        Number of time steps to be plotted. If None, all time steps are plotted.

    Returns
    -------
    tuple
        Tuple containing the x, y, t arrays and a dictionary with the data to be plotted.

    """
    # Load data
    data = np.load(data_path) 
    # Load parameters
    with open(parameters_path, 'rb') as fp:
        parameters = pickle.load(fp)
    # Get the domain
    x, y, t = data['x'], data['y'], data['t']
    # Get the spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x, y = np.meshgrid(x, y) # Create meshgrid for plotting
    t = t[:tn] if tn is not None else t
    # Add each variable 'phi' to the data_plots dictionary
    # The format is {'phi': {'data': [phi, ...], 'bounds': [phi_min, phi_max], 'ticks': phi_ticks}, ...}
    # Some variables need extra data. For instance 'Y' needs the terrain, so we add it as a list in 'data' key.
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
            'bounds': [T_min - 100 * 0, T_max + 100],
            'ticks': np.array(np.ceil(np.linspace(T_min, T_max + 1, 5, dtype=int) / 100.0) * 100, dtype=int)
        }
    if 'Y' in plots:
        Y, Y_min, Y_max = get_variable(data, 'Y', tn)
        # Get IBM nodes
        dead_nodes = parameters['dead_nodes']
        terrain = np.zeros_like(Y[0])
        terrain[:] = np.nan
        terrain[dead_nodes] = 1
        data_plots['Y'] = {
            'data': [Y, terrain],
            'bounds': [Y_min, Y_max], #[Y_min - 0.01, Y_max + 0.01],
            'ticks': np.linspace(Y_min, Y_max, 5)
        }
    if 'p' in plots:
        p, p_min, p_max = get_variable(data, 'p', tn)
        gradP = compute_gradient_plots(p, dx, dy)
        data_plots['p'] = {
            'data': [p, gradP],
            'bounds': [p_min - 1, p_max + 1],
            'ticks': np.round(np.linspace(p_min, p_max, 5), 1)
        }

    # If velocity computations are needed, add them to the plots
    if any([V in plots for V in U_comp]):
        # Get velocity components
        u, _, _ = get_variable(data, 'u', tn)
        v, _, _ = get_variable(data, 'v', tn)
        U = np.array([u, v])

    # Add vector field computations
    if 'modU' in plots:
        # Compute speed
        modU = np.sqrt(u**2 + v**2)
        modU_min, modU_max = modU.min(), modU.max()
        data_plots['modU'] = {
            'data': [modU, u, v],
            'bounds': [modU_min, modU_max],
            'ticks': np.round(np.linspace(modU_min, modU_max, 5), 1)
        }
    if 'divU' in plots:
        divU = compute_divergence_plots(U, dx, dy)
        divU_min, divU_max = divU.min(), divU.max()
        data_plots['divU'] = {
            'data': divU,
            'bounds': [divU_min - 1, divU_max + 1],
            'ticks': np.round(np.linspace(divU_min, divU_max, 5), 1)
        }
    if 'curlU' in plots:
        curlU = compute_curl_plots(U, dx, dy)
        curlU_min, curlU_max = curlU.min(), curlU.max()
        data_plots['curlU'] = {
            'data': curlU,
            'bounds': [curlU_min - 1, curlU_max + 1],
            'ticks': np.round(np.linspace(curlU_min, curlU_max, 5), 1)
        }
    return x, y, t, data_plots


def plot_scalar_field(fig: plt.Figure, ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, cmap: plt.cm, 
        z_bounds: list, ticks: list, title: str = None, label: str = None, alpha: float = 1, plot_type: str = 'imshow') -> None:
    """
    Plot a scalar field as a contour plot or an image.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to plot on.
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x : numpy.ndarray (Nx) or (Ny, Nx)
        The x-coordinates of the scalar field.
    y : numpy.ndarray (Ny) or (Ny, Nx)
        The y-coordinates of the scalar field.
    z : numpy.ndarray (Ny, Nx)
        The scalar field to plot.
    cmap : matplotlib.colors.Colormap
        The colormap to use.
    z_bounds : list
        The minimum and maximum values of the scalar field to plot.
    ticks : list
        The tick locations for the colorbar.
    title : str, optional
        The title of the plot.
    label : str, optional
        The label for the colorbar.
    contour : bool, optional
        If True, plot the scalar field as a contour plot. If False, plot as an image.
    alpha : float, optional
        The alpha value for the plot.

    Returns
    -------
    None
    """
    z_min, z_max = z_bounds
    if plot_type == 'contour':
        ax.contourf(x, y, z,cmap=cmap, vmin=z_min, vmax=z_max, alpha=alpha, antialised=True)
    elif plot_type == 'imshow':
        ax.imshow(z, cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', 
            interpolation='bilinear', vmin=z_min, vmax=z_max, alpha=alpha)
        im_ratio = z.shape[0] / z.shape[1]
    elif plot_type == 'pcolormesh':
        ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max, alpha=alpha)
    if title is not None:
        ax.set_title(title)
    if label is not None:
        m = plt.cm.ScalarMappable(cmap=cmap)
        m.set_array(z)
        m.set_clim(z_min, z_max)
        if plot_type == 'imshow':
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="1%", pad=0.1)
            fig.colorbar(m, ax=ax, cax=cax, ticks=ticks) # fraction=0.046, pad=0.04)
        else:
            fig.colorbar(m, ax=ax, ticks=ticks, label=label)
    return None

def plot_vector_field(ax: plt.Axes, x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, 
        streamplot: bool = True, qs: int = 1, density: float = 1.2, linewidth: float = .5, arrowsize: float = .3, color: str = 'k') -> None:
    """
    Plot a vector field on a given axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the vector field.
    x : numpy.ndarray
        The x-coordinates of the grid points.
    y : numpy.ndarray
        The y-coordinates of the grid points.
    u : numpy.ndarray
        The x-components of the vector field.
    v : numpy.ndarray
        The y-components of the vector field.
    streamplot : bool, optional
        If True, plot the vector field using a streamplot. Otherwise, plot using a quiver plot.
    qs : int, optional
        The stride of the quiver plot. Only used if `streamplot` is False.
    density : float, optional
        Controls the closeness of streamlines in a streamplot. Only used if `streamplot` is True.
    linewidth : float, optional
        The linewidth of the streamlines in a streamplot. Only used if `streamplot` is True.
    arrowsize : float, optional
        The size of the arrows in a streamplot or quiver plot.
    color : str, optional
        The color of the streamlines or arrows.

    Returns
    -------
    None
    """
    if streamplot: 
        ax.streamplot(x, y, u, v, density=density, linewidth=linewidth, arrowsize=arrowsize, color=color)
    else: 
        ax.quiver(x[::qs,::qs], y[::qs,::qs], u[::qs,::qs], v[::qs,::qs], scale=1, scale_units='xy', color=color)
    return None

# Plot one time step
def plot(n, t, x, y, plots, plot_lims, title=True, filename=None, dpi=200, streamplot=True, qs=1):
    n_plots = len(plots)
    x_min, x_max, y_min, y_max = plot_lims
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(12, n_plots * 2))#, dpi=dpi)
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
        # Plot speed
        plot_scalar_field(
            fig, axes[i], x, y, plots['modU']['data'][0][n], plt.cm.viridis, 
            plots['modU']['bounds'], plots['modU']['ticks'], r'Velocity $\mathbf{u}$, Speed $||\mathbf{u}||_2$', r'm s$^{-1}$', .8
        )
        # Plot velocity
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
        # Plot terrain (IBM nodes)
        plot_scalar_field(
            fig, axes[i], x, y, plots['Y']['data'][1], black_cmap, (0, 1), None, None, None
        )
        i += 1

    if "p" in plots:
        # Plot pressure
        plot_scalar_field(
            fig, axes[i], x, y, plots['p']['data'][0][n], plt.cm.viridis, 
            plots['p']['bounds'], plots['p']['ticks'], r'Pressure $p$', r"kg m$^{-1}s^{-2}$", .8
        )
        # Plot pressure gradient
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