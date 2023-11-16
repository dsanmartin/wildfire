import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from derivatives import compute_gradient_plots, compute_curl_plots, compute_divergence_plots

vvariable = r'$z$ (m)' # Vertical variable
hvariable = r'$x$ (m)' # Horizontal variable
U_comp = ['modU', 'divU', 'curlU'] # Computation over velocity field

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
            'ticks': np.array(np.ceil(np.linspace(T_min, T_max, 5, dtype=int) / 100.0) * 100, dtype=int)
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
            'bounds': [modU_min - 1, modU_max + 1],
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
        The title of the plot. Default is None.
    label : str, optional
        The label for the colorbar. Default is None.
    plot_type : str, optional
        Type of plot. Options: 'contour', 'imshow', 'pcolormesh'. Default is 'imshow'.
    alpha : float, optional
        The alpha value for the plot. Default is 1.

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
            fig.colorbar(m, ax=ax, cax=cax, ticks=ticks, label=label) # fraction=0.046, pad=0.04)
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
        The size of the arrows in a streamplot or quiver plot. Default is .3.
    color : str, optional
        The color of the streamlines or arrows. Default is 'k' (black).

    Returns
    -------
    None
    """
    if streamplot: 
        ax.streamplot(x, y, u, v, density=density, linewidth=linewidth, arrowsize=arrowsize, color=color)
    else: 
        ax.quiver(x[::qs,::qs], y[::qs,::qs], u[::qs,::qs], v[::qs,::qs], scale=1, scale_units='xy', color=color)
    return None

def plot(n: int, t: np.ndarray, x: np.ndarray, y: np.ndarray, plots: dict, plot_lims: list, 
        title: bool = True, filename: str = None, dpi: int = 200, streamplot: bool = True, qs: int = 1) -> None:
    """
    Plot simulation data for time step `n`.

    Parameters
    ----------
    n : int
        Index of the time step to plot.
    t : numpy.ndarray (Nt)
        Time value of the time step to plot.
    x : numpy.ndarray (Nx) or (Ny, Nx)
        1D or 2D array with the x-coordinates of the grid.
    y : numpy.ndarray
        1D or 2D array with the y-coordinates of the grid.
    plots : dict
        Dictionary with the data to plot. The keys are the names of the variables to plot and the values are dictionaries with the following keys:
        - 'data': numpy.ndarray with the data to plot.
        - 'bounds': list with the minimum and maximum values of the data.
        - 'ticks': list with the tick values of the colorbar.
    plot_lims : list
        Tuple with the limits of the variable 'phi' (phi_min, phi_max).
    title : bool, optional
        Whether to add a title to the plot. Default is True. Title is the time value of the time step.
    filename : str, optional
        Name of the file to save the plot. If None, the plot is displayed instead of saved. Default is None.
    dpi : int, optional
        Resolution of the saved figure in dots per inch. Default is 200.
    streamplot : bool, optional
        Whether to plot streamlines for vector fields. Default is True.
    qs : int, optional
        Density of streamlines for vector fields. Default is 1.

    Returns
    -------
    None
    """
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
            fig, axes[i], x, y, plots['Y']['data'][1], ListedColormap(['black']), [0, 1], None, None, None
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
    
    fig.tight_layout() # Adjust spacing between subplots
    
    # Save or show plot
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', transparent=True)
        plt.close()
    else:
        plt.show()
    return None

def plot_initial_conditions(x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, s: np.ndarray, T: np.ndarray, Y: np.ndarray, plot_lims: list = None) -> None:
    """
    Plots the initial conditions of a wildfire simulation.

    Parameters
    ----------
    x : numpy.ndarray (Ny, Nx) 
        Array of x-coordinates.
    y : numpy.ndarray (Ny, Nx) 
        Array of y-coordinates.
    u : numpy.ndarray (Ny, Nx) 
        Array of velocity component u.
    v : numpy.ndarray (Ny, Nx)
        Array of velocity component v.
    s : numpy.ndarray (Ny, Nx)
        Array of speed.
    T : numpy.ndarray (Ny, Nx)
        Array of temperature.
    Y : numpy.ndarray (Ny, Nx)
        Array of fuel.
    plot_lims : list, optional
        List containing the limits of the plot, in the form [[xmin, xmax], [ymin, ymax]], by default None.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 8))

    # Axis labels
    axes[-1].set_xlabel('x')
    for i in range(len(axes)):
        axes[i].set_ylabel('y')

    # Set limits if given
    if plot_lims is not None:
        for i in range(len(axes)):
            axes[i].set_xlim(plot_lims[0])
            axes[i].set_ylim(plot_lims[1])

    # Plot speed
    plot_scalar_field(fig, axes[0], x, y, s, plt.cm.viridis, 
        [s.min(), s.max()], None, r'Speed $\|\mathbf{u}\|$', r'm s$^{-1}$') 
    
    # Plot velocity u component
    plot_scalar_field(fig, axes[1], x, y, u, plt.cm.viridis,
        [u.min(), u.max()], None, r'Velocity component $u$', r'm s$^{-1}$')
    
    # Plot velocity v component
    plot_scalar_field(fig, axes[2], x, y, v, plt.cm.viridis,
        [v.min(), v.max()], None, r'Velocity component $v$', r'm s$^{-1}$')
    
    # Plot temperature
    plot_scalar_field(fig, axes[3], x, y, T, plt.cm.jet,
        [T.min(), T.max()], None, r'Temperature $T$', r'K')
    
    # Plot fuel
    plot_scalar_field(fig, axes[4], x, y, Y, plt.cm.Oranges,
        [Y.min(), Y.max()], None, r'Fuel $Y$', r'\%')

    fig.tight_layout() # Adjust spacing between subplots
    plt.show()
    return None

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