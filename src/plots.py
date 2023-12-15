import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap
from derivatives import compute_gradient_plots, compute_curl_plots, compute_divergence_plots

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
    # Create meshgrid for plotting
    # x, y = np.meshgrid(x, y) 
    if 'z' in data:
        z = data['z']
        dz = z[1] - z[0]
        # x, y, z = np.meshgrid(x, y, z) # Create meshgrid for plotting
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
    if 'w' in plots:
        w, w_min, w_max = get_variable(data, 'w', tn)
        data_plots['w'] = {
            'data': w,
            'bounds': [w_min - 1, w_max + 1],
            'ticks': np.round(np.linspace(w_min, w_max, 5), 1)
        }
    if 'T' in plots:
        T, T_min, T_max = get_variable(data, 'T', tn)
        T_mean = (T_min + T_max) / 2
        data_plots['T'] = {
            'data': T,
            'bounds': [T_min - 100 * 0, T_max + 100 * 0],
            'ticks': np.array(np.ceil(np.linspace(T_min, T_max, 5, dtype=int) / 100.0) * 100, dtype=int)
            # 'ticks': [300, 475, 650, 825, 1000]
            # 'ticks': np.round(np.linspace(T_min, T_max, 5))
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
        if 'z' in data:
            w, _, _ = get_variable(data, 'w', tn)
            U = np.array([u, v, w])
        
    # Add vector field computations
    if 'modU' in plots:
        # Compute speed
        if 'z' in data:
            modU = np.sqrt(u**2 + v**2 + w**2)
            data_ = [modU, u, v, w]
        else:
            modU = np.sqrt(u**2 + v**2)
            data_ = [modU, u, v]
        modU_min, modU_max = modU.min(), modU.max()
        data_plots['modU'] = {
            'data': data_,
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
    if 'z' in data:
        domain = (x, y, z, t)
    else:
        domain = (x, y, t)
    return domain, data_plots

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

def plot_scalar_field_3D(fig: plt.Figure, ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, f: np.ndarray, cmap: plt.cm, 
        f_bounds: list, ticks: list, title: str = None, label: str = None, alpha: float = 1, plot_type: str = 'imshow') -> None:
    # X, Y, Z = np.meshgrid(x, y, z)
    f_min, f_max = f_bounds
    # levels = np.linspace(f_min, f_max, 100)  #(z_min,z_max,number of contour),
    X, Y, Z = x, y, z
    # f_min, f_max = f.min(), f.max()
    # print(f_min, f_max)
    # label = 'Temperature'
    # m = plt.cm.ScalarMappable(cmap=cmap)
    # m.set_array(f)
    # m.set_clim(f_min, f_max)
    # # Creating plot
    m = ax.scatter(X, Y, Z, c=f, alpha=0.5, marker='o', cmap=cmap)
    fig.colorbar(m, ax=ax, ticks=ticks, label=label)
    # print(x.shape, y.shape, z.shape, f.shape)
    # ax.plot_surface(x[0, :, 0], y[:, 0, 0], f[:, :, 1], cmap=cmap, vmin=f_min, vmax=f_max, alpha=alpha)
    # ax.contourf3D(x[0, :, 0], y[:, 0, 0], f[:, :, 1], cmap=cmap, vmin=f_bounds[0], vmax=f_bounds[1], alpha=alpha)#, levels=levels)
    # ax.contourf(x[:,:,0], y[:,:,0], f[:,:,2], 1+levels, cmap=cmap, vmin=f_bounds[0], vmax=f_bounds[1], alpha=.75)#, levels=[0, 1])
    # ax.imshow(f[:,:,0], cmap=cmap, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', interpolation='bilinear', vmin=f_min, vmax=f_max, alpha=alpha)
    # ax.plot_surface(x[0, :, 0], z[0, 0, :], f[f.shape[1] // 2, :, :], cmap=cmap, vmin=f_bounds[0], vmax=f_bounds[1], alpha=alpha)


    return None

def plot_vector_field(ax: plt.Axes, x: np.ndarray, y: np.ndarray, u: np.ndarray, v: np.ndarray, 
        streamplot: bool = True, qs: int = 1, density: float = .6, linewidth: float = .5, arrowsize: float = .3, color: str = 'k') -> None:
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

def plot_vector_field_3D(ax: plt.Axes, x: np.ndarray, y: np.ndarray, z: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
        streamplot: bool = False, qs: int = 1, density: float = .6, linewidth: float = .5, arrowsize: float = .3, color: str = 'k') -> None:
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
        X = x[::qs,::qs,::qs]
        Y = y[::qs,::qs,::qs]
        Z = z[::qs,::qs,::qs]
        U = u[::qs,::qs,::qs]
        V = v[::qs,::qs,::qs]
        W = w[::qs,::qs,::qs]
        ax.quiver(X, Y, Z, U, V, W, color=color, normalize=True, length=1)
    return None

def plot_2D(n: int, domain: tuple, plots: dict, plot_lims: list, visualization: str = 'vertical',
        title: bool = True, filename: str = None, dpi: int = 200, streamplot: bool = True, qs: int = 1, density: float = 1) -> None:
    """
    Plot simulation data for time step `n`.

    Parameters
    ----------
    n : int
        Index of the time step to plot.
    domain: tuple
        Tuple with the domain (x, y, t) or (x, y, z, t).
        Each element is a numpy.ndarray with the coordinates of the domain.
    plots : dict
        Dictionary with the data to plot. The keys are the names of the variables to plot and the values are dictionaries with the following keys:
        - 'data': numpy.ndarray with the data to plot.
        - 'bounds': list with the minimum and maximum values of the data.
        - 'ticks': list with the tick values of the colorbar.
    plot_lims : list
        List with plot's limits. [x_min, x_max, y_min, y_max] or [x_min, x_max, y_min, y_max, z_min, z_max].
    visualization : str, optional
        Visualization type. Options: 'vertical, 'horizontal' or 'longitudinal'. Default is 'vertical'.
        Only for 3D simulations..
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
    # figsize = (n_plots * 2, 12)
    if visualization == 'horizontal':
        figsize = (n_plots * 4, 4)
        fig, axes = plt.subplots(1, n_plots, sharey=True, figsize=figsize)#, dpi=dpi)
        # figsize = (n_plots * 2, 12)
    else:
        figsize = (12, n_plots * 2)
        fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=figsize)#, dpi=dpi)
    
    
    if len(domain) == 3:
        x_min, x_max, y_min, y_max = plot_lims
        x, y, t = domain
        h_min, h_max = x_min, x_max
        v_min, v_max = y_min, y_max
        plot_title = r'Simulation at $t=%.1f$ s' % (t[n])
        h_variable = 'x'
        v_variable = 'z'
    elif len(domain) == 4:
        x_min, x_max, y_min, y_max, z_min, z_max = plot_lims
        x, y, z, t = domain
        if visualization == 'vertical':
            h_min, h_max = x_min, x_max
            v_min, v_max = z_min, z_max
            y_j = y.shape[0] // 2 # Show vertical slice at the middle
            plot_title = r'Simulation at $y=%.1f$ m and $t=%.1f$ s' % (y[y_j], t[n])
            h_variable = 'x'
            v_variable = 'z'
            y = z # Plot in z-axis
        elif visualization == 'horizontal':
            h_min, h_max = x_min, x_max
            v_min, v_max = y_min, y_max
            z_j = 3 #z.shape[0] // 2 # Show horizontal slice at the middle
            plot_title = r'Simulation at $z=%.1f$ m and $t=%.1f$ s' % (z[z_j], t[n])
            h_variable = 'x'
            v_variable = 'y'
        elif visualization == 'longitudinal':
            h_min, h_max = y_min, y_max
            v_min, v_max = z_min, z_max
            x_i = x.shape[0] // 2
            plot_title = r'Simulation at $x=%.1f$ m and $t=%.1f$ s' % (x[x_i], t[n])
            h_variable = 'y'
            v_variable = 'z'
            x = y


    # Just to keep the indexing 
    if n_plots == 1: 
        axes = [axes]

    # Set axes limits and labels
    if visualization == 'horizontal':
        axes[-1].set_ylabel(r'${}$ (m)'.format(v_variable))
        axes[-1].set_ylim(v_min, v_max)
        for i in range(len(axes)):            
            axes[i].set_xlabel(r'${}$ (m)'.format(h_variable))
            axes[i].set_xlim(h_min, h_max)
            
    elif visualization == 'vertical':
        axes[-1].set_xlabel(r'${}$ (m)'.format(h_variable))
        axes[-1].set_xlim(h_min, h_max)
        for i in range(len(axes)):
            axes[i].set_ylabel(r'${}$ (m)'.format(v_variable))
            axes[i].set_ylim(v_min, v_max)
    
    
        

    if title:
        fig.suptitle(plot_title)
        fig.subplots_adjust(top=0.88)

    i = 0

    if "u" in plots: # Plot velocity component u
        u = plots['u']['data'][n]
        if len(domain) == 4:
            if visualization == 'vertical':
                u = u[y_j].T
            elif visualization == 'horizontal':
                u = u[:,:,z_j]
            elif visualization == 'longitudinal':
                u = u[:,x_i,:].T
        plot_scalar_field(
            fig, axes[i], x, y, u, plt.cm.viridis, 
            plots['u']['bounds'], plots['u']['ticks'], r'Velocity component  $u$', r'm s$^{-1}$'
        )
        i += 1

    if "v" in plots: # Plot velocity component v
        v = plots['v']['data'][n]
        if len(domain) == 4:
            if visualization == 'vertical':
                v = v[y_j].T
            elif visualization == 'horizontal':
                v = v[:,:,z_j]
            elif visualization == 'longitudinal':
                v = v[:,x_i,:].T
        plot_scalar_field(
            fig, axes[i], x, y, v, plt.cm.viridis, 
            plots['v']['bounds'], plots['v']['ticks'], r'Velocity component  $v$', r'm s$^{-1}$'
        )
        i += 1
    
    if "w" in plots: # Plot velocity component w
        w = plots['w']['data'][n]
        if len(domain) == 4:
            if visualization == 'vertical':
                w = w[y_j].T
            elif visualization == 'horizontal':
                w = w[:,:,z_j]
            elif visualization == 'longitudinal':
                w = w[:,x_i,:].T
        plot_scalar_field(
            fig, axes[i], x, y, w, plt.cm.viridis, 
            plots['w']['bounds'], plots['w']['ticks'], r'Velocity component  $w$', r'm s$^{-1}$'
        )
        i += 1

    if "modU" in plots: # Plot speed
        modU = plots['modU']['data'][0][n]
        u, v = plots['modU']['data'][1][n], plots['modU']['data'][2][n]
        if len(domain) == 4:
            w = plots['modU']['data'][3][n]
            if visualization == 'vertical': 
                modU = modU[y_j].T
                u = u[y_j].T
                w = w[y_j].T
                v = w
            elif visualization == 'horizontal':
                modU = modU[:,:,z_j]
                u = u[:,:,z_j]
                w = w[:,:,z_j]
                v = w
            elif visualization == 'longitudinal':
                modU = modU[:,x_i,:].T
                u = u[:,x_i,:].T
                v = v[:,x_i,:].T
                u = v
        plot_scalar_field(
            fig, axes[i], x, y, modU, plt.cm.viridis, 
            plots['modU']['bounds'], plots['modU']['ticks'], r'Velocity $\mathbf{u}$, Speed $||\mathbf{u}||_2$', r'm s$^{-1}$', .8
        )
        # Plot velocity
        plot_vector_field(
            axes[i], x, y, u, v, streamplot=streamplot, qs=qs, density=density
        )
        i += 1

    if "divU" in plots: # Plot divergence
        divU = plots['divU']['data'][n]
        if len(domain) == 4:
            if visualization == 'vertical':
                divU = divU[y_j].T  
            elif visualization == 'horizontal':
                divU = divU[:,:,z_j]
            elif visualization == 'longitudinal':
                divU = divU[:,x_i,:].T      
        plot_scalar_field(
            fig, axes[i], x, y, divU, plt.cm.viridis, 
            plots['divU']['bounds'], plots['divU']['ticks'], r'Divergence $\nabla\cdot\mathbf{u}$', r's$^{-1}$'
        )
        i += 1

    if "curlU" in plots: # Plot curl
        curlU = plots['curlU']['data'][n]
        if len(domain) == 4:
            if visualization == 'vertical':
                curlU = curlU[y_j].T
            elif visualization == 'horizontal':
                curlU = curlU[:,:,z_j]
            elif visualization == 'longitudinal':
                curlU = curlU[:,x_i,:].T       
        plot_scalar_field(
            fig, axes[i], x, y, curlU, plt.cm.viridis, 
            plots['curlU']['bounds'], plots['curlU']['ticks'], r'Vorticity $\nabla\times\mathbf{u}$', r's$^{-1}$'
        )
        i += 1

    if "T" in plots: # Plot temperature
        T = plots['T']['data'][n]
        if len(domain) == 4:
            if visualization == 'vertical':
                T = T[y_j].T
            elif visualization == 'horizontal':
                T = T[:,:,z_j]
            elif visualization == 'longitudinal':
                T = T[:,x_i,:].T
        plot_scalar_field(
            fig, axes[i], x, y, T, plt.cm.jet, 
            plots['T']['bounds'], plots['T']['ticks'], r'Temperature $T$', r'K'
        )
        i += 1
    
    if "Y" in plots: # Plot fuel and terrain (IBM nodes)
        Y = plots['Y']['data'][0][n]
        terrain = plots['Y']['data'][1]
        if len(domain) == 4:
            if visualization == 'vertical':
                Y = Y[y_j].T
                terrain = terrain[y_j].T
            elif visualization == 'horizontal':
                Y = Y[:,:,z_j]
                terrain = terrain[:,:,z_j]
            elif visualization == 'longitudinal':
                Y = Y[:,x_i,:].T
                terrain = terrain[:,x_i,:].T
        plot_scalar_field(
            fig, axes[i], x, y, Y, plt.cm.Oranges, 
            plots['Y']['bounds'], plots['Y']['ticks'], r'Fuel $Y$', r'\%'
        )
        plot_scalar_field(
            fig, axes[i], x, y, terrain, ListedColormap(['black']), [0, 1], None, None, None
        )
        i += 1

    if "p" in plots: # Plot pressure and pressure gradient
        p = plots['p']['data'][0][n]
        px = plots['p']['data'][1][0][n]
        py = plots['p']['data'][1][1][n]
        if len(domain) == 4:
            pz = plots['p']['data'][1][2][n]
            if visualization == 'vertical':
                p = p[y_j].T
                px = px[y_j].T
                pz = pz[y_j].T
                py = pz
            elif visualization == 'horizontal':
                p = p[:,:,z_j]
                px = px[:,:,z_j]
                py = py[:,:,z_j]
            elif visualization == 'longitudinal':
                p = p[:,x_i,:].T
                py = py[:,x_i,:].T
                pz = pz[:,x_i,:].T
                px = py
        plot_scalar_field(
            fig, axes[i], x, y, p, plt.cm.viridis, 
            plots['p']['bounds'], plots['p']['ticks'], r'Pressure $p$', r"kg m$^{-1}s^{-2}$", .8
        )
        plot_vector_field(
            axes[i], x, y, px, py, streamplot=streamplot, qs=qs, density=density
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

def plot_initial_conditions_3D(x: np.ndarray, y: np.ndarray, z: np.ndarray, u: np.ndarray, v: np.ndarray, w: np.ndarray,
        s: np.ndarray, T: np.ndarray, Y: np.ndarray, plot_lims: list = None) -> None:
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
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), subplot_kw=dict(projection='3d'))
    # axes.set_aspect('equal')
    # for i in range(len(axes)):
    #     # axes[i].set_box_aspect(aspect=(1, 1, .1))
    #     axes[i].set_xlim3d(plot_lims[0])
    #     axes[i].set_ylim3d(plot_lims[1])
    #     axes[i].set_zlim3d(plot_lims[2])

    T_mask = T > 300
    Xp = x[T_mask]
    Yp = y[T_mask]
    Zp = z[T_mask]
    Tp = T[T_mask]
    plot_scalar_field_3D(fig, axes[0], Xp, Yp, Zp, Tp, plt.cm.jet, 
        [T.min(), T.max()], None, r'Temperature $T$', r'K') 
    
    # Plot velocity 
    plot_vector_field_3D(axes[1], x, y, z, u, v, w, streamplot=False, qs=4, density=1)
    # Plot speed
    # plot_scalar_field_3D(fig, axes[1], x, y, z, s, plt.cm.viridis, [s.min(), s.max()], None, r'Speed $\|\mathbf{u}\|$', r'm s$^{-1}$')
    
    # Set limits
    
    # axes[0].set_xlim(plot_lims[0])
    # axes[0].set_ylim(plot_lims[1])
    # axes[1].set_ylim(plot_lims[2])
    # axes[1].set_xlim(plot_lims[0])
    """
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 10))

    # Axis labels
    axes[-1].set_xlabel('x')
    for i in range(len(axes)):
        axes[i].set_ylabel('z')

    # Set limits if given
    if plot_lims is not None:
        for i in range(len(axes)):
            axes[i].set_xlim(plot_lims[0])
            axes[i].set_ylim(plot_lims[2])

    # Plot speed
    plot_scalar_field(fig, axes[0], x, z, s.T, plt.cm.viridis, 
        [s.min(), s.max()], None, r'Speed $\|\mathbf{u}\|$', r'm s$^{-1}$') 
    
    # Plot velocity u component
    plot_scalar_field(fig, axes[1], x, z, u.T, plt.cm.viridis,
        [u.min(), u.max()], None, r'Velocity component $u$', r'm s$^{-1}$')
    
    # Plot velocity v component
    plot_scalar_field(fig, axes[2], x, z, v.T, plt.cm.viridis,
        [v.min(), v.max()], None, r'Velocity component $v$', r'm s$^{-1}$')
    
    # Plot velocity w component
    plot_scalar_field(fig, axes[3], x, z, w.T, plt.cm.viridis,
        [w.min(), w.max()], None, r'Velocity component $w$', r'm s$^{-1}$')
    
    # Plot temperature
    plot_scalar_field(fig, axes[4], x, z, T.T, plt.cm.jet,
        [T.min(), T.max()], None, r'Temperature $T$', r'K')
    
    # Plot fuel
    plot_scalar_field(fig, axes[5], x, z, Y.T, plt.cm.Oranges,
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

def plot_2D_old(x, y, z, cmap=plt.cm.jet):
    plt.figure(figsize=(12, 6))
    plt.contourf(x, y, z, cmap=cmap)
    plt.colorbar()
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()