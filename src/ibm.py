import numpy as np

def boundary(u, v, cut_nodes, dead_nodes):
    """Boundary condition for IBM

    Parameters
    ----------
    u : array (Ny, Nx)
        First component of the velocity field
    v : array (Ny, Nx)
        Second component of the velocity field
    cut_nodes : array
        Node tag for cut nodes
    dead_nodes : array
        Node tag for dead nodes

    Returns
    -------
    array (2, Ny, Nx)
        Return velocity field with IBM 
    """
    # Get node's tags
    cut_nodes_y, cut_nodes_x = cut_nodes
    dead_nodes_y, dead_nodes_x = dead_nodes
    # Get vector field
    U = u.copy()
    V = v.copy()
    # Apply restrictions 
    U[cut_nodes_y, cut_nodes_x] = 0
    V[cut_nodes_y, cut_nodes_x] = 0
    U[dead_nodes_y, dead_nodes_x] = 0
    V[dead_nodes_y, dead_nodes_x] = 0
    # Return new vector field
    return np.array([U, V])


def cylinder(x, y, x_0, y_0, R, dx, dy):
    # Get circle nodes
    circle = (x - x_0) ** 2 + (y - y_0) ** 2 <= R ** 2
    # Gen nodes next to circle
    next_circle = (x - x_0) ** 2 + (y - y_0) ** 2 <= (R + (dx ** 2 + dy ** 2) ** 0.5) ** 2 
    # Get dead and cut nodes
    circle = circle.astype(int)
    next_circle = next_circle.astype(int)
    cut_nodes = np.where((next_circle - circle) == 1)
    dead_nodes = np.where(circle == 1)
    # Return nodes
    return cut_nodes, dead_nodes

def cylinders(x, y, centers, radiuses, dx, dy):
    cut_nodes = [np.array([]), np.array([])] 
    dead_nodes = [np.array([]), np.array([])]  #[[], []] 
    for center, radius in zip(centers, radiuses):
        cut_nodes_, dead_nodes_ = cylinder(x, y, center[0], center[1], radius, dx, dy)
        cut_nodes_y, cut_nodes_x = cut_nodes_[0], cut_nodes_[1]
        dead_nodes_y, dead_nodes_x = dead_nodes_[0], dead_nodes_[1]
        cut_nodes[0] = np.append(cut_nodes[0], cut_nodes_y)
        cut_nodes[1] = np.append(cut_nodes[1], cut_nodes_x)
        dead_nodes[0] = np.append(dead_nodes[0], dead_nodes_y)
        dead_nodes[1] = np.append(dead_nodes[1], dead_nodes_x)
    cut_nodes = np.array(cut_nodes, dtype=int)
    dead_nodes = np.array(dead_nodes, dtype=int)
    return cut_nodes, dead_nodes

def cavity(x, y, walls_lims, dx, dy):
    # Nx, Ny = x.shape[0], y.shape[0]
    Nx, Ny = x.shape[1], y.shape[0]
    x_min, x_max = walls_lims
    # Create a periodic walls in the x direction
    left_wall_end = x_min
    right_wall_start = x_max
    wall = np.zeros((Ny, Nx))
    wall[(x <= left_wall_end) | (x >= right_wall_start)] = 1
    wall[(x < left_wall_end) | (x > right_wall_start)] = 0.5
    dead_nodes = np.where(wall == 0.5)
    cut_nodes = np.where(wall == 1)
    return cut_nodes, dead_nodes

def topography_nodes(x, y, f, dx, dy):
    """Topography for IBM

    Parameters
    ----------
    x : array (Nx)
        x-coordinates of the mesh
    y : array (Ny)
        y-coordinates of the mesh
    f : array (Ny, Nx)
        Topography
    dx : float
        x-spacing of the mesh
    dy : float
        y-spacing of the mesh

    Returns
    -------
    array (Ny, Nx)
        Topography with IBM
    """
    # Old
    # # Get topography
    # topo = y <= f(x)#, y)
    # # import matplotlib.pyplot as plt
    # # plt.imshow(topo.astype(int), origin='lower')
    # # plt.show()
    # # Get nodes next to topography
    # next_f = y <= f(x) + dy #(dx ** 2 + dy ** 2) ** 0.5
    # topo = topo.astype(int)
    # next_f = next_f.astype(int)
    # # Get dead and cut nodes
    # cut_nodes = np.where((next_f - topo) == 1)
    # dead_nodes = np.where(topo == 1)
    # New
    # Get topography
    topo = (y >= (f(x) - dy)) & (y <= f(x))    
    dead = y < f(x) - dy
    #topo = topo.astype(int)
    #dead = dead.ast
    # Get dead and cut nodes
    cut_nodes = np.where(topo == True)
    dead_nodes = np.where(dead == True)
    # Return nodes
    return cut_nodes, dead_nodes

def topography_nodes_3D(x, y, z, f, dx, dy, dz):
    """Topography for IBM

    Parameters
    ----------
    x : array (Nx)
        x-coordinates of the mesh
    y : array (Ny)
        y-coordinates of the mesh
    z : array (Nz)
        z-coordinates of the mesh
    f : callable
        Topography
    dx : float
        x-spacing of the mesh
    dy : float
        y-spacing of the mesh
    dx : float
        z-spacing of the mesh

    Returns
    -------
    array (Ny, Nx, Nz)
        Topography with IBM
    """
    # Get topography
    topo = (z >= (f(x, y) - dz)) & (z <= f(x, y))    
    dead = z < f(x, y) - dz
    # Get dead and cut nodes
    cut_nodes = np.where(topo == True)
    dead_nodes = np.where(dead == True)
    # Return nodes
    return cut_nodes, dead_nodes


def building_circle(x, y, x_lims, y_lims, dx, dy):
    Nx, Ny = x.shape[0], y.shape[0]
    x_min, x_max = x_lims
    y_min, y_max = y_lims
    # Rectangle
    rectangle = np.zeros_like(x)
    rectangle[(y >= y_min) & (y <= y_max) & (x >= x_min) & (x <= x_max)] = 1
    inside_rectangle = np.zeros_like(x)
    inside_rectangle[(y >= y_min + dy) & (y <= y_max - dy) & (x >= x_min + dx) & (x <= x_max - dx)] = 1
    # Circle
    # Gen nodes next to circle
    x_med = (x_min + x_max) / 2
    R = x_max - x_med
    circle = (x - x_med) ** 2 + (y - y_max) ** 2 <= R ** 2 
    inside_circle = (x - x_med) ** 2 + (y - y_max) ** 2 <= (R - (dx ** 2 + dy ** 2) ** 0.5) ** 2
    building = rectangle + circle
    inside_building = inside_rectangle + inside_circle
    building[building >= 1] = 1
    building[building < 1] = 0
    border = building - inside_building
    border[border >= 1] = 1
    border[border < 1] = 0
    cut_nodes = np.where(border == 1)
    dead_nodes = np.where(inside_building == 1)
    return cut_nodes, dead_nodes

def building(x, y, x_lims, y_lims, dx, dy):
    Nx, Ny = x.shape[0], y.shape[0]
    x_min, x_max = x_lims
    y_min, y_max = y_lims
    # Rectangle
    rectangle = np.zeros_like(x)
    rectangle[(y >= y_min) & (y <= y_max) & (x >= x_min) & (x <= x_max)] = 1
    inside_rectangle = np.zeros_like(x)
    inside_rectangle[(y >= y_min ) & (y <= (y_max - dy)) & (x >= (x_min + dx)) & (x <= (x_max - dx))] = 1
    # Circle
    # Gen nodes next to circle
    x_med = (x_min + x_max) / 2
    R = x_max - x_med
    ellipse = ((x - x_med) / R) ** 2 + ((y - y_max) / (3 * dy)) ** 2 <= 1
    inside_ellipse = ((x - x_med) / (R - dx)) ** 2 + ((y - y_max) / (3 * dy - dy)) ** 2 <= 1
    building = rectangle + ellipse * 0
    inside_building = inside_rectangle + inside_ellipse * 0
    building[building >= 1] = 1
    building[building < 1] = 0
    border = building - inside_building
    border[border >= 1] = 1
    border[border < 1] = 0
    cut_nodes = np.where(border == 1)
    dead_nodes = np.where(inside_building == 1)
    return cut_nodes, dead_nodes


def topography_distance(x, y, f):
    return y - f(x)

def topography_distance_3D(x, y, z, f):
    return z - f(x, y)
