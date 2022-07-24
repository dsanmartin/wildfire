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
    cut_nodes_x, cut_nodes_y = cut_nodes
    dead_nodes_x, dead_nodes_y = dead_nodes
    # Get vector field
    U = u.copy()
    V = v.copy()
    # Apply restrictions 
    U[cut_nodes_x, cut_nodes_y] = 0
    V[cut_nodes_x, cut_nodes_y] = 0
    U[dead_nodes_x, dead_nodes_y] = 0
    V[dead_nodes_x, dead_nodes_y] = 0
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
    cut_nodes = [[], []] 
    dead_nodes = [[], []] 
    for center, radius in zip(centers, radiuses):
        cut_nodes_, dead_nodes_ = cylinder(x, y, center[0], center[1], radius, dx, dy)
        cut_nodes_x, cut_nodes_y = cut_nodes_[0], cut_nodes_[1]
        dead_nodes_x, dead_nodes_y = dead_nodes_[0], dead_nodes_[1]
        cut_nodes[0] = np.append(cut_nodes[0], cut_nodes_x)
        cut_nodes[1] = np.append(cut_nodes[1], cut_nodes_y)
        dead_nodes[0] = np.append(dead_nodes[0], dead_nodes_x,)
        dead_nodes[1] = np.append(dead_nodes_y, dead_nodes[1])
    cut_nodes = np.array(cut_nodes, dtype=int)
    dead_nodes = np.array(dead_nodes, dtype=int)
    return cut_nodes, dead_nodes