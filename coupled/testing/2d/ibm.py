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


#def cylinder(x, y, x_0, y_0, R):
