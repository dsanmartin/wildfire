import os
import pickle
import numpy as np

# Default output directory
OUTPUT_DIR = './output/'

def create_simulation_folder(save_path: str) -> None:
    """
    Create a folder for saving simulation outputs.

    Parameters
    ----------
    save_path : str
        Path to the folder where outputs will be saved.

    Returns
    -------
    str
        Path to the folder where outputs will be saved.
    """
    # Create output save path
    # save_path = OUTPUT_DIR + sim_name + '/'
    if not os.path.exists(save_path): # Create folder if it doesn't exist
        os.makedirs(save_path)
    # return save_path # Return save path for outputs
    return None

def save_approximation(save_path: str, x: np.ndarray, y: np.ndarray, z: np.ndarray, t: np.ndarray, 
        u: np.ndarray, v: np.ndarray, w: np.ndarray, T: np.ndarray, Y: np.ndarray, p: np.ndarray) -> None:
    """
    Save the approximation data to a file in npz format.

    Parameters
    ----------
    save_path : str
        The path where the data will be saved.
    x : numpy.ndarray (Nx,) or (Ny, Nx)
        The x-coordinates of the grid.
    y : numpy.ndarray (Ny,) or (Ny, Nx)
        The y-coordinates of the grid.
    z : numpy.ndarray (Nz,) or (Ny, Nx)
        The z-coordinates of the grid.
    t : numpy.ndarray (Nt,)
        The time steps of the simulation.
    u : numpy.ndarray (Nt, Ny, Nx)
        The velocity in the x-direction.
    v : numpy.ndarray (Nt, Ny, Nx)
        The velocity in the y-direction.
    w : numpy.ndarray (Nt, Ny, Nx)
        The velocity in the z-direction.
    T : numpy.ndarray (Nt, Ny, Nx)
        The temperature field.
    Y : numpy.ndarray (Nt, Ny, Nx, Ns)
        The species mass fractions.
    p : numpy.ndarray (Nt, Ny, Nx)
        The pressure field.

    Returns
    -------
    None
    """
    # Create folder for saving simulation outputs
    create_simulation_folder(save_path)
    # Save approximation data
    filename = save_path + 'data.npz'
    np.savez(filename, u=u, v=v, w=w, T=T, Y=Y, p=p, x=x, y=y, z=z, t=t)
    return None

def save_parameters(save_path: str, params: dict) -> None:
    """
    Save parameters to a pickle file.

    Parameters
    ----------
    save_path : str
        The path to the directory where the parameters file will be saved.
    params : dict
        A dictionary containing the parameters to be saved.

    Returns
    -------
    None
    """
    # Create folder for saving simulation outputs
    create_simulation_folder(save_path)
    # Create filename
    filename = save_path + 'parameters.pkl'
    # Save parameters
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    return None