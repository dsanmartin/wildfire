import os
import pickle
import numpy as np

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
    if not os.path.exists(save_path): # Create folder if it doesn't exist
        os.makedirs(save_path)
    # return save_path # Return save path for outputs
    return None

def save_approximation(parameters: dict, data: dict) -> None:
    """
    Save the approximation data to a file in npz format.

    Parameters
    ----------
    parameters : dict
        A dictionary containing the parameters of the model.
    data : dict
        A dictionary containing the approximation data.

    Returns
    -------
    None
    """
    # Get save path
    save_path = parameters['save_path']
    # Create folder for saving simulation outputs
    create_simulation_folder(save_path)
    # Save approximation data
    filename = save_path + 'data.npz'
    if len(data) == 5:
        x, y, t = parameters['x'], parameters['y'], parameters['t']
        u, v, T, Y, p = data['u'], data['v'], data['T'], data['Y'], data['p']
        NT = parameters['NT'] # Subsampling rate
        np.savez(filename, u=u, v=v, T=T, Y=Y, p=p, x=x, y=y, t=t[::NT])
    # elif len(data) == 6:
    else:
        x, y, z, t = parameters['x'], parameters['y'], parameters['z'], parameters['t']
        u, v, w, T, Y, p = data['u'], data['v'], data['w'], data['T'], data['Y'], data['p']
        u_, v_, w_, T_, Y_, p_ = data['u_'], data['v_'], data['w_'], data['T_'], data['Y_'], data['p_']
        NT = parameters['NT'] # Subsampling rate
        # np.savez(filename, u=u, v=v, w=w, T=T, Y=Y, p=p, x=x, y=y, z=z, t=t[::NT])
        np.savez(filename, u=u, v=v, w=w, T=T, Y=Y, p=p, x=x, y=y, z=z, t=t[::NT], u_=u_, v_=v_, w_=w_, T_=T_, Y_=Y_, p_=p_)
    return None

def save_parameters(parameters: dict) -> None:
    """
    Save parameters to a pickle file.

    Parameters
    ----------
    parameters : dict
        A dictionary containing the parameters to be saved.

    Returns
    -------
    None
    """
    # Get save path
    save_path = parameters['save_path']
    # Create folder for saving simulation outputs
    create_simulation_folder(save_path)
    # Create filename
    filename = save_path + 'parameters.pkl'
    # Save parameters
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)
    return None