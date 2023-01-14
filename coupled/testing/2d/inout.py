import os
import pickle
import numpy as np
from datetime import datetime

INPUT_DIR = './input/'
OUTPUT_DIR = './output/'

def create_simulation_folder(sim_name):
    # Create output save path
    save_path = OUTPUT_DIR + sim_name + '/'
    if not os.path.exists(save_path): # Create folder if it doesn't exist
        os.makedirs(save_path)
    return save_path # Return save path for outputs

def save_approximation(save_path, x, y, t, u, v, T, Y, p):
    filename = save_path + 'data.npz'
    np.savez(filename, u=u, v=v, T=T, Y=Y, p=p, x=x, y=y, t=t)
    return None

def save_parameters(save_path, params):
    filename = save_path + 'parameters.pkl'
    # Save parameters
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    return None