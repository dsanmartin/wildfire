from wildfire import Wildfire
from arguments import * # Include default parameters + command line arguments

def main():
    if spatial_dims == 2:
        domain = [(x_min, x_max, Nx), (y_min, y_max, Ny), (t_min, t_max, Nt, NT)]
    elif spatial_dims == 3:
        domain = [(x_min, x_max, Nx), (y_min, y_max, Ny), (z_min, z_max, Nz), (t_min, t_max, Nt, NT)]
    wildfire = Wildfire(domain, parameters)
    wildfire.initialize()
    wildfire.solve()
    wildfire.postprocess()

if __name__ == "__main__":
    main()
