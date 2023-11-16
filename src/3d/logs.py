import sys
from utils import non_dimensional_numbers
from inout import create_simulation_folder

def log_params(params: dict, dir_path: str = None) -> None:
    """
    Log simulation parameters to stdout or file.

    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters.
    dir_path : str, optional
        Path to directory where the parameters file will be saved. If None, the parameters will be printed to stdout.

    Returns
    -------
    None
    """
    # Get data
    x_min, x_max = params['x'][0], params['x'][-1]
    y_min, y_max = params['y'][0], params['y'][-1]
    z_min, z_max = params['z'][0], params['z'][-1]
    t_min, t_max = params['t'][0], params['t'][-1]
    Nx, Ny, Nz, Nt, NT = params['Nx'],params['Ny'], params['Nz'], params['Nt'], params['NT']
    dx, dy, dz, dt = params['dx'], params['dy'], params['dt'], params['dt']
    # U_0, V_0, W_0, T_0 = params['u0'], params['v0'], params['w0'], params['T0']
    sim_name = params['sim_name']
    method = params['method']
    rho, T_hot, T_inf, T_pc = params['rho'], params['T_hot'], params['T_inf'], params['T_pc']
    nu, alpha, Pr, g = params['nu'], params['alpha'], params['Pr'], params['g']
    A, T_act, H_R, h, a_v = params['A'], params['T_act'], params['H_R'], params['h'], params['a_v']
    Y_f, Y_D, C_p = params['Y_f'], params['Y_D'], params['C_p']
    turb, conser = params['turbulence'], params['conservative']
    S_top, S_bot, Sx = params['S_top'], params['S_bot'], params['Sx']
    source_filter = params['source_filter']
    radiation = params['radiation']
    include_source = params['include_source']
    initial_u_type = params['initial_u_type']
    u_z_0, d, u_ast, kappa = params['u_z_0'], params['d'], params['u_ast'], params['kappa']
    u_r, z_r, alpha = params['u_r'], params['z_r'], params['alpha']
    T0_shape = params['T0_shape']
    T0_x_start, T0_x_end = params['T0_x_start'], params['T0_x_end']
    T0_y_start, T0_y_end = params['T0_y_start'], params['T0_y_end']
    T0_z_start, T0_z_end = params['T0_z_start'], params['T0_z_end']
    T0_x_center, T0_y_center = params['T0_x_center'], params['T0_y_center']
    T0_length, T0_width, T0_height = params['T0_length'], params['T0_width'], params['T0_height']
    topography_shape = params['topography_shape']
    hill_center_x, hill_center_y = params['hill_center_x'], params['hill_center_y']
    hill_length, hill_height, hill_width = params['hill_length'], params['hill_height'], params['hill_width']
    fuel_height = params['fuel_height']
    sutherland_law = params['sutherland_law']
    S_T_0, S_k_0, S_k = params['S_T_0'], params['S_k_0'], params['S_k']
    bound = params['bound']
    T_min, T_max = params['T_min'], params['T_max']
    Y_min, Y_max = params['Y_min'], params['Y_max']

    # Non dimensional numbers calculation
    Re, Gr, Ra, Sr, Ste, St, Ze = non_dimensional_numbers(params)
    
    if dir_path is None: # Print to stdout or file
        f = sys.stdout
    else: # Print to file
        create_simulation_folder(dir_path) # Create simulation folder if it doesn't exist
        f = open(dir_path + 'parameters.txt', 'w')

    print("Simulation name:", sim_name, file=f)
    print("Domain: [%.4f, %.4f] x [%.4f, %.4f] x [%.4f, %.4f] x [%.4f, %.4f]" % (x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max), file=f)
    print("Grid size: Nx: %d, Ny: %d, Nz: %d, Nt: %d" % (Nx, Ny, Nz, Nt), file=f)
    print("dx: %.4f, dy: %.4f, dz: %.4f, dt: %.4f" % (dx, dy, dz, dt), file=f)
    print("Time integration: %s" % method, file=f)
    print("Time samples: %d" % NT, file=f)
    print("nu: %.2e, g: (%.4f, %.4f, %.4f)" % (nu, g[0], g[1], g[2]), file=f)
    print("alpha: %.2e, C_p: %.4f, T_inf: %.4f, T_hot: %.4f" % (alpha, C_p, T_inf, T_hot), file=f)
    print("rho: %.4f, T_pc: %.4f, A: %.4f, T_act: %.4f" % (rho, T_pc, A, T_act), file=f)
    print("H_R: %.4f, h: %.4f, a_v: %.4f, Y_D: %.4f, Y_f: %.4f" % (H_R, h, a_v, Y_D, Y_f), file=f)
    print("Initial u type: %s" % initial_u_type, file=f)
    if initial_u_type == 'log':
        print("z_0: %.4f, d: %.4f, u_ast: %.4f, kappa: %.4f" % (u_z_0, d, u_ast, kappa), file=f)
    else:
        print("  u_r: %.4f, z_r: %.4f, kappa: %.4f" % (u_r, z_r, kappa), file=f)
    print("Initial temperature shape: %s" % T0_shape, file=f)
    print("  x: [%.4f, %.4f], y: [%.4f, %.4f], z: [%.4f, %.4f]" % (T0_x_start, T0_x_end, T0_y_start, T0_y_end, T0_z_start, T0_z_end), file=f)
    print("  x center: %.4f, y center: %.4f, length: %.4f, width: %.4f, height: %.4f" % (T0_x_center, T0_y_center, T0_length, T0_width, T0_height), file=f)
    print("Topography shape: %s" % topography_shape, file=f)
    if topography_shape == 'hill':
        print("    Center: [%.4f, %.4f], Length: %.4f, Height: %.4f, Width: %.4f" % (hill_center_x, hill_center_y, hill_length, hill_height, hill_width), file=f)
    print("Fuel height: %.4f" % fuel_height, file=f)
    print("Include source: %r" % include_source, file=f)
    print("Source filter: %r" % source_filter, file=f)
    if source_filter:
        print("  S_top: %.4f, S_bot: %.4f, Sx: %.4f" % (S_top, S_bot, Sx), file=f)
    print("Turbulence: %r" % turb, file=f)
    print("Conservative: %r" % conser, file=f)
    print("Radiation: %r" % radiation, file=f)
    print("Sutherland's law: %r" % sutherland_law, file=f)
    if sutherland_law:
        print("  T_0: %.4f, k_0: %.4f, S_k: %.4f" % (S_T_0, S_k_0, S_k), file=f)
    print("Bound: %r" % bound, file=f)
    if bound:
        print("  Temperature: [%.4f, %.4f]" % (T_min, T_max), file=f)
        print("  Fuel: [%.4f, %.4f]" % (Y_min, Y_max), file=f)
    print("Reynolds: %.2e" %  Re, file=f)
    print("Prandtl: %.4f" % Pr, file=f)
    print("Grashof: %.2e" % Gr, file=f)
    print("Rayleigh: %.2e" % Ra, file=f)
    print("Strouhal: %.2e" % Sr, file=f)
    print("Stefan: %.4f" % Ste, file=f)
    print("Stanton: %.4f" % St, file=f)
    print("Zeldovich: %.4f\n" % Ze, file=f)

    if dir_path is not None:
        f.close()

    return None