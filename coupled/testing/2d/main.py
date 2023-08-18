import time
import datetime
import numpy as np
from arguments import * # Include default parameters + command line arguments
from utils import domain
from initial_conditions import u0, v0, T0, Y0, p0, F, topo, shape#, T_mask
from ibm import topography_nodes, topography_distance
from pde import solve_pde
from inout import create_simulation_folder, save_approximation, save_parameters
from plots import plot_2D, plot_ic, plot_1D
from logs import log_params

def main():
    # Get arguments
    input_ic = False
    # Create arrays
    x, y, t, Xm, Ym, dx, dy, dt = domain(x_min, x_max, y_min, y_max, t_min, t_max, Nx, Ny, Nt)
    # Evaluate initial conditions 
    U_0 = u0(Xm, Ym)
    V_0 = v0(Xm, Ym)
    T_0 = T0(Xm, Ym) 
    Y_0 = Y0(Xm, Ym)
    P_0 = p0(Xm, Ym)
    F_e = F(Xm, Ym)
    # Topography effect
    cut_nodes, dead_nodes = topography_nodes(Xm, Ym, topo, dx, dy)
    values_dead_nodes = np.array([u_dead_nodes, v_dead_nodes, T_dead_nodes, Y_dead_nodes])
    topo_distance = topography_distance(Xm, Ym, topo)
    # Temperature mask
    T_mask = T_0 > 300 #plate(Xm, Ym) #shape(Xm, Ym) > 0.01
    #T_mask = shape(Xm, Ym) == 1
    ### THIS CAN BE IMPROVED MAYBE CREATING INITIAL CONDITIONS INCLUDING TOPOGRAPHY :')
    if input_ic is not True:
        U_0[dead_nodes] = 0
        V_0[dead_nodes] = 0
        T_0[dead_nodes] = T_inf
        Y_0[dead_nodes] = 1
        # Y_0 = Y_0 + (Ym) <= topo(Xm) + 2 * dy
    if show_ic:
        test = shape(Xm, Ym)
        plot_ic(Xm, Ym, U_0, V_0, T_0, Y_0)
        # mask = (Ym >= 1.9) & (Ym <= 2.1)
        # print(U_0[mask])
        # plot_1D(Xm[0], T_0[0])
        # print(T_0.min(), T_0.max())
        # plot_2D(Xm, Ym, T_0)
        # plot_2D(Xm, Ym, T_0)
        # plot_2D(Xm, Ym, T_0[np.array(T_mask)])
    if debug:
        return False
    # Dirichlet boundary conditions
    # We are using the boundary of initial conditions as BC. We can test other values
    u_y_min = U_0[0]
    u_y_max = U_0[-1]
    v_y_min = V_0[0]
    v_y_max = V_0[-1]
    p_y_min = P_0[0]
    p_y_max = P_0[-1]
    T_y_min = T_0[0]
    T_y_max = T_0[-1]
    Y_y_min = Y_0[0]
    Y_y_max = Y_0[-1]
    dirichlet_y_bc = np.array([
        [u_y_min, u_y_max],
        [v_y_min, v_y_max],
        [T_y_min, T_y_max],
        [Y_y_min, Y_y_max],
        [p_y_min, p_y_max]
    ])
    # Create parameter diccionary
    params = {
        # Domain
        'x': x, 'y': y, 'Xm': Xm, 'Ym': Ym, 'dx': dx, 'dy': dy, 'Nx': Nx, 'Ny': Ny, # Space
        't': t, 'dt': dt, 'Nt': Nt, 'NT': NT, # Time
        # Fluid
        'nu': nu, 'rho': rho, 'g': g, 'T_inf': T_inf, 'F': F_e,
        'Pr': Pr, 'C_s': C_s, # Turbulence
        'C_D': C_D, 'a_v': a_v, # Drag force
        'turbulence': turb,
        'conservative': conser,
        'radiation': radiation,
        'include_source': include_source,
        # Temperature
        'k': k, 'C_p': C_p, 
        'delta': delta, 'sigma': sigma, # Radiation
        # Fuel 
        'A': A, 'T_act': T_act, 'T_pc': T_pc, 'H_R': H_R, 'h': h, 'Y_thr': Y_thr, 'Y_f': Y_f,
        # Boundary conditions just in y (because it's periodic on x)
        'bc_on_y': dirichlet_y_bc,    
        # IBM
        'cut_nodes': cut_nodes, 'dead_nodes': dead_nodes, 'values_dead_nodes': values_dead_nodes, 'Y_top': topo_distance,
        # 'mask': mask,
        'method': method, # IVP solver Initial u type
        'T_hot': T_hot,
        'T_mask': T_mask, 
        'T_0': T_0,
        't_source': t_source,
        'T_source': T_0,
        # 'T_mask': plate(Xm, Ym),
        # 'ST': ST,
        # 'u_y_min': u_y_min,
        # 'u_y_max': u_y_max,
        # 'v_y_min': v_y_min,
        # 'v_y_max': v_y_max,
        # 'p_y_max': p_y_max,
        # 'Y_y_min': Y_y_min,
        # 'Y_y_max': Y_y_max,
        'u0': U_0,
        'v0': V_0,
        'T0': T_0,
        'Y0': Y_0,
        'sim_name': sim_name,
        # Source/sink bounds
        'S_top': S_top,
        'S_bot': S_bot,
        'Sx': Sx,
        'debug': debug_pde,
        'source_filter': source_filter,
        # Initial conditions
        # Wind
        'initial_u_type': initial_u_type,
        'u_z_0': u_z_0,
        'd': d,
        'u_ast': u_ast,
        'kappa': kappa,
        'u_r': u_r,
        'y_r': y_r,
        'alpha': alpha,
        # Temperature
        'T0_shape': T0_shape,
        'T0_x_start': T0_x_start,
        'T0_x_end': T0_x_end,
        'T0_y_start': T0_y_start,
        'T0_y_end': T0_y_end,
        'T0_x_center': T0_x_center,
        'T0_width': T0_width,
        'T0_height': T0_height,
        # Topography
        'topography_shape': topography_shape,
        'hill_center': hill_center,
        'hill_height': hill_height,
        'hill_width': hill_width,
        # Fuel
        'fuel_height': fuel_height,
        # Sutherland's law
        'sutherland_law': sutherland_law,
        'S_T_0': S_T_0,
        'S_k_0': S_k_0,
        'S_k': S_k,
        'truncate': truncate,
        'T_min': T_min,
        'T_max': T_max,
        'Y_min': Y_min,
        'Y_max': Y_max,
    }
    # Show parameters
    log_params(params)
    # Solve PDE
    z_0 = np.array([U_0, V_0, T_0, Y_0])
    time_start = time.time()
    u, v, T, Y, p  = solve_pde(z_0, params)
    time_end = time.time()
    solve_time = (time_end - time_start)
    # print("Solver time: ", solve_time, "s\n")
    print("Solver time: ", str(datetime.timedelta(seconds=round(solve_time))), "\n")
    # Create simulation folder
    if save_path is None:
        save_path = create_simulation_folder(sim_name)
    log_params(params, save_path)
    # Save outputs
    save_approximation(save_path, x, y, t[::NT], u, v, T, Y, p)
    # Remove Soruce temperature!
    # del args['ST']
    save_parameters(save_path, params)
    print("Simulation name:", sim_name) # To easily get the name of the simulation for visualization

if __name__ == "__main__":
    main()
