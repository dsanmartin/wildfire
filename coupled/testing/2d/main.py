import time
import datetime
import numpy as np
from parameters import *
from utils import domain
from initial_conditions import u0, v0, T0, Y0, p0, F, topo, shape#, T_mask
# from topography import topo
from ibm import topography_nodes, topography_distance
from pde import solve_pde
from inout import create_simulation_folder, save_approximation, save_parameters
from plots import plot_2D, plot_ic, plot_1D
from arguments import args
from logs import show_info, save_info

def main():
    # Get arguments
    input_ic = False
    show_ic = args.show_initial_condition
    debug = args.debug
    sim_name = args.name
    save_path = args.save_path
    Nx = args.x_nodes
    Ny = args.y_nodes
    Nt = args.t_nodes
    x_min, x_max = args.x_min, args.x_max
    y_min, y_max = args.y_min, args.y_max
    t_min, t_max = args.t_min, args.t_max
    nu = args.viscosity
    k = args.diffusivity
    Pr = args.prandtl
    Y_f = args.fuel_consumption
    H_R = args.heat_energy
    h = args.convective_coefficient
    A = args.pre_exponential_coefficient
    T_act = args.activation_temperature
    Y_thr = args.fuel_threshold
    T_hot = args.hot_temperature
    S_top = args.source_top
    S_bot = args.source_bottom
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
    T_mask = T_0 > T_inf #plate(Xm, Ym) #shape(Xm, Ym) > 0.01
    T_mask = T_mask.astype(int)
    T_mask = None
    ### THIS CAN BE IMPROVED MAYBE CREATING INITIAL CONDITIONS INCLUDING TOPOGRAPHY :')
    if input_ic is not True:
        U_0[dead_nodes] = 0
        V_0[dead_nodes] = 0
        T_0[dead_nodes] = T_inf
        Y_0[dead_nodes] = 1
        # Y_0 = Y_0 + (Ym) <= topo(Xm) + 2 * dy
    if show_ic:
        plot_ic(Xm, Ym, U_0, V_0, T_0, Y_0)
        # mask = (Ym >= 1.9) & (Ym <= 2.1)
        # print(U_0[mask])
        # plot_1D(Xm[0], T_0[0])
        # print(Y_0.min(), Y_0.max())
        # plot_2D(Xm, Ym, T_mask)
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
        # Temperature
        'k': k, 'C_p': C_p, 
        'delta': delta, 'sigma': sigma, 
        # Fuel 
        'A': A, 'T_act': T_act, 'T_pc': T_pc, 'H_R': H_R, 'h': h, 'Y_thr': Y_thr, 'Y_f': Y_f,
        # Boundary conditions just in y (because it's periodic on x)
        'bc_on_y': dirichlet_y_bc,    
        # IBM
        'cut_nodes': cut_nodes, 'dead_nodes': dead_nodes, 'values_dead_nodes': values_dead_nodes, 'Y_top': topo_distance,
        # 'mask': mask,
        'method': method, # IVP solver 
        'T_hot': T_hot,
        'T_mask': T_mask, 
        'T_0': T_0,
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
    }
    # Show parameters
    # print(params)
    show_info(params)
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
        save_path = create_simulation_folder(args.name)
    save_info(params, save_path)
    # Save outputs
    save_approximation(save_path, x, y, t[::NT], u, v, T, Y, p)
    # Remove Soruce temperature!
    # del args['ST']
    save_parameters(save_path, params)
    print("Simulation name:", sim_name) # To easily get the name of the simulation for visualization

if __name__ == "__main__":
    main()
