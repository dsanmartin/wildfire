import time
import numpy as np
from parameters import *
from utils import domain
from initial_conditions import u0, v0, T0, Y0, p0, F, plate
from topography import topo
from ibm import topography_nodes, topography_distance
from pde import solve_pde
from inout import create_simulation_folder, save_approximation, save_parameters
from plots import plot_2D, plot_ic, plot_1D
from arguments import args
from logs import show_info

def main():
    # Others... To check
    input_ic = False
    show_ic = False
    debug = False
    # Get arguments
    sim_name = args.name
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
    ### THIS CAN BE IMPROVED MAYBE CREATING INITIAL CONDITIONS INCLUDING TOPOGRAPHY :')
    if input_ic is not True:
        U_0[dead_nodes] = 0
        V_0[dead_nodes] = 0
        T_0[dead_nodes] = T_inf
        Y_0 = Y_0 + (Ym) <= topo(Xm) + 2 * dy
    if show_ic:
        plot_ic(Xm, Ym, U_0, V_0, T_0, Y_0)
        #plot_1D(Xm[0], T_0[0])
        # plot_2D(Xm, Ym, U_0)
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
        'k': k,
        # Fuel 
        'A': A, 'B': B, 'T_pc': T_pc, 'H_R': H_R, 'h': h, 'Y_thr': Y_thr, 'Y_f': Y_f,
        # Boundary conditions just in y (because it's periodic on x)
        'bc_on_y': dirichlet_y_bc,    
        # IBM
        'cut_nodes': cut_nodes, 'dead_nodes': dead_nodes, 'values_dead_nodes': values_dead_nodes, 'Y_top': topo_distance,
        # 'mask': mask,
        'method': method, # IVP solver 
        'TA': TA,
        'T_mask': None, 
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
    print("Solver time: ", solve_time, "s\n")
    # Create simulation folder
    save_path = create_simulation_folder(args.name)
    # Save outputs
    save_approximation(save_path, x, y, t[::NT], u, v, T, Y, p)
    # Remove Soruce temperature!
    # del args['ST']
    save_parameters(save_path, params)

if __name__ == "__main__":
    main()
