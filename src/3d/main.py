import numpy as np
from arguments import * # Include default parameters + command line arguments
from utils import domain
from initial_conditions import u0, v0, w0, T0, Y0, p0, F, topo, shape#, T_mask
from ibm import topography_nodes, topography_distance
from pde import solve_pde
from inout import create_simulation_folder, save_approximation, save_parameters
from plots import plot_initial_conditions, plot_2D, plot_1D
from logs import log_params
import matplotlib.pyplot as plt

def main():
    # Get arguments
    input_ic = False
    # Create arrays
    x, y, z, t, Xm, Ym, Zm, dx, dy, dz, dt = domain(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, Nx, Ny, Nz, Nt)
    # Evaluate initial conditions 
    U_0 = u0(Xm, Ym, Zm)
    V_0 = v0(Xm, Ym, Zm)
    W_0 = w0(Xm, Ym, Zm)
    T_0 = T0(Xm, Ym, Zm) 
    Y_0 = Y0(Xm, Ym, Zm)
    P_0 = p0(Xm, Ym, Zm)
    F_e = F(Xm, Ym, Zm)
    # Topography effect
    cut_nodes, dead_nodes = topography_nodes(Xm, Ym, Zm, topo, dx, dy, dz)
    # cut_nodes_y, cut_nodes_x, cut_nodes_z = cut_nodes
    # Scatter 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x[cut_nodes_x], y[cut_nodes_y], z[cut_nodes_z])
    # # ax.scatter(cut_nodes_x, cut_nodes_y, cut_nodes_z, c='b', marker='o', alpha=.5)
    # # ax.plot_surface(x[cut_nodes_x], y[cut_nodes_y], z[cut_nodes_z], alpha=.5)
    # plt.show()
    # print(asd)
    values_dead_nodes = np.array([u_dead_nodes, v_dead_nodes, w_dead_nodes, T_dead_nodes, Y_dead_nodes])
    topo_distance = topography_distance(Xm, Ym, Zm, topo)
    # Temperature mask
    T_mask = T_0 > 300 #plate(Xm, Ym) #shape(Xm, Ym) > 0.01
    #T_mask = shape(Xm, Ym) == 1
    ### THIS CAN BE IMPROVED MAYBE CREATING INITIAL CONDITIONS INCLUDING TOPOGRAPHY :')
    if input_ic is not True:
        U_0[dead_nodes] = u_dead_nodes
        V_0[dead_nodes] = v_dead_nodes
        W_0[dead_nodes] = w_dead_nodes
        T_0[dead_nodes] = T_dead_nodes
        Y_0[dead_nodes] = Y_dead_nodes
        # Y_0 = Y_0 + (Ym) <= topo(Xm) + 2 * dy
    if show_ic:
        # test = shape(Xm, Ym)
        S_0 = np.sqrt(U_0**2 + V_0**2)
        plot_initial_conditions(Xm, Ym, U_0, V_0, S_0, T_0, Y_0, plot_lims=[[0, 200], [0, 20]])
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
    u_z_min = U_0[:,:, 0]
    u_z_max = U_0[:,:,-1]
    v_z_min = V_0[:,:, 0]
    v_z_max = V_0[:,:,-1]
    w_z_min = W_0[:,:, 0]
    w_z_max = W_0[:,:,-1]
    p_z_min = P_0[:,:, 0]
    p_z_max = P_0[:,:,-1]
    T_z_min = T_0[:,:, 0]
    T_z_max = T_0[:,:,-1]
    Y_z_min = Y_0[:,:, 0]
    Y_z_max = Y_0[:,:,-1]
    dirichlet_z_bc = np.array([
        [u_z_min, u_z_max],
        [v_z_min, v_z_max],
        [w_z_min, w_z_max],
        [T_z_min, T_z_max],
        [Y_z_min, Y_z_max],
        [p_z_min, p_z_max]
    ])
    # Create parameter diccionary
    params = {
        # Domain
        'x': x, 'y': y, 'z': z, 'Xm': Xm, 'Ym': Ym, 'Zm': Zm, 
        'dx': dx, 'dy': dy, 'dz': dz, 'Nx': Nx, 'Ny': Ny, 'Nz': Nz, 
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
        'alpha': alpha, 'C_p': C_p, 
        'delta': delta, 'sigma': sigma, # Radiation
        # Fuel 
        'A': A, 'T_act': T_act, 'T_pc': T_pc, 'H_R': H_R, 'h': h, 'Y_D': Y_D, 'Y_f': Y_f,
        # Boundary conditions just in y (because it's periodic on x)
        'bc_on_z': dirichlet_z_bc,    
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
        'w0': W_0,
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
        'z_r': z_r,
        'alpha_u': alpha_u,
        # Temperature
        'T0_shape': T0_shape,
        'T0_x_start': T0_x_start,
        'T0_x_end': T0_x_end,
        'T0_y_start': T0_y_start,
        'T0_y_end': T0_y_end,
        'T0_z_start': T0_z_start,
        'T0_z_end': T0_z_end,
        'T0_x_center': T0_x_center,
        'T0_y_center': T0_y_center,
        'T0_length': T0_length, 
        'T0_width': T0_width,
        'T0_height': T0_height,
        # Topography
        'topography_shape': topography_shape,
        'hill_center_x': hill_center_x,
        'hill_center_y': hill_center_y,
        'hill_length': hill_length,
        'hill_height': hill_height,
        'hill_width': hill_width,
        # Fuel
        'fuel_height': fuel_height,
        # Sutherland's law
        'sutherland_law': sutherland_law,
        'S_T_0': S_T_0, 'S_k_0': S_k_0, 'S_k': S_k,
        'bound': bound,
        'T_min': T_min, 'T_max': T_max,
        'Y_min': Y_min, 'Y_max': Y_max,
        'periodic_axes': periodic_axes,
        'save_path': save_path,
    }
    # Show parameters
    log_params(params)
    # Solve PDE
    z_0 = np.array([U_0, V_0, W_0, T_0, Y_0])
    u, v, w, T, Y, p  = solve_pde(z_0, params)
    # Create simulation folder
    # if save_path is None:
    #     save_path_ = create_simulation_folder(sim_name)
    # else:
    #     save_path_ = save_path
    log_params(params, save_path)
    # Save outputs
    save_approximation(save_path, x, y, z, t[::NT], u, v, w, T, Y, p)
    save_parameters(save_path, params)
    print("Simulation name:", sim_name) # To easily get the name of the simulation for visualization

if __name__ == "__main__":
    main()
