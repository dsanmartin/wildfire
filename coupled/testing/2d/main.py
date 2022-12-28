import sys
import numpy as np
from parameters import *
from utils import domain
from initial_conditions import u0, v0, T0, Y0, p0, F, plate
from topography import topo
from ibm import topography_nodes
from pde import solve_pde
from inout import create_simulation_folder, save_approximation, save_parameters
from plots import plot_2d

def main():
    sim_name = None
    if len(sys.argv) > 1:
        sim_name = sys.argv[1]
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
    ### THIS CAN BE IMPROVED MAYBE CREATING INITIAL CONDITIONS INCLUDING TOPOGRAPHY :')
    U_0[dead_nodes] = 0
    V_0[dead_nodes] = 0
    T_0[dead_nodes] = T_inf
    Y_0 = Y_0 + (Ym) <= topo(Xm) + 2 * dy
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
        'turb': turb,
        'conservative': conser,
        # Temperature
        'k': k,
        # Fuel 
        'A': A, 'B': B, 'T_pc': T_pc, 'H_R': H_R, 'h': h, 'Y_thr': Y_thr, 'Y_f': Y_f,
        # Boundary conditions just in y (because it's periodic on x)
        'bc_on_y': dirichlet_y_bc,    
        # IBM
        'cut_nodes': cut_nodes, 'dead_nodes': dead_nodes, 'values_dead_nodes': values_dead_nodes,
        # 'mask': mask,
        'method': method, # IVP solver 
        'TA': TA,
        # 'T_mask': None, 
        'T_mask': plate(Xm, Ym),
        # 'ST': ST,
        # 'u_y_min': u_y_min,
        # 'u_y_max': u_y_max,
        # 'v_y_min': v_y_min,
        # 'v_y_max': v_y_max,
        # 'p_y_max': p_y_max,
        # 'Y_y_min': Y_y_min,
        # 'Y_y_max': Y_y_max,
        # 'u0': U_0,
        # 'v0': V_0,
    }
    # Solve PDE
    z_0 = np.array([U_0, V_0, T_0, Y_0])
    u, v, T, Y, p  = solve_pde(z_0, params)
    # Create simulation folder
    save_path = create_simulation_folder(sim_name)
    # Save outputs
    save_approximation(save_path, x, y, t[::NT], u, v, T, Y, p)
    # Remove Soruce temperature!
    # del args['ST']
    save_parameters(save_path, params)

if __name__ == "__main__":
    main()