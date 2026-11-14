import numpy as np
# from arguments import * # Include default parameters + command line arguments
from utils import domain_2D, domain_3D
from initial_conditions import U0, T0, Y0, p0, F, topo
from ibm import topography_nodes, topography_nodes_3D, topography_distance, topography_distance_3D, cavity, cylinder
from pde import solve_pde_2D, solve_pde_3D
from logs import log_params
from input_output import save_approximation, save_parameters
from plots import plot_initial_conditions, plot_initial_conditions_3D

class Wildfire:

    def __init__(self, domain: list, parameters: dict):
        self.domain = domain # Domain (x, y, z, t), z optional
        self.parameters = parameters # Physical parameters
        self.ndims = len(domain) # Number of dimensions 3 for 2D in space, 4 for 3D in space
        self.data = None # Data from the simulation

    def initialize(self):
        if self.ndims == 3:
            self.initialize_2D()
        elif self.ndims == 4:
            self.initialize_3D()
        else:
            raise Exception('Number of dimensions not supported')

    def initialize_2D(self):
        experiment = self.parameters['experiment']
        # Extract parameters to initialize
        (x_min, x_max, Nx), (y_min, y_max, Ny), (t_min, t_max, Nt, NT) = self.domain
        # Create arrays
        if experiment == 'cylinder':
            periodic = (True, True)
        else:
            periodic = (True, False)
        x, y, t, Xm, Ym, dx, dy, dt = domain_2D(x_min, x_max, y_min, y_max, t_min, t_max, Nx, Ny, Nt, periodic)
        # Evaluate initial conditions
        u0, v0 = U0 
        U_0 = u0(Xm, Ym)
        V_0 = v0(Xm, Ym)
        T_0 = T0(Xm, Ym) 
        Y_0 = Y0(Xm, Ym)
        P_0 = p0(Xm, Ym)
        F_e = F(Xm, Ym)
        # Topography effect
        if experiment == 'cavity':
            T0_x_start, T0_x_end = self.parameters['T0_x_start'], self.parameters['T0_x_end']
            T0_y_start, T0_y_end = self.parameters['T0_y_start'], self.parameters['T0_y_end']
            cut_nodes, dead_nodes = cavity(Xm, Ym, [T0_x_start, T0_x_end], dx, dy)
        elif experiment == 'cavity_diff':
            T0_x_start, T0_x_end = self.parameters['T0_x_start'], self.parameters['T0_x_end']
            T0_y_start, T0_y_end = self.parameters['T0_y_start'], self.parameters['T0_y_end']
            cut_nodes, dead_nodes = cavity(Xm, Ym, [T0_x_start, T0_y_end], dx, dy)
        elif experiment == 'cylinder':
            x_0 = 2#(x_max + x_min) / 2
            y_0 = (y_max + y_min) / 2
            R = 1
            cut_nodes, dead_nodes = cylinder(Xm, Ym, x_0, y_0, R, dx, dy)
        else:
            cut_nodes, dead_nodes = topography_nodes(Xm, Ym, topo, dx, dy)
        dead_nodes_values = np.array([
            self.parameters['dead_nodes_values'][0], 
            self.parameters['dead_nodes_values'][1], 
            self.parameters['dead_nodes_values'][-2], 
            self.parameters['dead_nodes_values'][-1]
        ])
        topo_distance = topography_distance(Xm, Ym, topo)
        # Set values for dead nodes (IBM)
        if self.parameters['input_ic'] is not True:
            U_0[dead_nodes] = dead_nodes_values[0]
            V_0[dead_nodes] = dead_nodes_values[1]
            T_0[dead_nodes] = dead_nodes_values[2]
            Y_0[dead_nodes] = dead_nodes_values[3]
        if self.parameters['show_ic']:
            plot_lims = [[0, 200], [y_min, y_max]]
            plot_lims = [[x_min, x_max], [y_min, y_max]]
            if experiment == 'cylinder':
                Xm_plot = Xm
                U_plot = U_0
                V_plot = V_0
                T_plot = T_0
                Y_plot = Y_0
            else: # Add last columns to plot
                Xm_plot = np.c_[Xm, np.ones(Ny) * x_max]
                U_plot = np.c_[U_0, U_0[:,0]]
                V_plot = np.c_[V_0, V_0[:,0]]
                T_plot = np.c_[T_0, T_0[:,0]]
                Y_plot = np.c_[Y_0, Y_0[:,0]]
            # Add last columns to plot
            # if experiment == 'fire' or experiment == 'cavity':
            #     Xm_plot = np.c_[Xm, np.ones(Ny) * x_max]
            #     U_plot = np.c_[U_0, U_0[:,0]]
            #     V_plot = np.c_[V_0, V_0[:,0]]
            #     T_plot = np.c_[T_0, T_0[:,0]]
            #     Y_plot = np.c_[Y_0, Y_0[:,0]]
            # else:
            #     Xm_plot = Xm
            #     U_plot = U_0
            #     V_plot = V_0
            #     T_plot = T_0
            #     Y_plot = Y_0
            S_plot = np.sqrt(U_plot ** 2 + V_plot ** 2)
            # Plots
            all_plots = {
                'u': U_plot,
                'v': V_plot,
                's': S_plot,
                'T': T_plot,
                'Y': Y_plot
            }
            selected_plots = ['s', 'T']#, 'Y']
            plots_to_show = {key: value for key, value in all_plots.items() if key in selected_plots}
            plot_initial_conditions(Xm_plot, Ym, plots_to_show, plot_lims=plot_lims, visualization='vertical')
        if self.parameters['debug']:  
            raise SystemExit()
        # We assume Dirichlet boundary conditions on the beginning
        dirichlet_y_bc = np.array([
            [U_0[0], U_0[-1]],
            [V_0[0], V_0[-1]],
            [T_0[0], T_0[-1]],
            [Y_0[0], Y_0[-1]],
            [P_0[0], P_0[-1]]
        ])
        # Add parameters to the dictionary
        self.parameters['x'] = x
        self.parameters['y'] = y
        self.parameters['t'] = t
        self.parameters['Xm'] = Xm
        self.parameters['Ym'] = Ym
        self.parameters['dx'] = dx
        self.parameters['dy'] = dy
        self.parameters['dt'] = dt
        self.parameters['Nx'] = Nx
        self.parameters['Ny'] = Ny
        self.parameters['Nt'] = Nt
        self.parameters['NT'] = NT
        self.parameters['u0'] = U_0
        self.parameters['v0'] = V_0
        self.parameters['T0'] = T_0
        self.parameters['Y0'] = Y_0
        self.parameters['F'] = F_e
        self.parameters['bc_on_z'] = dirichlet_y_bc
        self.parameters['cut_nodes'] = cut_nodes
        self.parameters['dead_nodes'] = dead_nodes
        self.parameters['dead_nodes_values'] = dead_nodes_values
        self.parameters['Y_top'] = topo_distance
        if self.parameters['t_source'] > 0:
            T_mask = T_0 != self.parameters['T_inf'] #plate(Xm, Ym) #shape(Xm, Ym) > 0.01
            T_source = T_0
        else:
            T_mask = T_source = None
        self.parameters['T_mask'] = T_mask
        self.parameters['T_source'] = T_source
        # Show parameters
        log_params(self.parameters)

    def initialize_3D(self):
        # Extract parameters to initialize
        (x_min, x_max, Nx), (y_min, y_max, Ny), (z_min, z_max, Nz), (t_min, t_max, Nt, NT) = self.domain
        # Create arrays
        x, y, z, t, Xm, Ym, Zm, dx, dy, dz, dt = domain_3D(x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max, Nx, Ny, Nz, Nt)
        # Evaluate initial conditions 
        u0, v0, w0 = U0
        U_0 = u0(Xm, Ym, Zm)
        V_0 = v0(Xm, Ym, Zm)
        W_0 = w0(Xm, Ym, Zm)
        T_0 = T0(Xm, Ym, Zm) 
        Y_0 = Y0(Xm, Ym, Zm)
        P_0 = p0(Xm, Ym, Zm)
        F_e = F(Xm, Ym, Zm)
        # Topography effect
        cut_nodes, dead_nodes = topography_nodes_3D(Xm, Ym, Zm, topo, dx, dy, dz)
        topo_distance = topography_distance_3D(Xm, Ym, Zm, topo)
        if self.parameters['input_ic'] is not True:
            U_0[dead_nodes] = self.parameters['dead_nodes_values'][0]
            V_0[dead_nodes] = self.parameters['dead_nodes_values'][1]
            W_0[dead_nodes] = self.parameters['dead_nodes_values'][2]
            T_0[dead_nodes] = self.parameters['dead_nodes_values'][3]
            Y_0[dead_nodes] = self.parameters['dead_nodes_values'][4]
        if self.parameters['show_ic']:
            plot_lims = [[0, 200], [0, 200], [0, 20]]
            # plot_lims = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            S_0 = np.sqrt(U_0 ** 2 + V_0 ** 2 + W_0 ** 2)
            # T_mask = T_0 > 300
            # Xp = Xm[T_mask]
            # Yp = Ym[T_mask]
            # Zp = Zm[T_mask]
            # Tp = T_0[T_mask]
            # plot_initial_conditions_3D(Xp, Yp, Zp, U_0, V_0, W_0, S_0, Tp, Y_0, plot_lims=[[0, 200], [0, 200], [0, 20]])
            # plot_initial_conditions_3D(Xm, Ym, Zm, U_0, V_0, W_0, S_0, T_0, Y_0, plot_lims=[[0, 200], [0, 200], [0, 20]])
            y_j = Ny // 2 
            z_k = 1
            plot_initial_conditions_3D(Xm[y_j], Ym[y_j], Zm[y_j], U_0[y_j], V_0[y_j], W_0[y_j], S_0[y_j], T_0[y_j], Y_0[y_j], plot_lims=plot_lims)
            plot_initial_conditions(Xm[:,:,z_k], Ym[:,:,z_k], U_0[:,:,z_k], V_0[:,:,z_k], S_0[:,:,z_k], T_0[:,:,z_k], Y_0[:,:,z_k], plot_lims=plot_lims)
            # plot_initial_conditions(Xm[:,:,z_k], Ym[:,:,z_k], U_0[:,:,z_k], V_0[:,:,z_k], S_0[:,:,z_k], T_0[:,:,z_k], Y_0[:,:,z_k], plot_lims=[[x_min, x_max], [y_min, y_max]])
        if self.parameters['debug']: 
            raise SystemExit()
        # We assume Dirichlet boundary conditions on the beginning
        dirichlet_z_bc = np.array([
            [U_0[:,:,0], U_0[:,:,-1]],
            [V_0[:,:,0], V_0[:,:,-1]],
            [W_0[:,:,0], W_0[:,:,-1]],
            [T_0[:,:,0], T_0[:,:,-1]],
            [Y_0[:,:,0], Y_0[:,:,-1]],
            [P_0[:,:,0], P_0[:,:,-1]]
        ])
        # Get Poisson problem indices
        # indices = pre_computation(Nx, Ny, dz, x_max, y_max)
        indices = None
        # Add parameters to the dictionary
        self.parameters['x'] = x
        self.parameters['y'] = y
        self.parameters['z'] = z
        self.parameters['t'] = t
        self.parameters['Xm'] = Xm
        self.parameters['Ym'] = Ym
        self.parameters['Zm'] = Zm
        self.parameters['dx'] = dx
        self.parameters['dy'] = dy
        self.parameters['dz'] = dz
        self.parameters['dt'] = dt
        self.parameters['Nx'] = Nx
        self.parameters['Ny'] = Ny
        self.parameters['Nz'] = Nz
        self.parameters['Nt'] = Nt
        self.parameters['NT'] = NT
        self.parameters['u0'] = U_0
        self.parameters['v0'] = V_0
        self.parameters['w0'] = W_0
        self.parameters['T0'] = T_0
        self.parameters['Y0'] = Y_0
        self.parameters['F'] = F_e
        self.parameters['bc_on_z'] = dirichlet_z_bc
        self.parameters['cut_nodes'] = cut_nodes
        self.parameters['dead_nodes'] = dead_nodes
        self.parameters['indices'] = indices
        # for gamma in indices:
        #     print("gamma:", gamma)
        #     print("r:", indices[gamma]['r'])
        #     print("s:", indices[gamma]['s'])
        # for i in range(len(indices)):
        #     print("gamma:", indices[i][0])
        #     print("r:", indices[i][1])
        #     print("s:", indices[i][2])
        # print(asd)
        # self.parameters['dead_nodes_values'] = dead_nodes_values
        self.parameters['Y_top'] = topo_distance
        if self.parameters['t_source'] > 0:
            T_mask = T_0 > 300 #plate(Xm, Ym) #shape(Xm, Ym) > 0.01
            T_source = T_0
        else:
            T_mask = T_source = None
        self.parameters['T_mask'] = T_mask
        self.parameters['T_source'] = T_source
        # Show parameters
        log_params(self.parameters)

    def solve(self):
        # Initial conditions
        if self.ndims == 3:
            r_0 = np.array([
                self.parameters['u0'], 
                self.parameters['v0'],
                self.parameters['T0'],
                self.parameters['Y0']
            ])
            self.data = solve_pde_2D(r_0, self.parameters)
        elif self.ndims == 4:
            r_0 = np.array([
                self.parameters['u0'], 
                self.parameters['v0'],
                self.parameters['w0'],
                self.parameters['T0'],
                self.parameters['Y0']
            ])
            self.data = solve_pde_3D(r_0, self.parameters)

    def postprocess(self):
        log_params(self.parameters, True)
        # Save outputs
        save_approximation(self.parameters, self.data)
        save_parameters(self.parameters)
        print("Simulation name:", self.parameters['sim_name']) # To easily get the name of the simulation for visualization