import numpy as np

def show_info(params):
    # Get data
    x_min, x_max = params['x'][0], params['x'][-1]
    y_min, y_max = params['y'][0], params['y'][-1]
    t_min, t_max = params['t'][0], params['t'][-1]
    Nx, Ny, Nt, NT = params['Nx'],params['Ny'], params['Nt'], params['NT']
    dx, dy, dt = params['dx'], params['dy'], params['dt']
    U_0, V_0, T_0 = params['u0'], params['v0'], params['T0']
    sim_name = params['sim_name']
    method = params['method']
    TA, T_inf, T_pc = params['TA'], params['T_inf'], params['T_pc']
    nu, k, Pr, g = params['nu'], params['k'], params['Pr'], params['g']
    A, B, H_R, h = params['A'], params['B'], params['H_R'], params['h']
    Y_f, Y_thr = params['Y_f'], params['Y_thr']
    turb, conser = params['turbulence'], params['conservative']

    L = ((x_max - x_min) * (y_max - y_min)) ** 0.5
    Re = np.mean(U_0[:,0]) * L / nu
    T_avg = np.mean(T_0[:, Nx // 2])
    beta = 1 / T_avg
    L_v = (y_max - y_min)
    Gr = abs(g[-1]) * beta * (TA - T_inf) * L_v ** 3 / nu ** 2 
    Ra = Gr * Pr
    print("Simulation name:", sim_name)
    print("Domain: [%.4f, %.4f] x [%.4f, %.4f] x [%.4f, %.4f]" % (x_min, x_max, y_min, y_max, t_min, t_max))
    print("Grid: Nx: %d, Ny: %d, Nt: %d" % (Nx, Ny, Nt))
    print("dx: %.4f, dy: %.4f, dt: %.4f" % (dx, dy, dt))
    print("Time integration: %s" % method)
    print("Samples: %d" % NT)
    print("nu: %.2e, g: (%.4f, %.4f)" % (nu, g[0], g[1]))
    print("k: %.2e, T_inf: %.4f, T_hot: %.4f" % (k, T_inf, TA))
    print("T_pc: %.4f, A: %.4f, B: %.4f" % (T_pc, A, B))
    print("H_R: %.4f, h: %.4f, Y_thr: %.4f, Y_f: %.4f" % (H_R, h, Y_thr, Y_f))
    print("Turbulence: %r" % turb)
    print("Conservative: %r" % conser)
    print("Reynolds: %.4f" %  Re)
    print("Prandtl: %.4f" % Pr)
    print("Grashof: %.2e" % Gr)
    print("Rayleigh: %.2e\n" % Ra)