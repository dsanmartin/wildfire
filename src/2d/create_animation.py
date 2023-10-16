import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse
import pickle
from matplotlib.colors import ListedColormap
from plots import plot_scalar_field, plot_vector_field
import warnings
warnings.filterwarnings("ignore") # To remove warnings from contourf using NaNs

black_cmap = ListedColormap(['black'])

parser = argparse.ArgumentParser(description='Visualization of numerical simulations')
parser.add_argument('-v', '--visualization', type=str, 
    help='Type of visualization. Options: "horizontal" or "vertical". Default: "vertical".', default="vertical")
parser.add_argument('-p', '--plots', type=str, 
    help='Plots to show. Options: u, v, modU, divU, curlU, T, Y, p. Default: modU T p.', default="modU T Y")
parser.add_argument('-s', '--show', type=str, 
    help='Show, PDF, video or GIF. Options: "plot", "pdf", "video" or "GIF". Default: "plot".', default="plot")
parser.add_argument('-t', '--time-sample', type=int, help='Time sample step. Default 1', default=1)
parser.add_argument('-n', '--time-step', type=int, help='Up to time step n. Default 0 (all data)', default=0)
parser.add_argument('-i', '--input', type=str, help='Simulation directory.', required=True)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default='')
# parser.add_argument('-xmin', '--x-min', type=float, default=666, help="Left boundary of domain in x.")
# parser.add_argument('-xmax', '--x-max', type=float, default=666, help="Right boundary of domain in x.")
# parser.add_argument('-ymin', '--y-min', type=float, default=666, help="Bottom boundary of domain in y.")
# parser.add_argument('-ymax', '--y-max', type=float, default=666, help="Top boundary of domain in y.")
parser.add_argument('-xmin', '--x-min', type=float, default=0, help="Left boundary of domain in x.")
parser.add_argument('-xmax', '--x-max', type=float, default=200, help="Right boundary of domain in x.")
parser.add_argument('-ymin', '--y-min', type=float, default=0, help="Bottom boundary of domain in y.")
parser.add_argument('-ymax', '--y-max', type=float, default=20, help="Top boundary of domain in y.")
args = parser.parse_args()

# Default values
visualization = args.visualization #"horizontal" # or vertical
plots = args.plots.split() #"modU T p"
input_dir = args.input
output_dir = args.output
show = args.show # plot, video or gif
streamplot = True
qs = 2 # Quiver samples
ts = args.time_sample # Time samples
stop = args.time_step # Up to time step. -1 for all
if input_dir[-1] != "/":
    input_dir += "/"
if output_dir == "":
    output_dir = input_dir
if output_dir[-1] != "/":
    output_dir += "/"
filename = input_dir + "/data.npz"
U_comp = ['modU', 'divU', 'curlU'] # Computation

# Load data
data = np.load(filename)
with open(input_dir + 'parameters.pkl', 'rb') as fp:
    parameters = pickle.load(fp)

# Create name for video
if show != "plot":
    sim_id = parameters['sim_name']
    gif_name = output_dir + sim_id + ".gif"
    video_name = output_dir + sim_id + ".mp4"

# Domain
x = data['x']
y = data['y']
t = data['t']
# Computational domain
x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]
dx = x[1] - x[0]
dy = y[1] - y[0]
x, y = np.meshgrid(x, y)
# Plot domain
x_min, x_max = args.x_min, args.x_max
y_min, y_max = args.y_min, args.y_max

if stop != 0:
    t = t[:stop]

if "u" in plots or any(x in plots for x in U_comp):
    u = data['u']
    if stop != 0:
        u = u[:stop]
    u_min, u_max = np.min(u) - 1, np.max(u) + 1
    u_ticks = np.linspace(np.min(u), np.max(u), 5)
if "v" in plots or any(x in plots for x in U_comp):
    v = data['v']
    if stop != 0:
        v = v[:stop]
    v_min, v_max = np.min(v) - 1, np.max(v) + 1
    v_ticks = np.linspace(np.min(v), np.max(v), 5)
if "T" in plots:
    T = data['T']
    if stop != 0:
        T = T[:stop]
    T_min, T_max = np.min(T) - 100, np.max(T) + 100
    T_ticks = np.linspace(np.min(T), np.max(T), 5)
if "Y" in plots:
    Y = data['Y']
    if stop != 0:
        Y = Y[:stop]
    Y_min, Y_max = np.min(Y) - 0.01, np.max(Y) + 0.01
    Y_ticks = np.linspace(np.min(Y), np.max(Y), 5)
    # Load terrain
    dead_nodes = parameters['dead_nodes']
    terrain = np.zeros_like(Y[0])
    terrain[:] = np.nan
    terrain[dead_nodes] = 1
if "p" in plots:
    p = data['p']
    if stop != 0:
        p = p[:stop]
    p_min, p_max = np.min(p) - 1, np.max(p) + 1
    p_ticks = np.linspace(np.min(p), np.max(p), 5)

# Vector field details
# Derivatives
if "divU" in plots or "curlU" in plots:
    ux = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dx)
    uy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    vx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
    vy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    uy[:, 0, :] = (-3 * u[:, 0, :] + 4 * u[:, 1, :] - u[:, 2, :]) / (2 * dy) # Forward at y=y_min
    uy[:, -1,:] = (3 * u[:, -1, :] - 4 * u[:, -2,:] + u[:, -3,:]) / (2 * dy) # Backward at y=y_max
    vy[:, 0, :] = (-3 * v[:, 0, :] + 4 * v[:, 1, :] - v[:, 2, :]) / (2 * dy) # Forward at y=y_min
    vy[:, -1,:] = (3 * v[:, -1, :] - 4 * v[:, -2,:] + v[:, -3,:]) / (2 * dy) # Backward at y=y_max
    # Divergence
    divU =  ux + vy
    # Curl
    curlU = vx - uy
    divU_min, divU_max = np.min(divU) - 1, np.max(divU) + 1
    curlU_min, curlU_max = np.min(curlU) - 1, np.max(curlU) + 1
    divU_ticks = np.linspace(np.min(divU), np.max(divU), 5)
    curlU_ticks = np.linspace(np.min(curlU), np.max(curlU), 5)
if "p" in plots:
    px = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2 * dx)
    py = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)
    py[:, 0, :] = (-3 * p[:, 0, :] + 4 * p[:, 1, :] - p[:, 2, :]) / (2 * dy) # Forward at y=y_min
    py[:, -1,:] = (3 * p[:, -1, :] - 4 * p[:, -2,:] + p[:, -3,:]) / (2 * dy) # Backward at y=y_max
# Speed
if "modU" in plots:
    modU = np.sqrt(u ** 2 + v ** 2) 
    modU_min, modU_max = np.min(modU) - 1, np.max(modU) + 1
    modU_ticks = np.linspace(np.min(modU), np.max(modU), 5)

# Filenames for output
filenames = []

# Plots 
n_plots = len(plots)

args = {}
if show == "plot":
    args['figsize'] = (12, n_plots * 2)
else:
    args['dpi'] = 400

# Axis labels
vvariable = r'$z$ (m)'
hvariable = r'$x$ (m)'

# Number of samples to show
Nt = t.shape[0]
# Plot
for n in range(0, Nt, ts):
# for n in [0, Nt // 2, Nt - 1]:
    if show != 'plot':
        print("Creating figure %d/%d" % (n+1, Nt))

    if visualization == "horizontal":
        fig, axes = plt.subplots(1, n_plots, sharey=True, **args)
        axes[0].set_ylabel(vvariable)
        axes[0].set_ylim(y_min, y_max)
        for i in range(len(axes)):
            axes[i].set_xlabel(hvariable)
            axes[i].set_xlim(x_min, x_max)
    else: # Vertical
        fig, axes = plt.subplots(n_plots, 1, sharex=True, **args)
        axes[-1].set_xlabel(hvariable)
        axes[-1].set_xlim(x_min, x_max)
        for i in range(len(axes)):
            axes[i].set_ylabel(vvariable)
            axes[i].set_ylim(y_min, y_max)

    fig.suptitle(r'Simulation at $t=%.1f$ s' % (t[n]))
    fig.subplots_adjust(top=0.88)

    i = 0

    if "u" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, u[n], plt.cm.viridis, 
            (u_min, u_max), u_ticks, r'Velocity component  $u$', r'm s$^{-1}$'
        )
        i += 1

    if "v" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, v[n], plt.cm.viridis, 
            (v_min, v_max), v_ticks, r'Velocity component  $v$', r'm s$^{-1}$'
        )
        i += 1

    if "modU" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, modU[n], plt.cm.viridis, 
            (modU_min, modU_max), modU_ticks, r'Velocity $\mathbf{u}$, Speed $||\mathbf{u}||_2$', r'm s$^{-1}$', .8
        )
        plot_vector_field(
            axes[i], x, y, u[n], v[n], streamplot=streamplot, qs=qs
        )
        i += 1

    if "divU" in plots:
        # Plot divergence
        plot_scalar_field(
            fig, axes[i], x, y, divU[n], plt.cm.viridis, 
            (divU_min, divU_max), divU_ticks, r'Divergence $\nabla\cdot\mathbf{u}$', r's$^{-1}$'
        )
        i += 1

    if "curlU" in plots:
        # Plot curl
        plot_scalar_field(
            fig, axes[i], x, y, curlU[n], plt.cm.viridis, 
            (curlU_min, curlU_max), curlU_ticks, r'Vorticity $\nabla\times\mathbf{u}$', r's$^{-1}$'
        )
        i += 1

    if "T" in plots:
        # Plot temperature
        plot_scalar_field(
            fig, axes[i], x, y, T[n], plt.cm.jet, 
            (T_min, T_max), T_ticks, r'Temperature $T$', r'K'
        )
        i += 1
    
    if "Y" in plots:
        # Plot fuel
        plot_scalar_field(
            fig, axes[i], x, y, Y[n], plt.cm.Oranges, (Y_min, Y_max), Y_ticks, r'Fuel $Y$', r'%'
        )
        # Plot terrain
        plot_scalar_field(
            fig, axes[i], x, y, terrain, black_cmap, (0, 1), None, None, None
        )
        i += 1

    if "p" in plots:
        plot_scalar_field(
            fig, axes[i], x, y, p[n], plt.cm.viridis, 
            (p_min, p_max), p_ticks, r'Pressure $p$', r"kg m$^{-1}s^{-2}$", .8
        )
        plot_vector_field(
            axes[i], x, y, px[n], py[n], streamplot=streamplot, qs=qs
        )
        i += 1
    

    fig.tight_layout()

    # Save figures
    if show == "plot":
        plt.show()
    elif show == 'pdf':
        name = f'{n}.pdf'
        plt.savefig(input_dir + name, transparent=True, dpi=400)
        plt.close()
    else:
        name = f'{n}.png'
        filenames.append(input_dir + name)
        plt.savefig(input_dir + name)
        plt.close()

# Build video or GIF
if show not in ["plot", "pdf"]:
    if show == "video":
        io_writer = imageio.get_writer(video_name)
    else:
        io_writer = imageio.get_writer(gif_name, mode='I')

    with io_writer as writer:
        for i, filename in enumerate(filenames):
            print("Processing figure %d/%d" % (i+1, Nt))
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)
            
    print("Done!")