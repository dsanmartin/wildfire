import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import argparse

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
parser.add_argument('-xmin', '--x-min', type=float, default=666, help="Left boundary of domain in x.")
parser.add_argument('-xmax', '--x-max', type=float, default=666, help="Right boundary of domain in x.")
parser.add_argument('-ymin', '--y-min', type=float, default=666, help="Bottom boundary of domain in y.")
parser.add_argument('-ymax', '--y-max', type=float, default=666, help="Top boundary of domain in y.")
args = parser.parse_args()

# Default values
visualization = args.visualization #"horizontal" # or vertical
plots = args.plots.split() #"modU T p"
filename = args.input
show = args.show # plot, video or gif
streamplot = True
qs = 2 # Quiver samples
ts = args.time_sample # Time samples
stop = args.time_step # Up to time step. -1 for all

# Get base directory
if show != "plot":
    filename_split = filename.split("/")
    file = filename_split[-1]
    sim_id = filename_split[-2]
    base_dir = "/".join(filename_split[:-1]) + "/"
    # Ouput format
    gif_name = base_dir + sim_id + "_" + file.replace('npz', 'gif')
    video_name = base_dir + sim_id + "_" + file.replace('npz', 'mp4')

# Load data
data = np.load(filename)

if "u" in plots:
    u = data['u']
    if stop != 0:
        u = u[:stop]
if "v" in plots:
    v = data['v']
    if stop != 0:
        v = v[:stop]
if "T" in plots:
    T = data['T']
    if stop != 0:
        T = T[:stop]
if "Y" in plots:
    Y = data['Y']
    if stop != 0:
        Y = Y[:stop]
if "p" in plots:
    p = data['p']
    if stop != 0:
        p = p[:stop]
if "modU" in plots or "divU" in plots or "curlU" in plots:
    u = data['u']
    v = data['v']
    if stop != 0:
        u = u[:stop]
        v = v[:stop]

# Domain
x = data['x']
y = data['y']
t = data['t']
x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]
dx = x[1] - x[0]
dy = y[1] - y[0]
x, y = np.meshgrid(x, y)

if stop != 0:
    t = t[:stop]
    
# Change domain
if args.x_min != 666:
    x_min = args.x_min
if args.x_max != 666:
    x_max = args.x_max
if args.y_min != 666:
    y_min = args.y_min
if args.y_max != 666:
    y_max = args.y_max
# x_min, x_max = 0, 200
# y_min, y_max = 0, 20


# print(np.min(u), np.max(u))

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
if "p" in plots:
    px = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2 * dx)
    py = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)
    py[:, 0, :] = (-3 * p[:, 0, :] + 4 * p[:, 1, :] - p[:, 2, :]) / (2 * dy) # Forward at y=y_min
    py[:, -1,:] = (3 * p[:, -1, :] - 4 * p[:, -2,:] + p[:, -3,:]) / (2 * dy) # Backward at y=y_max

# Speed
modU = np.sqrt(u ** 2 + v ** 2) 

# Filenames for output
filenames = []

# Plots 
n_plots = len(plots)

args = {}
if show == "plot":
    args['figsize'] = (12, n_plots * 2)
else:
    args['dpi'] = 400

vvariable = r'$z$ (m)'
hvariable = r'$x$ (m)'

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
        pi = axes[i].contourf(x, y, u[n], cmap=plt.cm.viridis, vmin=np.min(u), vmax=np.max(u))
        fig.colorbar(pi, ax=axes[i], label=r'm s$^{-1}$')
        axes[i].set_title(r'Velocity component  $u$')
        i += 1

    if "v" in plots:
        pi = axes[i].contourf(x, y, v[n], cmap=plt.cm.viridis, vmin=np.min(v), vmax=np.max(v))
        fig.colorbar(pi, ax=axes[i], label=r'm s$^{-1}$')
        axes[i].set_title(r'Velocity component $v$')
        i += 1

    if "modU" in plots:
        pi = axes[i].contourf(x, y, modU[n], cmap=plt.cm.viridis, alpha=.8, vmin=np.min(modU), vmax=np.max(modU))
        fig.colorbar(pi, ax=axes[i], label=r'm s$^{-1}$')
        if streamplot: axes[i].streamplot(x, y, u[n], v[n], density=1.2, linewidth=.5, arrowsize=.3, color='k')
        else: axes[i].quiver(x[::qs,::qs], y[::qs,::qs], u[n,::qs,::qs], v[n,::qs,::qs])
        axes[i].set_title(r'Velocity $\mathbf{u}$, Speed $||\mathbf{u}||_2$')
        i += 1

    if "divU" in plots:
        pi = axes[i].contourf(x, y, divU[n], cmap=plt.cm.viridis, vmin=np.min(divU), vmax=np.max(divU))
        fig.colorbar(pi, ax=axes[i], label=r's$^{-1}$')
        axes[i].set_title(r'Divergence $\nabla\cdot\mathbf{u}$')
        i += 1

    if "curlU" in plots:
        pi = axes[i].contourf(x, y, curlU[n], cmap=plt.cm.viridis, vmin=np.min(curlU), vmax=np.max(curlU))
        fig.colorbar(pi, ax=axes[i], label=r's$^{-1}$')
        axes[i].set_title(r'Vorticity $\nabla\times\mathbf{u}$')
        i += 1

    if "T" in plots:
        pi = axes[i].contourf(x, y, T[n],cmap=plt.cm.jet, vmin=np.min(T), vmax=np.max(T))
        # pi = axes[i].imshow(T[n],cmap=plt.cm.jet, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', vmin=np.min(T), vmax=np.max(T))
        axes[i].set_title(r'Temperature $T$')
        fig.colorbar(pi, ax=axes[i], label='K')
        i += 1
    
    if "Y" in plots:
        pi = axes[i].contourf(x, y, Y[n], cmap=plt.cm.Oranges, vmin=np.min(Y), vmax=np.max(Y))
        axes[i].set_title(r'Fuel $Y$')
        fig.colorbar(pi, ax=axes[i], label="%")
        i += 1

    if "p" in plots:
        pi = axes[i].contourf(x, y, p[n], cmap=plt.cm.viridis, alpha=.8, vmin=np.min(p), vmax=np.max(p))
        if streamplot: axes[i].streamplot(x, y, px[n], py[n], density=1.2, linewidth=.5, arrowsize=.3, color='k')
        else: axes[i].quiver(x[::qs,::qs], y[::qs,::qs], px[n,::qs,::qs], py[n,::qs,::qs])
        axes[i].set_title("Pressure " + r'$p$')
        fig.colorbar(pi, ax=axes[i], label=r"kg m$^{-1}s^{-2}$")
        i += 1
    

    fig.tight_layout()

    # Save figures
    if show == "plot":
        plt.show()
    elif show == 'pdf':
        name = f'{n}.pdf'
        plt.savefig(base_dir + name, transparent=True, dpi=400)
        plt.close()
    else:
        name = f'{n}.png'
        filenames.append(base_dir + name)
        plt.savefig(base_dir + name)
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
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)