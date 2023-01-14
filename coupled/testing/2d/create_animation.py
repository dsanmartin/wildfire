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
    help='Plots to show. Options: u, v, modU, divU, curlU, T, Y, p. Default: modU T p.', default="modU T p")
parser.add_argument('-s', '--show', type=str, 
    help='Show, video or GIF. Options: "plot", "video" or "GIF". Default: "plot".', default="plot")
parser.add_argument('-t', '--ts', type=int, help='Time sample. Default 1', default=1)
parser.add_argument('-i', '--input', type=str, help='Simulation directory.', required=True)
args = parser.parse_args()

# Default values
visualization = args.visualization #"horizontal" # or vertical
plots = args.plots.split() #"modU T p"
filename = args.input
show = args.show # plot, video or gif
streamplot = True
qs = 2 # Quiver samples
ts = args.ts # Time samples

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
if "v" in plots:
    v = data['v']
if "T" in plots:
    T = data['T']
if "Y" in plots:
    Y = data['Y']
if "p" in plots:
    p = data['p']
if "modU" in plots or "divU" in plots or "curlU" in plots:
    u = data['u']
    v = data['v']

# Domain
x = data['x']
y = data['y']
t = data['t']
x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]
dx = x[1] - x[0]
dy = y[1] - y[0]
x, y = np.meshgrid(x, y)


# stop = 200
# t = t[:stop]
# u = u[:stop]
# v = v[:stop]
# T = T[:stop]
# Y = Y[:stop]

# print(np.min(u), np.max(u))

# Vector field details
# Derivatives
if "divU" in plots or "curlU" in plots:
    ux = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dx)
    uy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    vx = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dx)
    vy = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dy)
    # Divergence
    divU =  ux + vy
    # Curl
    curlU = vx - uy
if "p" in plots:
    px = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2 * dx)
    py = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)

# Speed
modU = np.sqrt(u ** 2 + v ** 2) 

# Filenames for output
filenames = []

# Plots 
n_plots = len(plots)

args = {}
if show == "plot":
    args['figsize'] = (12, 4)
else:
    args['dpi'] = 400

vvariable = r'$z$ (m)'
hvariable = r'$x$ (m)'

# Plot
for n in range(0, t.shape[0], ts):
# for n in [10]:

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

    fig.suptitle(r'Simulation a $t=%.1f$ [s]' % (t[n]))
    fig.subplots_adjust(top=0.88)

    i = 0

    if "u" in plots:
        pi = axes[i].contourf(x, y, u[n], cmap=plt.cm.viridis, vmin=np.min(u), vmax=np.max(u))
        fig.colorbar(pi, ax=axes[i], label=r'm s$^{-1}$')
        axes[i].set_title("Velocity component " + r'$u$')
        i += 1

    if "v" in plots:
        pi = axes[i].contourf(x, y, v[n], cmap=plt.cm.viridis, vmin=np.min(v), vmax=np.max(v))
        fig.colorbar(pi, ax=axes[i], label=r'm s$^{-1}$')
        axes[i].set_title("Velocity component " + r'$v$')
        i += 1

    if "modU" in plots:
        pi = axes[i].contourf(x, y, modU[n], cmap=plt.cm.viridis, alpha=.8, vmin=np.min(modU), vmax=np.max(modU))
        fig.colorbar(pi, ax=axes[i], label=r'm s$^{-1}$')
        if streamplot: axes[i].streamplot(x, y, u[n], v[n], density=1.2, linewidth=.5, arrowsize=.3, color='k')
        else: axes[i].quiver(x[::qs,::qs], y[::qs,::qs], u[n,::qs,::qs], v[n,::qs,::qs])
        axes[i].set_title("Velocity " + r'$\mathbf{u}, ||\mathbf{u}||_2$')
        i += 1

    if "divU" in plots:
        pi = axes[i].contourf(x, y, divU[n], cmap=plt.cm.viridis, vmin=np.min(divU), vmax=np.max(divU))
        fig.colorbar(pi, ax=axes[i], label=r's$^{-1}$')
        axes[i].set_title(r'$\nabla\cdot\mathbf{u}$')
        i += 1

    if "curlU" in plots:
        pi = axes[i].contourf(x, y, curlU[n], cmap=plt.cm.viridis, vmin=np.min(curlU), vmax=np.max(curlU))
        fig.colorbar(pi, ax=axes[i], label=r's$^{-1}$')
        axes[i].set_title(r'$\nabla\times\mathbf{u}$')
        i += 1

    if "T" in plots:
        pi = axes[i].contourf(x, y, T[n],cmap=plt.cm.jet, vmin=np.min(T), vmax=np.max(T))
        axes[i].set_title("Temperature " + r'$T$')
        fig.colorbar(pi, ax=axes[i], label='K')
        i += 1
    
    if "Y" in plots:
        pi = axes[i].contourf(x, y, Y[n], cmap=plt.cm.Oranges, vmin=np.min(Y), vmax=np.max(Y))
        axes[i].set_title(r'$Y$')
        fig.colorbar(pi, ax=axes[i], label="%")

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
    else:
        name = f'{n}.png'
        filenames.append(base_dir + name)
        plt.savefig(base_dir + name)
        plt.close()

# Build video or GIF
if show != "plot":
    if show == "video":
        io_writer = imageio.get_writer(video_name)
    else:
        io_writer = imageio.get_writer(gif_name, mode='I')

    with io_writer as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)