import os
import imageio
import argparse
from plots import plot, load_data_for_plots
import warnings
warnings.filterwarnings("ignore") # To remove warnings from contourf using NaNs

# Get arguments
parser = argparse.ArgumentParser(description='Visualization of numerical simulations')
parser.add_argument('-l', '--layout', type=str, help='Visualization layout. Options: "horizontal" or "vertical". Default: "vertical".', default="vertical")
parser.add_argument('-p', '--plots', type=str, help='Plots to show. Options: u, v, modU, divU, curlU, T, Y, p. Default: modU T p.', default="modU T Y")
parser.add_argument('-s', '--show', type=str, help='Show, PDF, video or GIF. Options: "plot", "pdf", "video" or "gif". Default: "plot".', default="plot")
parser.add_argument('-ts', '--time-sample', type=int, help='Time sample step. Default 1', default=1)
parser.add_argument('-tn', '--time-step', type=int, help='Show up to time step n. Default None (all data)', default=None)
parser.add_argument('-i', '--input', type=str, help='Simulation directory.', required=True)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default='')
parser.add_argument('-xmin', '--x-min', type=float, default=0, help="Left boundary of domain in x.")
parser.add_argument('-xmax', '--x-max', type=float, default=200, help="Right boundary of domain in x.")
parser.add_argument('-ymin', '--y-min', type=float, default=0, help="Bottom boundary of domain in y.")
parser.add_argument('-ymax', '--y-max', type=float, default=20, help="Top boundary of domain in y.")
args = parser.parse_args()

# Default values
layout = args.layout #"horizontal" # or vertical
plots = args.plots.split() #"modU T p"
input_dir = args.input
output_dir = args.output
show = args.show # plot, video or gif
streamplot = True
qs = 1 # Quiver samples
ts = args.time_sample # Time samples
tn = args.time_step # Up to time step. -1 for all
if input_dir[-1] != "/":
    input_dir += "/"
if output_dir == "":
    output_dir = input_dir
if output_dir[-1] != "/":
    output_dir += "/"
data_path = input_dir + "/data.npz"
parameters_path = input_dir + "/parameters.pkl"
U_comp = ['modU', 'divU', 'curlU'] # Computation

# Create name for video
if show != "plot":
    sim_id = input_dir.split("/")[-2]
    gif_name = output_dir + sim_id + ".gif"
    video_name = output_dir + sim_id + ".mp4"

# Load data
x, y, t, data_plots = load_data_for_plots(data_path, parameters_path, plots, tn=None)

# Computational domain
x_min, x_max = x[0], x[-1]
y_min, y_max = y[0], y[-1]
# Plot domain. Overwrite if specified in arguments
x_min, x_max = args.x_min, args.x_max
y_min, y_max = args.y_min, args.y_max
plot_lims = [x_min, x_max, y_min, y_max]

# Filenames for output
filenames = []

filename = None # Assume it is just a plot
# File extensions
if show != "plot":
    ext = '.png' if show in ['gif', 'video'] else '.pdf'

# Number of samples to show
Nt = t.shape[0]
# Plot
for n in range(0, Nt, ts):
# for n in [0, Nt // 2, Nt - 1]:
    if show != 'plot':
        print("Creating figure %d/%d" % (n+1, Nt))
        filename = output_dir + str(n) + ext 
        filenames.append(filename)
    plot(n, t, x, y, data_plots, plot_lims, title=True, filename=filename, dpi=400)

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