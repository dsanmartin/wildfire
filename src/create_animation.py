import os
import imageio
import argparse
import numpy as np
from plots import load_data_for_plots, plot_2D
from skimage.transform import resize
import warnings
warnings.filterwarnings("ignore") # To remove warnings from contourf using NaNs

# Get arguments
parser = argparse.ArgumentParser(description='Visualization of numerical simulations')
parser.add_argument('-l', '--layout', type=str, help='Visualization layout. Options: "horizontal" or "vertical". Default: "vertical".', default="vertical")
parser.add_argument('-p', '--plots', type=str, help='Plots to show. Options: u, v, modU, divU, curlU, T, Y, p. Default: modU,T,p.', default="modU,T,Y")
parser.add_argument('-s', '--show', type=str, help='Show, PDF, video or GIF. Options: "plot", "pdf", "video" or "gif". Default: "plot".', default="plot")
parser.add_argument('-ts', '--time-sample', type=int, help='Time sample step. Default 1', default=1)
parser.add_argument('-tn', '--time-step', type=int, help='Show up to time step n. Default None (all data)', default=None)
parser.add_argument('-i', '--input', type=str, help='Simulation directory.', required=True)
parser.add_argument('-o', '--output', type=str, help='Output directory.', default='')
parser.add_argument('-xmin', '--x-min', type=float, default=0, help="Left boundary of domain in x.")
parser.add_argument('-xmax', '--x-max', type=float, default=200, help="Right boundary of domain in x.")
parser.add_argument('-ymin', '--y-min', type=float, default=0, help="Bottom boundary of domain in y.")
parser.add_argument('-ymax', '--y-max', type=float, default=20, help="Top boundary of domain in y.")
parser.add_argument('-zmin', '--z-min', type=float, default=-1, help="Bottom boundary of domain in z.")
parser.add_argument('-zmax', '--z-max', type=float, default=-1, help="Top boundary of domain in z.")
parser.add_argument('-v', '--visualization', type=str, default='vertical', 
    help="Slice to show. Options: 'vertical', 'horizontal' or 'longitudinal'. Default: 'vertical'.")
parser.add_argument('-b', '--bounds', type=int, default=1, help="Use scalar bounds in plots. Default: True.")
parser.add_argument('-fps', '--fps', type=int, default=10, help="Frames per second for video. Default: 10.")
args = parser.parse_args()

# Default values
layout = args.layout #"horizontal" # or vertical
plots = args.plots.split(',') #"modU T p"
input_dir = args.input
output_dir = args.output
show = args.show # plot, video or gif
visualization = args.visualization # vertical, horizontal or longitudinal
streamplot = True 
density = 5 # Streamplot density
if visualization == "longitudinal" or visualization == "horizontal":
    density = 5
else:
    density = 0.6 
qs = 1 # Quiver samples
ts = args.time_sample # Time samples
tn = args.time_step # Up to time step. -1 for all
if input_dir[-1] != "/":
    input_dir += "/"
if output_dir == "":
    output_dir = input_dir
if output_dir[-1] != "/":
    output_dir += "/"
data_path = input_dir + "data.npz"
parameters_path = input_dir + "/parameters.pkl"
U_comp = ['modU', 'divU', 'curlU'] # Computation
dpi = 200
filename = None
bounds = ticks = args.bounds
fps = args.fps

# Parameters for video or GIF
if show != "plot":
    sim_id = input_dir.split("/")[-2]
    name = output_dir + sim_id + "_" + visualization[0]
    gif_name = name + ".gif"
    video_name = name + ".mp4"
    ext = '.png' if show in ['gif', 'video'] else '.pdf'
    dpi = 400

# Load data
domain, data_plots = load_data_for_plots(data_path, parameters_path, plots, tn=None)

if len(domain) == 3:
    x, y, t = domain
    z = None
    slices = None
elif len(domain) == 4:
    x, y, z, t = domain
    slices = [x.shape[0] // 2, y.shape[0] // 2, 1]

# Computational domain
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
# Plot domain. Overwrite if specified in arguments
x_min, x_max = args.x_min, args.x_max
y_min, y_max = args.y_min, args.y_max
plot_lims = [x_min, x_max, y_min, y_max]
if z is not None:
    z_min, z_max = args.z_min, args.z_max
    if z_min == z_max == -1:
        z_min, z_max = z.min(), z.max()
    plot_lims = [x_min, x_max, y_min, y_max, z_min, z_max]

# Filenames for output
filenames = []

# Number of samples to show
if tn == -1:
    Nt = t.shape[0]
else:
    Nt = tn

# Plot
for n in range(0, Nt, ts):
# for n in [0, Nt // 2, Nt - 1]:
    if show != 'plot':
        print("Creating figure %d/%d" % (n+1, Nt))
        filename = output_dir + str(n) + ext 
        filenames.append(filename)
    plot_2D(n, domain, data_plots, plot_lims, visualization=visualization, title=True, 
            streamplot=streamplot, qs=qs, density=density, 
            filename=filename, dpi=dpi, 
            bounds=bounds, ticks=ticks, slices=slices)

# Build video or GIF
if show not in ["plot", "pdf"]:
    if show == "video":
        io_writer = imageio.get_writer(video_name, fps=fps)
    else:
        io_writer = imageio.get_writer(gif_name, mode='I')

    with io_writer as writer:
        for i, filename in enumerate(filenames):
            print("Processing figure %d/%d" % (i+1, Nt))
            image = imageio.imread(filename)
            if i == 0:
                height, width, _ = image.shape
            image = resize(image, (height, width))
            image = (255 * image).astype(np.uint8)
            writer.append_data(image)
            # image = Image.open(filename)
            # resized_image = image.resize((width, height))
            # writer.append_data(np.array(resized_image))
            os.remove(filename)
            
    print("Done!")