import argparse
from parameters import *
from datetime import datetime

parser = argparse.ArgumentParser(description='2D simplified wildfire numerical simulation')
# Parameters
parser.add_argument('-n', '--name', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="Numerical simulation name. Default: Current datetime with format 'YYYYMMDDhhmmss'")
parser.add_argument('-k', '--diffusivity', type=float, default=k,
    help="Thermal diffusivity parameter. Default: {}".format(k))
parser.add_argument('-nu', '--viscosity', type=float, default=nu,
    help="Kinematic viscosity parameter. Default: {}".format(nu))
parser.add_argument('-Pr', '--prandtl', type=float, default=Pr,
    help="Prandtl number parameter. Default: {}".format(Pr))
parser.add_argument('-fc', '--fuel-consumption', type=float, default=Y_f,
    help="Fuel consumption parameter. Default: {}".format(Y_f))
parser.add_argument('-Yt', '--fuel-threshold', type=float, default=Y_thr,
    help="Solid fuel threshold force. Default: {}".format(Y_thr))
parser.add_argument('-HR', '--heat-energy', type=float, default=H_R,
    help="Heat energy per unit of mass parameter. Default: {}".format(H_R))
parser.add_argument('-hc', '--heat-coefficient', type=float, default=h,
    help="Convective heat coefficient. Default: {}".format(h))
parser.add_argument('-Ta', '--activation-temperature', type=float, default=T_act,
    help="Activation energy. Default: {}".format(T_act))
parser.add_argument('-Th', '--hot-temperature', type=float, default=T_hot,
    help="Hot temperature. Default: {}".format(T_hot))
parser.add_argument('-St', '--source-top', type=float, default=S_top,
    help="Source top bound. Default: {}".format(S_top))
parser.add_argument('-Sb', '--source-bottom', type=float, default=S_bot,
    help="Source bottom bound. Default: {}".format(S_bot))
# Domain #
parser.add_argument('-xmin', '--x-min', type=float, default=x_min,
    help="Left boundary of domain in x. Default: {}".format(x_min))
parser.add_argument('-xmax', '--x-max', type=float, default=x_max,
    help="Right boundary of domain in x. Default: {}".format(x_max))
parser.add_argument('-ymin', '--y-min', type=float, default=y_min,
    help="Bottom boundary of domain in y. Default: {}".format(y_min))
parser.add_argument('-ymax', '--y-max', type=float, default=y_max,
    help="Top boundary of domain in y. Default: {}".format(y_max))
parser.add_argument('-tmin', '--t-min', type=float, default=t_min,
    help="Initial time. Default: {}".format(t_min))
parser.add_argument('-tmax', '--t-max', type=float, default=t_max,
    help="End time. Default: {}".format(t_max))
parser.add_argument('-Nx', '--x-nodes', type=int, default=Nx,
    help="Number of nodes in x. Default: {}".format(Nx))
parser.add_argument('-Ny', '--y-nodes', type=int, default=Ny,
    help="Number of nodes in y. Default: {}".format(Ny))
parser.add_argument('-Nt', '--t-nodes', type=int, default=Nt,
    help="Number of nodes in t. Default: {}".format(Nt))
# Others parameters for debugging #
parser.add_argument('-d', '--debug', type=int, default=0,
    help="Debug. 1 for debugging. Default: {}".format(0))
parser.add_argument('-sic', '--show-initial-condition', type=int, default=0,
    help="Show (plot) initial conditions. 1 for plot. Default: {}".format(0))

""""
parser.add_argument('-v', '--visualization', type=str, 
    help='Type of visualization. Options: "horizontal" or "vertical". Default: "vertical".', default="vertical")
parser.add_argument('-p', '--plots', type=str, 
    help='Plots to show. Options: u, v, modU, divU, curlU, T, Y, p. Default: modU T p.', default="modU T p")
parser.add_argument('-s', '--show', type=str, 
    help='Show, video or GIF. Options: "plot", "video" or "GIF". Default: "plot".', default="plot")
parser.add_argument('-t', '--ts', type=int, help='Time sample. Default 1', default=1)
parser.add_argument('-i', '--input', type=str, help='Simulation directory.', required=True)
"""
args = parser.parse_args()