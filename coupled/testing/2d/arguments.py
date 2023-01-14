import argparse
from parameters import *
from datetime import datetime

parser = argparse.ArgumentParser(description='2D simplified wildfire numerical simulation')
# Parameters
parser.add_argument('-n', '--name', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"),
    help="Numerical simulation name. Default: Current datetime with format 'YYYYMMDDhhmmss'")
parser.add_argument('-fc', '--fuel-consumption', type=float, default=Y_f,
    help="Fuel consumption parameter. Default: {}".format(Y_f))


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