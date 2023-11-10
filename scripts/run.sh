#!/bin/bash
# Get args
args=("$@");
simulation_id=$(date '+%Y%m%d%H%M%S'); # Unique simulation ID
path="./data/output/2d/${simulation_id}/"; # Path to save simulation data
mkdir ${path} # Create directory to save simulation data
nohup python -u src/2d/main.py -param data/input/2d/parameters.ini -path ${path} 1> ${path}output.out 2> ${path}error.err &
# nohup python -u src/2d/main.py -path ${path} 1> ${path}output.out 2> ${path}error.err &
if [ "${args[0]}" = "log" ]; then
    sleep .5
    tail -f ${path}output.out
fi