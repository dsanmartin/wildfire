#!/bin/bash
# Get args
if [ $# -eq 0 ]
  then
    echo "Error: The parameters file path is required."
    python src/2d/main.py -h
fi
args=("$@");
parameters_file=${args[0]}; # Path to parameters file
simulation_id=$(date '+%Y%m%d%H%M%S'); # Unique simulation ID
path="./data/output/${simulation_id}/"; # Path to save simulation data
mkdir ${path} # Create directory to save simulation data
nohup python -u src/main.py ${parameters_file} -path ${path} -n ${simulation_id} 1> ${path}output.out 2> ${path}error.err &
if [ "${args[1]}" = "log" ]; then
    # sleep .5
    tail -f ${path}output.out
fi