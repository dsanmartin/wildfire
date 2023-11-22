#!/bin/bash
args=("$@");
# sim_name=${args[0]};
input_dir=${args[0]};
ts=${args[1]};
tn=${args[2]};
xmin=${args[3]};
xmax=${args[4]};
ymin=${args[5]};
ymax=${args[6]};
# Data
# data="${OUTPUT_DIR}${sim_name}/data.npz";
# data="${data_dir}data.npz";
# Check if t is not empty
if [ -z "${t}" ]; then
    t=1;
fi
# Check if n is not empty
if [ -z "${n}" ]; then
    n=0;
fi
# Check if xmin is not empty
if [ -z "${xmin}" ]; then
    # xmin=666;
    xmin=0;
fi
# Check if xmax is not empty
if [ -z "${xmax}" ]; then
    # xmax=666;
    xmax=200;
fi
# Check if ymin is not empty
if [ -z "${ymin}" ]; then
    # ymin=666;
    ymin=0;
fi
# Check if ymax is not empty
if [ -z "${ymax}" ]; then
    # ymax=666;
    ymax=20;
fi
# Create animation
python src/2d/create_animation.py -i ${input_dir} -ts ${ts} -tn ${tn} -xmin ${xmin} -xmax ${xmax} -ymin ${ymin} -ymax ${ymax} -p "modU,T,Y";