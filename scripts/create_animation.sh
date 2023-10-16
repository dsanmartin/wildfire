#!/bin/bash
# OUTPUT_DIR="./output/";
# OUTPUT_DIR="";
args=("$@");
input_dir=${args[0]};
output_dir=${args[1]};
xmin=0;
xmax=200;
ymin=0;
ymax=20;
show='video';
n_max=10001;
if [ -z "${output_dir}" ]; then
    output_dir=$input_dir;
fi
# Create animation
python -u src/2d/create_animation.py -i ${input_dir} -o ${output_dir} -s ${show} -xmin ${xmin} -xmax ${xmax} -ymin ${ymin} -ymax ${ymax} -n ${n_max} > ${input_dir}log.log &