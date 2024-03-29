#!/bin/bash
# OUTPUT_DIR="./output/";
# OUTPUT_DIR="";
args=("$@");
input_dir=${args[0]};
log=${args[1]};
visualization=${args[2]};
output_dir=${args[3]};
xmin=0;
xmax=200;
ymin=0;
ymax=20;
show='video';
plots="modU,T,Y";
n_max=-1;
ts=1;
if [ -z "${visualization}" ]; then
    visualization="vertical";
fi
# Check if visualization is "horizontal"
if [ "${visualization}" = "horizontal" ]; then
    ymax=200;
fi
if [ -z "${output_dir}" ]; then
    output_dir=$input_dir;
fi
# Create animation
python -u src/create_animation.py -i ${input_dir} -o ${output_dir} -s ${show} -xmin ${xmin} -xmax ${xmax} -ymin ${ymin} -ymax ${ymax} -ts ${ts} -tn ${n_max} -p ${plots} -v ${visualization} > ${output_dir}animation.log &
if [ "${log}" = "log" ]; then
    sleep .5
    tail -f ${output_dir}animation.log
fi