#!/bin/bash
usage() { echo "Usage: $0 [-i <input_dir>] [-s <ts>] [-n <tn>] [-v <vis>] [-x <xmin>] [-X <xmax>] [-y <ymin>] [-Y <ymax>] [-z <zmin>] [-Z <zmax>] [-b <bounds>]" 1>&2; exit 1; }
# Default values
ts=1;
tn=-1;
vis="vertical";
xmin=0;
xmax=200;
ymin=0;
ymax=200;
zmin=0;
zmax=10;
plots="modU,T,Y";
plots="modU,T";
bounds=1;
# Parse arguments
while getopts ":i:s:n:v:x:X:y:Y:z:Z:b:" opt; do
  case "${opt}" in
    i) input_dir=${OPTARG} ;;
    s) ts=${OPTARG} ;;
    n) tn=${OPTARG} ;;
    v) vis=${OPTARG} ;;
    x) xmin=${OPTARG} ;;
    X) xmax=${OPTARG} ;;
    y) ymin=${OPTARG} ;;
    Y) ymax=${OPTARG} ;;
    z) zmin=${OPTARG} ;;
    Z) zmax=${OPTARG} ;;
    b) bounds=${OPTARG} ;;
    *) echo "Invalid option -$OPTARG" >&2
      exit 1
    ;;
  esac
done
if [ -z "$input_dir" ]; then
    usage;
fi
# Create animation
python src/create_animation.py -i "${input_dir}" \
                                -ts ${ts} \
                                -tn ${tn} \
                                -xmin ${xmin} \
                                -xmax ${xmax} \
                                -ymin ${ymin} \
                                -ymax ${ymax} \
                                -zmin ${zmin} \
                                -zmax ${zmax} \
                                -p ${plots} \
                                -v ${vis} \
                                -b ${bounds};