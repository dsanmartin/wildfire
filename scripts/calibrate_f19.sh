#!/bin/bash
# Calibration runs for the CSIRO F19 grass fire (2D x-z, extended model, 140 s, dx = 1.17 m).
#
#   bash scripts/calibrate_f19.sh write       # only write the parameter files
#   bash scripts/calibrate_f19.sh             # write them and run one after another (in the background)
#   bash scripts/calibrate_f19.sh parallel    # write them and run all at the same time (shares the GPU)
#
# Runs (output in data/output/<name>/, log in <name>.out):
#   cal_base        H_C = 21 MJ/kg, kappa_gas = 0.1   (reference, current defaults)
#   cal_hc156       H_C = 15.6 MJ/kg                  (heat of combustion of the pyrolysis gas, Mell et al. 2007)
#   cal_kg03        kappa_gas = 0.3 1/m               (stronger flame radiation)
#   cal_hc156_kg03  both
#   cal_long        reference on a domain twice as long (x up to 1900 m), to check the effect of the
#                   periodic x boundary on the late slowdown
# Common: 1.17 m grid, moisture 0.058, turbulence filter width 0.30 m, ignition and wind of the paper case.
set -e
cd "$(dirname "$0")/.."
mode=${1:-sequential}
dir=data/input/2D/calibration
mkdir -p "$dir"

write_ini() {   # name H_C kappa_gas x_max Nx
    cat > "$dir/$1.ini" <<EOF
[domain]
x_min = -500
x_max = $4
y_min = 0
y_max = 20
t_min = 0
t_max = 140
[numerics]
Nx = $5
Ny = 256
Nt = 140000
NT = 1000
[temperature]
T_hot = 900
[wind]
u_r = 4.8
[fuel]
Y_h = 0.51
C_d = 0.15
alpha_s = 0.002
sigma_s = 4000
moisture = 0.058
H_C = $2
[radiation]
kappa_gas = $3
[turbulence]
filter_width = 0.30
EOF
}

write_ini cal_base       21e6   0.1  700  1025
write_ini cal_hc156      15.6e6 0.1  700  1025
write_ini cal_kg03       21e6   0.3  700  1025
write_ini cal_hc156_kg03 15.6e6 0.3  700  1025
write_ini cal_long       21e6   0.1  1900 2049
runs="cal_base cal_hc156 cal_kg03 cal_hc156_kg03 cal_long"
echo "parameter files in $dir: $runs"

if [ "$mode" = "write" ]; then
    exit 0
elif [ "$mode" = "parallel" ]; then
    for r in $runs; do
        nohup python -u src/main.py "$dir/$r.ini" -n "$r" --device cuda > "$r.out" 2>&1 &
    done
    echo "started all runs in parallel; follow with: tail -f cal_base.out"
else
    nohup bash -c "for r in $runs; do python -u src/main.py $dir/\$r.ini -n \$r --device cuda > \$r.out 2>&1; done" \
        > calibration.out 2>&1 &
    echo "started the runs one after another; follow with: tail -f cal_base.out"
fi
