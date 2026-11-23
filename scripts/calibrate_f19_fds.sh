#!/bin/bash
# Calibration runs for the CSIRO F19 grass fire with the FDS-like atmosphere (2D x-z, 140 s, dx = 1.17 m).
#
#   bash scripts/calibrate_f19_fds.sh write        # only write the parameter files
#   bash scripts/calibrate_f19_fds.sh              # write them and run all at once (default; on the 4090
#                                                  # parallel runs barely slow each other on this grid)
#   bash scripts/calibrate_f19_fds.sh sequential   # write them and run one after another
#   bash scripts/calibrate_f19_fds.sh parallel graph   # add "graph" to run with --cuda-graph
#
# Common to all runs:
#   wind: Monin-Obukhov profile of FDS Case_F19 (4.9 m/s at 2 m, z_0 = 0.03 m, L = -500 m), kept by the
#         profile forcing (1 s), log-law wall model at the ground
#   gas:  mixture specific heat (N2, O2, CO2, H2O), Sutherland viscosity and conductivity
#   fuel: moisture 0.058; turbulence filter width 0.30 m; adaptive time step (CFL 0.4)
# Runs (output in data/output/<name>/, log in <name>.out):
#   fds_hc21_kg01    H_C = 21 MJ/kg,   kappa_gas = 0.1 1/m
#   fds_hc21_kg03    H_C = 21 MJ/kg,   kappa_gas = 0.3 1/m
#   fds_hc156_kg01   H_C = 15.6 MJ/kg, kappa_gas = 0.1 1/m
#   fds_hc156_kg03   H_C = 15.6 MJ/kg, kappa_gas = 0.3 1/m
set -e
cd "$(dirname "$0")/.."
mode=${1:-parallel}
extra=""
if [ "$2" = "graph" ] || [ "$1" = "graph" ]; then
    extra="--cuda-graph"
    [ "$1" = "graph" ] && mode=parallel
fi
dir=data/input/2D/calibration_fds
mkdir -p "$dir"

write_ini() {   # name H_C kappa_gas
    cat > "$dir/$1.ini" <<EOF
[domain]
x_min = -500
x_max = 700
y_min = 0
y_max = 20
t_min = 0
t_max = 140
[numerics]
Nx = 1025
Ny = 256
Nt = 140000
NT = 1000
cfl = 0.4
[temperature]
T_hot = 900
[wind]
shape = monin-obukhov
u_r = 4.9
z_r = 2
z_0 = 0.03
L = -500
forcing = profile
forcing_time = 1
[wall]
model = log
[gas]
cp_model = mixture
transport = sutherland
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

write_ini fds_hc21_kg01  21e6   0.1
write_ini fds_hc21_kg03  21e6   0.3
write_ini fds_hc156_kg01 15.6e6 0.1
write_ini fds_hc156_kg03 15.6e6 0.3
runs="fds_hc21_kg01 fds_hc21_kg03 fds_hc156_kg01 fds_hc156_kg03"
echo "parameter files in $dir: $runs"

if [ "$mode" = "write" ]; then
    exit 0
elif [ "$mode" = "sequential" ]; then
    nohup bash -c "for r in $runs; do python -u src/main.py $dir/\$r.ini -n \$r --device cuda $extra > \$r.out 2>&1; done" \
        > calibration_fds.out 2>&1 &
    echo "started the runs one after another; follow with: tail -f fds_hc21_kg01.out"
else
    for r in $runs; do
        nohup python -u src/main.py "$dir/$r.ini" -n "$r" --device cuda $extra > "$r.out" 2>&1 &
    done
    echo "started all runs in parallel${extra:+ with CUDA graphs}; follow with: tail -f fds_hc21_kg01.out"
fi
