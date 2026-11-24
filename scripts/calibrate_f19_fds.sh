#!/bin/bash
# Calibration runs for the CSIRO F19 grass fire with the FDS Case_F19 setup (2D x-z, 140 s, dx = 1.17 m).
#
#   bash scripts/calibrate_f19_fds.sh write        # only write the parameter files
#   bash scripts/calibrate_f19_fds.sh              # write them and run all at once (default; on the 4090
#                                                  # parallel runs barely slow each other on this grid)
#   bash scripts/calibrate_f19_fds.sh sequential   # write them and run one after another
#   bash scripts/calibrate_f19_fds.sh parallel direct  # add "direct" to run without CUDA graphs
#
# The model defaults are the FDS-like setup (consistent divergence, mixture c_p, Sutherland transport,
# Monin-Obukhov wind with forcing and log-law wall model, moisture 0.058, F19 fuel of FDS Case_F19:
# sigma_s = 12240 1/m, alpha_s = 0.00123, C_d = 0.7, Y_h = 0.49 m; CUDA graphs on a CUDA device).
# Specific to F19 here: wind 4.9 m/s at 2 m with L = -500 m, ambient 34 C (FDS TMPA), grass plot
# x = 0-200 m, and the FDS ignition burner ([ignition] model = burner, whose defaults are the FDS
# Case_F19 burner: 1460 kW/m2 on x = 0-3.3 m with its 12 s ramp, lit at 1.56 m/s from the center).
# Common to all runs: turbulence filter width 0.30 m, adaptive time step (CFL 0.4).
# H_C: FDS burns 75% of the dry fuel as gas with 17.4 MJ/kg (the rest is char), i.e. 13.1 MJ per kg of
# dry fuel in the gas phase; this model gasifies all of it, so 13.1 matches the FDS gas-phase heat and
# 17.4 is the upper bound (all fuel burning as FDS fuel vapor).
# Runs (output in data/output/<name>/, log in logs/<name>.out):
#   f19_hc131_kg01   H_C = 13.1 MJ/kg, kappa_gas = 0.1 1/m
#   f19_hc131_kg03   H_C = 13.1 MJ/kg, kappa_gas = 0.3 1/m
#   f19_hc174_kg01   H_C = 17.4 MJ/kg, kappa_gas = 0.1 1/m
#   f19_hc174_kg03   H_C = 17.4 MJ/kg, kappa_gas = 0.3 1/m
set -e
cd "$(dirname "$0")/.."
mkdir -p logs                                  # console output of the runs: logs/<name>.out
mode=${1:-parallel}
extra=""
if [ "$2" = "direct" ] || [ "$1" = "direct" ]; then
    extra="--no-cuda-graph"
    [ "$1" = "direct" ] && mode=parallel
fi
[ "$1" = "graph" ] && mode=parallel          # graphs are the default on CUDA now
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
T_inf = 307.15
[ignition]
model = burner
[wind]
u_r = 4.9
z_r = 2
L = -500
[fuel]
H_C = $2
x_start = 0
x_end = 200
[radiation]
kappa_gas = $3
[turbulence]
filter_width = 0.30
EOF
}

write_ini f19_hc131_kg01 13.1e6 0.1
write_ini f19_hc131_kg03 13.1e6 0.3
write_ini f19_hc174_kg01 17.4e6 0.1
write_ini f19_hc174_kg03 17.4e6 0.3
runs="f19_hc131_kg01 f19_hc131_kg03 f19_hc174_kg01 f19_hc174_kg03"
echo "parameter files in $dir: $runs"

if [ "$mode" = "write" ]; then
    exit 0
elif [ "$mode" = "sequential" ]; then
    nohup bash -c "for r in $runs; do python -u src/main.py $dir/\$r.ini -n \$r --device cuda $extra > logs/\$r.out 2>&1; done" \
        > logs/calibration_fds.out 2>&1 &
    echo "started the runs one after another; follow with: tail -f logs/f19_hc131_kg01.out"
else
    for r in $runs; do
        nohup python -u src/main.py "$dir/$r.ini" -n "$r" --device cuda $extra > "logs/$r.out" 2>&1 &
    done
    echo "started all runs in parallel${extra:+ without CUDA graphs}; follow with: tail -f logs/f19_hc131_kg01.out"
fi
