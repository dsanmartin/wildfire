#!/bin/bash
# F19 in 2D: effect of the FDS pyrolysis kinetics and fuel heat capacity (same setup as f19_hc131_kg01 of
# scripts/calibrate_f19_fds.sh: H_C = 13.1 MJ/kg, kappa_gas = 0.1 1/m, burner, 34 C, F19 wind).
#
#   bash scripts/calibrate_f19_pyro.sh write        # only write the parameter files
#   bash scripts/calibrate_f19_pyro.sh              # write them and run all at once (default)
#   bash scripts/calibrate_f19_pyro.sh sequential   # one after another
#
# Runs (output in data/output/<name>/, log in logs/<name>.out); the baseline (linear pyrolysis between 400 and
# 500 K, constant c_s = 1500 J/(kg K)) is f19_hc131_kg01:
#   f19_lin_csfds   linear pyrolysis, FDS heat capacity ramp (1100 -> 2000 J/(kg K) from 0 to 200 C)
#   f19_arr_cs      FDS Arrhenius kinetics (A = 1040 1/s, E = 61 kJ/mol: T_act = 7342 K), constant c_s
#   f19_arr_csfds   FDS Arrhenius kinetics and FDS heat capacity ramp
# FDS kinetics: dY/dt = -A exp(-T_act / T_s) Y, i.e. Y_f = 1 and no threshold (T_pc below ambient).
# Compare: python src/firefront.py data/output/f19_hc131_kg01 data/output/f19_lin_csfds \
#              data/output/f19_arr_cs data/output/f19_arr_csfds --obs data/input/f19_front_fds.csv --window 20 50
set -e
cd "$(dirname "$0")/.."
mkdir -p logs                                  # console output of the runs: logs/<name>.out
mode=${1:-parallel}
dir=data/input/2D/calibration_pyro
mkdir -p "$dir"

ARR="pyrolysis = arrhenius
A = 1040
T_act = 7342
Y_f = 1"

write_ini() {   # name, extra [fuel] lines, T_pc
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
T_pc = $3
[ignition]
model = burner
[wind]
u_r = 4.9
z_r = 2
L = -500
[fuel]
H_C = 13.1e6
x_start = 0
x_end = 200
$2
[radiation]
kappa_gas = 0.1
[turbulence]
filter_width = 0.30
EOF
}

write_ini f19_lin_csfds "c_s_model = fds" 523
write_ini f19_arr_cs "$ARR" 300
write_ini f19_arr_csfds "$ARR
c_s_model = fds" 300
runs="f19_lin_csfds f19_arr_cs f19_arr_csfds"
echo "parameter files in $dir: $runs"

if [ "$mode" = "write" ]; then
    exit 0
elif [ "$mode" = "sequential" ]; then
    nohup bash -c "for r in $runs; do python -u src/main.py $dir/\$r.ini -n \$r --device cuda > logs/\$r.out 2>&1; done" \
        > logs/calibration_pyro.out 2>&1 &
    echo "started the runs one after another; follow with: tail -f logs/f19_lin_csfds.out"
else
    for r in $runs; do
        nohup python -u src/main.py "$dir/$r.ini" -n "$r" --device cuda > "logs/$r.out" 2>&1 &
    done
    echo "started all runs in parallel; follow with: tail -f logs/f19_arr_cs.out"
fi
