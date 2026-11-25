#!/bin/bash
# 3D run of the CSIRO F19 grass fire with the FDS Case_F19 setup (burner lit along the line at 1.56 m/s).
#
#   bash scripts/f19_3d.sh write    # only write the parameter files
#   bash scripts/f19_3d.sh smoke    # 3 s test: speed, GPU memory, CUDA graphs, budgets (a few minutes)
#   bash scripts/f19_3d.sh full     # 140 s run in the background (several hours)
#   bash scripts/f19_3d.sh sz_smoke # stretched vertical grid, exact fuel load (below)
#   bash scripts/f19_3d.sh sz_full
#   bash scripts/f19_3d.sh ux_full  # uniform grid, exact fuel load: the reference for sz_full
#
# Grid: x -60..260 m, y -180..180 m, z 0..20 m; 256 x 288 x 160 cells of 1.25 x 1.25 x 0.125 m (11.8 M cells,
# about 4 cells across the 0.49 m fuel bed). Grass plot x = 0-200 m, y = -100..100 m as in FDS; periodic in x
# and y. The wind is driven by a sponge zone at x = -60..-20 m (velocity relaxed to the initial profile, T and
# the species to ambient), so air leaving downstream re-enters as ambient air and the fire itself is not
# forced; the plot covers 56% of the width (the periodic images of the fire are 160 m apart).
# Fuel: FDS Arrhenius pyrolysis (A = 1040 1/s, E = 61 kJ/mol: T_act = 7342 K, no threshold) and the FDS
# fuel heat capacity ramp (1100 -> 2000 J/(kg K) from 0 to 200 C); in 2D these halved the spread rate
# (4.8 -> 2.3 m/s, scripts/calibrate_f19_pyro.sh).
# Speed: float32 (--dtype float32, pressure solver tolerance 1e-6) and radiation angle increment 4 (each
# update solves a quarter of the directions): about 3 times faster than float64 with the full radiation
# (profile: 870 -> ~290 ms per step, memory 17 -> 9 GiB). In 2D both left the front position (within 1 m)
# and the heat released (0.4%) unchanged; float32 conserves mass to ~1e-5 of the domain mass.
# Adaptive time step (CFL 0.4, at most 5 ms). Case: H_C = 13.1 MJ/kg (FDS
# gas-phase equivalent), kappa_gas = 0.1 1/m, ambient 34 C, F19 wind (4.9 m/s at 2 m, L = -500 m), grid-based
# filter width ((dx dy dz)^(1/3) = 0.58 m; a fixed 0.30 m would be far below the horizontal spacing).
# Output: planes every 1 s (x-z at y = 10 m, x-y at z = 1 m, y-z through the head fire), full fields every 20 s.
# The side view is off the center line: the setup is symmetric in y and the front has a narrow notch at y = 0.
# Output in data/output/f19_3d_sponge (the earlier run without sponge, 240 m wide, is data/output/f19_3d).
# Stretched vertical grid (f19_3d_sz): 57 rows instead of 161, cell heights growing by 4% per row from 0.13 m
# at the ground (4 nodes in the fuel bed, as on the uniform grid) to at most 0.5 m (0.17 m at 1 m, 0.21 m at
# 2 m, 0.33 m at 5 m, 0.5 m above 9.5 m); filter width and mixing time follow the local cell height.
# Both grid-comparison runs (f19_3d_sz, f19_3d_ux) use [fuel] fill = exact, so that both hold the 0.49 m bed
# (with fill = nodes the uniform grid holds 0.4375 m and the stretched one 0.475 m).
# Compare with: python src/firefront.py data/output/f19_3d_sponge data/output/f19_3d --obs data/input/f19_front_fds.csv
set -e
cd "$(dirname "$0")/.."
mkdir -p logs                                  # console output of the runs: logs/<name>.out
mode=${1:-smoke}
dir=data/input/3D/f19
mkdir -p "$dir"

write_ini() {   # name t_max Nt NT full_interval [Nz] [extra numerics lines] [extra fuel lines]
    cat > "$dir/$1.ini" <<EOF
[domain]
x_min = -60
x_max = 260
y_min = -180
y_max = 180
z_min = 0
z_max = 20
t_min = 0
t_max = $2
[numerics]
Nx = 257
Ny = 289
Nz = ${6:-161}
Nt = $3
NT = $4
cfl = 0.4
$7
[pressure]
tol = 1e-6
[temperature]
T_inf = 307.15
T_pc = 300
[ignition]
model = burner
[wind]
u_r = 4.9
z_r = 2
L = -500
forcing = sponge
sponge_start = -60
sponge_end = -20
[fuel]
H_C = 13.1e6
pyrolysis = arrhenius
A = 1040
T_act = 7342
Y_f = 1
c_s_model = fds
x_start = 0
x_end = 200
y_start = -100
y_end = 100
$8
[radiation]
kappa_gas = 0.1
angle_increment = 4
[output]
y_slice = 10
z_slice = 1
full_interval = $5
EOF
}

write_ini f19_3d_sponge_smoke 3 600 100 0         # dt_max 5 ms, output every 0.5 s, no full snapshots
write_ini f19_3d_sponge 140 28000 200 20          # dt_max 5 ms, output every 1 s, full fields every 20 s
SZ="z_ratio = 1.04
z_dz_max = 0.5"
write_ini f19_3d_sz_smoke 3 600 100 0 57 "$SZ" "fill = exact"
write_ini f19_3d_sz 140 28000 200 20 57 "$SZ" "fill = exact"
write_ini f19_3d_ux 140 28000 200 20 161 "" "fill = exact"
echo "parameter files in $dir: f19_3d_sponge_smoke.ini, f19_3d_sponge.ini, f19_3d_sz_smoke.ini, f19_3d_sz.ini, f19_3d_ux.ini"

case "$mode" in
    write) ;;
    smoke)
        python -u src/main.py "$dir/f19_3d_sponge_smoke.ini" -n f19_3d_sponge_smoke --device cuda --dtype float32 2>&1 | tee logs/f19_3d_sponge_smoke.out
        echo
        echo "check: 'CUDA graphs: captured ...', mean step time, GPU memory peak, mass residual near 0%" ;;
    full)
        nohup python -u src/main.py "$dir/f19_3d_sponge.ini" -n f19_3d_sponge --device cuda --dtype float32 > logs/f19_3d_sponge.out 2>&1 &
        echo "started; follow with: tail -f logs/f19_3d_sponge.out" ;;
    sz_smoke)
        python -u src/main.py "$dir/f19_3d_sz_smoke.ini" -n f19_3d_sz_smoke --device cuda --dtype float32 2>&1 | tee logs/f19_3d_sz_smoke.out ;;
    sz_full)
        nohup python -u src/main.py "$dir/f19_3d_sz.ini" -n f19_3d_sz --device cuda --dtype float32 > logs/f19_3d_sz.out 2>&1 &
        echo "started; follow with: tail -f logs/f19_3d_sz.out" ;;
    ux_full)
        nohup python -u src/main.py "$dir/f19_3d_ux.ini" -n f19_3d_ux --device cuda --dtype float32 > logs/f19_3d_ux.out 2>&1 &
        echo "started; follow with: tail -f logs/f19_3d_ux.out" ;;
    *) echo "usage: bash scripts/f19_3d.sh [write|smoke|full|sz_smoke|sz_full|ux_full]"; exit 1 ;;
esac
