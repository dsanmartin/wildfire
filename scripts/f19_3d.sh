#!/bin/bash
# 3D run of the CSIRO F19 grass fire with the FDS Case_F19 setup (burner lit along the line at 1.56 m/s).
#
#   bash scripts/f19_3d.sh write    # only write the parameter files
#   bash scripts/f19_3d.sh smoke    # 3 s test: speed, GPU memory, CUDA graphs, budgets (a few minutes)
#   bash scripts/f19_3d.sh full     # 140 s run in the background (several hours)
#
# Grid: x -100..300 m, y -120..120 m, z 0..20 m; 320 x 192 x 160 cells of 1.25 x 1.25 x 0.125 m (9.8 M cells,
# about 4 cells across the 0.49 m fuel bed). Grass plot x = 0-200 m, y = -100..100 m as in FDS; periodic in x
# and y with room around the plot. Adaptive time step (CFL 0.4, at most 5 ms). Case: H_C = 13.1 MJ/kg (FDS
# gas-phase equivalent), kappa_gas = 0.1 1/m, ambient 34 C, F19 wind (4.9 m/s at 2 m, L = -500 m), grid-based
# filter width ((dx dy dz)^(1/3) = 0.58 m; a fixed 0.30 m would be far below the horizontal spacing).
# Output: planes every 1 s (x-z at y = 0, x-y at z = 1 m, y-z through the head fire), full fields every 20 s.
# Compare with: python src/firefront.py data/output/f19_3d --obs data/input/f19_front_fds.csv --window 56 138
set -e
cd "$(dirname "$0")/.."
mode=${1:-smoke}
dir=data/input/3D/f19
mkdir -p "$dir"

write_ini() {   # name t_max Nt NT full_interval
    cat > "$dir/$1.ini" <<EOF
[domain]
x_min = -100
x_max = 300
y_min = -120
y_max = 120
z_min = 0
z_max = 20
t_min = 0
t_max = $2
[numerics]
Nx = 321
Ny = 193
Nz = 161
Nt = $3
NT = $4
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
H_C = 13.1e6
x_start = 0
x_end = 200
y_start = -100
y_end = 100
[radiation]
kappa_gas = 0.1
[output]
y_slice = 0
z_slice = 1
full_interval = $5
EOF
}

write_ini f19_3d_smoke 3 600 100 0         # dt_max 5 ms, output every 0.5 s, no full snapshots
write_ini f19_3d 140 28000 200 20          # dt_max 5 ms, output every 1 s, full fields every 20 s
echo "parameter files in $dir: f19_3d_smoke.ini, f19_3d.ini"

case "$mode" in
    write) ;;
    smoke)
        python -u src/main.py "$dir/f19_3d_smoke.ini" -n f19_3d_smoke --device cuda 2>&1 | tee f19_3d_smoke.out
        echo
        echo "check: 'CUDA graphs: captured ...', mean step time, GPU memory peak, mass residual near 0%" ;;
    full)
        nohup python -u src/main.py "$dir/f19_3d.ini" -n f19_3d --device cuda > f19_3d.out 2>&1 &
        echo "started; follow with: tail -f f19_3d.out" ;;
    *) echo "usage: bash scripts/f19_3d.sh [write|smoke|full]"; exit 1 ;;
esac
