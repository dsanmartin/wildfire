#!/bin/bash
# 3D run of the CSIRO C064 grass fire with the FDS Case_C064 setup (burner lit along the line at 1.0 m/s),
# the same model settings as scripts/f19_3d.sh (second validation case).
#
#   bash scripts/c064_3d.sh write    # only write the parameter files
#   bash scripts/c064_3d.sh smoke    # 3 s test: speed, GPU memory, CUDA graphs, budgets (a few minutes)
#   bash scripts/c064_3d.sh full     # 120 s run in the background
#   bash scripts/c064_3d.sh sz_smoke # stretched vertical grid, exact fuel load (below)
#   bash scripts/c064_3d.sh sz_full
#   bash scripts/c064_3d.sh ux_full  # uniform grid, exact fuel load: the reference for sz_full
#   bash scripts/c064_3d.sh st_smoke # stretched grid with the four burning houses of the FDS structures case
#   bash scripts/c064_3d.sh st_full
#
# Case (Case_C064.fds): ambient 32 C, wind 4.6 m/s at 2 m (L = -500 m, z_0 = 0.03 m); grass plot x = 0-100 m,
# y = -50..50 m, bed height 0.21 m, bulk density 1.33 kg/m3 (packing ratio 1.33 / 500), surface-to-volume
# ratio 9770 1/m, moisture 0.063; burner 1500 kW/m2 on x = 0-1.6 m, y = -25..25 m, 5 s at full power
# (ramp 0-1-5-6 s), lit from the center at 1.0 m/s. Same vegetation model as F19 (FDS Arrhenius kinetics,
# heat capacity ramp, drag coefficient 0.7 here), H_C = 13.1 MJ/kg, kappa_gas = 0.1 1/m.
# Grid: x -40..160 m, y -90..90 m, z 0..20 m; 200 x 180 x 200 cells of 1 x 1 x 0.1 m (7.2 M cells; three
# nodes in the 0.21 m bed). Sponge zone at x = -40..-20 m; the plot covers 56% of the width.
# float32, pressure solver tolerance 1e-6, radiation angle increment 4; adaptive time step (CFL 0.4, at most
# 5 ms). Output: planes every 1 s (x-z at y = 10 m, x-y at z = 1 m), full fields every 20 s. The side view is
# off the center line: the setup is symmetric in y and the front has a narrow notch at y = 0.
# Stretched vertical grid (c064_3d_sz): 65 rows instead of 201, cell heights growing by 4.75% per row from
# 0.10 m at the ground (3 nodes in the bed, as on the uniform grid) to at most 0.4 m (0.15 m at 1 m, 0.20 m at
# 2 m, 0.34 m at 5 m, 0.4 m above 6.4 m). The grid-comparison runs (c064_3d_sz, c064_3d_ux) use [fuel]
# fill = exact, so that both hold the 0.21 m bed (with fill = nodes, as in c064_3d, the uniform grid holds
# 0.25 m, 19% more, and the stretched one 0.26 m).
# Structures (c064_3d_st): the stretched-grid run with the four 15 x 15 x 5 m houses of the FDS case
# Case_C064_structures (x = 25-40 and 60-75 m, y = -25..-10 and 10..25 m) as burning structures ([structures]
# model = ignition: walls ignite at 150 C with 750 kW/m2, roofs at 160 C with 600 kW/m2, 10 mm inert wood on
# 89 mm insulation, ramp 0-2-8-100-120 s, bulk density 150 kg/m3). Side view through houses 3 and 4 (y = 17.5 m),
# plan view at z = 2 m, as the FDS slices. Compare: python scripts/compare_structures.py data/output/c064_3d_st
# --fds path/to/Case_C064_structures_cat_devc.csv
# Measured: head at 15, 42 and 100 m at 27, 53 and 100 s (validation/csiro/C064_front.csv).
# Compare: python src/firefront.py data/output/c064_3d --obs validation/csiro/C064_front.csv
#          python scripts/compare_perimeters.py data/output/c064_3d --case C064
set -e
cd "$(dirname "$0")/.."
mkdir -p logs                                  # console output of the runs: logs/<name>.out
mode=${1:-smoke}
dir=data/input/3D/c064
mkdir -p "$dir"

write_ini() {   # name t_max Nt NT full_interval [Nz] [extra numerics lines] [extra fuel lines]
    cat > "$dir/$1.ini" <<EOF
[domain]
x_min = -40
x_max = 160
y_min = -90
y_max = 90
z_min = 0
z_max = 20
t_min = 0
t_max = $2
[numerics]
Nx = 201
Ny = 181
Nz = ${6:-201}
Nt = $3
NT = $4
cfl = 0.4
$7
[pressure]
tol = 1e-6
[temperature]
T_inf = 305.15
T_pc = 300
[ignition]
model = burner
hrrpua = 1500e3
x_start = 0
x_end = 1.6
y_start = -25
y_end = 25
ramp = 0 0, 1 1, 5 1, 6 0
spread_rate = 1.0
[wind]
u_r = 4.6
z_r = 2
L = -500
forcing = sponge
sponge_start = -40
sponge_end = -20
[fuel]
H_C = 13.1e6
pyrolysis = arrhenius
A = 1040
T_act = 7342
Y_f = 1
c_s_model = fds
Y_h = 0.21
rho_bulk = 1.33
alpha_s = 0.00266
sigma_s = 9770
moisture = 0.063
x_start = 0
x_end = 100
y_start = -50
y_end = 50
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

write_ini c064_3d_smoke 3 600 100 0         # dt_max 5 ms, output every 0.5 s, no full snapshots
write_ini c064_3d 120 24000 200 20          # dt_max 5 ms, output every 1 s, full fields every 20 s
SZ="z_ratio = 1.0475
z_dz_max = 0.4"
write_ini c064_3d_sz_smoke 3 600 100 0 65 "$SZ" "fill = exact"
write_ini c064_3d_sz 120 24000 200 20 65 "$SZ" "fill = exact"
write_ini c064_3d_ux 120 24000 200 20 201 "" "fill = exact"
HOUSES="[buildings]
h1 = 25, 40, -25, -10, 5
h2 = 60, 75, -25, -10, 5
h3 = 25, 40, 10, 25, 5
h4 = 60, 75, 10, 25, 5
[structures]
model = ignition"
for name in c064_3d_st_smoke c064_3d_st; do
    if [ "$name" = c064_3d_st_smoke ]; then write_ini $name 3 600 100 0 65 "$SZ" "fill = exact"
    else write_ini $name 120 24000 200 20 65 "$SZ" "fill = exact"; fi
    # side and plan views through the houses; restart checkpoints every 10 outputs (10 s)
    sed -i.bak 's/^y_slice = .*/y_slice = 17.5/; s/^z_slice = .*/z_slice = 2/; s/^full_interval = \(.*\)/full_interval = \1\ncheckpoint_every = 10/' "$dir/$name.ini" && rm -f "$dir/$name.ini.bak"
    echo "$HOUSES" >> "$dir/$name.ini"
done
echo "parameter files in $dir: c064_3d_smoke.ini, c064_3d.ini, c064_3d_sz_smoke.ini, c064_3d_sz.ini, c064_3d_ux.ini, c064_3d_st_smoke.ini, c064_3d_st.ini"

case "$mode" in
    write) ;;
    smoke)
        python -u src/main.py "$dir/c064_3d_smoke.ini" -n c064_3d_smoke --device cuda --dtype float32 2>&1 | tee logs/c064_3d_smoke.out
        echo
        echo "check: 'CUDA graphs: captured ...', mean step time, GPU memory peak, mass residual near 0%" ;;
    full)
        nohup python -u src/main.py "$dir/c064_3d.ini" -n c064_3d --device cuda --dtype float32 > logs/c064_3d.out 2>&1 &
        echo "started; follow with: tail -f logs/c064_3d.out" ;;
    sz_smoke)
        python -u src/main.py "$dir/c064_3d_sz_smoke.ini" -n c064_3d_sz_smoke --device cuda --dtype float32 2>&1 | tee logs/c064_3d_sz_smoke.out ;;
    sz_full)
        nohup python -u src/main.py "$dir/c064_3d_sz.ini" -n c064_3d_sz --device cuda --dtype float32 > logs/c064_3d_sz.out 2>&1 &
        echo "started; follow with: tail -f logs/c064_3d_sz.out" ;;
    ux_full)
        nohup python -u src/main.py "$dir/c064_3d_ux.ini" -n c064_3d_ux --device cuda --dtype float32 > logs/c064_3d_ux.out 2>&1 &
        echo "started; follow with: tail -f logs/c064_3d_ux.out" ;;
    st_smoke)
        python -u src/main.py "$dir/c064_3d_st_smoke.ini" -n c064_3d_st_smoke --device cuda --dtype float32 2>&1 | tee logs/c064_3d_st_smoke.out ;;
    st_full)
        nohup python -u src/main.py "$dir/c064_3d_st.ini" -n c064_3d_st --device cuda --dtype float32 > logs/c064_3d_st.out 2>&1 &
        echo "started; follow with: tail -f logs/c064_3d_st.out" ;;
    *) echo "usage: bash scripts/c064_3d.sh [write|smoke|full|sz_smoke|sz_full|ux_full|st_smoke|st_full]"; exit 1 ;;
esac
