#!/bin/bash
# 2D tests of burning structures ([structures] model = ignition) in the C064 grass fire.
#
#   bash scripts/houses_2d.sh write    # only write the parameter files
#   bash scripts/houses_2d.sh          # write them and run all three in the background (default)
#   bash scripts/houses_2d.sh tall     # only the 40 m variant
#
# Runs (output in data/output/<name>/, log in logs/<name>.out):
#   houses_2d_one   one house, x = 30-45 m (the first Mac test)
#   houses_2d_two   two houses 20 m apart, x = 30-45 m and 65-80 m: does the burning first house ignite the
#                   second one, and how much does it shelter it from the grass fire?
#   houses_2d_two_tall  the same with the domain top at 40 m instead of 20 m (115 rows: the nodes below 20 m
#                   are those of the 20 m grid): effect of the domain height on the burning houses and their plume
#                   (second house ignited at 47.6 s instead of 59.6 s: use the 40 m top for 2D structure tests)
# Houses 15 m long and 5 m tall (2D: per metre of depth, infinitely long across the wind), FDS 'IGNITABLE' surfaces
# (defaults of [structures]). Grass of C064 (0.21 m, exact fuel load) from x = 0 to 150 m, also between and behind
# the houses; burner as in C064; wind 4.6 m/s at 2 m; sponge zone at x = -40..-20 m. Stretched vertical grid as in
# scripts/c064_3d.sh (0.10 m at the ground to at most 0.4 m), 1 m horizontal cells, 120 s.
# In 2D a house blocks the whole width, so the grass fire cannot pass around it, and the plume is bent onto the
# houses more than in 3D: these runs test the coupling, not the ignition times.
# Compare: python scripts/compare_structures.py data/output/houses_2d_two
#          python src/visualize.py data/output/houses_2d_two --field T HRR --xlim -5 160 --zlim 0 20
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
mode=${1:-run}
dir=data/input/2D/structures
mkdir -p "$dir"

write_ini() {   # name, [buildings] lines, [domain top (m), rows]
    cat > "$dir/$1.ini" <<EOF
[domain]
x_min = -40
x_max = 200
y_min = 0
y_max = ${3:-20}
t_min = 0
t_max = 120
[numerics]
Nx = 241
Ny = ${4:-65}
Nt = 24000
NT = 200
cfl = 0.4
z_ratio = 1.0475
z_dz_max = 0.4
[pressure]
tol = 1e-8
[temperature]
T_inf = 305.15
T_pc = 300
[ignition]
model = burner
hrrpua = 1500e3
x_start = 0
x_end = 1.6
ramp = 0 0, 1 1, 5 1, 6 0
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
x_end = 150
fill = exact
[radiation]
kappa_gas = 0.1
[buildings]
$2
[structures]
model = ignition
EOF
}

write_ini houses_2d_one "house1 = 30, 45, 5"
TWO="house1 = 30, 45, 5
house2 = 65, 80, 5"
write_ini houses_2d_two "$TWO"
write_ini houses_2d_two_tall "$TWO" 40 115
runs="houses_2d_one houses_2d_two houses_2d_two_tall"
echo "parameter files in $dir: houses_2d_one.ini, houses_2d_two.ini, houses_2d_two_tall.ini"

if [ "$mode" = "write" ]; then
    exit 0
elif [ "$mode" = "tall" ]; then
    runs="houses_2d_two_tall"
fi
for r in $runs; do
    nohup python -u src/main.py "$dir/$r.ini" -n "$r" --device cuda > "logs/$r.out" 2>&1 &
done
echo "started $runs; follow with: tail -f logs/<name>.out"
