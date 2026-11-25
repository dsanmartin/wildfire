"""Consistency check of the stretched vertical grid code: on a uniform grid, the stretched code path (forced with
grid['force_stretched']) must give the same solution as the uniform path, to round-off.

    python scripts/check_stretched.py [--steps 300]

Runs small 2D and 3D fires (burner, solid fuel, radiation, turbulence, wall model, face projection, buildings) with
both paths in float64 on the CPU for a fixed number of steps and prints the largest difference of every
field, relative to its range. Expected: about 1e-12 or less (the two paths differ only in the order of
floating-point operations: Fornberg weights instead of the fixed stencils, F/(dz/2) instead of 2F/dz, ...).
"""
import argparse
import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from config import load_config  # noqa: E402
from setup import build  # noqa: E402
from model import Model  # noqa: E402

CASES = {
    '2D': """[domain]
x_min = -60
x_max = 100
y_min = 0
y_max = 10
t_min = 0
t_max = 1
[numerics]
Nx = 129
Ny = 41
Nt = 1000
[ignition]
model = burner
[wind]
u_r = 4.9
z_r = 2
L = -500
forcing = sponge
sponge_start = -60
sponge_end = -30
[wall]
model = log
[fuel]
x_start = 0
x_end = 80
[radiation]
kappa_gas = 0.1
""",
    '2D buildings': """[domain]
x_min = -60
x_max = 100
y_min = 0
y_max = 10
t_min = 0
t_max = 1
[numerics]
Nx = 129
Ny = 41
Nt = 1000
[ignition]
model = burner
[wind]
u_r = 4.9
z_r = 2
forcing = sponge
sponge_start = -60
sponge_end = -30
[fuel]
x_start = 0
x_end = 80
[buildings]
house = 30, 40, 4
[radiation]
kappa_gas = 0.1
""",
    '3D': """[domain]
x_min = -40
x_max = 80
y_min = -30
y_max = 30
z_min = 0
z_max = 8
t_min = 0
t_max = 1
[numerics]
Nx = 49
Ny = 25
Nz = 33
Nt = 1000
[ignition]
model = burner
[wind]
u_r = 4.9
z_r = 2
forcing = sponge
sponge_start = -40
sponge_end = -20
[fuel]
x_start = 0
x_end = 60
y_start = -20
y_end = 20
[buildings]
house = 30, 45, -10, 5, 4
[radiation]
kappa_gas = 0.1
angle_increment = 2
""",
}


def run(text, steps, force):
    with tempfile.NamedTemporaryFile('w', suffix='.ini', delete=False) as f:
        f.write(text)
    cfg = load_config(f.name, device='cpu', dtype='float64', cuda_graph='off')
    os.unlink(f.name)
    grid = build(cfg)
    grid['force_stretched'] = force
    model = Model(cfg, grid, torch.device('cpu'), torch.float64)
    assert model.stretched == force
    y = torch.as_tensor(grid['state'], dtype=torch.float64)
    p = torch.zeros(model.shape, dtype=torch.float64)
    t = 0.0
    with torch.no_grad():
        for _ in range(steps):
            model.set_dt(model.dt)
            model.set_time(t)
            y, p = model.step(t, y, p)
            t += model.dt
    return y, p, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300)
    args = ap.parse_args()
    worst = 0.0
    for name, text in CASES.items():
        ya, pa, ma = run(text, args.steps, False)
        yb, pb, mb = run(text, args.steps, True)
        nd = ma.nd
        labels = (['u', 'w'] if nd == 2 else ['u', 'v', 'w']) + ['T', 'Y', 'Ts', 'W', 'YF', 'YO2']
        print(f"{name}: {args.steps} steps of {ma.dt:.0e} s, grid {'x'.join(map(str, ma.shape))}, "
              f"T max {float(ya[nd].max()):.0f} K")
        for k, lab in enumerate(labels[:ya.shape[0]]):
            rng = float(ya[k].max() - ya[k].min()) or 1.0
            d = float((ya[k] - yb[k]).abs().max()) / rng
            worst = max(worst, d)
            print(f"  {lab:4s} max |uniform - stretched path| / range = {d:.1e}")
        d = float((pa - pb).abs().max()) / (float(pa.max() - pa.min()) or 1.0)
        worst = max(worst, d)
        print(f"  p    max |uniform - stretched path| / range = {d:.1e}")
        ea, eb = float(ma.energy['combustion']), float(mb.energy['combustion'])
        print(f"  combustion energy: {ea:.6e} vs {eb:.6e} (rel diff {abs(ea - eb) / ea:.1e})")
    print("PASS" if worst < 1e-9 else "FAIL", f"(largest relative difference {worst:.1e})")


if __name__ == '__main__':
    main()
