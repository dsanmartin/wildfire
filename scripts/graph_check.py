"""Step-by-step check of the CUDA-graph runner against direct execution.

Three copies of the model start from the same state and take the same fixed time step:
  A, C  direct execution (A vs C: is direct execution itself reproducible on this device?)
  B     CUDA graphs, captured after 3 direct steps (as in main.py)
After every step it prints the largest relative difference of u and T between A and C and between
A and B. Roundoff gives ~1e-15 after the first graph step, with occasional jumps where a threshold
flips (ignition temperature, fuel top, limiter); a bug gives a large difference right away.

    python -u scripts/graph_check.py data/input/2D/bench.ini [steps] [--device cuda]
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
from config import load_config
from setup import build
from model import Model
from main import pick_device
from cudagraph import GraphRunner


def new_model(cfg, device, dtype):
    grid = build(cfg)
    m = Model(cfg, grid, device, dtype)
    y = torch.as_tensor(grid['state'], dtype=dtype, device=device)
    if m.has_terrain and not cfg.legacy:
        m.apply_bc(y)
    return m, y, torch.zeros(m.shape, dtype=dtype, device=device)


def rel(a, b, i):
    return float((a[i] - b[i]).abs().max() / a[i].abs().max().clamp(min=1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parameters_file')
    ap.add_argument('steps', nargs='?', type=int, default=200)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()
    cfg = load_config(args.parameters_file, sim_name='graph_check', save_path='./data/output/graph_check/',
                      device=args.device)
    device, dtype = pick_device(cfg.device), getattr(torch, cfg.dtype)
    nd = cfg.ndim
    iT = nd                                           # u, (v,) w, T, ...
    (A, ya, pa), (B, yb, pb), (C, yc, pc) = (new_model(cfg, device, dtype) for _ in range(3))
    with torch.no_grad():
        dt = A.stable_dt(ya, A.dt) if cfg.cfl > 0 else A.dt
        for m in (A, B, C):
            m.set_dt(dt)
        print(f"device {device}, dt {dt:.3e} s (fixed), {args.steps} steps")
        for _ in range(3):
            ya, pa = A.step(0.0, ya, pa)
            yb, pb = B.step(0.0, yb, pb)
            yc, pc = C.step(0.0, yc, pc)
        print(f"after 3 direct steps: A-B u {rel(ya, yb, 0):.1e}, A-C u {rel(ya, yc, 0):.1e}")
        runner = GraphRunner(B, yb, pb)
        print(f"captured: {', '.join(runner.graphs) or 'none (emulated)'}")
        print(f"{'step':>5} {'A-C u':>9} {'A-C T':>9} {'A-B u':>9} {'A-B T':>9}  PCG its A/B")
        for n in range(1, args.steps + 1):
            ya, pa = A.step(0.0, ya, pa)
            yc, pc = C.step(0.0, yc, pc)
            yb, pb = runner.step()
            if n <= 10 or n % 10 == 0:
                print(f"{n:5d} {rel(ya, yc, 0):9.1e} {rel(ya, yc, iT):9.1e} {rel(ya, yb, 0):9.1e} "
                      f"{rel(ya, yb, iT):9.1e}  {A.pressure_iterations}/{B.pressure_iterations}", flush=True)


if __name__ == '__main__':
    main()
