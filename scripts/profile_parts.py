"""Time each part of a time step: radiation, prediction (right-hand side, provisional state, PCG setup),
pressure solve (PCG iterations) and finalization (projection correction, boundary conditions, budgets).

The model takes `warm` direct steps from the initial state (so that the fire, the face fluxes and the
radiation field exist), then `steps` timed steps with a fixed dt. Each part runs either as its CUDA graph
(--graphs, the production path) or directly. GPU times are measured with CUDA events.

    python -u scripts/profile_parts.py data/input/3D/f19/f19_3d_smoke.ini --device cuda --graphs
    python -u scripts/profile_parts.py data/input/3D/f19/f19_3d_smoke.ini --device cuda --graphs --dtype float32 --tol 1e-6
    python -u scripts/profile_parts.py data/input/3D/f19/f19_3d_smoke.ini --device cuda --graphs --angle-increment 4

Options override the input file: --dtype, --tol (pressure solver; float32 cannot reach 1e-10),
--update-every and --angle-increment (radiation), --n-polar and --n-azimuth (directions).
"""
import argparse
import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import torch
from config import load_config
from setup import build
from model import Model
from main import pick_device
from cudagraph import GraphRunner


class Timer:
    """Elapsed time of a block in ms: CUDA events on a GPU, wall clock otherwise."""
    def __init__(self, device):
        self.cuda = device.type == 'cuda'

    def __call__(self, fn):
        if self.cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); out = fn(); b.record(); torch.cuda.synchronize()
            return out, a.elapsed_time(b)
        t0 = time.perf_counter(); out = fn()
        return out, (time.perf_counter() - t0) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parameters_file')
    ap.add_argument('--device', default=None)
    ap.add_argument('--dtype', default=None, choices=['float64', 'float32'])
    ap.add_argument('--graphs', action='store_true', help='replay each part as its CUDA graph (GPU)')
    ap.add_argument('--warm', type=int, default=20)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--tol', type=float, default=None)
    ap.add_argument('--update-every', type=int, default=None)
    ap.add_argument('--angle-increment', type=int, default=None)
    ap.add_argument('--n-polar', type=int, default=None)
    ap.add_argument('--n-azimuth', type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.parameters_file, sim_name='profile', save_path='./data/output/profile/',
                      device=args.device, dtype=args.dtype)
    for key, attr in (('tol', 'tol'), ('update_every', 'rad_every'), ('angle_increment', 'angle_increment'),
                      ('n_polar', 'n_polar'), ('n_azimuth', 'n_azimuth')):
        if getattr(args, key) is not None:
            setattr(cfg, attr, getattr(args, key))
    device, dtype = pick_device(cfg.device), getattr(torch, cfg.dtype)
    timer = Timer(device)
    grid = build(cfg)
    m = Model(cfg, grid, device, dtype)
    y = torch.as_tensor(grid['state'], dtype=dtype, device=device)
    p = torch.zeros(m.shape, dtype=dtype, device=device)
    dt = m.dt
    print(f"{args.parameters_file}: grid {'x'.join(map(str, m.shape))} ({m.Zm.numel() / 1e6:.2f} M nodes), "
          f"device {device}, {cfg.dtype}, {'CUDA graphs' if args.graphs else 'direct'}; dt {dt:.1e} s, "
          f"tol {cfg.tol:g}, radiation every {cfg.rad_every} steps, angle increment {cfg.angle_increment}, "
          f"{cfg.n_polar}x{cfg.n_azimuth} directions", flush=True)
    t = 0.0
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(args.warm):
            m.set_dt(dt); m.set_time(t)
            y, p = m.step(t, y, p)
            t += dt
        print(f"warm-up: {args.warm} direct steps in {time.perf_counter() - t0:.1f} s (to t = {t:.2f} s)", flush=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        runner = GraphRunner(m, y, p, log=lambda *a: None)
        if args.graphs and device.type == 'cuda' and not runner.graphs:
            raise SystemExit("no CUDA graph could be captured")
        run = runner._run if args.graphs else (lambda name: runner.functions[name]())
        P = m.face_proj
        total = {'radiation': 0.0, 'predict': 0.0, 'pcg': 0.0, 'finalize': 0.0}
        calls = {k: 0 for k in total}
        its = 0
        wall0 = time.perf_counter()
        for n in range(args.steps):
            m.set_dt(dt); m.set_time(t)
            if runner.radiation_names and m.radiation_due():
                name = runner.radiation_names[0] if len(runner.radiation_names) == 1 else \
                    runner.radiation_names[m.next_radiation_group()]
                _, ms = timer(lambda: run(name)); total['radiation'] += ms; calls['radiation'] += 1
            m.step_count += 1
            _, ms = timer(lambda: run('predict')); total['predict'] += ms; calls['predict'] += 1

            def pcg():
                k = 0
                while True:
                    run('pcg'); k += runner.chunk
                    if k >= runner.max_iter or float(P.pcg_residual()) < runner.tol:
                        return k
            k, ms = timer(pcg); total['pcg'] += ms; calls['pcg'] += 1; its += k
            _, ms = timer(lambda: run('finalize')); total['finalize'] += ms; calls['finalize'] += 1
            t += dt
        wall = (time.perf_counter() - wall0) / args.steps * 1e3
    step = sum(total.values()) / args.steps
    print(f"\n{'part':<11} {'ms/call':>9} {'calls/step':>11} {'ms/step':>9} {'share':>7}")
    for k in total:
        if calls[k]:
            per_step = total[k] / args.steps
            print(f"{k:<11} {total[k] / calls[k]:9.2f} {calls[k] / args.steps:11.2f} {per_step:9.2f} {100 * per_step / step:6.1f}%")
    print(f"{'step':<11} {'':9} {'':11} {step:9.2f}   (wall clock {wall:.2f} ms/step; "
          f"{its / args.steps:.1f} PCG iterations per step)")
    if device.type == 'cuda':
        print(f"GPU memory peak: {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB")


if __name__ == '__main__':
    main()
