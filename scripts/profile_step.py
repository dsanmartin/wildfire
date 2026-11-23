"""Time steps with the fuel slab on and off, directly and with CUDA graphs.

Each variant starts from the initial state, takes `warm` untimed steps (plus the capture for graphs)
and then times `steps` steps with a fixed dt. The variants are interleaved over `repeat` rounds and
the best round is reported, to reduce the effect of other load on the machine. It also times the
solid-fuel source terms alone (solid_rates).

    python -u scripts/profile_step.py data/input/2D/bench.ini --device cuda [--steps 100] [--repeat 3]
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


def setup(cfg, slab, device, dtype, warm):
    cfg.fuel_slab = slab
    grid = build(cfg)
    m = Model(cfg, grid, device, dtype)
    y = torch.as_tensor(grid['state'], dtype=dtype, device=device)
    if m.has_terrain and not cfg.legacy:
        m.apply_bc(y)
    p = torch.zeros(m.shape, dtype=dtype, device=device)
    m.set_dt(m.dt)
    for _ in range(warm):
        y, p = m.step(0.0, y, p)
    return m, y, p


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parameters_file')
    ap.add_argument('--device', default=None)
    ap.add_argument('--steps', type=int, default=100)
    ap.add_argument('--warm', type=int, default=5)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--dtype', default=None, choices=['float64', 'float32'])
    args = ap.parse_args()
    cfg = load_config(args.parameters_file, sim_name='profile', save_path='./data/output/profile/',
                      device=args.device, dtype=args.dtype)
    device, dtype = pick_device(cfg.device), getattr(torch, cfg.dtype)
    graphs = device.type == 'cuda' and not args.no_graph and cfg.projection == 'face'
    modes = [('direct', False)] + ([('graphs', True)] if graphs else [])
    best = {}
    with torch.no_grad():
        for r in range(args.repeat):
            for slab in (False, True):
                m, y, p = setup(cfg, slab, device, dtype, args.warm)
                # solid source terms alone
                if cfg.solid_temperature:
                    rho = m.rho(y[cfg.ndim])
                    sync(device); t0 = time.perf_counter()
                    for _ in range(50):
                        m.solid_rates(y, rho)
                    sync(device)
                    key = ('solid_rates', slab)
                    best[key] = min(best.get(key, 1e9), (time.perf_counter() - t0) / 50)
                for mode, use_graph in modes:
                    if use_graph:
                        mm, yy, pp = setup(cfg, slab, device, dtype, args.warm)
                        runner = GraphRunner(mm, yy, pp, log=lambda *a: None)
                        stepper = lambda: runner.step()
                    else:
                        mm, yy, pp = m, y.clone(), p.clone()
                        state = {'y': yy, 'p': pp}
                        def stepper(mm=mm, state=state):
                            state['y'], state['p'] = mm.step(0.0, state['y'], state['p'])
                    sync(device); t0 = time.perf_counter()
                    for _ in range(args.steps):
                        stepper()
                    sync(device)
                    key = (mode, slab)
                    best[key] = min(best.get(key, 1e9), (time.perf_counter() - t0) / args.steps)
            print(f"round {r + 1}/{args.repeat} done", flush=True)
    m, _, _ = setup(cfg, True, device, dtype, 0)
    print(f"\n{args.parameters_file}: grid {'x'.join(map(str, m.shape))}, device {device}, "
          f"fuel slab {m.k_fuel} of {m.shape[0]} rows")
    print(f"{'':12s} {'full grid':>12s} {'slab':>12s} {'saving':>8s}")
    for name in ['solid_rates'] + [md for md, _ in modes]:
        if (name, False) in best:
            a, b = best[(name, False)] * 1e3, best[(name, True)] * 1e3
            print(f"{name + (' (ms)' if name == 'solid_rates' else ' (ms/step)'):12s} "
                  f"{a:12.3f} {b:12.3f} {100 * (1 - b / a):7.1f}%")


if __name__ == '__main__':
    main()
