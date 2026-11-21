"""Run a simulation: python src/main.py parameters.ini [-n NAME] [-path DIR] [--device cuda] ..."""
import argparse
import os
import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import torch
from config import load_config
from setup import build
from model import Model, ENERGY_TERMS


def pick_device(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # MPS only on request: it has no float64


def with_periodic_ends(a):
    """(Nz, [Ny-1,] Nx-1) -> (Nz, [Ny,] Nx): repeat the first node of every periodic axis at the end."""
    for axis in range(1, a.ndim):
        a = np.concatenate([a, np.take(a, [0], axis=axis)], axis=axis)
    return a


def run(cfg, verbose=True):
    device = pick_device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    if device.type == 'mps' and dtype == torch.float64:
        raise ValueError("the MPS backend has no float64; use --dtype float32 or --device cpu")
    grid = build(cfg)
    model = Model(cfg, grid, device, dtype)
    nd, Nt, NT, t = cfg.ndim, cfg.Nt, cfg.NT, grid['t']
    os.makedirs(cfg.save_path, exist_ok=True)
    log = open(os.path.join(cfg.save_path, 'log.txt'), 'w')

    def say(*lines):
        for line in lines:
            print(line, file=log, flush=True)
            if verbose:
                print(line)

    say(f"Simulation: {cfg.sim_name} ({nd}D), device: {device}, dtype: {cfg.dtype}, Poisson: {model.poisson.method}",
        f"Grid: {'x'.join(str(s) for s in model.shape)} (z, {'y, ' if nd == 3 else ''}x), "
        f"spacings (dx, {'dy, ' if nd == 3 else ''}dz): {model.h}, dt: {model.dt:.3e}, steps: {Nt}")
    y = torch.as_tensor(grid['state'], dtype=dtype, device=device)
    if model.has_terrain and not cfg.legacy:
        model.apply_bc(y)
    p = torch.zeros(model.shape, dtype=dtype, device=device)
    frames, pressures, times = [y.cpu().numpy()], [p.cpu().numpy()], [t[0]]
    unit = 'kJ/m' if nd == 2 else 'kJ'
    energy = {key: [0.0] for key in ENERGY_TERMS}
    start = last = time.perf_counter()
    with torch.no_grad():
        for n in range(Nt):
            y, p = model.step(t[n], y, p)
            if (n + 1) % NT == 0 or n == Nt - 1:
                cfl, T_min, T_max, Y_min, Y_max = model.diagnostics(y)     # also synchronises the device
                now = time.perf_counter()
                say(f"Time step: {n + 1:6d}, Simulation time: {t[n + 1]:.4f} s",
                    f"CFL: {cfl:.6f}",
                    f"Temperature: Min = {T_min:.2f} K, Max {T_max:.2f} K",
                    f"Fuel: Min = {Y_min:.2f}, Max {Y_max:.2f}",
                    f"Pressure iterations: {model.pressure_iterations}, "
                    f"mean step time: {(now - last) / ((n % NT) + 1):.6f} s")
                last = now
                E = {key: float(v) / 1e3 for key, v in model.energy.items()}
                dE = {key: E[key] - energy[key][-1] for key in E}
                ratio = lambda key, d: f"{100 * d[key] / d['combustion']:.1f}%" if d['combustion'] > 0 else "n/a"
                say(f"Energy since start [{unit}]: combustion {E['combustion']:.4g}, fuel overshoot {E['fuel_overshoot']:.4g} "
                    f"({ratio('fuel_overshoot', E)}), clip low {E['clip_low']:.4g} ({ratio('clip_low', E)}), "
                    f"clip high {E['clip_high']:.4g} ({ratio('clip_high', E)})",
                    f"Energy since last output: clip low {ratio('clip_low', dE)}, fuel overshoot {ratio('fuel_overshoot', dE)} "
                    f"of combustion")
                for key in E:
                    energy[key].append(E[key])
                if not np.isfinite(T_max):
                    say("Non-finite temperature: stopping")
                    break
                frames.append(y.cpu().numpy())
                pressures.append(p.cpu().numpy())
                times.append(t[n + 1])
    elapsed = time.perf_counter() - start
    say(f"\nSolver time: {timedelta(seconds=round(elapsed))} ({elapsed / (n + 1) * 1e3:.2f} ms per step)\n")
    log.close()

    z = np.stack(frames)                                             # (Nt_saved, ndim + 2, Nz, ...)
    names = ['u', 'v', 'T', 'Y'] if nd == 2 else ['u', 'v', 'w', 'T', 'Y']
    data = {name: np.stack([with_periodic_ends(f) for f in z[:, i]]) for i, name in enumerate(names)}
    data['p'] = np.stack([with_periodic_ends(f) for f in pressures])
    data['x'], data['t'] = grid['x'], np.array(times)
    n_saved = len(times)
    for key, values in energy.items():      # cumulative, in J (per metre in 2D), at the saved times
        data['E_' + key] = np.array(values[:n_saved]) * 1e3
    if nd == 2:
        data['y'] = grid['z']                                        # the vertical axis is called y in 2D
    else:
        data['y'], data['z'] = grid['y'], grid['z']
    np.savez(os.path.join(cfg.save_path, 'data.npz'), **data)
    parameters = cfg.to_dict()
    parameters.update(x=grid['x'], y=data['y'], t=grid['t'], dt=model.dt,
                      cut_nodes=np.where(grid['cut']), dead_nodes=np.where(grid['dead']))
    with open(os.path.join(cfg.save_path, 'parameters.pkl'), 'wb') as f:
        pickle.dump(parameters, f)
    return data, elapsed


def main():
    ap = argparse.ArgumentParser(description='Coupled wildfire-atmosphere simulation (PyTorch)')
    ap.add_argument('parameters_file')
    ap.add_argument('-n', '--name', default=datetime.now().strftime('%Y%m%d%H%M%S'))
    ap.add_argument('-path', '--save-path', default=None)
    ap.add_argument('--device', default=None, help='auto (cuda if available, else cpu), cpu, cuda, cuda:1, mps')
    ap.add_argument('--dtype', default=None, choices=['float64', 'float32'])
    ap.add_argument('--poisson', default=None, choices=['auto', 'thomas', 'eig'])
    ap.add_argument('--legacy', action='store_true', default=None,
                    help='reproduce the original src_legacy/ code, defects included (see src/README.md)')
    ap.add_argument('--threads', type=int, default=None, help='CPU threads')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    save_path = args.save_path or f'./data/output/{args.name}/'
    cfg = load_config(args.parameters_file, sim_name=args.name, save_path=save_path, device=args.device,
                      dtype=args.dtype, poisson=args.poisson, legacy=args.legacy)
    run(cfg)
    print("Simulation name:", cfg.sim_name)


if __name__ == '__main__':
    main()
