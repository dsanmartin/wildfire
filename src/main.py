"""Run a simulation: python src/main.py parameters.ini [-n NAME] [-path DIR] [--device cuda] ..."""
import argparse
import glob
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
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


def with_periodic_ends(a, axes=None):
    """Repeat the first node of every periodic axis at its end. By default all axes but the first
    (the vertical one): (Nz, [Ny-1,] Nx-1) -> (Nz, [Ny,] Nx)."""
    for axis in (range(1, a.ndim) if axes is None else axes):
        a = np.concatenate([a, np.take(a, [0], axis=axis)], axis=axis)
    return a


class Recorder:
    """Collects the output at the saved times (every NT steps): the state, p and the derived fields
    of Model.output_fields.

    2D: the full fields. 3D: three planes of every field: x-z through y = y_slice, x-y at
    z = z_slice, and y-z through the head fire (the most downwind burnt node of the ground row on the
    x-z plane), plus the fuel left in every column, and the full fields of the last output.
    Full-field snapshots at a chosen interval (``[output] full_interval``, both 2D and 3D) are written
    by ``save_full`` to separate files, so memory does not grow with them."""

    def __init__(self, cfg, grid, names):
        self.cfg, self.nd, self.names = cfg, cfg.ndim, names
        self.times, self.frames = [], []
        if self.nd == 3:
            y_nodes, z_nodes = grid['y'][:-1], grid['z']
            y0 = cfg.y_slice if cfg.y_slice is not None else 0.5 * (cfg.y_min + cfg.y_max)
            self.j = int(np.argmin(np.abs(y_nodes - y0)))
            self.k = int(np.argmin(np.abs(z_nodes - cfg.z_slice)))
            self.x_nodes = grid['x'][:-1]
            self.ix0 = int(np.argmin(np.abs(self.x_nodes - cfg.T0_x_end)))
            self.y_value, self.z_value = float(y_nodes[self.j]), float(z_nodes[self.k])
            self.had_fuel = grid['state'][cfg.ndim + 1][0, self.j] > 0.5     # ground row on the x-z plane

    def add(self, y, p, t, extra):
        """extra: dict of derived fields (same order at every call)."""
        self.times.append(t)
        full = torch.cat([y, p.unsqueeze(0)] + [v.unsqueeze(0) for v in extra.values()])
        if self.nd == 2:
            self.frames.append(full.cpu().numpy())
            return
        nd = self.nd
        Y = y[nd + 1]
        burnt = (Y[0, self.j] < 0.5).cpu().numpy() & self.had_fuel & (self.x_nodes >= self.cfg.T0_x_start)
        ix = int(np.flatnonzero(burnt).max()) if burnt.any() else self.ix0
        self.frames.append({'xz': full[:, :, self.j, :].cpu().numpy(), 'xy': full[:, self.k].cpu().numpy(),
                            'yz': full[:, :, :, ix].cpu().numpy(), 'yz_x': self.x_nodes[ix],
                            'fuel_column': Y.clamp(min=0).sum(0).cpu().numpy()})
        self.last = full.cpu().numpy()

    def arrays(self):
        names = self.names
        if self.nd == 2:
            z = np.stack(self.frames)
            return {name: np.stack([with_periodic_ends(f) for f in z[:, i]]) for i, name in enumerate(names)}
        out = {}
        for plane, axes in (('xz', [1]), ('xy', [0, 1]), ('yz', [1])):
            stack = np.stack([f[plane] for f in self.frames])            # (t, nfields, a, b)
            for i, name in enumerate(names):
                out[f'{plane}_{name}'] = np.stack([with_periodic_ends(s, axes) for s in stack[:, i]])
        out['yz_x'] = np.array([f['yz_x'] for f in self.frames])
        out['fuel_column'] = np.stack([with_periodic_ends(f['fuel_column'], [0, 1]) for f in self.frames])
        for i, name in enumerate(names):
            out['final_' + name] = with_periodic_ends(self.last[i])
        out['y_slice'], out['z_slice'] = self.y_value, self.z_value
        return out


def save_full(cfg, grid, names, y, p, extra, t):
    """Write all fields at time t to NAME/full/t_XXXXX.XXX.npz (precision: [output] full_dtype)."""
    folder = os.path.join(cfg.save_path, 'full')
    os.makedirs(folder, exist_ok=True)
    full = torch.cat([y, p.unsqueeze(0)] + [v.unsqueeze(0) for v in extra.values()]).cpu().numpy()
    data = {name: with_periodic_ends(full[i]).astype(cfg.full_dtype) for i, name in enumerate(names)}
    data['t'], data['x'] = t, grid['x']
    if cfg.ndim == 2:
        data['y'] = grid['z']
    else:
        data['y'], data['z'] = grid['y'], grid['z']
    np.savez(os.path.join(folder, f't_{t:09.3f}.npz'), **data)


def make_runner(model, y, p, say):
    """CUDA-graph runner, or None (with a message) if it cannot be used."""
    from cudagraph import GraphRunner, graph_problems
    problems = graph_problems(model)
    if problems:
        say(f"CUDA graphs disabled: {problems}")
        return None
    try:
        runner = GraphRunner(model, y, p, log=say, error_dir=model.cfg.save_path)
        if runner.emulate:
            say(f"CUDA graphs: emulated on {y.device} (PCG chunks of {runner.chunk} iterations)")
        return runner
    except Exception as e:                                       # fall back to direct execution
        say(f"CUDA graphs disabled: capture failed ({type(e).__name__}: {e})")
        return None


def run(cfg, verbose=True):
    device = pick_device(cfg.device)
    dtype = getattr(torch, cfg.dtype)
    if device.type == 'mps' and dtype == torch.float64:
        raise ValueError("the MPS backend has no float64; use --dtype float32 or --device cpu")
    if cfg.model == 'planview':
        from planview import run_planview
        return run_planview(cfg, device, dtype, verbose)
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
    if model.k_fuel < model.shape[0]:
        say(f"Fuel slab: rows 0-{model.k_fuel - 1} of {model.shape[0]} "
            f"(z <= {float(model.Zm[model.k_fuel - 1].max()):.2f} m)")
    y = torch.as_tensor(grid['state'], dtype=dtype, device=device)
    if model.has_terrain and not cfg.legacy:
        model.apply_bc(y)
    p = torch.zeros(model.shape, dtype=dtype, device=device)
    names = ['u', 'v', 'T', 'Y'] if nd == 2 else ['u', 'v', 'w', 'T', 'Y']
    if cfg.solid_temperature:
        names += ['Ts', 'W', 'YF', 'YO2']        # solid fuel temperature, moisture, fuel gas, oxygen
    extra = model.output_fields(y)
    names = names + ['p'] + list(extra)
    rec = Recorder(cfg, grid, names)
    rec.add(y, p, t[0], extra)
    next_full = None
    if cfg.full_interval and cfg.full_interval > 0:
        save_full(cfg, grid, names, y, p, extra, t[0])
        next_full = t[0] + cfg.full_interval
    unit = 'kJ/m' if nd == 2 else 'kJ'
    energy = {key: [0.0] for key in ENERGY_TERMS}
    # Time stepping. Fixed step dt_max = (t_max - t_min) / Nt, or adaptive ([numerics] cfl > 0): each step
    # takes the largest dt allowed by the CFL target and the diffusion limit, at most dt_max, and steps
    # are shortened to land exactly on the output times, which stay every NT * dt_max seconds.
    dt_max = model.dt
    out_dt = NT * dt_max
    t_end = t[-1]
    t_now, n, steps_since = t[0], 0, 0
    next_out = t[0] + out_dt
    dt_min_seen, dt_max_seen = np.inf, 0.0
    runner = None
    start = last = time.perf_counter()
    with torch.no_grad():
        while t_now < t_end - 1e-9 * out_dt:
            dt = model.stable_dt(y, dt_max) if cfg.cfl > 0 else dt_max
            dt = min(dt, next_out - t_now, t_end - t_now)
            model.set_dt(dt)
            dt_min_seen, dt_max_seen = min(dt_min_seen, dt), max(dt_max_seen, dt)
            if runner is None and cfg.cuda_graph and n == 3:     # capture after a few direct steps
                runner = make_runner(model, y, p, say)
            if runner is not None:
                y, p = runner.step()
            else:
                y, p = model.step(t_now, y, p)
            t_now += dt
            n += 1
            steps_since += 1
            if next_full is not None and t_now >= next_full - 0.5 * dt:
                save_full(cfg, grid, names, y, p, model.output_fields(y), t_now)
                next_full += cfg.full_interval
            if t_now >= next_out - 1e-9 * out_dt or t_now >= t_end - 1e-9 * out_dt:
                next_out += out_dt
                model.update_diagnostics()
                cfl, T_min, T_max, Y_min, Y_max = model.diagnostics(y)     # also synchronises the device
                now = time.perf_counter()
                say(f"Time step: {n:6d}, Simulation time: {t_now:.4f} s",
                    f"CFL: {cfl:.6f}, dt: {dt:.3e} s" + (f" (since last output: min {dt_min_seen:.3e}, max {dt_max_seen:.3e})"
                                                        if cfg.cfl > 0 else ""),
                    f"Gas temperature: Min = {T_min:.2f} K, Max {T_max:.2f} K",
                    f"Fuel: Min = {Y_min:.2f}, Max {Y_max:.2f}",
                    *([f"Fuel temperature (nodes with fuel): Min = {s[0]:.2f} K, Max {s[1]:.2f} K"]
                      if (s := model.solid_diagnostics(y) if cfg.solid_temperature else None) else []),
                    *([f"Heat release rate: {fd[0] / 1e6:.3f} {'MW/m' if nd == 2 else 'MW'}, "
                       f"HRR-weighted gas temperature: {fd[1]:.1f} K"] if (fd := model.flame_diagnostics(y)) else []),
                    *([f"Max discrete divergence error (face fluxes): {model.max_divergence:.3e} 1/s"]
                      if cfg.projection == 'face' else []),
                    *([f"Max divergence error next to buildings: {model.wall_divergence:.3e} 1/s"]
                      if getattr(model, 'has_solids', False) else []),
                    f"Pressure iterations: {model.pressure_iterations}, "
                    f"mean step time: {(now - last) / steps_since:.6f} s")
                last, steps_since = now, 0
                dt_min_seen, dt_max_seen = np.inf, 0.0
                E = {key: float(v) / 1e3 for key, v in model.energy.items()}
                dE = {key: E[key] - energy[key][-1] for key in E}
                ratio = lambda key, d: f"{100 * d[key] / d['combustion']:.1f}%" if d['combustion'] > 0 else "n/a"
                say(f"Energy since start [{unit}]: combustion {E['combustion']:.4g}, fuel overshoot {E['fuel_overshoot']:.4g} "
                    f"({ratio('fuel_overshoot', E)}), clip low {E['clip_low']:.4g} ({ratio('clip_low', E)}), "
                    f"clip high {E['clip_high']:.4g} ({ratio('clip_high', E)})",
                    f"Energy since last output: clip low {ratio('clip_low', dE)}, fuel overshoot {ratio('fuel_overshoot', dE)} "
                    f"of combustion", "")
                for key in E:
                    energy[key].append(E[key])
                if not np.isfinite(T_max):
                    say("Non-finite temperature: stopping")
                    break
                rec.add(y, p, t_now, model.output_fields(y))
    elapsed = time.perf_counter() - start
    say(f"\nSolver time: {timedelta(seconds=round(elapsed))} ({elapsed / max(n, 1) * 1e3:.2f} ms per step, {n} steps)\n")
    log.close()

    data = rec.arrays()
    data['x'], data['t'] = grid['x'], np.array(rec.times)
    if grid['solid'].any():
        data['solid'] = with_periodic_ends(grid['solid'])
    n_saved = len(rec.times)
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


def save_provenance(cfg, parameters_file, device):
    """Store what is needed to reproduce the run: the parameter file as given, all resolved
    parameters, the code that ran (src/*.py, which also covers uncommitted changes) and the git
    and library versions."""
    out = cfg.save_path
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(parameters_file, os.path.join(out, 'input.ini'))
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(cfg.to_dict(), f, indent=2)
    src = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.join(out, 'code')
    os.makedirs(code_dir, exist_ok=True)
    for path in glob.glob(os.path.join(src, '*.py')):
        shutil.copy2(path, code_dir)

    def git(*args):
        try:
            return subprocess.run(['git', *args], cwd=src, capture_output=True, text=True, timeout=10).stdout.rstrip()
        except Exception:
            return None
    status = git('status', '--porcelain', '--', '.')
    info = {
        'date': datetime.now().isoformat(timespec='seconds'),
        'command': ' '.join(sys.argv),
        'parameters_file': os.path.abspath(parameters_file),
        'git_branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'git_commit': git('rev-parse', 'HEAD'),
        'git_uncommitted_changes_in_src': bool(status) if status is not None else None,
        'git_status_src': status.splitlines() if status else [],
        'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
        'host': platform.node(),
    }
    with open(os.path.join(out, 'run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


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
    ap.add_argument('--cuda-graph', action='store_true', default=None,
                    help='replay each time step as CUDA graphs (GPU; emulated on CPU for testing)')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    save_path = args.save_path or f'./data/output/{args.name}/'
    cfg = load_config(args.parameters_file, sim_name=args.name, save_path=save_path, device=args.device,
                      dtype=args.dtype, poisson=args.poisson, legacy=args.legacy, cuda_graph=args.cuda_graph)
    save_provenance(cfg, args.parameters_file, pick_device(cfg.device))
    run(cfg)
    print("Simulation name:", cfg.sim_name)


if __name__ == '__main__':
    main()
