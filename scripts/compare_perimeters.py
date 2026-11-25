"""Compare a 3D run with the measured fire perimeters and head-fire positions of a CSIRO grassland fire.

    python scripts/compare_perimeters.py data/output/f19_3d_sponge --case F19 \
        [--fds FDS_0.5m=path/Case_F19_cat_devc.csv ...] [--xlim -20 240]

For each measured perimeter time the plan view shows the model's burnt area (the contour where half of
the fuel column is gone, from data.npz) and the measured perimeter (validation/csiro/<CASE>_perimeters.csv).
A second panel shows the head-fire position against time: the model (most downwind burnt point within
2 m of y = 0), the measured head (validation/csiro/<CASE>_front.csv) and, optionally, FDS device files
(column 'x': the location of the maximum temperature near the ground). Writes NAME/figures/perimeters.png
and prints, at each perimeter time, the measured and modeled head position and half-width.
Note: the FDS 'x' device and the model's burnt contour are different definitions of the head.
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
VALIDATION = os.path.join(HERE, '..', 'validation', 'csiro')


def read_csv(path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith('#')]
    return np.array([[float(v) for v in r] for r in rows[1:]])


def read_fds_devc(path):
    rows = list(csv.reader(open(path)))
    head = [c.strip() for c in rows[1]]
    data = np.array([[float(v) for v in r] for r in rows[2:] if r])
    return data[:, 0], data[:, head.index('x')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim_dir')
    ap.add_argument('--case', default='F19', choices=['F19', 'C064'])
    ap.add_argument('--fds', nargs='*', default=[], help='LABEL=path/to/CHID_devc.csv')
    ap.add_argument('--xlim', type=float, nargs=2)
    ap.add_argument('--threshold', type=float, default=0.5, help='fraction of the fuel column left that counts as burnt')
    args = ap.parse_args()

    d = np.load(os.path.join(args.sim_dir, 'data.npz'))
    if 'fuel_column' not in d:
        raise SystemExit("not a 3D run (no fuel_column in data.npz)")
    x, y, t = d['x'], d['y'], d['t']
    col = d['fuel_column']
    nx, ny = col.shape[2], col.shape[1]
    x, y = x[:nx], y[:ny]
    had = col[0] > 0
    frac = np.where(had, col / np.where(had, col[0], 1.0), 1.0)
    burnt = (frac < args.threshold) & had
    near = np.abs(y) <= 2.0
    head = np.array([x[b[near].any(axis=0)].max() if b[near].any() else np.nan for b in burnt])

    front = read_csv(os.path.join(VALIDATION, f'{args.case}_front.csv'))
    per = read_csv(os.path.join(VALIDATION, f'{args.case}_perimeters.csv'))
    times = sorted(set(per[:, 0]))

    print(f"{args.case}: measured vs model ({os.path.basename(os.path.normpath(args.sim_dir))})")
    print(f"{'t [s]':>6} {'head meas':>10} {'head model':>11} {'half-width meas':>16} {'half-width model':>17}")
    for tt in times:
        n = int(np.argmin(np.abs(t - tt)))
        p = per[per[:, 0] == tt]
        head_meas = np.interp(tt, front[:, 0], front[:, 1]) if front[:, 0].min() <= tt <= front[:, 0].max() else np.nan
        hw_meas = np.abs(p[:, 2]).max()
        rows = burnt[n].any(axis=1)
        hw_model = np.abs(y[rows]).max() if rows.any() else np.nan
        print(f"{tt:6.0f} {head_meas:10.1f} {head[n]:11.1f} {hw_meas:16.1f} {hw_model:17.1f}"
              + (f"   (model output at {t[n]:.1f} s)" if abs(t[n] - tt) > 0.5 else ""))

    xlim = args.xlim or (min(-10.0, per[:, 1].min() - 10), max(per[:, 1].max(), np.nanmax(head)) + 20)
    ylim = (min(-10.0, per[:, 2].min() - 15), max(10.0, per[:, 2].max() + 15))
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0])
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(times)))
    for tt, c in zip(times, colors):
        n = int(np.argmin(np.abs(t - tt)))
        ax.contour(x, y, burnt[n].astype(float), levels=[0.5], colors=[c], linewidths=1.6)
        p = per[per[:, 0] == tt]
        ax.plot(p[:, 1], p[:, 2], 'o--', color=c, ms=3.5, lw=1, label=f'{tt:.0f} s: measured (dots), model (line)')
    ax.set(xlim=xlim, ylim=ylim, xlabel='x [m]', ylabel='y [m]', aspect='equal',
           title=f'{args.case}: burnt area of the model vs measured perimeters')
    ax.legend(fontsize=8, loc='upper left')
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, head, 'C3', lw=2, label='model (burnt contour, |y| < 2 m)')
    for spec in args.fds:
        label, path = spec.split('=', 1)
        tf, xf = read_fds_devc(path)
        ax2.plot(tf, xf, lw=1, label=f'{label} (max temperature)')
    ax2.plot(front[:, 0], front[:, 1], 'ko', ms=6, label='measured')
    ax2.set(xlabel='t [s]', ylabel='head-fire position x [m]', title='head fire')
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    out = os.path.join(args.sim_dir, 'figures')
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, 'perimeters.png')
    fig.savefig(path, dpi=150)
    print('wrote', path)


if __name__ == '__main__':
    main()
