"""Fire front position and rate of spread, optionally compared with observations.

    python src/firefront.py data/output/RUN1 [data/output/RUN2 ...] [--obs f19_front.csv] [--window 20 120]

Head fire: the most downwind position where half of the fuel is gone; back fire: the most upwind
one. In 2D this uses the ground row of the fuel bed; in 3D the fuel left in each column along the
x-z plane (y = y_slice), and also reports the burnt area. The rate of spread (ROS) is a linear fit
of the head-fire position over the time window (default: the second half of the run).

--obs: CSV with columns t [s], x [m] (head-fire position; a header line is allowed). The model
front is interpolated at the observation times; the script reports the RMSE of the position and
both rates of spread over the window. x must use the same origin as the simulation.

Writes NAME/figures/fire_front.csv per run and a comparison plot fire_front.png in the first run.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fronts(d, threshold=0.5):
    x, t = d['x'], d['t']
    if 'plan_view' in d:                        # plan view: fuel map, front along y = y_slice
        Y = d['Y']
        j = int(np.argmin(np.abs(d['y'] - float(d['y_slice']))))
        burnt = (Y[:, j, :] < threshold) & (Y[0, j, :] > threshold)
        dx, dy = x[1] - x[0], d['y'][1] - d['y'][0]
        area = ((Y[:, :-1, :-1] < threshold) & (Y[0, :-1, :-1] > threshold)).sum(axis=(1, 2)) * dx * dy
    elif 'fuel_column' in d:                    # 3D
        col = d['fuel_column']
        frac = col / np.where(col[0] > 0, col[0], np.nan)
        j = int(np.argmin(np.abs(d['y'] - float(d['y_slice']))))
        burnt = np.nan_to_num(frac[:, j, :], nan=1.0) < threshold
        dx, dy = x[1] - x[0], d['y'][1] - d['y'][0]
        area = (np.nan_to_num(frac[:, :-1, :-1], nan=1.0) < threshold).sum(axis=(1, 2)) * dx * dy
    else:                                       # 2D: ground row
        burnt = (d['Y'][:, 0, :] < threshold) & (d['Y'][0, 0, :] > threshold)   # only where there was fuel
        area = None
    head = np.array([x[b].max() if b.any() else np.nan for b in burnt])
    back = np.array([x[b].min() if b.any() else np.nan for b in burnt])
    return t, head, back, area


def ros(t, xf, window):
    m = (t >= window[0]) & (t <= window[1]) & np.isfinite(xf)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(t[m], xf[m], 1)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sim_dirs', nargs='+')
    ap.add_argument('--obs', help='CSV of observed head-fire position: t [s], x [m]')
    ap.add_argument('--window', type=float, nargs=2, help='time window for the rate of spread [s]')
    ap.add_argument('--threshold', type=float, default=0.5, help='fuel fraction left that counts as burnt')
    args = ap.parse_args()

    obs = None
    if args.obs:
        obs = np.genfromtxt(args.obs, delimiter=',', comments='#', skip_header=0)
        obs = obs[np.all(np.isfinite(obs), axis=1)][:, :2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for sim in args.sim_dirs:
        d = dict(np.load(os.path.join(sim, 'data.npz')))
        t, head, back, area = fronts(d, args.threshold)
        window = args.window or (t[-1] / 2, t[-1])
        name = os.path.basename(os.path.normpath(sim))
        r = ros(t, head, window)
        line = f"{name}: head fire at end {head[-1]:.1f} m | ROS {window[0]:.0f}-{window[1]:.0f} s = {r:.2f} m/s"
        if area is not None:
            line += f" | burnt area at end {area[-1]:.0f} m^2"
        if obs is not None:
            xm = np.interp(obs[:, 0], t, head, left=np.nan, right=np.nan)
            ok = np.isfinite(xm)
            rmse = float(np.sqrt(np.mean((xm[ok] - obs[ok, 1]) ** 2))) if ok.any() else np.nan
            line += f" | RMSE vs observations {rmse:.1f} m ({ok.sum()} points)"
        print(line)
        out = os.path.join(sim, 'figures')
        os.makedirs(out, exist_ok=True)
        cols = [t, head, back] + ([area] if area is not None else [])
        np.savetxt(os.path.join(out, 'fire_front.csv'), np.column_stack(cols), delimiter=',',
                   header='t_s,head_x_m,back_x_m' + (',burnt_area_m2' if area is not None else ''), comments='')
        lh = axes[0].plot(t, head, label=f'{name} (ROS {r:.2f} m/s)')[0]
        axes[0].plot(t, back, color=lh.get_color(), ls=':', lw=1)
        dt = np.gradient(t)
        axes[1].plot(t, np.gradient(head, t) if len(t) > 2 else head * 0, color=lh.get_color(), label=name)
    if obs is not None:
        # the observations have their own times: fit over all of them (two points: their slope)
        r_obs = ros(obs[:, 0], obs[:, 1], (obs[:, 0].min(), obs[:, 0].max()))
        print(f"observations: ROS = {r_obs:.2f} m/s over {obs[:, 0].min():.0f}-{obs[:, 0].max():.0f} s")
        axes[0].plot(obs[:, 0], obs[:, 1], 'ko', ms=4, label=f'observed (ROS {r_obs:.2f} m/s)')
    axes[0].set(xlabel='t [s]', ylabel='x [m]', title='Head fire (solid) and back fire (dotted)')
    axes[1].set(xlabel='t [s]', ylabel='dx/dt [m/s]', title='Instantaneous head-fire speed')
    axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
    path = os.path.join(args.sim_dirs[0], 'figures', 'fire_front.png')
    fig.savefig(path, dpi=150)
    print('wrote', path)


if __name__ == '__main__':
    main()
