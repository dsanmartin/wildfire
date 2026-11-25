"""Compare the burning structures of a run with the FDS structures case.

    python scripts/compare_structures.py data/output/c064_3d_st [--fds Case_C064_structures_cat_devc.csv]
           [--head-fds Case_C064_structures_cat_devc.csv]

Reads the per-building diagnostics of the run (S_* arrays in data.npz, written with [structures] model = ignition)
and, optionally, the FDS device file of Case_C064_structures (devices 'H<n> HRR', 'H<n> burning rate',
'H<n> total gauge heat flux', 'H<n> mean wall temperature', 'H<n> max wall temperature'; the n-th building of the
run is FDS house n). Plots, per house, the gas-phase heat release around it, the fuel release (burning rate),
the surface-integrated gauge heat flux and the mean and maximum surface temperature against time, and prints the
ignition time of each house (first surface element at its ignition temperature). Writes NAME/figures/structures.png.
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_devc(path):
    rows = list(csv.reader(open(path)))
    head = [c.strip() for c in rows[1]]
    data = np.array([[float(v) for v in r] for r in rows[2:] if r])
    return {name: data[:, i] for i, name in enumerate(head)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sim_dir')
    ap.add_argument('--fds', help='FDS device file of the structures case (CHID_devc.csv)')
    args = ap.parse_args()
    d = np.load(os.path.join(args.sim_dir, 'data.npz'))
    if 'S_t' not in d:
        raise SystemExit('no structure diagnostics in data.npz (run with [structures] model = ignition)')
    t = d['S_t']
    nb = d['S_T_s_mean'].shape[1]
    fds = read_devc(args.fds) if args.fds else None
    print(f"{'house':>6} {'ignition [s]':>13} {'peak fuel release [MW]':>23} {'peak gas HRR around [MW]':>25}"
          + (f" {'FDS: first burning [s]':>23} {'FDS peak HRR [MW]':>18}" if fds else ""))
    for b in range(nb):
        line = (f"{b + 1:>6} {d['S_t_ignition'][-1, b]:13.1f} {d['S_release_hrr'][:, b].max() / 1e6:23.2f} "
                f"{d['S_gas_hrr'][:, b].max() / 1e6:25.2f}")
        if fds:
            br = fds[f'H{b + 1} burning rate']
            first = fds['Time'][np.argmax(br > 0)] if (br > 0).any() else np.nan
            line += f" {first:23.1f} {fds[f'H{b + 1} HRR'].max() / 1e3:18.2f}"
        print(line)
    panels = [('gas_hrr', 'H{} HRR', 1e6, 1e3, 'gas-phase HRR within 2 m [MW]'),
              ('release_hrr', None, 1e6, None, 'fuel release x H_C [MW]'),
              ('gauge_heat_flux', 'H{} total gauge heat flux', 1e6, 1e3, 'gauge heat flux, surface integral [MW]'),
              ('T_s_mean', 'H{} mean wall temperature', 1.0, None, 'mean surface temperature [K]'),
              ('T_s_max', 'H{} max wall temperature', 1.0, None, 'max surface temperature [K]')]
    fig, axes = plt.subplots(len(panels), nb, figsize=(3.6 * nb, 2.4 * len(panels)), sharex=True, squeeze=False,
                             constrained_layout=True)
    for r, (key, fkey, scale, fscale, label) in enumerate(panels):
        for b in range(nb):
            ax = axes[r, b]
            ax.plot(t, d['S_' + key][:, b] / scale, 'C3', lw=1.6, label='model')
            if fds and fkey:
                v = fds[fkey.format(b + 1)]
                v = v * 1e3 / scale if fscale else v + 273.15          # FDS: kW, C
                ax.plot(fds['Time'], v, 'k', lw=1, label='FDS')
            if fds and key == 'release_hrr':                        # FDS burning rate (kg/s) x the model's H_C
                Hc = d['S_release_hrr'][:, b] / np.where(d['S_release'][:, b] > 0, d['S_release'][:, b], np.nan)
                Hc = np.nanmax(Hc) if np.isfinite(Hc).any() else 13.1e6
                ax.plot(fds['Time'], fds[f'H{b + 1} burning rate'] * Hc / scale, 'k', lw=1, label='FDS')
            if b == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 0:
                ax.set_title(f'house {b + 1}')
            ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel('t [s]')
    axes[0, 0].legend(fontsize=8)
    out = os.path.join(args.sim_dir, 'figures')
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, 'structures.png'), dpi=130)
    print('wrote', os.path.join(out, 'structures.png'))


if __name__ == '__main__':
    main()
