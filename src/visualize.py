"""Snapshots and animations of a simulation saved by src/main.py.

    python src/visualize.py data/output/NAME                       # every field: 3 snapshots + MP4
    python src/visualize.py data/output/NAME --field T HRR --times 10 30 60
    python src/visualize.py data/output/NAME --field T --xlim -20 150 --vmax 1400 --no-anim
    python src/visualize.py data/output/NAME --field speed T --arrows      # with velocity arrows

2D runs: x-z plane. 3D runs: the three planes saved at every output, x-y (at z_slice), x-z (at
y_slice) and y-z (through the head fire). The fuel bed edge (half of the initial fuel left) is drawn
as a white contour, on the x-y plane from the fuel left in each column.

Animations are MP4 when ffmpeg is available (on the system, or bundled with the imageio-ffmpeg
package), GIF otherwise or with --gif. Figures are written to NAME/figures/.
"""
import argparse
import os
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

LABELS = {'T': 'Gas temperature [K]', 'Ts': 'Fuel temperature [K]', 'Y': 'Fuel fraction [-]',
          'YF': 'Fuel gas mass fraction [-]', 'YO2': 'Oxygen mass fraction [-]', 'W': 'Moisture [kg/kg]',
          'u': 'u [m/s]', 'v': 'v [m/s]', 'w': 'w [m/s]', 'p': 'p [Pa]', 'speed': 'Speed [m/s]',
          'HRR': 'Heat release rate [W/m$^3$]', 'pyrolysis': 'Pyrolysis rate [kg/(m$^3$ s)]',
          'Q_conv': 'Convective heating of the fuel [W/m$^3$]', 'G': 'Incident radiation [W/m$^2$]',
          'q_rad_gas': 'Net radiative heating of the gas [W/m$^3$]',
          'q_rad_fuel': 'Net radiative heating of the fuel [W/m$^3$]', 'nu_t': 'Turbulent viscosity [m$^2$/s]'}
CMAPS = {'T': 'inferno', 'Ts': 'inferno', 'Y': 'YlGn', 'YF': 'viridis', 'YO2': 'Blues_r', 'W': 'Blues',
         'speed': 'viridis', 'u': 'RdBu_r', 'v': 'RdBu_r', 'w': 'RdBu_r', 'p': 'RdBu_r',
         'HRR': 'hot', 'pyrolysis': 'hot', 'Q_conv': 'RdBu_r', 'G': 'magma', 'q_rad_gas': 'RdBu_r',
         'q_rad_fuel': 'RdBu_r', 'nu_t': 'viridis'}
ORDER = ['T', 'speed', 'u', 'v', 'w', 'p', 'Y', 'Ts', 'W', 'YF', 'YO2', 'HRR', 'pyrolysis', 'Q_conv',
         'G', 'q_rad_gas', 'q_rad_fuel', 'nu_t']


def available_fields(d):
    """Fields present in the run, in a fixed order; 'speed' is derived from the velocity."""
    is3d = 'z' in d
    has = (lambda n: f'xz_{n}' in d) if is3d else (lambda n: n in d)
    return [n for n in ORDER if (n == 'speed' and has('u')) or (n != 'speed' and has(n))]


def field2d(d, name):
    """(t, z, x) array of a 2D run."""
    if name == 'speed':
        return np.hypot(d['u'], d['v'])
    return d[name]


def plane3d(d, plane, name):
    if name == 'speed':
        return np.sqrt(sum(d[f'{plane}_{c}'] ** 2 for c in 'uvw'))
    return d[f'{plane}_{name}']


def head_fire_x(d):
    """Most downwind position where the ground fuel (2D) or the column fuel (3D, centreline) is half gone."""
    x = d['x']
    if 'plan_view' in d:
        j = np.argmin(np.abs(d['y'] - float(d['y_slice'])))
        burnt = (d['Y'][:, j, :] < 0.5) & (d['Y'][0, j, :] > 0.5)
    elif 'fuel_column' in d:
        col = d['fuel_column']
        frac = col / np.where(col[0] > 0, col[0], np.nan)
        j = np.argmin(np.abs(d['y'] - float(d['y_slice'])))
        burnt = frac[:, j, :] < 0.5
    else:
        burnt = (d['Y'][:, 0, :] < 0.5) & (d['Y'][0, 0, :] > 0.5)
    return np.array([x[b].max() if b.any() else np.nan for b in burnt])


def find_ffmpeg():
    """ffmpeg executables to try, in order: the system one, then the one bundled with imageio-ffmpeg."""
    paths = []
    if shutil.which('ffmpeg'):
        paths.append(shutil.which('ffmpeg'))
    try:
        import imageio_ffmpeg
        paths.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    return list(dict.fromkeys(paths))


def has_h264(ffmpeg):
    """True if this ffmpeg has the libx264 encoder. Only H.264 is used: other encoders (mpeg4, or the
    openh264 build of conda's ffmpeg) produce files that many players show as green or not at all."""
    try:
        out = subprocess.run([ffmpeg, '-hide_banner', '-encoders'], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    return ' libx264 ' in out


def save_anim(anim, out_dir, name, args, ffmpegs):
    """H.264 MP4 with the first ffmpeg that has libx264 and works; GIF otherwise (or with --gif)."""
    if not args.gif:
        path = os.path.join(out_dir, f'{name}.mp4')
        for ffmpeg in ffmpegs:
            matplotlib.rcParams['animation.ffmpeg_path'] = ffmpeg
            writer = animation.FFMpegWriter(fps=args.fps, codec='libx264',
                                            extra_args=['-pix_fmt', 'yuv420p', '-crf', '20',
                                                        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
            try:
                anim.save(path, writer=writer, dpi=120)
                print('wrote', path)
                return
            except Exception:
                if os.path.exists(path):
                    os.remove(path)
    path = os.path.join(out_dir, f'{name}.gif')
    anim.save(path, writer=animation.PillowWriter(fps=args.fps), dpi=90)
    print('wrote', path)


def solids(ax, X, Z, mask):
    """Grey fill of the building nodes in a plane."""
    if mask is not None and mask.any():
        ax.contourf(X, Z, mask.astype(float), levels=[0.5, 1.5], colors=['0.55'], zorder=3)


def arrows(ax, X, Z, U, V, xlim, zlim, scale):
    """Velocity arrows in the plane, about 40 x 14 of them over the visible window, same scale for all frames."""
    ix = np.flatnonzero((X >= xlim[0]) & (X <= xlim[1]))
    iz = np.flatnonzero((Z >= zlim[0]) & (Z <= zlim[1]))
    ix = ix[::max(1, len(ix) // 40)]
    iz = iz[::max(1, len(iz) // 14)]
    ax.quiver(X[ix], Z[iz], U[np.ix_(iz, ix)], V[np.ix_(iz, ix)], color='w', alpha=0.8,
              scale=scale, width=0.0018, headwidth=4)


def color_range(values, name, args):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    if args.vmin is not None or args.vmax is not None:
        return (args.vmin if args.vmin is not None else float(v.min()),
                args.vmax if args.vmax is not None else float(v.max()))
    if CMAPS.get(name) == 'RdBu_r':                       # signed field: symmetric range
        m = float(np.percentile(np.abs(v), 99.5)) or 1.0
        return -m, m
    vmin = 288.15 if name in ('T', 'Ts') else float(np.percentile(v, 0.1))
    vmax = float(np.percentile(v, 99.9))
    return vmin, (vmax if vmax > vmin else vmin + 1.0)


def render(d, name, args, out_dir, ffmpeg):
    is3d = 'z' in d
    t, x = d['t'], d['x']
    z = d['z'] if is3d else d['y']
    if is3d:
        planes = {p: plane3d(d, p, name) for p in ('xy', 'xz', 'yz')}
        vmin, vmax = color_range(np.concatenate([planes[p].ravel() for p in planes]), name, args)
    else:
        F = field2d(d, name)
        vmin, vmax = color_range(F.ravel(), name, args)
    cmap = CMAPS.get(name, 'viridis')
    label = LABELS.get(name, name)
    front = head_fire_x(d)
    if args.xlim:
        xlim = args.xlim
    else:
        f = front[np.isfinite(front)]
        xlim = (min(-20.0, float(f.min()) - 20) if f.size else -20.0, (f.max() if f.size else 50.0) + 40)
    zlim = args.zlim or (float(z[0]), float(z[-1]))
    times = args.times if args.times else [t[len(t) // 3], t[2 * len(t) // 3], t[-1]]
    if is3d:
        speed_ref = float(np.percentile(np.sqrt(d['xz_u'] ** 2 + d['xz_w'] ** 2), 99))
    elif 'u' not in d:   # plan view: no velocity field
        speed_ref = 1.0
    else:
        speed_ref = float(np.percentile(np.hypot(d['u'], d['v']), 99))
    arrow_scale = 25 * max(speed_ref, 1e-6)              # an arrow of speed_ref spans ~1/25 of the axis width
    idx = [int(np.argmin(np.abs(t - s))) for s in times]

    plan = 'plan_view' in d
    if plan:   # x-y map of the fuel-bed layer
        y = d['y']
        ylim = args.ylim or (float(y[0]), float(y[-1]))

        def draw(ax, n):
            ax.clear()
            m = ax.pcolormesh(x, y, F[n], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.contour(x, y, d['Y'][n], levels=[0.5], colors='w', linewidths=0.8)
            ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect('equal')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.set_title(f't = {t[n]:.1f} s', fontsize=10, loc='left')
            return m
        fig, axes = plt.subplots(1, len(idx), figsize=(5 * len(idx), 4.6), constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, idx):
            m = draw(ax, n)
        fig.colorbar(m, ax=axes, label=label, shrink=0.85)
        path = os.path.join(out_dir, f'{name}_snapshots.png')
        fig.savefig(path, dpi=150); plt.close(fig)
        print('wrote', path)
        if not args.no_anim:
            fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
            m = draw(ax, 0)
            fig.colorbar(m, ax=ax, label=label, shrink=0.85)
            anim = animation.FuncAnimation(fig, lambda n: draw(ax, n), frames=len(t), interval=1000 / args.fps)
            save_anim(anim, out_dir, name, args, ffmpeg)
            plt.close(fig)
        return

    if not is3d:
        def draw(ax, n):
            ax.clear()
            m = ax.pcolormesh(x, z, F[n], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax.contour(x, z, d['Y'][n], levels=[0.5], colors='w', linewidths=0.8)
            solids(ax, x, z, d.get('solid'))
            if args.arrows:
                arrows(ax, x, z, d['u'][n], d['v'][n], xlim, zlim, arrow_scale)
            ax.set_xlim(xlim); ax.set_ylim(zlim); ax.set_ylabel('z [m]')
            ax.set_title(f't = {t[n]:.1f} s' + (f',  head fire at x = {front[n]:.1f} m' if np.isfinite(front[n]) else ''),
                         fontsize=10, loc='left')
            return m
        fig, axes = plt.subplots(len(idx), 1, figsize=(11, 2.2 * len(idx) + 0.6), sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, idx):
            m = draw(ax, n)
        axes[-1].set_xlabel('x [m]')
        fig.colorbar(m, ax=axes, label=label, shrink=0.9)
        path = os.path.join(out_dir, f'{name}_snapshots.png')
        fig.savefig(path, dpi=150); plt.close(fig)
        print('wrote', path)
        if not args.no_anim:
            fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
            m = draw(ax, 0); ax.set_xlabel('x [m]')
            fig.colorbar(m, ax=ax, label=label)
            anim = animation.FuncAnimation(fig, lambda n: draw(ax, n), frames=len(t), interval=1000 / args.fps)
            save_anim(anim, out_dir, name, args, ffmpeg)
            plt.close(fig)
        return

    y = d['y']
    ylim = args.ylim or (float(y[0]), float(y[-1]))
    col = d['fuel_column']
    frac = col / np.where(col[0] > 0, col[0], np.nan)
    zs, ys = float(d['z_slice']), float(d['y_slice'])
    solid = d.get('solid')
    j_plane, k_plane = int(np.argmin(np.abs(y - ys))), int(np.argmin(np.abs(z - zs)))

    def draw(axs, n):
        for ax in axs:
            ax.clear()
        m = axs[0].pcolormesh(x, y, planes['xy'][n], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        axs[0].contour(x, y, np.nan_to_num(frac[n], nan=1.0), levels=[0.5], colors='w', linewidths=0.8)
        if solid is not None:
            ix = int(np.argmin(np.abs(x - d['yz_x'][n])))
            solids(axs[0], x, y, solid[k_plane]); solids(axs[1], x, z, solid[:, j_plane, :])
            solids(axs[2], y, z, solid[:, :, ix])
        if args.arrows:
            arrows(axs[0], x, y, d['xy_u'][n], d['xy_v'][n], xlim, ylim, arrow_scale)
            arrows(axs[1], x, z, d['xz_u'][n], d['xz_w'][n], xlim, zlim, arrow_scale)
            arrows(axs[2], y, z, d['yz_v'][n], d['yz_w'][n], ylim, zlim, arrow_scale)
        axs[0].axhline(ys, color='w', lw=0.5, ls=':')
        axs[0].axvline(d['yz_x'][n], color='w', lw=0.5, ls=':')
        axs[0].set(xlim=xlim, ylim=ylim, xlabel='x [m]', ylabel='y [m]', title=f'x-y plane, z = {zs:.2f} m')
        axs[1].pcolormesh(x, z, planes['xz'][n], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        axs[1].set(xlim=xlim, ylim=zlim, xlabel='x [m]', ylabel='z [m]', title=f'x-z plane, y = {ys:.1f} m')
        axs[2].pcolormesh(y, z, planes['yz'][n], cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        axs[2].set(xlim=ylim, ylim=zlim, xlabel='y [m]', ylabel='z [m]',
                   title=f'y-z plane, x = {d["yz_x"][n]:.1f} m (head fire)')
        for ax in axs:
            ax.title.set_fontsize(9)
        axs[0].figure.suptitle(f't = {t[n]:.1f} s', x=0.01, ha='left', fontsize=10)
        return m

    def new_figure():
        return plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True, gridspec_kw={'width_ratios': [1.3, 1.3, 1]})

    for n in idx:
        fig, axs = new_figure()
        m = draw(axs, n)
        fig.colorbar(m, ax=axs, label=label, shrink=0.9)
        path = os.path.join(out_dir, f'{name}_planes_t{t[n]:06.1f}.png')
        fig.savefig(path, dpi=150); plt.close(fig)
        print('wrote', path)
    if not args.no_anim:
        fig, axs = new_figure()
        m = draw(axs, 0)
        fig.colorbar(m, ax=axs, label=label, shrink=0.9)
        anim = animation.FuncAnimation(fig, lambda n: draw(axs, n), frames=len(t), interval=1000 / args.fps)
        save_anim(anim, out_dir, name, args, ffmpeg)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sim_dir')
    ap.add_argument('--field', nargs='*', help='fields to plot (default: all in the run): ' + ', '.join(ORDER))
    ap.add_argument('--times', type=float, nargs='*', help='snapshot times in s (default: 3 evenly spread)')
    ap.add_argument('--xlim', type=float, nargs=2, help='x range [m] (default: around the fire)')
    ap.add_argument('--ylim', type=float, nargs=2, help='y range [m], 3D (default: whole domain)')
    ap.add_argument('--zlim', type=float, nargs=2, help='height range [m] (default: whole domain)')
    ap.add_argument('--vmin', type=float, help='colour scale minimum (applies to every field plotted)')
    ap.add_argument('--vmax', type=float, help='colour scale maximum (applies to every field plotted)')
    ap.add_argument('--no-anim', action='store_true', help='snapshots only')
    ap.add_argument('--arrows', action='store_true', help='overlay velocity arrows (in-plane components)')
    ap.add_argument('--gif', action='store_true', help='GIF animations even if ffmpeg is available')
    ap.add_argument('--fps', type=int, default=8)
    args = ap.parse_args()

    d = dict(np.load(os.path.join(args.sim_dir, 'data.npz')))
    out_dir = os.path.join(args.sim_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)
    fields = available_fields(d)
    if args.field:
        missing = [f for f in args.field if f not in fields]
        if missing:
            raise SystemExit(f"not in this run: {', '.join(missing)}. Available: {', '.join(fields)}")
        fields = args.field
    ffmpeg = [] if (args.no_anim or args.gif) else [f for f in find_ffmpeg() if has_h264(f)]
    if not args.no_anim:
        print('animations:', 'GIF' if args.gif else ('MP4 (H.264) with ' + ffmpeg[0] if ffmpeg else
              'GIF: no ffmpeg with the H.264 encoder (libx264) found. For MP4: pip install imageio-ffmpeg'))
    for name in fields:
        render(d, name, args, out_dir, ffmpeg)


if __name__ == '__main__':
    main()
