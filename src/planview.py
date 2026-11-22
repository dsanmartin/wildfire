"""Plan-view (x-y) fire-spread model of the fuel-bed layer.

A reduced 2D model in the horizontal plane, of the reaction-diffusion-convection type (Asensio &
Ferragut 2002). It has no vertical direction, so the processes the coupled model resolves in the
vertical (plume, indraft, flame tilt) are parameterised: the wind is prescribed, part of the heat
of combustion leaves the bed, and the bed loses heat to the air above it.

State: T, the temperature of the fuel-bed layer (gas and fuel together), and Y, the fuel fraction.
With C = rho c_p + rho_b c_s Y the heat capacity of air plus fuel per unit volume,

    C dT/dt = -rho c_p U . grad T + div(k(T) grad T) + chi rho_b H_C (-dY/dt) - h_top (T - T_inf) / Y_h
    dY/dt   = -Y_f R(T) Y,          R(T) = A exp(-T_act / T) H(T - T_pc)

- U: mid-flame wind, speed u_r * wind_factor (default: the power law at the bed height,
  (Y_h / z_r)^alpha_u), direction wind_direction.
- k(T) = kappa + 4 sigma delta_bed T^3: conduction plus radiation inside the fuel bed, which is
  optically thick; delta_bed defaults to the photon mean free path of the bed, 4 / (alpha_s sigma_s).
- chi = heat_retained: fraction of the heat of combustion that stays in the bed.
- h_top: heat loss coefficient to the air above the bed, per unit ground area.
- R is integrated exactly over each step (reaction = exact), as in the vertical model.

Boundary conditions: periodic in x and y. The ignition and fuel-bed extent use the same keys as the
3D model ([temperature] shape, x_start..y_end; [fuel] x_start..y_end).
"""
import os
import time
from datetime import timedelta
import numpy as np
import torch
from operators import Operators


def build_planview(cfg):
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.Nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.Ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.Nt + 1)
    Ym, Xm = np.meshgrid(y[:-1], x[:-1], indexing='ij')                 # (Ny-1, Nx-1)
    x_c, y_c = (cfg.T0_x_start + cfg.T0_x_end) / 2, (cfg.T0_y_start + cfg.T0_y_end) / 2
    L, Wd = cfg.T0_x_end - cfg.T0_x_start, cfg.T0_y_end - cfg.T0_y_start
    in_x = (Xm >= cfg.T0_x_start) & (Xm <= cfg.T0_x_end)
    in_y = (Ym >= cfg.T0_y_start) & (Ym <= cfg.T0_y_end)
    if cfg.T0_shape == 'plate':
        shape = (in_x & in_y).astype(np.float64)
    elif cfg.T0_shape == 'gaussian':
        shape = np.exp(-((Xm - x_c) / L) ** 2 - ((Ym - y_c) / Wd) ** 2)
    elif cfg.T0_shape == 'line':
        shape = np.exp(-((Xm - x_c) / L) ** 2) * in_y
    else:
        raise ValueError(f"unknown plan-view ignition shape '{cfg.T0_shape}'")
    T = cfg.T_inf + shape * (cfg.T_hot - cfg.T_inf)
    Y = np.ones_like(T)
    for coord, lo, hi in ((Xm, cfg.fuel_x_start, cfg.fuel_x_end), (Ym, cfg.fuel_y_start, cfg.fuel_y_end)):
        if lo is not None and hi is not None:
            Y *= (coord >= lo) & (coord <= hi)
    return {'x': x, 'y': y, 't': t, 'dt': t[1] - t[0], 'spacings': (x[1] - x[0], y[1] - y[0]),
            'state': np.stack([T, Y]), 'source_mask': shape > 0.01}


class PlanViewModel:
    def __init__(self, cfg, grid, device, dtype):
        if cfg.topography_shape != 'flat':
            raise NotImplementedError("the plan-view model supports flat terrain only")
        self.cfg = cfg
        self.dt = float(grid['dt'])
        self.h = tuple(float(s) for s in grid['spacings'])           # (dx, dy)
        self.op = Operators(self.h, all_periodic=True)
        self.dV = self.h[0] * self.h[1] * cfg.Y_h                     # bed volume per node, m^3
        factor = cfg.wind_factor if cfg.wind_factor is not None else (cfg.Y_h / cfg.z_r) ** cfg.alpha_u
        speed = cfg.u_r * factor
        theta = np.deg2rad(cfg.wind_direction)
        self.U = (speed * np.cos(theta), speed * np.sin(theta))
        self.wind_speed = speed
        self.delta = cfg.delta_bed if cfg.delta_bed else 4.0 / (cfg.alpha_s * cfg.sigma_s)
        self.T0 = torch.as_tensor(grid['state'][0], dtype=dtype, device=device)
        self.source_mask = torch.as_tensor(grid['source_mask'], device=device)
        self.energy = {k: torch.zeros((), dtype=dtype, device=device) for k in ('combustion', 'clip_low', 'clip_high')}

    def rho(self, T):
        c = self.cfg
        return torch.full_like(T, c.rho_inf) if c.density_constant else c.rho_inf * c.T_inf / T

    def reaction(self, T):
        c = self.cfg
        R = c.A * torch.exp(-c.T_act / T) * (T > c.T_pc).to(T.dtype)
        if c.reaction == 'exact' and c.Y_f > 0:
            R = -torch.expm1(-c.Y_f * self.dt * R) / (c.Y_f * self.dt)
        return R

    def rates(self, y):
        """dT/dt, dY/dt and the heat release rate per unit bed volume (W/m^3)."""
        c, op = self.cfg, self.op
        T, Y = y[0], y[1].clamp(min=0)
        rho_cp = self.rho(T) * c.c_p
        C = rho_cp + c.rho_bulk * c.c_s * Y
        k = c.kappa + 4 * c.sigma * self.delta * T ** 3
        dkdT = 12 * c.sigma * self.delta * T ** 2
        gx, gy = op.grad(T)
        conduction = dkdT * (gx ** 2 + gy ** 2) + k * op.lap(T)
        scheme = op.limited if c.T_advection == 'koren' else op.upwind
        ux, uy = (torch.full_like(T, self.U[0]), torch.full_like(T, self.U[1]))
        adv = scheme(ux, T, 0) + scheme(uy, T, 1)
        dY = -c.Y_f * self.reaction(T) * Y
        hrr = c.rho_bulk * c.H_C * (-dY)
        dT = (-rho_cp * adv + conduction + c.heat_retained * hrr - c.h_top * (T - c.T_inf) / c.Y_h) / C
        return dT, dY, hrr

    def rhs(self, y):
        dT, dY, _ = self.rates(y)
        return torch.stack([dT, dY])

    def step(self, t_n, y):
        c, dt = self.cfg, self.dt
        if c.method == 'euler':
            y1 = y + dt * self.rhs(y)
        elif c.method == 'RK2':
            k1 = self.rhs(y)
            y1 = y + 0.5 * dt * (k1 + self.rhs(y + dt * k1))
        elif c.method == 'RK4':
            k1 = self.rhs(y); k2 = self.rhs(y + 0.5 * dt * k1)
            k3 = self.rhs(y + 0.5 * dt * k2); k4 = self.rhs(y + dt * k3)
            y1 = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown time method '{c.method}'")
        burned = (y[1] - y1[1]).clamp(min=0)
        self.energy['combustion'] += burned.sum() * (c.rho_bulk * c.H_C * self.dV)
        if c.bound:
            T_raw = y1[0].clone()
            y1[0].clamp_(c.T_min, c.T_max)
            y1[1].clamp_(c.Y_min, c.Y_max)
            dT = y1[0] - T_raw
            C = self.rho(y1[0]) * c.c_p + c.rho_bulk * c.c_s * y1[1]
            self.energy['clip_low'] += (C * dT.clamp(min=0)).sum() * self.dV
            self.energy['clip_high'] += (C * dT.clamp(max=0)).sum() * self.dV
        if t_n <= c.t_source:
            y1[0][self.source_mask] = self.T0[self.source_mask]
        return y1


def run_planview(cfg, device, dtype, verbose=True):
    from main import with_periodic_ends            # avoid a circular import at module load
    grid = build_planview(cfg)
    model = PlanViewModel(cfg, grid, device, dtype)
    Nt, NT, t = cfg.Nt, cfg.NT, grid['t']
    os.makedirs(cfg.save_path, exist_ok=True)
    log = open(os.path.join(cfg.save_path, 'log.txt'), 'w')

    def say(*lines):
        for line in lines:
            print(line, file=log, flush=True)
            if verbose:
                print(line)

    say(f"Simulation: {cfg.sim_name} (plan view x-y), device: {device}, dtype: {cfg.dtype}",
        f"Grid: {cfg.Ny - 1}x{cfg.Nx - 1} (y, x), spacings (dx, dy): {model.h}, dt: {model.dt:.3e}, steps: {Nt}",
        f"Mid-flame wind: {model.wind_speed:.2f} m/s towards {cfg.wind_direction:.0f} deg, "
        f"radiative path length in the bed: {model.delta:.3f} m, heat retained: {cfg.heat_retained}, "
        f"h_top: {cfg.h_top} W/(m^2 K)")
    y = torch.as_tensor(grid['state'], dtype=dtype, device=device)
    had_fuel = y[1] > 0.5
    frames, times, hrr_frames, energy = [], [], [], {k: [] for k in model.energy}

    def record(y, tn):
        _, _, hrr = model.rates(y)
        frames.append(y.cpu().numpy()); hrr_frames.append(hrr.cpu().numpy()); times.append(tn)
        for k, v in model.energy.items():
            energy[k].append(float(v))

    record(y, t[0])
    start = last = time.perf_counter()
    with torch.no_grad():
        for n in range(Nt):
            y = model.step(t[n], y)
            if (n + 1) % NT == 0 or n == Nt - 1:
                T, Y = y[0], y[1]
                now = time.perf_counter()
                E = {k: float(v) / 1e3 for k, v in model.energy.items()}
                say(f"Time step: {n + 1:6d}, Simulation time: {t[n + 1]:.4f} s",
                    f"Bed temperature: Min = {float(T.min()):.2f} K, Max {float(T.max()):.2f} K",
                    f"Fuel: Min = {float(Y.min()):.2f}, Max {float(Y.max()):.2f}, burnt area "
                    f"{float(((Y < 0.5) & had_fuel).sum()) * model.h[0] * model.h[1]:.0f} m^2",
                    f"Energy since start [kJ]: combustion {E['combustion']:.4g}, clip low {E['clip_low']:.4g}, "
                    f"clip high {E['clip_high']:.4g}",
                    f"mean step time: {(now - last) / ((n % NT) + 1):.6f} s")
                last = now
                if not np.isfinite(float(T.max())):
                    say("Non-finite temperature: stopping")
                    break
                record(y, t[n + 1])
    elapsed = time.perf_counter() - start
    say(f"\nSolver time: {timedelta(seconds=round(elapsed))} ({elapsed / (n + 1) * 1e3:.2f} ms per step)\n")
    log.close()

    z = np.stack(frames)
    data = {'T': np.stack([with_periodic_ends(f, [0, 1]) for f in z[:, 0]]),
            'Y': np.stack([with_periodic_ends(f, [0, 1]) for f in z[:, 1]]),
            'HRR': np.stack([with_periodic_ends(f, [0, 1]) for f in hrr_frames]),
            'x': grid['x'], 'y': grid['y'], 't': np.array(times), 'plan_view': True,
            'y_slice': float(cfg.y_slice if cfg.y_slice is not None else 0.5 * (cfg.y_min + cfg.y_max))}
    for k, v in energy.items():
        data['E_' + k] = np.array(v) * 1e3
    np.savez(os.path.join(cfg.save_path, 'data.npz'), **data)
    return data, elapsed
