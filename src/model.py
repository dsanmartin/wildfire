"""Wildfire model: right-hand side, boundary conditions and the projection time step.

One implementation for 2D and 3D. The state is a single tensor ``y`` of shape
(ndim + 2, Nz, [Ny-1,] Nx-1) holding the velocity components (u, [v,] w), the
temperature T and the fuel fraction Y, in that order. The last velocity component is
always the vertical one. The physics is the published 2D model (src_legacy/pde.py::Phi_2D,
src_legacy/pressure.py::solve_pressure_2D, src_legacy/turbulence.py::turbulence_2D).
"""
import numpy as np
import torch
from operators import Operators
from pressure import PoissonSolver


ENERGY_TERMS = {
    'combustion': 'heat of the fuel actually burned',
    'fuel_overshoot': 'extra heat released by burning more fuel than there was (Y < 0 reset to 0)',
    'clip_low': 'energy added by raising T to T_min',
    'clip_high': 'energy removed by lowering T to T_max',
}


class Model:
    def __init__(self, cfg, grid, device, dtype):
        self.cfg = cfg
        self.nd = nd = cfg.ndim
        self.device, self.dtype = device, dtype
        self.dt = float(grid['dt'])
        self.h = tuple(float(s) for s in grid['spacings'])
        self.op = Operators(self.h, legacy=cfg.legacy)
        to = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        self.Zm = to(grid['Zm'])
        self.shape = tuple(self.Zm.shape)
        # Distance to the wall for the turbulence damping (the original uses the height above z_min)
        self.wall_distance = self.Zm if cfg.legacy else to(np.clip(grid['Zm'] - grid['ground'], 0.0, None))
        self.gravity = [0.0] * (nd - 1) + [cfg.g]
        # Filter width: tied to the grid by default, so the subgrid mixing length C_s Delta shrinks
        # under refinement and the solution keeps changing. A fixed physical width makes the model
        # independent of the grid, so results can converge as the grid is refined.
        self.Delta = cfg.filter_width if cfg.filter_width else float(np.prod(self.h)) ** (1.0 / nd)
        self.Lz = cfg.z_max - cfg.z_min
        # Boundary data
        state0 = grid['state']
        self.vel_bottom = to(state0[:nd, 0])                      # Dirichlet values at z = z_min
        self.T0 = to(state0[nd])
        self.source_mask = torch.as_tensor(grid['source_mask'], device=device)
        cut, dead = grid['cut'], grid['dead']
        # Flat terrain: the cut nodes are exactly the bottom row, already handled by the z = z_min condition
        self.has_terrain = bool(dead.any() or cut[1:].any() or not cut[0].all())
        if self.has_terrain:
            if cut[-2:].any():
                raise ValueError("terrain reaches the top of the domain")
            idx = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
            row = int(np.prod(self.shape[1:]))                    # nodes per horizontal plane
            cut_flat = np.flatnonzero(cut)
            self.cut, self.cut_up1, self.cut_up2 = idx(cut_flat), idx(cut_flat + row), idx(cut_flat + 2 * row)
            self.cut_h = idx(cut_flat % row)                      # horizontal position of each cut node
            self.dead = idx(np.flatnonzero(dead))
            self.dead_values = to([0.0] * nd + [cfg.T_inf, 0.0]).unsqueeze(1)
            # z index of the terrain surface in every column (highest cut node), for the wall shear stress
            levels = np.arange(self.shape[0]).reshape((-1,) + (1,) * (nd - 1))
            surface = np.where(cut, levels, 0).max(axis=0)
            self.surface = idx(surface).unsqueeze(0)
            # Surface node above every dead node: dead nodes copy its temperature (see apply_bc)
            dead_flat = np.flatnonzero(dead)
            self.dead_surface = idx(surface.reshape(-1)[dead_flat % row] * row + dead_flat % row)
        # Pressure solver
        lengths = [cfg.x_max - cfg.x_min] + ([cfg.y_max - cfg.y_min] if nd == 3 else [])
        legacy_length = cfg.x_max if (cfg.legacy and nd == 2) else None
        self.poisson = PoissonSolver(self.shape, self.h, lengths, device, dtype, cfg.poisson, legacy_length)
        self.pressure_iterations = 0
        # Energy budget, accumulated on the device (J per metre of depth in 2D, J in 3D)
        self.dV = float(np.prod(self.h))
        self.energy = {key: torch.zeros((), dtype=dtype, device=device) for key in ENERGY_TERMS}

    # ---------------------------------------------------------------- physics
    def rho(self, T):
        if self.cfg.density_constant:
            return torch.full_like(T, self.cfg.rho_inf)
        return self.cfg.rho_inf * self.cfg.T_inf / T

    def heat(self, T, Y, rho, grad_T, lap_T):
        """Conduction + radiation (diffusion approximation) + combustion source + convective sink, in K/s."""
        c = self.cfg
        rho_cp = rho * c.c_p
        k = (c.kappa + 4 * c.sigma * c.delta * T ** 3) / rho_cp
        dkdT = 12 * c.sigma * c.delta * T ** 2 / rho_cp
        grad2 = grad_T[0] ** 2
        for g in grad_T[1:]:
            grad2 = grad2 + g ** 2
        return dkdT * grad2 + k * lap_T + self.q(T, Y)

    def reaction(self, T):
        """Rate R(T) of the fuel equation dY/dt = -Y_f R Y: Arrhenius K(T) times the phase-change step H(T).

        K grows to ~1e5 1/s in a flame, far above 1/dt, so the explicit rate overshoots and the bounds on
        T and Y end up doing the integration. With ``reaction = exact`` the rate is replaced by its
        average over one step for frozen T, R = (1 - exp(-Y_f K H dt)) / (Y_f dt). Euler then gives the
        exact decay Y exp(-Y_f K H dt), Y stays in [0, 1], and the heat released in a step is exactly
        H_C / (c_p Y_f) times the fuel burned in it. For dt -> 0 this is the original rate."""
        c = self.cfg
        R = c.A * torch.exp(-c.T_act / T) * (T > c.T_pc).to(T.dtype)
        if c.reaction == 'exact' and c.Y_f > 0:
            R = -torch.expm1(-c.Y_f * self.dt * R) / (c.Y_f * self.dt)
        return R

    def q(self, T, Y):
        """Combustion heat source plus convective heat loss, in K/s."""
        c = self.cfg
        source = c.H_C * Y * self.reaction(T) / c.c_p
        if c.sink == 'surface':
            return source + self.surface_exchange(T, Y)
        # Original term: zero inside the fuel, and in the node above it a loss towards 0 K
        fuel = Y > 0
        fuel_plus_one = (torch.roll(fuel, 1, 0) | fuel).to(T.dtype)   # fuel nodes and the node above them
        T_g = c.T_inf + (T - c.T_inf) * fuel_plus_one
        diff = T_g - T * fuel.to(T.dtype)
        nz1 = self.shape[0] - 1
        sink = (-c.h_c * self.Lz * (torch.abs(diff) * nz1 / self.Lz) ** (1 / 4) * c.alpha_s * c.sigma_s * diff
                / (self.rho(T_g) * c.c_p * nz1))
        return source + sink * fuel_plus_one

    def surface_exchange(self, T, Y):
        """Natural convection between the top of the fuel bed and the gas above it, in K/s.

        In every column the top fuel node (highest node with Y > fuel_threshold, so the surface moves
        down as the fuel burns) exchanges a heat flux per unit ground area
            Q = h (T_g - T_s),   h = h_c (|T_g - T_s| / Y_h)^(1/4)   [W/m^2]
        with the gas node directly above it (T_s, T_g: temperatures of the two nodes). The flux is
        spread over one cell height dz in each node: +Q / (rho c_p dz) in the fuel node and
        -Q / (rho c_p dz) in the gas node, so energy is conserved and the total exchange does not
        depend on the grid."""
        c = self.cfg
        nz = self.shape[0]
        levels = torch.arange(nz, device=T.device).view((-1,) + (1,) * (self.nd - 1))
        top = torch.where(Y > c.fuel_threshold, levels, -1).amax(0)          # (..., Nx), -1: no fuel
        has = (top >= 0) & (top < nz - 1)
        i_s = top.clamp(0, nz - 2).unsqueeze(0)
        i_g = i_s + 1
        T_s, T_g = T.gather(0, i_s)[0], T.gather(0, i_g)[0]
        dT = T_g - T_s
        Q = c.h_c * (torch.abs(dT) / c.Y_h) ** (1 / 4) * dT * has.to(T.dtype)
        rho_cp_dz = self.rho(T) * c.c_p * self.h[-1]
        out = torch.zeros_like(T)
        out.scatter_add_(0, i_s, (Q / rho_cp_dz.gather(0, i_s)[0]).unsqueeze(0))
        out.scatter_add_(0, i_g, (-Q / rho_cp_dz.gather(0, i_g)[0]).unsqueeze(0))
        return out

    def sgs(self, vel, T, rho, grad_T, lap_T):
        """Smagorinsky subgrid terms with Van Driest wall damping. Returns ([sgs_u...], sgs_T)."""
        c, op, nd = self.cfg, self.op, self.nd
        G = [op.grad(u) for u in vel]                                 # G[i][j] = d u_i / d x_j
        div = G[0][0]
        for i in range(1, nd):
            div = div + G[i][i]
        S = [[None] * nd for _ in range(nd)]
        mod2 = -2 / 3 * div ** 2
        for i in range(nd):
            S[i][i] = G[i][i] - div / 3
            mod2 = mod2 + 2 * G[i][i] ** 2
            for j in range(i + 1, nd):
                S[i][j] = S[j][i] = 0.5 * (G[i][j] + G[j][i])
                mod2 = mod2 + (G[i][j] + G[j][i]) ** 2
        mod_S = torch.sqrt(mod2.clamp_(min=0))
        # Wall shear stress from the bottom row
        w = nd - 1                                                    # vertical component / direction
        at_wall = (lambda a: a.gather(0, self.surface)[0]) if (self.has_terrain and not c.legacy) else (lambda a: a[0])
        tau2 = at_wall(G[0][w] + G[w][0]) ** 2
        if nd == 3:
            tau2 = tau2 + at_wall(G[1][w] + G[w][1]) ** 2
        tau_w = c.mu * torch.sqrt(tau2) + 1e-16
        u_tau = torch.sqrt(tau_w / rho)
        z_plus = self.wall_distance * rho * u_tau / c.mu
        fw = 1 - torch.exp(-z_plus / 26)
        mu_sgs = rho * (c.C_s * self.Delta * fw) ** 2 * mod_S
        grad_mu = op.grad(mu_sgs)
        out = []
        for i in range(nd):
            acc = grad_mu[0] * S[i][0] + mu_sgs * op.d1(S[i][0], 0)
            for j in range(1, nd):
                acc = acc + grad_mu[j] * S[i][j] + mu_sgs * op.d1(S[i][j], j)
            out.append(-2 * acc / rho)
        acc = mu_sgs * lap_T
        for j in range(nd):
            acc = acc + grad_mu[j] * grad_T[j]
        return out, -acc / (c.Pr * rho)

    def rhs(self, y):
        """Time derivative of the state before the pressure correction."""
        c, op, nd = self.cfg, self.op, self.nd
        vel, T, Y = y[:nd], y[nd], y[nd + 1]
        rho = self.rho(T)
        nu = c.mu / rho
        grad_T, lap_T = op.grad(T), op.lap(T)
        heat = self.heat(T, Y, rho, grad_T, lap_T)
        speed2 = vel[0] ** 2
        for u in vel[1:]:
            speed2 = speed2 + u ** 2
        drag = 0.5 * c.C_d * c.alpha_s * c.sigma_s * Y * torch.sqrt(speed2)
        buoyancy = (rho - c.rho_inf) / rho
        # Viscous term of the dilatation: nu / 3 * grad(div u), with div u from the energy equation
        grad_div = None if c.density_constant else op.grad(heat / (c.c_p * rho * T))
        sgs_u, sgs_T = self.sgs(vel, T, rho, grad_T, lap_T) if c.turbulence else ([0.0] * nd, 0.0)
        out = torch.empty_like(y)
        for i in range(nd):
            conv = op.upwind(vel[0], vel[i], 0)
            for j in range(1, nd):
                conv = conv + op.upwind(vel[j], vel[i], j)
            force = self.gravity[i] * buoyancy - drag * vel[i]
            if grad_div is not None:
                force = force + nu * grad_div[i] / 3
            out[i] = nu * op.lap(vel[i]) - conv + force - sgs_u[i]
        if c.T_advection == 'central':   # original: oscillates at the fire front (cell Peclet number >> 1)
            adv = vel[0] * grad_T[0]
            for j in range(1, nd):
                adv = adv + vel[j] * grad_T[j]
        else:          # Koren-limited third order (default) or second-order upwind (as for the momentum)
            scheme = op.limited if c.T_advection == 'koren' else op.upwind
            adv = scheme(vel[0], T, 0)
            for j in range(1, nd):
                adv = adv + scheme(vel[j], T, j)
        out[nd] = heat - adv - sgs_T
        out[nd + 1] = -c.Y_f * self.reaction(T) * Y
        self.apply_bc(out, homogeneous=True)
        return out

    # ----------------------------------------------------- boundary conditions
    def apply_bc(self, y, homogeneous=False):
        """Bottom: velocity Dirichlet, dT/dz = 0. Top: zero vertical gradient.
        Terrain: cut nodes behave like the bottom. Dead nodes have zero velocity and fuel, and the
        temperature of the surface node above them: the ground is adiabatic, as at z = z_min. (The
        original holds them at T_inf, an artificial density jump next to the fire that makes the
        pressure fixed point diverge.)
        The fuel Y is not transported (its equation is a local ODE), so it gets no boundary condition.
        The original extrapolated it like T, which overwrote the ground row with (4 Y_1 - Y_2) / 3
        every step: fuel vanished without releasing heat, or went negative, wherever the bed burned
        unevenly in z. ``legacy`` keeps that behaviour.
        ``homogeneous=True`` is the version for time derivatives (Dirichlet rows do not change)."""
        nd = self.nd
        last = y.shape[0] if self.cfg.legacy else nd + 1     # rows nd .. last-1 get the Neumann condition
        y[:nd, 0] = 0.0 if homogeneous else self.vel_bottom
        y[nd:last, 0] = (4 * y[nd:last, 1] - y[nd:last, 2]) / 3
        y[:last, -1] = (4 * y[:last, -2] - y[:last, -3]) / 3
        if self.has_terrain:
            flat = y.view(y.shape[0], -1)
            flat[nd:last, self.cut] = (4 * flat[nd:last, self.cut_up1] - flat[nd:last, self.cut_up2]) / 3
            flat[:nd, self.cut] = 0.0 if homogeneous else self.vel_bottom.reshape(nd, -1)[:, self.cut_h]
            flat[:, self.dead] = 0.0 if homogeneous else self.dead_values
            if not self.cfg.legacy:
                flat[nd, self.dead] = flat[nd, self.dead_surface]
        return y

    # -------------------------------------------------------------- time step
    def project(self, y1, T_n, p):
        """Solve for the pressure and correct the provisional velocity in ``y1`` in place."""
        c, op, nd, dt = self.cfg, self.op, self.nd, self.dt
        T1, Y1 = y1[nd], y1[nd + 1]
        rho_n = self.rho(T_n)
        div = op.half(y1[0], 0)
        for j in range(1, nd):
            div = div + op.half(y1[j], j)
        if c.density_constant:
            p = self.poisson.solve(rho_n / dt * div)
            self.pressure_iterations = 1
        else:
            # Low Mach number: div(u^{n+1}) is prescribed by the energy equation
            rho1 = self.rho(T1)
            target = self.heat(T1, Y1, rho1, op.grad(T1), op.lap(T1)) / (c.c_p * rho1 * T1)
            f0 = rho_n / dt * (div - target)
            grad_rho = [g / rho_n for g in op.grad_half(rho_n)]
            # Fixed point on the variable-coefficient term grad(rho) . grad(p) / rho
            for i in range(c.max_iter):
                f = f0
                for j in range(nd):
                    f = f + grad_rho[j] * op.d1(p, j)
                p_new = self.poisson.solve(f)
                err = (p_new - p).abs().max()
                p = p_new
                if c.log_solver:
                    print(f"Fixed-point iteration: {i}, L-inf: {float(err):.2e}")
                if err < c.tol:
                    break
            self.pressure_iterations = i + 1
        grad_p = op.grad(p)
        for j in range(nd):
            y1[j] -= dt / rho_n * grad_p[j]
        return p

    def step(self, t_n, y, p):
        c, nd, dt = self.cfg, self.nd, self.dt
        if c.method == 'euler':
            y1 = y + dt * self.rhs(y)
        elif c.method == 'RK2':
            k1 = self.rhs(y)
            k2 = self.rhs(y + dt * k1)
            y1 = y + 0.5 * dt * (k1 + k2)
        elif c.method == 'RK4':
            k1 = self.rhs(y)
            k2 = self.rhs(y + 0.5 * dt * k1)
            k3 = self.rhs(y + 0.5 * dt * k2)
            k4 = self.rhs(y + dt * k3)
            y1 = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"unknown time method '{c.method}'")
        p = self.project(y1, y[nd], p)
        self.apply_bc(y1)
        heat_per_fuel = c.H_C / c.Y_f if c.Y_f > 0 else 0.0     # J/kg released per unit decrease of Y
        if c.bound:
            T_raw, Y_raw = y1[nd].clone(), y1[nd + 1].clone()
            y1[nd].clamp_(c.T_min, c.T_max)
            y1[nd + 1].clamp_(c.Y_min, c.Y_max)
            # Energy the bounds add or remove: rho c_p dT for the gas left in the node, and for
            # Y < 0 the heat that the reaction term released from fuel that did not exist
            rho1 = self.rho(y1[nd])
            dT = y1[nd] - T_raw
            self.energy['clip_low'] += (rho1 * dT.clamp(min=0)).sum() * (c.c_p * self.dV)
            self.energy['clip_high'] += (rho1 * dT.clamp(max=0)).sum() * (c.c_p * self.dV)
            self.energy['fuel_overshoot'] += (rho1 * (-Y_raw).clamp(min=0)).sum() * (heat_per_fuel * self.dV)
        burned = (y[nd + 1] - y1[nd + 1]).clamp(min=0)
        self.energy['combustion'] += (self.rho(y1[nd]) * burned).sum() * (heat_per_fuel * self.dV)
        if t_n <= c.t_source:
            y1[nd][self.source_mask] = self.T0[self.source_mask]
        return y1, p

    def diagnostics(self, y):
        nd = self.nd
        cfl = sum(float(y[j].abs().max()) / self.h[j] for j in range(nd)) * self.dt
        T, Y = y[nd], y[nd + 1]
        return cfl, float(T.min()), float(T.max()), float(Y.min()), float(Y.max())
