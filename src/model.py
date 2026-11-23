"""Wildfire model: right-hand side, boundary conditions and the projection time step.

One implementation for 2D and 3D. The state is a single tensor ``y`` of shape
(ncomp, Nz, [Ny-1,] Nx-1) holding the velocity components (u, [v,] w), the gas
temperature T and the fuel fraction Y, and with the solid fuel model also the fuel
temperature T_s and moisture W (kg water per kg dry fuel), in that order. The last velocity
component is always the vertical one. With ``solid_temperature = False`` and
``radiation = diffusion`` the physics is the published 2D model (src_legacy/).
"""
import numpy as np
import torch
from operators import Operators
from pressure import PoissonSolver
from radiation import Radiation
from projection import FaceProjection
from operators import face_values


# Specific heat of air at constant pressure, J/(kg K), ideal-gas tables (e.g. Cengel & Boles, Table A-17).
# Combustion products (CO2, H2O) have somewhat higher values; air is used for the whole gas.
CP_AIR_T = np.array([250.0, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000, 2500, 3000])
CP_AIR = np.array([1003.0, 1005, 1013, 1029, 1051, 1099, 1142, 1173, 1211, 1253, 1286, 1312])


# Molar specific heats, J/(mol K), of N2, O2, CO2 and H2O (JANAF tables, rounded), for cp_model = mixture
CP_SPECIES_T = np.array([300.0, 500, 1000, 1500, 2000, 2500, 3000])
CP_SPECIES = {   # (molar mass kg/mol, cp at CP_SPECIES_T)
    'N2': (0.028013, [29.12, 29.58, 32.70, 34.85, 36.00, 36.60, 37.00]),
    'O2': (0.031999, [29.39, 31.09, 34.88, 36.56, 37.78, 39.00, 40.00]),
    'CO2': (0.04401, [37.22, 44.63, 54.31, 58.38, 60.35, 61.50, 62.20]),
    'H2O': (0.018015, [33.60, 35.23, 41.27, 46.98, 51.18, 54.00, 55.70]),
}
# Products of burning cellulose, C6H10O5 + 6 O2 -> 6 CO2 + 5 H2O: mass fractions of CO2 and H2O
PRODUCT_CO2, PRODUCT_H2O = 264.06 / 354.15, 90.08 / 354.15
SUTHERLAND_MU, SUTHERLAND_KAPPA = 110.4, 194.0     # Sutherland constants (K) of air


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
        self.dt_t = torch.tensor(self.dt, dtype=dtype, device=device)   # dt on the device (CUDA graphs)
        self._p = None                                            # pressure of the last step (warm start)
        self._target = None                                       # divergence target of the last projection
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
            extra = [cfg.T_inf, 0.0, 0.0, cfg.Y_O2_amb] if cfg.solid_temperature else []
            self.dead_values = to([0.0] * nd + [cfg.T_inf, 0.0] + extra).unsqueeze(1)
            # z index of the terrain surface in every column (highest cut node), for the wall shear stress
            levels = np.arange(self.shape[0]).reshape((-1,) + (1,) * (nd - 1))
            surface = np.where(cut, levels, 0).max(axis=0)
            self.surface = idx(surface).unsqueeze(0)
            # Surface node above every dead node: dead nodes copy its temperature (see apply_bc)
            dead_flat = np.flatnonzero(dead)
            self.dead_surface = idx(surface.reshape(-1)[dead_flat % row] * row + dead_flat % row)
        self._init_solids(grid, to)
        # Fuel slab: rows 0..k_fuel-1 hold every node with fuel or moisture. Neither is transported and
        # both only decrease, so no fuel ever appears above; the solid fuel terms and the drag are
        # computed on this slab only (the fuel bed is a few percent of the domain height)
        nz = self.shape[0]
        self.k_fuel = nz
        if cfg.fuel_slab and not cfg.legacy:
            fuel0 = state0[nd + 1] > 0
            if cfg.solid_temperature:
                fuel0 = fuel0 | (state0[nd + 3] > 0)
            rows = np.flatnonzero(fuel0.reshape(nz, -1).any(axis=1))
            self.k_fuel = min(nz, int(rows.max()) + 2) if rows.size else 1    # one spare row
        # Face-based projection (default): discretely conservative face fluxes, also used for transport
        self.faces = None
        if cfg.projection == 'face':
            self.face_proj = FaceProjection(self.shape, self.h, device, dtype,
                                            grid['solid'] if self.has_solids else None)
        self.max_divergence = 0.0
        # Mass budget (face projection): control volume = fluid nodes of rows 0..Nz-2 (row 0: half
        # cell), whose only open boundary is the vertical face below the top row
        self.mass = None
        if cfg.projection == 'face':
            P = self.face_proj
            keep = ~(grid['solid'] | grid['dead'])
            self._mass_w = P.unknown * P.weight * to(keep.astype(np.float64))
            self.dA = float(np.prod(self.h[:-1]))                 # area of a vertical face
            self.mass = {k: torch.zeros((), dtype=dtype, device=device) for k in ('outflow', 'pyrolysis')}
        self._last_mdot = None
        # Log-law wall model at the ground (flat ground: row 0; terrain: the cut node of each column)
        if cfg.wall_model == 'log':
            from setup import psi_m
            H = int(np.prod(self.shape[1:]))
            if self.has_terrain:
                W, U = np.flatnonzero(grid['cut']), np.flatnonzero(grid['cut']) + H
                cols = W % H
            else:
                W, U, cols = np.arange(H), np.arange(H) + H, np.arange(H)
            z1 = grid['Zm'].reshape(-1)[U] - grid['ground'].reshape(-1)[cols]   # height of the first node
            z0 = cfg.wall_z0 if cfg.wall_z0 else cfg.u_z_0
            denom = np.log(np.maximum(z1 / z0, np.e))
            if cfg.L_obukhov:
                denom = denom - psi_m(z1 / cfg.L_obukhov) + psi_m(z0 / cfg.L_obukhov)
            lidx = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
            self.wall_W, self.wall_U, self.wall_cols = lidx(W), lidx(U), lidx(cols)
            self.wall_coef = to(cfg.von_karman / denom)               # u_tau = coef |u_1|
            self.wall_dz = self.h[-1]
            self._u_tau_wall = None
        # Wind forcing (periodic domains): target = horizontal mean of the initial velocity at each height
        if cfg.forcing == 'profile':
            hdims = tuple(range(1, nd))
            solid0 = torch.as_tensor(grid['solid'] | grid['dead'], device=device)
            self.force_mask = (~solid0).to(dtype)
            count = self.force_mask.sum(hdims, keepdim=True).clamp(min=1)
            vel0 = to(state0[:nd])
            self.force_target = [(v * self.force_mask).sum(hdims, keepdim=True) / count for v in vel0]
            self.force_count = count
        # Pressure solver
        lengths = [cfg.x_max - cfg.x_min] + ([cfg.y_max - cfg.y_min] if nd == 3 else [])
        legacy_length = cfg.x_max if (cfg.legacy and nd == 2) else None
        self.poisson = PoissonSolver(self.shape, self.h, lengths, device, dtype, cfg.poisson, legacy_length)
        self.pressure_iterations = 0
        # Gas-phase mixing time and the components with a zero-gradient condition at the walls
        self.tau_mix = cfg.tau_mix or float(np.sqrt(2 * self.Delta / abs(cfg.g)))
        ncomp = state0.shape[0]
        if cfg.legacy:
            self.neumann = list(range(nd, ncomp))
        else:   # T and the gas species; Y, T_s and W are not transported and get no condition
            self.neumann = [nd] + ([nd + 4, nd + 5] if cfg.solid_temperature else [])
        self.top_rows = list(range(nd)) + self.neumann
        # Radiation
        self.G = None                                             # incident radiation, W/m^2
        self.step_count = 0
        self._first_stage, self._last_omega = True, None
        if cfg.radiation == 'transport':
            self.rad = Radiation(self.shape, self.h, cfg.n_polar, cfg.n_azimuth, device, dtype)
            if self.has_terrain:
                self.dead_mask = torch.zeros(self.shape, dtype=torch.bool, device=device)
                self.dead_mask.view(-1)[self.dead] = True
        # Energy budget, accumulated on the device (J per metre of depth in 2D, J in 3D)
        self.dV = float(np.prod(self.h))
        self.energy = {key: torch.zeros((), dtype=dtype, device=device) for key in ENERGY_TERMS}

    def _init_solids(self, grid, to):
        """Buildings: direct-forcing immersed boundary on the grid nodes.

        Solid nodes have zero velocity. Solid nodes with at least one fluid face neighbour ('wall
        nodes') take, for T and the gas species, the average of their fluid neighbours: zero normal
        gradient, i.e. adiabatic, impermeable walls. Interior solid nodes hold ambient values.
        """
        solid = grid.get('solid')
        self.has_solids = solid is not None and bool(solid.any())
        if not self.has_solids:
            return
        cfg, nd = self.cfg, self.nd
        shape, N = solid.shape, solid.size
        fluid = ~solid
        index = np.arange(N).reshape(shape)
        neighbours, is_fluid = [], []
        for axis in range(nd):
            for s in (1, -1):
                nb = np.roll(index, -s, axis=axis)
                valid = np.ones(shape, dtype=bool)
                if axis == 0:                                     # vertical: no wrap-around
                    (valid[-1] if s == 1 else valid[0])[...] = False
                neighbours.append(nb.ravel())
                is_fluid.append((fluid.ravel()[nb.ravel()] & valid.ravel()))
        neighbours, is_fluid = np.stack(neighbours, 1), np.stack(is_fluid, 1)
        wall = solid.ravel() & is_fluid.any(1)
        weights = is_fluid[wall].astype(np.float64)
        idx = lambda a: torch.as_tensor(a, dtype=torch.long, device=self.device)
        self.solid_idx = idx(np.flatnonzero(solid))
        self.wall_idx = idx(np.flatnonzero(wall))
        self.wall_nb = idx(neighbours[wall])
        self.wall_w = to(weights / weights.sum(1, keepdims=True))
        values = [0.0] * nd + [cfg.T_inf, 0.0] + ([cfg.T_inf, 0.0, 0.0, cfg.Y_O2_amb] if cfg.solid_temperature else [])
        self.solid_values = to(values).unsqueeze(1)
        self.solid_mask = torch.as_tensor(solid, device=self.device)
        self.fluid_mask = to(fluid.astype(np.float64))
        near = np.zeros(N, dtype=bool)                            # fluid nodes touching a wall
        near[neighbours[wall].ravel()[is_fluid[wall].ravel()]] = True
        self.near_wall = torch.as_tensor(near.reshape(shape), device=self.device)
        # Wall distance for the turbulence damping: also the distance to the nearest building
        try:
            from scipy.ndimage import distance_transform_edt
            d = distance_transform_edt(fluid, sampling=self.h[::-1])
            self.wall_distance = torch.minimum(self.wall_distance, to(d))
        except ImportError:
            print("scipy not available: turbulence wall damping ignores building walls")
        self.wall_divergence = 0.0

    def apply_solids(self, y, homogeneous=False):
        """Wall conditions of the buildings (see _init_solids)."""
        flat = y.view(y.shape[0], -1)
        if homogeneous:                       # index_fill_: no host-to-device copy (CUDA graphs)
            flat.index_fill_(1, self.solid_idx, 0.0)
        else:
            flat[:, self.solid_idx] = self.solid_values
        for k in self.neumann:
            flat[k, self.wall_idx] = (flat[k][self.wall_nb] * self.wall_w).sum(1)

    # ---------------------------------------------------------------- physics
    @staticmethod
    def _interp(T, Tt, values):
        """Linear interpolation in a table (clamped at its ends). Tt, values: 1D tensors."""
        Tc = T.clamp(Tt[0], Tt[-1])
        i = torch.searchsorted(Tt, Tc.contiguous()).clamp(1, len(Tt) - 1)
        T0, T1, c0, c1 = Tt[i - 1], Tt[i], values[i - 1], values[i]
        return c0 + (c1 - c0) * (Tc - T0) / (T1 - T0)

    def cp(self, T, y=None):
        """Specific heat of the gas, J/(kg K).
        constant: c_p. variable: c_p(T) of air (table CP_AIR). mixture: sum_k Y_k c_p,k(T) over N2, O2,
        CO2 and H2O (JANAF), with the products inferred from the oxygen consumed: the deficit
        Y_O2,amb (1 - Y_F) - Y_O2 (dilution by the fuel gas accounted for) burned (1 + 1/s) kg of products
        per kg of oxygen, split into CO2 and H2O by the cellulose stoichiometry; the fuel gas and the
        rest are counted as N2. The energy equation rho c_p DT/Dt = q and the low-Mach constraint
        div(u) = q / (rho c_p T) keep their form."""
        c = self.cfg
        if c.cp_model == 'constant':
            return c.c_p
        if not hasattr(self, '_cp_tables'):
            mk = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=T.dtype, device=T.device)
            self._cp_tables = {'air': (mk(CP_AIR_T), mk(CP_AIR))}
            for k, (M, values) in CP_SPECIES.items():
                self._cp_tables[k] = (mk(CP_SPECIES_T), mk(np.asarray(values) / M))
        if c.cp_model == 'variable' or y is None:
            return self._interp(T, *self._cp_tables['air'])
        nd = self.nd
        Y_F, Y_O = y[nd + 4].clamp(0, 1), y[nd + 5].clamp(0, 1)
        Y_P = ((c.Y_O2_amb * (1 - Y_F) - Y_O).clamp(min=0) * (1 + 1 / c.s_O2)).clamp(max=1)
        Y_CO2, Y_H2O = PRODUCT_CO2 * Y_P, PRODUCT_H2O * Y_P
        Y_N2 = (1 - Y_O - Y_CO2 - Y_H2O).clamp(min=0)
        cp = lambda k: self._interp(T, *self._cp_tables[k])
        return Y_N2 * cp('N2') + Y_O * cp('O2') + Y_CO2 * cp('CO2') + Y_H2O * cp('H2O')

    def _sutherland(self, T, ref, S):
        """Sutherland's law scaled to the value ref at 288.15 K (mu and kappa are given for air at
        15 C, whatever the ambient temperature); returns the value and its derivative."""
        T0 = 288.15
        v = ref * (T / T0) ** 1.5 * (T0 + S) / (T + S)
        return v, v * (1.5 / T - 1 / (T + S))

    def viscosity(self, T):
        c = self.cfg
        return c.mu if c.transport == 'constant' else self._sutherland(T, c.mu, SUTHERLAND_MU)[0]

    def conductivity(self, T):
        """Molecular thermal conductivity and its temperature derivative."""
        c = self.cfg
        if c.transport == 'constant':
            return c.kappa, 0.0
        return self._sutherland(T, c.kappa, SUTHERLAND_KAPPA)

    def rho(self, T):
        if self.cfg.density_constant:
            return torch.full_like(T, self.cfg.rho_inf)
        return self.cfg.rho_inf * self.cfg.T_inf / T

    def heat(self, y, rho, grad_T, lap_T, solid=None):
        """Heating rate of the gas in K/s: conduction, radiation, and the combustion and exchange terms.

        Radiation: ``diffusion`` adds 4 sigma delta T^3 to the conductivity (the published model);
        ``transport`` adds kappa_g (G - 4 sigma T^4), the absorbed minus the emitted radiation.
        Fuel: with the solid model the gas gains the heat of combustion of the pyrolysed fuel and
        loses what convection carries into the fuel; otherwise the single-temperature source and
        sink of the published model (``q``) are used."""
        c, nd = self.cfg, self.nd
        T, Y = y[nd], y[nd + 1]
        rho_cp = rho * self.cp(T, y)
        kap, dkap = self.conductivity(T)                         # molecular conduction: div(kap grad T)
        grad2 = grad_T[0] ** 2
        for g in grad_T[1:]:
            grad2 = grad2 + g ** 2
        if c.radiation == 'diffusion':
            k = (kap + 4 * c.sigma * c.delta * T ** 3) / rho_cp
            dkdT = (dkap + 12 * c.sigma * c.delta * T ** 2) / rho_cp
            out = dkdT * grad2 + k * lap_T
        else:
            kappa_g = c.kappa_gas
            if not c.solid_temperature:   # one temperature: the fuel absorbs and emits at T as well
                kappa_g = kappa_g + c.alpha_s * c.sigma_s * Y.clamp(min=0) / 4
            out = (kap * lap_T + dkap * grad2) / rho_cp + kappa_g * (self.G - 4 * c.sigma * T ** 4) / rho_cp
        if c.solid_temperature:
            s = solid if solid is not None else self.solid_rates(y, rho)
            net = s['Q_comb'].clone()
            net[:s['k']] -= s['Q_conv']                           # fuel slab
            return out + net / rho_cp
        return out + self.q(T, Y)

    def wall_stress(self, y):
        """Log-law wall model. u_tau = kappa |u_1| / [ln(z_1/z_0) - psi_m(z_1/L) + psi_m(z_0/L)] from the
        horizontal velocity u_1 of the first node above the ground (height z_1). Returns, for each
        horizontal component, the momentum sink -u_tau^2 u_i / |u_1| / dz to add at those nodes, and
        stores u_tau per column for the Van Driest damping."""
        nd = self.nd
        flat = y.view(y.shape[0], -1)
        uh = torch.sqrt(sum(flat[i, self.wall_U] ** 2 for i in range(nd - 1)))
        u_tau = self.wall_coef * uh
        H = int(np.prod(self.shape[1:]))
        cols = torch.zeros(H, dtype=y.dtype, device=y.device)
        cols[self.wall_cols] = u_tau
        if self._u_tau_wall is None:          # persistent buffer: read by the turbulence model of the next
            self._u_tau_wall = cols.view(self.shape[1:]).clone()   # evaluation (CUDA graphs keep addresses)
        else:
            self._u_tau_wall.copy_(cols.view(self.shape[1:]))
        scale = -(u_tau ** 2) / uh.clamp(min=1e-12) / self.wall_dz
        return [scale * flat[i, self.wall_U] for i in range(nd - 1)]

    def wind_forcing(self, vel):
        """Body force that relaxes the horizontal-mean velocity at each height toward the initial
        profile, (target - mean) / forcing_time, uniform in the horizontal (so it does not distort the
        local flow around the fire) and applied in the fluid only. In a periodic domain nothing else
        drives the wind, and ground friction would slowly decelerate it."""
        c, nd = self.cfg, self.nd
        hdims = tuple(range(1, nd))
        out = []
        for i in range(nd):
            mean = (vel[i] * self.force_mask).sum(hdims, keepdim=True) / self.force_count
            out.append((self.force_target[i] - mean) / c.forcing_time * self.force_mask)
        return out

    def flux_advection(self, phi, F, divF):
        """u . grad(phi) in conservative form with the face fluxes F: div(F phi_face) - phi div(F),
        phi_face Koren-limited and upwinded with F. Exact for a uniform phi and conservative."""
        nd = self.nd
        fluxes = [F[j] * face_values(F[j], phi, nd - 1 - j, nd - 1 - j == 0) for j in range(nd)]
        return self.face_proj.divergence(fluxes) - phi * divF

    def dilatation(self, heat, rho, T, solid, y=None):
        """div(u) imposed by the low Mach number approximation, plus the volume of gas produced by
        pyrolysis, mdot / rho. heat is the gas heating rate in K/s (molecular and subgrid conduction,
        radiation, combustion, exchange with the fuel). With rho T constant, continuity gives the
        thermal expansion (1/T) DT/Dt = heat / T (consistent). The published model divides this
        again by rho c_p (published, kept for --legacy), which makes the expansion 300-1000 times
        too small, so the mass of the heated gas is not conserved."""
        if self.cfg.dilatation == 'published':
            out = heat / (self.cp(T, y) * rho * T)
        else:
            out = heat / T
        if solid is not None:                                     # mdot: fuel slab only
            k = solid['k']
            out[:k] += solid['mdot'] / rho[:k]
        return out

    def target_divergence(self, y1, rho1, solid1):
        """div(u^{n+1}) from the provisional state. The subgrid heat flux uses the eddy viscosity of the
        last right-hand side evaluation (the one of this step), so the strain rate is not recomputed."""
        c, op, nd = self.cfg, self.op, self.nd
        T1 = y1[nd]
        grad_T1, lap_T1 = op.grad(T1), op.lap(T1)
        heating = self.heat(y1, rho1, grad_T1, lap_T1, solid1)
        if c.dilatation == 'consistent' and c.turbulence and getattr(self, '_mu_sgs', None) is not None:
            acc = self._mu_sgs * lap_T1
            for j in range(nd):
                acc = acc + self._grad_mu[j] * grad_T1[j]
            heating = heating + acc / (c.Pr * rho1)
        return self.dilatation(heating, rho1, T1, solid1, y1)

    def mass_total(self, y):
        """Gas mass in the control volume of the mass budget, kg (kg/m in 2D)."""
        return (self.rho(y[self.nd]) * self._mass_w).sum() * self.dV

    def top_outflow(self, T, F_top):
        """Mass flux out through the vertical faces below the top row, kg/s (kg/(m s) in 2D), with
        the density upwinded by the face flux F_top (upward positive)."""
        nz = self.shape[0]
        rho_face = torch.where(F_top > 0, self.rho(T[nz - 2]), self.rho(T[nz - 1]))
        return (rho_face * F_top).sum() * self.dA

    def slab_to_full(self, a):
        """A field computed on the fuel slab, extended by zeros to the whole grid."""
        out = torch.zeros(self.shape, dtype=a.dtype, device=a.device)
        out[:a.shape[0]] = a
        return out

    def solid_rates(self, y, rho):
        """Solid fuel and gas-phase combustion. Rates per unit volume.

        Heat to the fuel elements (surface per volume a = alpha_s sigma_s Y, shrinking as the fuel is
        consumed): convection Q_conv = h a (T - T_s), with h the larger of a natural-convection
        estimate h_c |T - T_s|^(1/3) and the forced convection of a cylinder of diameter 4 / sigma_s
        (Hilpert: Nu = 0.683 Re^0.466 Pr^(1/3)); radiation Q_rad = a/4 (G - 4 sigma T_s^4).
        Wet fuel at T_boil spends the net heat on evaporating its moisture. Dry fuel pyrolyses:
          linear (Mell et al. 2007, FDS): between T_pyr and T_pyr_end a fraction
            (T_s - T_pyr) / (T_pyr_end - T_pyr) of the net heat pyrolyses fuel at H_pyr J/kg, all of it
            above T_pyr_end; the rest heats the fuel;
          arrhenius: dY/dt = -Y_f A exp(-T_act / T_s) Y above T_pc, absorbing H_pyr J/kg.
        The pyrolysis gas Y_F burns with the oxygen Y_O2 at the mixing-limited rate
          omega = rho min(Y_F, Y_O2 / s_O2) / tau_mix  [kg fuel/(m^3 s)],  releasing H_C per kg."""
        c, nd, k = self.cfg, self.nd, self.k_fuel
        # Gas-phase combustion: the whole domain (the fuel gas burns in the plume)
        Y_F, Y_O = y[nd + 4].clamp(min=0), y[nd + 5].clamp(min=0)
        omega = rho * torch.minimum(Y_F, Y_O / c.s_O2) / self.tau_mix
        # Solid fuel: the fuel slab only (rows 0..k-1); Q_conv, mdot, dY, dT_s and dW have k rows
        ys, rho = y[:, :k], rho[:k]
        T, Y, T_s, W = ys[nd], ys[nd + 1].clamp(min=0), ys[nd + 2], ys[nd + 3].clamp(min=0)
        speed2 = ys[0] ** 2
        for u in ys[1:nd]:
            speed2 = speed2 + u ** 2
        a = c.alpha_s * c.sigma_s * Y
        diameter = 4 / c.sigma_s
        dT = T - T_s
        if c.transport == 'sutherland':   # gas properties at the film temperature
            T_film = 0.5 * (T + T_s)
            Re = torch.sqrt(speed2) * diameter * self.rho(T_film) / self.viscosity(T_film)
            k_film = self.conductivity(T_film)[0]
        else:
            Re = torch.sqrt(speed2) * diameter * rho / c.mu
            k_film = c.kappa
        h = torch.maximum(k_film / diameter * 0.683 * Re ** 0.466 * c.Pr ** (1 / 3),
                          c.h_c * torch.abs(dT) ** (1 / 3))
        has = (Y > c.fuel_threshold) | (W > 0)
        Q_conv = h * a * dT * has
        Q_rad = (a / 4 * (self.G[:k] - 4 * c.sigma * T_s ** 4) * has) if c.radiation == 'transport' else 0.0
        Q_in = Q_conv + Q_rad
        dry = (W <= 0) & (Y > c.fuel_threshold)
        if c.pyrolysis == 'linear':
            share = ((T_s - c.T_pyr) / (c.T_pyr_end - c.T_pyr)).clamp(0, 1) * dry
            mdot = share * Q_in.clamp(min=0) / c.H_pyr
        else:
            mdot = c.rho_bulk * c.Y_f * self.reaction(T_s) * Y * dry
        mdot = torch.minimum(mdot, c.rho_bulk * Y / self.dt_t)      # cannot pyrolyse more than is left
        Q_net = Q_in - mdot * c.H_pyr
        evap = (W > 0) & (T_s >= c.T_boil)
        dW = torch.where(evap, -Q_net.clamp(min=0) / (c.rho_bulk * c.L_v), torch.zeros_like(W))
        heat_capacity = (c.rho_bulk * (Y * c.c_s + W * c.c_w)).clamp(min=1e-30)
        dT_s = torch.where(evap, Q_net.clamp(max=0), Q_net) / heat_capacity * has
        return {'k': k,'Q_conv': Q_conv, 'Q_comb': omega * c.H_C, 'dY': -mdot / c.rho_bulk, 'dT_s': dT_s,
                'dW': dW, 'mdot': mdot, 'omega': omega}

    def update_radiation(self, y):
        """Solve the radiative transfer equation for the current state and store G."""
        c, nd = self.cfg, self.nd
        T, Y = y[nd], y[nd + 1].clamp(min=0)
        k_s = c.alpha_s * c.sigma_s * Y / 4                       # absorption by the fuel elements
        T_s = y[nd + 2] if c.solid_temperature else T
        kappa = (c.kappa_gas + k_s).clamp(min=1e-12)
        Ib = c.sigma * (c.kappa_gas * T ** 4 + k_s * T_s ** 4) / (np.pi * kappa)
        if self.has_terrain:                                      # terrain: opaque, at its own temperature
            kappa = torch.where(self.dead_mask, torch.full_like(kappa, 1e3), kappa)
            Ib = torch.where(self.dead_mask, c.sigma * T ** 4 / np.pi, Ib)
        if self.has_solids:                                       # buildings: opaque, at their wall temperature
            kappa = torch.where(self.solid_mask, torch.full_like(kappa, 1e3), kappa)
            Ib = torch.where(self.solid_mask, c.sigma * T ** 4 / np.pi, Ib)
        I_ground = c.sigma * T[0] ** 4 / np.pi
        G = self.rad.solve(kappa, Ib, I_ground, c.sigma * c.T_inf ** 4 / np.pi)
        if self.G is None:
            self.G = G
        else:                                                     # in place: a persistent buffer
            self.G.copy_(G)

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
            R = -torch.expm1(-c.Y_f * self.dt_t * R) / (c.Y_f * self.dt_t)
        return R

    def q(self, T, Y):
        """Combustion heat source plus convective heat loss, in K/s."""
        c = self.cfg
        source = c.H_C * Y * self.reaction(T) / self.cp(T)
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
        rho_cp_dz = self.rho(T) * self.cp(T) * self.h[-1]
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
        mu = self.viscosity(T)
        mu_wall = at_wall(mu) if torch.is_tensor(mu) else mu
        tau_w = mu_wall * torch.sqrt(tau2) + 1e-16
        u_tau = torch.sqrt(tau_w / rho)
        if c.wall_model == 'log' and self._u_tau_wall is not None:   # modeled friction velocity
            u_tau = self._u_tau_wall
        z_plus = self.wall_distance * rho * u_tau / mu
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
        if getattr(self, '_mu_sgs', None) is None:                    # reused for the species and the time step
            self._mu_sgs, self._grad_mu = mu_sgs, list(grad_mu)
        else:                                                         # in place: persistent buffers
            self._mu_sgs.copy_(mu_sgs)
            for g_old, g_new in zip(self._grad_mu, grad_mu):
                g_old.copy_(g_new)
        return out, -acc / (c.Pr * rho)

    def rhs(self, y):
        """Time derivative of the state before the pressure correction."""
        c, op, nd = self.cfg, self.op, self.nd
        vel, T, Y = y[:nd], y[nd], y[nd + 1]
        rho = self.rho(T)
        nu = self.viscosity(T) / rho                                  # d(mu)/dx terms neglected
        grad_T, lap_T = op.grad(T), op.lap(T)
        solid = self.solid_rates(y, rho) if c.solid_temperature else None
        if solid is not None and self._first_stage:
            self._last_omega, self._first_stage = solid['omega'], False
            self._last_mdot = solid['mdot']
        heat = self.heat(y, rho, grad_T, lap_T, solid)
        speed2 = vel[0] ** 2
        for u in vel[1:]:
            speed2 = speed2 + u ** 2
        k = self.k_fuel                                               # drag: fuel slab only
        drag = 0.5 * c.C_d * c.alpha_s * c.sigma_s * Y[:k] * torch.sqrt(speed2[:k])
        buoyancy = (rho - c.rho_inf) / rho
        sgs_u, sgs_T = self.sgs(vel, T, rho, grad_T, lap_T) if c.turbulence else ([0.0] * nd, 0.0)
        # Viscous term of the dilatation: nu / 3 * grad(div u), with div u from the energy equation
        # (consistent: the whole heating rate of T, including the subgrid heat flux)
        heating = heat - sgs_T if c.dilatation == 'consistent' else heat
        grad_div = None if c.density_constant else op.grad(self.dilatation(heating, rho, T, solid, y))
        out = torch.empty_like(y)
        face_mode = c.projection == 'face'
        if face_mode:
            F = self.faces if self.faces is not None else self.face_proj.to_faces(vel)
            divF = self.face_proj.divergence(F)
            self._F, self._divF = F, divF
        forcing = self.wind_forcing(vel) if c.forcing == 'profile' else None
        wall_drag = self.wall_stress(y) if c.wall_model == 'log' else None
        for i in range(nd):
            if face_mode:
                conv = self.flux_advection(vel[i], F, divF)
            else:
                conv = op.upwind(vel[0], vel[i], 0)
                for j in range(1, nd):
                    conv = conv + op.upwind(vel[j], vel[i], j)
            force = self.gravity[i] * buoyancy                       # a new tensor
            force[:k] -= drag * vel[i][:k]
            if grad_div is not None:
                force = force + nu * grad_div[i] / 3
            if forcing is not None:
                force = force + forcing[i]
            out[i] = nu * op.lap(vel[i]) - conv + force - sgs_u[i]
        if wall_drag is not None:   # wall stress as a momentum sink in the first cell above the ground
            for i in range(nd - 1):
                out[i].view(-1).index_add_(0, self.wall_U, wall_drag[i])
        if face_mode:
            adv = self.flux_advection(T, F, divF)
        elif c.T_advection == 'central':   # original: oscillates at the fire front (cell Peclet number >> 1)
            adv = vel[0] * grad_T[0]
            for j in range(1, nd):
                adv = adv + vel[j] * grad_T[j]
        else:          # Koren-limited third order (default) or second-order upwind (as for the momentum)
            scheme = op.limited if c.T_advection == 'koren' else op.upwind
            adv = scheme(vel[0], T, 0)
            for j in range(1, nd):
                adv = adv + scheme(vel[j], T, j)
        out[nd] = heat - adv - sgs_T
        if solid is not None:
            out[nd + 1:nd + 4, k:] = 0.0                              # above the fuel slab
            out[nd + 1, :k], out[nd + 2, :k], out[nd + 3, :k] = solid['dY'], solid['dT_s'], solid['dW']
            # Fuel gas and oxygen: transported like T (same scheme, unit Lewis number, turbulent
            # Schmidt number = Pr), produced by pyrolysis and consumed by combustion
            scheme = op.limited if c.T_advection == 'koren' else op.upwind
            D = self.conductivity(T)[0] / (rho * self.cp(T, y))
            # The pyrolysis gas is added mass: it brings fuel gas and dilutes what is already there
            mdot, omega = solid['mdot'], solid['omega']               # mdot: fuel slab only
            num_F, num_O = -omega, c.s_O2 * omega
            num_F[:k] += mdot * (1 - y[nd + 4, :k])
            num_O[:k] += mdot * y[nd + 5, :k]
            sources = {nd + 4: num_F / rho, nd + 5: -num_O / rho}
            for i_sp, src in sources.items():
                Yk = y[i_sp]
                if face_mode:
                    adv = self.flux_advection(Yk, F, divF)
                else:
                    adv = scheme(vel[0], Yk, 0)
                    for j in range(1, nd):
                        adv = adv + scheme(vel[j], Yk, j)
                lap = op.lap(Yk)
                rate = D * lap - adv + src
                if c.turbulence:
                    grad_Yk = op.grad(Yk)
                    acc = self._mu_sgs * lap
                    for j in range(nd):
                        acc = acc + self._grad_mu[j] * grad_Yk[j]
                    rate = rate + acc / (c.Pr * rho)
                out[i_sp] = rate
        else:
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
        legacy = self.cfg.legacy
        # Zero gradient. Second-order extrapolation (4 phi_1 - phi_2) / 3 leaves the range of the interior
        # values where the profile is curved: at the fire front, with a hot layer just above cold ground
        # (T_1 = 348 K, T_2 = 584 K), it gave 270 K, below ambient, and the bound then added energy at
        # every step. Scalars therefore use the bounded first-order form phi_0 = phi_1 (legacy: original).
        extrap = lambda a, b: (4 * a - b) / 3
        y[:nd, 0] = 0.0 if homogeneous else self.vel_bottom
        for k in self.neumann:
            y[k, 0] = extrap(y[k, 1], y[k, 2]) if legacy else y[k, 1]
        for k in self.top_rows:
            y[k, -1] = extrap(y[k, -2], y[k, -3]) if (legacy or k < nd) else y[k, -2]
        if self.has_terrain:
            flat = y.view(y.shape[0], -1)
            for k in self.neumann:
                flat[k, self.cut] = extrap(flat[k, self.cut_up1], flat[k, self.cut_up2]) if legacy \
                    else flat[k, self.cut_up1]
            if homogeneous:                   # index_fill_: no host-to-device copy (CUDA graphs)
                flat[:nd].index_fill_(1, self.cut, 0.0)
                flat.index_fill_(1, self.dead, 0.0)
            else:
                flat[:nd, self.cut] = self.vel_bottom.reshape(nd, -1)[:, self.cut_h]
                flat[:, self.dead] = self.dead_values
            if not self.cfg.legacy:
                flat[nd, self.dead] = flat[nd, self.dead_surface]
        if self.cfg.wall_model == 'log':
            # Wall model: tangential velocity at the wall copied from the first node (the resolved stress
            # vanishes there; the wall stress is applied as a sink instead), no flow through the wall
            flat = y.view(y.shape[0], -1)
            for i in range(nd - 1):
                flat[i, self.wall_W] = flat[i, self.wall_U]
            flat[nd - 1].index_fill_(0, self.wall_W, 0.0)
        if self.has_solids:
            self.apply_solids(y, homogeneous)
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
        target = 0.0
        if c.density_constant:
            p = self.poisson.solve(rho_n / dt * div)
            self.pressure_iterations = 1
        else:
            # Low Mach number: div(u^{n+1}) is prescribed by the energy equation
            rho1 = self.rho(T1)
            solid1 = self.solid_rates(y1, rho1) if c.solid_temperature else None
            target = self.target_divergence(y1, rho1, solid1)
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
        if self.has_solids:
            p = self.wall_correction(y1, rho_n, target, p)
        return p

    def wall_correction(self, y1, rho_n, target, p):
        """Remove the flow through the buildings. Zeroing the velocity inside the solids after the
        projection breaks the divergence constraint in the fluid next to the walls; each iteration
        zeroes it again and solves one more Poisson problem for the remaining divergence error in the
        fluid (an iterative direct-forcing correction, as in FDS)."""
        c, op, nd, dt = self.cfg, self.op, self.nd, self.dt
        flat = y1.view(y1.shape[0], -1)
        for _ in range(c.wall_iterations):
            flat[:nd].index_fill_(1, self.solid_idx, 0.0)
            div = op.half(y1[0], 0)
            for j in range(1, nd):
                div = div + op.half(y1[j], j)
            err = (div - target) * self.fluid_mask
            dp = self.poisson.solve(rho_n / dt * err)
            grad = op.grad(dp)
            for j in range(nd):
                y1[j] -= dt / rho_n * grad[j]
            p = p + dp
        flat[:nd].index_fill_(1, self.solid_idx, 0.0)
        div = op.half(y1[0], 0)
        for j in range(1, nd):
            div = div + op.half(y1[j], j)
        self.wall_divergence = float(((div - target) * self.near_wall).abs().max())
        return p

    def set_dt(self, dt):
        """Set the time step (host float and device tensor)."""
        self.dt = float(dt)
        self.dt_t.fill_(self.dt)

    def project_begin(self, y1, T_n, p0):
        """Face projection, first half: divergence target and pressure equation (PCG setup)."""
        c, op, nd = self.cfg, self.op, self.nd
        T1 = y1[nd]
        rho_n = self.rho(T_n)
        if c.density_constant:
            target = torch.zeros_like(T1)
        else:
            rho1 = self.rho(T1)
            solid1 = self.solid_rates(y1, rho1) if c.solid_temperature else None
            target = self.target_divergence(y1, rho1, solid1)
        if self._target is None:
            self._target = target.clone()
        else:
            self._target.copy_(target)
        self._rho_n = rho_n
        self.face_proj.project_begin(y1[:nd], rho_n, target, self.dt_t, p0)

    def project_end(self, y1):
        """Face projection, second half: projected face fluxes, and node velocities corrected by the
        average of the corrections of their two faces (Rhie-Chow). Updates y1 in place."""
        nd, P = self.nd, self.face_proj
        F, p = P.project_end()
        for j in range(nd):
            dF = F[j] - P._F_star[j]
            y1[j] += 0.5 * (dF + P._shift(dF, nd - 1 - j, -1))
        if self.faces is None:
            self.faces = [f.clone() for f in F]
        else:                                                     # in place: persistent buffers
            for f_old, f_new in zip(self.faces, F):
                f_old.copy_(f_new)
        if self._p is None:
            self._p = p.clone()
        else:
            self._p.copy_(p)
        return self._p

    def update_diagnostics(self):
        """Divergence errors of the last face projection (host synchronization: call at outputs)."""
        if self.cfg.projection != 'face' or self.faces is None:
            return
        P = self.face_proj
        m = P.unknown > 0
        err = (P.divergence(self.faces) - self._target).abs()
        self.max_divergence = float(err[m].max())
        if self.has_solids:
            self.wall_divergence = float(err[self.near_wall & m].max())

    def predict(self, y):
        """Provisional state y1 of a step (time integration of everything but the pressure), and for
        the face projection the start of the pressure solve."""
        c, nd, dt = self.cfg, self.nd, self.dt_t
        self._first_stage = True
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
        if c.solid_temperature:   # heat of combustion released this step (exact for Euler)
            self.energy['combustion'] += self._last_omega.sum() * (c.H_C * dt * self.dV)
        if self.mass is not None:  # mass budget: fluxes of the start of the step (exact for Euler)
            self.mass['outflow'] += self.top_outflow(y[nd], self._F[nd - 1][-2]) * dt
            if c.solid_temperature:
                w = self._mass_w[:self.k_fuel]
                self.mass['pyrolysis'] += (self._last_mdot * w).sum() * (self.dV * dt)
        if c.projection == 'face':
            self.project_begin(y1, y[nd], self._p)
        return y1

    def finalize(self, y, y1):
        """End of a step after the projection: boundary conditions, solid fuel fixes, bounds and the
        energy budget. Updates y1 in place and returns it."""
        c, nd = self.cfg, self.nd
        self.apply_bc(y1)
        if c.solid_temperature:
            y1[nd + 3].clamp_(min=0)
            y1[nd + 4:nd + 6].clamp_(0, 1)
            # Where the fuel is gone the solid temperature has no meaning: it follows the gas
            k = self.k_fuel
            has = (y1[nd + 1, :k] > c.fuel_threshold) | (y1[nd + 3, :k] > 0)
            y1[nd + 2, :k] = torch.where(has, y1[nd + 2, :k], y1[nd, :k])
            y1[nd + 2, k:] = y1[nd, k:]                           # no fuel above the slab
            if c.bound:
                y1[nd + 2].clamp_(c.T_min, c.T_max)
        # Heat released per unit decrease of Y: rho_bulk H_C [J/m^3] with the solid model; in the
        # published model rho H_C / Y_f, with rho the gas density
        heat_per_fuel = c.H_C / c.Y_f if c.Y_f > 0 else 0.0
        fuel_density = (lambda T: torch.full_like(T, c.rho_bulk * c.H_C)) if c.solid_temperature \
            else (lambda T: self.rho(T) * heat_per_fuel)
        if c.bound:
            T_raw, Y_raw = y1[nd].clone(), y1[nd + 1].clone()
            y1[nd].clamp_(c.T_min, c.T_max)
            y1[nd + 1].clamp_(c.Y_min, c.Y_max)
            # Energy the bounds add or remove: rho c_p dT for the gas left in the node, and for
            # Y < 0 the heat that the reaction term released from fuel that did not exist
            rho1 = self.rho(y1[nd])
            dT = y1[nd] - T_raw
            rcp = rho1 * self.cp(y1[nd], y1)
            self.energy['clip_low'] += (rcp * dT.clamp(min=0)).sum() * self.dV
            self.energy['clip_high'] += (rcp * dT.clamp(max=0)).sum() * self.dV
            self.energy['fuel_overshoot'] += (fuel_density(y1[nd]) * (-Y_raw).clamp(min=0)).sum() * self.dV
        if not c.solid_temperature:
            burned = (y[nd + 1] - y1[nd + 1]).clamp(min=0)
            self.energy['combustion'] += (fuel_density(y1[nd]) * burned).sum() * self.dV
        return y1

    def radiation_due(self):
        c = self.cfg
        return c.radiation == 'transport' and (self.G is None or self.step_count % c.rad_every == 0)

    def step(self, t_n, y, p):
        """One time step, executed directly (see cudagraph.GraphRunner for the captured version)."""
        c, nd = self.cfg, self.nd
        if self.radiation_due():
            self.update_radiation(y)
        self.step_count += 1
        if c.projection == 'face':
            if self._p is None:
                self._p = p.clone()
            y1 = self.predict(y)
            P = self.face_proj
            for it in range(1, max(c.max_iter, 200) + 1):
                P.pcg_iterate(1)
                if float(P.pcg_residual()) < c.tol:
                    break
            self.pressure_iterations = it
            p = self.project_end(y1)
        else:
            y1 = self.predict(y)
            p = self.project(y1, y[nd], p)
        y1 = self.finalize(y, y1)
        if t_n <= c.t_source:
            y1[nd][self.source_mask] = self.T0[self.source_mask]
        return y1, p

    def output_fields(self, y):
        """Derived fields written with the state (all per unit volume unless stated):
        HRR      heat release rate of combustion, W/m^3 (FDS: HRRPUV)
        pyrolysis  pyrolysis rate, kg/(m^3 s)                         (solid fuel model)
        Q_conv   convective heating of the fuel by the gas, W/m^3      (solid fuel model)
        G        incident radiation, W/m^2                             (radiation transport)
        q_rad_gas, q_rad_fuel  net radiative heating of gas and fuel, W/m^3
        nu_t     subgrid (turbulent) viscosity, m^2/s                  (turbulence model)"""
        c, nd = self.cfg, self.nd
        T, Y = y[nd], y[nd + 1].clamp(min=0)
        rho = self.rho(T)
        out = {}
        if c.radiation == 'transport' and self.G is None:
            self.update_radiation(y)
        if c.solid_temperature:
            s = self.solid_rates(y, rho)
            out['HRR'] = s['Q_comb']
            out['pyrolysis'], out['Q_conv'] = self.slab_to_full(s['mdot']), self.slab_to_full(s['Q_conv'])
        else:
            out['HRR'] = rho * c.H_C * Y * self.reaction(T)
        if c.radiation == 'transport':
            kappa_g = c.kappa_gas if c.solid_temperature else c.kappa_gas + c.alpha_s * c.sigma_s * Y / 4
            out['G'] = self.G
            out['q_rad_gas'] = kappa_g * (self.G - 4 * c.sigma * T ** 4)
            if c.solid_temperature:
                has = (Y > c.fuel_threshold) | (y[nd + 3] > 0)
                out['q_rad_fuel'] = c.alpha_s * c.sigma_s * Y / 4 * (self.G - 4 * c.sigma * y[nd + 2] ** 4) * has
        if c.turbulence:
            self.sgs(y[:nd], T, rho, self.op.grad(T), self.op.lap(T))
            out['nu_t'] = self._mu_sgs / rho
        return out

    def stable_dt(self, y, dt_max):
        """Largest time step allowed by the advective CFL target (sum over directions of max|u_j| / h_j)
        and by the explicit diffusion limit dt <= 0.25 / (nu_eff sum 1/h_j^2), with nu_eff the largest
        molecular plus subgrid kinematic viscosity of the last right-hand side evaluation."""
        c, nd = self.cfg, self.nd
        s = sum(float(y[j].abs().max()) / self.h[j] for j in range(nd))
        dt = min(dt_max, c.cfl / s) if s > 0 else dt_max
        rho = self.rho(y[nd])
        nu = self.viscosity(y[nd]) / rho
        nu_max = float(nu.max()) if torch.is_tensor(nu) else nu
        if getattr(self, '_mu_sgs', None) is not None:
            nu_max += float((self._mu_sgs / rho).max())
        inv_h2 = sum(1 / h ** 2 for h in self.h)
        if nu_max > 0:
            dt = min(dt, 0.25 / (nu_max * inv_h2))
        return dt

    def diagnostics(self, y):
        nd = self.nd
        cfl = sum(float(y[j].abs().max()) / self.h[j] for j in range(nd)) * self.dt
        T, Y = y[nd], y[nd + 1]
        return cfl, float(T.min()), float(T.max()), float(Y.min()), float(Y.max())

    def flame_diagnostics(self, y):
        """Total heat release rate (W per metre of depth in 2D, W in 3D) and the gas temperature
        weighted by the local heat release rate: a flame temperature comparable with measurements,
        unlike the single-cell maximum."""
        c, nd = self.cfg, self.nd
        T, Y = y[nd], y[nd + 1].clamp(min=0)
        rho = self.rho(T)
        if c.solid_temperature:
            hrr = rho * torch.minimum(y[nd + 4].clamp(min=0), y[nd + 5].clamp(min=0) / c.s_O2) * c.H_C / self.tau_mix
        else:
            hrr = rho * c.H_C * Y * self.reaction(T)
        total = float(hrr.sum())
        T_flame = float((hrr * T).sum()) / total if total > 0 else float('nan')
        return total * self.dV, T_flame

    def solid_diagnostics(self, y):
        """Fuel temperature range over the nodes that still hold fuel, and the total heat release rate
        (W per metre of depth in 2D, W in 3D)."""
        c, nd = self.cfg, self.nd
        Y, T_s = y[nd + 1], y[nd + 2]
        fuel = Y > c.fuel_threshold
        rho = self.rho(y[nd])
        hrr = float((rho * torch.minimum(y[nd + 4].clamp(min=0), y[nd + 5].clamp(min=0) / c.s_O2)).sum()) \
            * c.H_C / self.tau_mix * self.dV
        if not bool(fuel.any()):
            return float('nan'), float('nan'), hrr
        return float(T_s[fuel].min()), float(T_s[fuel].max()), hrr
