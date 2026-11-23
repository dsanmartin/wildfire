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
        # Face-based projection (default): discretely conservative face fluxes, also used for transport
        self.faces = None
        if cfg.projection == 'face':
            self.face_proj = FaceProjection(self.shape, self.h, device, dtype,
                                            grid['solid'] if self.has_solids else None)
        self.max_divergence = 0.0
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
        flat[:, self.solid_idx] = 0.0 if homogeneous else self.solid_values
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
        """Sutherland's law scaled to the value ref at T_inf; returns the value and its derivative."""
        T0 = self.cfg.T_inf
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
            return out + (s['Q_comb'] - s['Q_conv']) / rho_cp
        return out + self.q(T, Y)

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
        """div(u) imposed by the low Mach number approximation: thermal expansion heat / (rho c_p T),
        plus the volume of gas produced by pyrolysis, mdot / rho."""
        out = heat / (self.cp(T, y) * rho * T)
        if solid is not None:
            out = out + solid['mdot'] / rho
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
        c, nd = self.cfg, self.nd
        T, Y, T_s, W = y[nd], y[nd + 1].clamp(min=0), y[nd + 2], y[nd + 3].clamp(min=0)
        Y_F, Y_O = y[nd + 4].clamp(min=0), y[nd + 5].clamp(min=0)
        speed2 = y[0] ** 2
        for u in y[1:nd]:
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
        Q_rad = (a / 4 * (self.G - 4 * c.sigma * T_s ** 4) * has) if c.radiation == 'transport' else 0.0
        Q_in = Q_conv + Q_rad
        dry = (W <= 0) & (Y > c.fuel_threshold)
        if c.pyrolysis == 'linear':
            share = ((T_s - c.T_pyr) / (c.T_pyr_end - c.T_pyr)).clamp(0, 1) * dry
            mdot = share * Q_in.clamp(min=0) / c.H_pyr
        else:
            mdot = c.rho_bulk * c.Y_f * self.reaction(T_s) * Y * dry
        mdot = torch.minimum(mdot, c.rho_bulk * Y / self.dt)        # cannot pyrolyse more than is left
        Q_net = Q_in - mdot * c.H_pyr
        evap = (W > 0) & (T_s >= c.T_boil)
        dW = torch.where(evap, -Q_net.clamp(min=0) / (c.rho_bulk * c.L_v), torch.zeros_like(W))
        heat_capacity = (c.rho_bulk * (Y * c.c_s + W * c.c_w)).clamp(min=1e-30)
        dT_s = torch.where(evap, Q_net.clamp(max=0), Q_net) / heat_capacity * has
        omega = rho * torch.minimum(Y_F, Y_O / c.s_O2) / self.tau_mix
        return {'Q_conv': Q_conv, 'Q_comb': omega * c.H_C, 'dY': -mdot / c.rho_bulk, 'dT_s': dT_s,
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
        self.G = self.rad.solve(kappa, Ib, I_ground, c.sigma * c.T_inf ** 4 / np.pi)

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
        self._mu_sgs, self._grad_mu = mu_sgs, grad_mu                # reused for the species
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
        heat = self.heat(y, rho, grad_T, lap_T, solid)
        speed2 = vel[0] ** 2
        for u in vel[1:]:
            speed2 = speed2 + u ** 2
        drag = 0.5 * c.C_d * c.alpha_s * c.sigma_s * Y * torch.sqrt(speed2)
        buoyancy = (rho - c.rho_inf) / rho
        # Viscous term of the dilatation: nu / 3 * grad(div u), with div u from the energy equation
        grad_div = None if c.density_constant else op.grad(self.dilatation(heat, rho, T, solid, y))
        sgs_u, sgs_T = self.sgs(vel, T, rho, grad_T, lap_T) if c.turbulence else ([0.0] * nd, 0.0)
        out = torch.empty_like(y)
        face_mode = c.projection == 'face'
        if face_mode:
            F = self.faces if self.faces is not None else self.face_proj.to_faces(vel)
            divF = self.face_proj.divergence(F)
            self._F, self._divF = F, divF
        forcing = self.wind_forcing(vel) if c.forcing == 'profile' else None
        for i in range(nd):
            if face_mode:
                conv = self.flux_advection(vel[i], F, divF)
            else:
                conv = op.upwind(vel[0], vel[i], 0)
                for j in range(1, nd):
                    conv = conv + op.upwind(vel[j], vel[i], j)
            force = self.gravity[i] * buoyancy - drag * vel[i]
            if grad_div is not None:
                force = force + nu * grad_div[i] / 3
            if forcing is not None:
                force = force + forcing[i]
            out[i] = nu * op.lap(vel[i]) - conv + force - sgs_u[i]
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
            out[nd + 1], out[nd + 2], out[nd + 3] = solid['dY'], solid['dT_s'], solid['dW']
            # Fuel gas and oxygen: transported like T (same scheme, unit Lewis number, turbulent
            # Schmidt number = Pr), produced by pyrolysis and consumed by combustion
            scheme = op.limited if c.T_advection == 'koren' else op.upwind
            D = self.conductivity(T)[0] / (rho * self.cp(T, y))
            # The pyrolysis gas is added mass: it brings fuel gas and dilutes what is already there
            mdot, omega = solid['mdot'], solid['omega']
            sources = {nd + 4: (mdot * (1 - y[nd + 4]) - omega) / rho,
                       nd + 5: -(c.s_O2 * omega + mdot * y[nd + 5]) / rho}
            for k, src in sources.items():
                Yk = y[k]
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
                out[k] = rate
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
            flat[:nd, self.cut] = 0.0 if homogeneous else self.vel_bottom.reshape(nd, -1)[:, self.cut_h]
            flat[:, self.dead] = 0.0 if homogeneous else self.dead_values
            if not self.cfg.legacy:
                flat[nd, self.dead] = flat[nd, self.dead_surface]
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
        if c.projection == 'face':
            if not c.density_constant:
                rho1 = self.rho(T1)
                solid1 = self.solid_rates(y1, rho1) if c.solid_temperature else None
                target = self.dilatation(self.heat(y1, rho1, op.grad(T1), op.lap(T1), solid1), rho1, T1, solid1, y1)
            P = self.face_proj
            F_star = P.to_faces(y1[:nd])
            F, p, its = P.project(y1[:nd], rho_n, target, dt, c.tol, max(c.max_iter, 200), p0=p)
            # Node velocities: the average of the corrections of the two adjacent faces (Rhie-Chow)
            for j in range(nd):
                dF = F[j] - F_star[j]
                y1[j] += 0.5 * (dF + P._shift(dF, nd - 1 - j, -1))
            self.faces, self.pressure_iterations = F, its
            m = P.unknown > 0
            self.max_divergence = float((P.divergence(F) - target)[m].abs().max())
            if self.has_solids:
                self.wall_divergence = float((P.divergence(F) - target)[self.near_wall & m].abs().max())
            return p
        if c.density_constant:
            p = self.poisson.solve(rho_n / dt * div)
            self.pressure_iterations = 1
        else:
            # Low Mach number: div(u^{n+1}) is prescribed by the energy equation
            rho1 = self.rho(T1)
            solid1 = self.solid_rates(y1, rho1) if c.solid_temperature else None
            target = self.dilatation(self.heat(y1, rho1, op.grad(T1), op.lap(T1), solid1), rho1, T1, solid1, y1)
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
            flat[:nd, self.solid_idx] = 0.0
            div = op.half(y1[0], 0)
            for j in range(1, nd):
                div = div + op.half(y1[j], j)
            err = (div - target) * self.fluid_mask
            dp = self.poisson.solve(rho_n / dt * err)
            grad = op.grad(dp)
            for j in range(nd):
                y1[j] -= dt / rho_n * grad[j]
            p = p + dp
        flat[:nd, self.solid_idx] = 0.0
        div = op.half(y1[0], 0)
        for j in range(1, nd):
            div = div + op.half(y1[j], j)
        self.wall_divergence = float(((div - target) * self.near_wall).abs().max())
        return p

    def step(self, t_n, y, p):
        c, nd, dt = self.cfg, self.nd, self.dt
        if c.radiation == 'transport' and (self.G is None or self.step_count % c.rad_every == 0):
            self.update_radiation(y)
        self.step_count += 1
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
        p = self.project(y1, y[nd], p)
        self.apply_bc(y1)
        if c.solid_temperature:
            y1[nd + 3].clamp_(min=0)
            y1[nd + 4:nd + 6].clamp_(0, 1)
            # Where the fuel is gone the solid temperature has no meaning: it follows the gas
            has = (y1[nd + 1] > c.fuel_threshold) | (y1[nd + 3] > 0)
            y1[nd + 2] = torch.where(has, y1[nd + 2], y1[nd])
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
            out['HRR'], out['pyrolysis'], out['Q_conv'] = s['Q_comb'], s['mdot'], s['Q_conv']
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
