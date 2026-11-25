"""Burning structures: the buildings of [buildings] as ignitable surfaces, as in FDS with IGNITION_TEMPERATURE and
HRRPUA on the surfaces of an obstruction (structures model = ignition).

Every face between a solid node of a building and a fluid neighbour is a surface element: a roof (fluid above)
or a wall (fluid beside or below). Each element has
  - a 1D wall of layers (thickness, density, conductivity, specific heat), heated at the exposed side and
    insulated at the back, solved implicitly (backward Euler, cells graded towards the surface);
  - the surface heat balance q_net = h (T_g - T_s) + eps (q_inc - sigma T_s^4), with T_g the gas temperature of
    the fluid node, the FDS convection coefficient h = max(C |T_g - T_s|^(1/3), k/L 0.037 Re^0.8 Pr^(1/3))
    (C = 1.52 for roofs, 1.31 for walls, L = 1 m) and the radiation incident from the gas side
    q_inc = G / 2 - sigma T_s^4 (G at the fluid node; the other half of the directions come from the wall,
    which emits at T_s: in equilibrium q_inc = sigma T^4);
  - ignition when T_s reaches the ignition temperature; from then on the element releases fuel gas at
    hrrpua * ramp(t - t_ign) / H_C per unit area into its fluid node, where it burns in the gas phase like the
    pyrolysis gas, until the building's combustible mass (bulk density times volume) is used up.
The gas loses the convected heat h (T_g - T_s) A / V in the fluid node. The element areas of each building are
scaled so that its roof and walls have their exact areas (the boxes are snapped to the grid nodes).
"""
import numpy as np
import torch

SIGMA = 5.670374419e-8


def parse_layers(text):
    """'thickness density conductivity specific_heat, ...' (m, kg/m^3, W/(m K), J/(kg K)) -> list of tuples."""
    layers = []
    for part in text.split(','):
        values = [float(v) for v in part.split()]
        if len(values) != 4 or values[0] <= 0 or min(values[1:]) <= 0:
            raise ValueError(f"[structures] layers: expected 'thickness density conductivity specific_heat' "
                             f"per layer, separated by commas, got '{text}'")
        layers.append(tuple(values))
    return layers


def parse_ramp(text):
    points = sorted(tuple(float(v) for v in pair.split()) for pair in text.split(','))
    if any(len(pt) != 2 for pt in points) or len(points) < 2:
        raise ValueError(f"[structures] ramp: expected 'time fraction' pairs separated by commas, got '{text}'")
    return points


def wall_cells(layers, first=2e-5, ratio=1.3):
    """Cells through the wall, from the exposed surface: graded geometrically from `first` in the first layer,
    then uniform cells no smaller than the last one in the others. Returns (dx, k, rho_c) arrays."""
    dx, k, rc = [], [], []
    for i, (L, rho, cond, cp) in enumerate(layers):
        if i == 0:
            cells, size = [], min(first, L / 4)
            while sum(cells) + size < L:
                cells.append(size)
                size *= ratio
            rest = L - sum(cells)
            if rest > 0.3 * cells[-1]:
                cells.append(rest)
            else:                                                 # merge a sliver into the last cell
                cells[-1] += rest
        else:
            n = int(np.clip(np.ceil(L / max(dx[-1], L / 20)), 3, 20))
            cells = [L / n] * n
        dx += cells
        k += [cond] * len(cells)
        rc += [rho * cp] * len(cells)
    return np.array(dx), np.array(k), np.array(rc)


class Structures:
    """Surface elements of the buildings, their wall temperatures, ignition and fuel gas release."""

    def __init__(self, model, grid, to):
        cfg, nd = model.cfg, model.nd
        self.model, self.nd, self.cfg = model, nd, cfg
        shape = model.shape
        solid, dead = grid['solid'], grid['dead']
        fluid = ~(solid | dead)
        z = np.asarray(grid['z'], dtype=np.float64)
        hf = np.diff(z)
        hc = np.concatenate([[0.5 * hf[0]], 0.5 * (hf[:-1] + hf[1:]), [hf[-1]]])     # control-volume heights
        dx = model.h[0]
        dy = model.h[1] if nd == 3 else 1.0
        pairs = model._building_masks(grid, with_boxes=True)     # (box, solid nodes) of each building
        masks = [mask for _, mask in pairs]
        self.n = len(masks)
        if self.n == 0:
            raise ValueError("structures need [buildings]")
        owner = np.full(shape, -1)
        for b, m in enumerate(masks):
            owner[m & (owner < 0)] = b
        index = np.arange(solid.size).reshape(shape)
        levels = np.broadcast_to(np.arange(shape[0]).reshape((-1,) + (1,) * (nd - 1)), shape)
        faces = []          # (solid flat index, fluid flat index, kind 0 roof / 1 wall, area, building)
        for axis in range(nd):
            for s in (1, -1):
                nb = np.roll(index, -s, axis=axis)
                valid = np.ones(shape, dtype=bool)
                if axis == 0:
                    (valid[-1] if s == 1 else valid[0])[...] = False
                sel = solid & valid & fluid.ravel()[nb.ravel()].reshape(shape)
                if not sel.any():
                    continue
                k_rows = levels[sel]
                if axis == 0:                                     # horizontal face: roof (up) or underside
                    area = np.full(k_rows.shape, dx * dy)
                    kind = 0 if s == 1 else 1
                elif axis == nd - 1:                              # x-normal wall
                    area = dy * hc[k_rows]
                    kind = 1
                else:                                             # y-normal wall
                    area = dx * hc[k_rows]
                    kind = 1
                faces.append(np.stack([index[sel], nb[sel], np.full(k_rows.shape, kind), area, owner[sel]], 1))
        F = np.concatenate(faces)
        solid_idx, fluid_idx = F[:, 0].astype(np.int64), F[:, 1].astype(np.int64)
        kind, area, bld = F[:, 2].astype(int), F[:, 3], F[:, 4].astype(int)
        # Exact roof and wall areas of each building (boxes standing on the ground)
        self.boxes, self.volume = [], []
        named = {tuple(box): name for name, box in cfg.buildings}
        self.names = [named.get(tuple(box), str(b + 1)) for b, (box, _) in enumerate(pairs)]
        for b, (box, _) in enumerate(pairs):
            if nd == 2:
                x0, x1, height = box
                roof, walls, volume = x1 - x0, 2 * height, (x1 - x0) * height
                self.boxes.append((x0, x1, None, None, height))
            else:
                x0, x1, y0, y1, height = box
                roof, walls = (x1 - x0) * (y1 - y0), 2 * height * ((x1 - x0) + (y1 - y0))
                volume = (x1 - x0) * (y1 - y0) * height
                self.boxes.append((x0, x1, y0, y1, height))
            for kd, exact in ((0, roof), (1, walls)):
                sel = (bld == b) & (kind == kd)
                if sel.any() and area[sel].sum() > 0:
                    area[sel] *= exact / area[sel].sum()
            self.volume.append(volume)
        # Volume of each fluid node (the source per unit volume is flux * A / V)
        row = int(np.prod(shape[1:]))
        V = dx * dy * hc[fluid_idx // row]
        self.k_needed = int(fluid_idx.max() // row) + 2           # fuel slab must reach the fluid nodes
        idx = lambda a: torch.as_tensor(a, dtype=torch.long, device=model.device)
        self.solid_idx, self.fluid_idx, self.bld = idx(solid_idx), idx(fluid_idx), idx(bld)
        self.area = to(area)
        self.area_per_volume = to(area / V)
        self.kind_np, self.bld_np, self.area_np = kind, bld, area
        # Surface properties of roofs and walls; elements of the same kind share the wall discretisation
        self.groups = []
        for kd, name in ((0, 'roof'), (1, 'wall')):
            sel = np.flatnonzero(kind == kd)
            if sel.size == 0:
                continue
            dxw, kw, rcw = wall_cells(parse_layers(getattr(cfg, f'structures_{name}_layers')))
            # interface conductances between cells; back side insulated
            Gi = 1.0 / (dxw[:-1] / (2 * kw[:-1]) + dxw[1:] / (2 * kw[1:]))
            self.groups.append({
                'kind': kd, 'sel': idx(sel), 'n': sel.size,
                'C_nat': 1.52 if kd == 0 else 1.31,
                'T_ign': getattr(cfg, f'structures_{name}_T_ign'),
                'hrrpua': getattr(cfg, f'structures_{name}_hrrpua'),
                'cap': to(rcw * dxw), 'G': to(Gi), 'N': dxw.size,
                'T': torch.full((sel.size, dxw.size), cfg.T_inf, dtype=model.dtype, device=model.device),
            })
        self.ramp = parse_ramp(cfg.structures_ramp)
        self.eps = cfg.structures_emissivity
        nF = solid_idx.size
        self.T_s = torch.full((nF,), cfg.T_inf, dtype=model.dtype, device=model.device)
        self.t_ign = torch.full((nF,), 1e9, dtype=model.dtype, device=model.device)    # not ignited
        self.mass = to(cfg.structures_bulk_density * np.array(self.volume))             # combustible mass, kg
        # per-element constants, set once (writing Python numbers into device tensors at every step would be a
        # host-to-device copy, which CUDA graphs cannot capture)
        self.hrrpua = torch.zeros(nF, dtype=model.dtype, device=model.device)
        self.C_nat = torch.zeros(nF, dtype=model.dtype, device=model.device)
        for g in self.groups:
            self.hrrpua[g['sel']] = g['hrrpua']
            self.C_nat[g['sel']] = g['C_nat']
        # Surface temperature of each solid node seen by the radiation (mean of its elements; T_inf inside)
        counts = np.bincount(solid_idx, minlength=solid.size).astype(np.float64)
        self.node_weight = to(np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0))
        self.T_node = torch.full(shape, cfg.T_inf, dtype=model.dtype, device=model.device)
        self.T_gauge = cfg.structures_T_gauge if cfg.structures_T_gauge else cfg.T_inf
        # Diagnostics: envelope boxes around each building for the gas-phase heat release
        self.envelopes = []
        for (x0, x1, y0, y1, height) in self.boxes:
            m = cfg.structures_envelope
            inside = (grid['Xm'] >= x0 - m) & (grid['Xm'] <= x1 + m) & (grid['Zm'] <= height + cfg.structures_envelope_top)
            if nd == 3:
                inside &= (grid['Ym'] >= y0 - m) & (grid['Ym'] <= y1 + m)
            self.envelopes.append(torch.as_tensor(inside, device=model.device))
        # Last surface fluxes (diagnostics), per element
        self.q_net = torch.zeros(nF, dtype=model.dtype, device=model.device)
        self.q_gauge = torch.zeros(nF, dtype=model.dtype, device=model.device)

    # ------------------------------------------------------------------ state carried between steps
    def state(self):
        return {'T_s': self.T_s, 't_ign': self.t_ign, 'mass': self.mass, 'T_node': self.T_node,
                'walls': [g['T'] for g in self.groups]}

    def load_state(self, state):
        for key in ('T_s', 't_ign', 'mass', 'T_node'):
            getattr(self, key).copy_(state[key].to(getattr(self, key)))
        for g, T in zip(self.groups, state['walls']):
            g['T'].copy_(T.to(g['T']))

    # ------------------------------------------------------------------ physics
    def ramp_value(self, tau):
        frac = torch.full_like(tau, self.ramp[0][1])
        for (ta, fa), (tb, fb) in zip(self.ramp, self.ramp[1:]):
            if tb > ta:
                frac = frac + (fb - fa) * ((tau - ta) / (tb - ta)).clamp(0, 1)
        return frac

    def release_flux(self):
        """Fuel gas released per unit area of each element, kg/(m^2 s)."""
        burning = (self.t_ign < 1e8) & (self.mass[self.bld] > 0)
        tau = (self.model.t_t - self.t_ign).clamp(min=0)
        return self.hrrpua * self.ramp_value(tau) * burning.to(self.T_s.dtype) / self.cfg.H_C

    def surface_fluxes(self, y, G):
        """Convective coefficient and fluxes at every element: (h, T_g, q_conv, q_inc)."""
        m, nd = self.model, self.nd
        flat = y.view(y.shape[0], -1)
        T_g = flat[nd, self.fluid_idx]
        speed2 = flat[0, self.fluid_idx] ** 2
        for i in range(1, nd):
            speed2 = speed2 + flat[i, self.fluid_idx] ** 2
        rho = m.rho(T_g)
        kap = m.conductivity(T_g)[0]
        nu = m.viscosity(T_g) / rho
        Re = torch.sqrt(speed2) / nu                              # L = 1 m
        dT = T_g - self.T_s
        h = torch.maximum(self.C_nat * dT.abs() ** (1 / 3), kap * 0.037 * Re ** 0.8 * self.cfg.Pr ** (1 / 3))
        q_conv = h * dT
        q_inc = (0.5 * G.view(-1)[self.fluid_idx] - SIGMA * self.T_s ** 4).clamp(min=0)
        return h, T_g, q_conv, q_inc

    def add_sources(self, y, rho, mdot, Q_wall):
        """Add the fuel gas released (kg/(m^3 s)) to mdot and the heat convected to the walls (W/m^3, lost by
        the gas) to Q_wall; both are fuel-slab fields (rows 0..k-1)."""
        if self.model.G is None:
            return
        _, _, q_conv, _ = self.surface_fluxes(y, self.model.G)
        mdot.view(-1).index_add_(0, self.fluid_idx, self.release_flux() * self.area_per_volume)
        Q_wall.view(-1).index_add_(0, self.fluid_idx, q_conv * self.area_per_volume)

    def advance(self, y, dt):
        """Wall temperatures over one step (implicit conduction, explicit surface flux), ignition and the
        combustible mass. In place (CUDA graphs)."""
        m = self.model
        if m.G is None:
            return
        flux = self.release_flux()
        burned = torch.zeros_like(self.mass).index_add_(0, self.bld, flux * self.area)
        self.mass.sub_(burned * dt).clamp_(min=0)
        h, T_g, q_conv, q_inc = self.surface_fluxes(y, m.G)
        q_net = q_conv + self.eps * (q_inc - SIGMA * self.T_s ** 4)
        self.q_net.copy_(q_net)
        # gauge heat flux as in FDS: a sensor held at T_gauge (incident radiation minus its own emission)
        self.q_gauge.copy_(q_inc - SIGMA * self.T_gauge ** 4 + h * (T_g - self.T_gauge))
        for g in self.groups:
            T = g['T']
            cap, Gi, N = g['cap'], g['G'], g['N']
            # backward Euler: cap_i (T_i' - T_i) / dt = G_{i-1}(T_{i-1}' - T_i') + G_i (T_{i+1}' - T_i') + [i=0] q
            lower = torch.cat([Gi.new_zeros(1), -Gi])            # coefficient of T_{i-1}
            upper = torch.cat([-Gi, Gi.new_zeros(1)])            # coefficient of T_{i+1}
            diag = cap / dt - lower - upper
            rhs = T * (cap / dt)
            rhs[:, 0] = rhs[:, 0] + q_net[g['sel']]
            # Thomas algorithm over the cells, vectorised over the elements
            c_ = [None] * N
            d_ = [None] * N
            c_[0] = upper[0] / diag[0]
            d_[0] = rhs[:, 0] / diag[0]
            for i in range(1, N):
                denom = diag[i] - lower[i] * c_[i - 1]
                c_[i] = upper[i] / denom
                d_[i] = (rhs[:, i] - lower[i] * d_[i - 1]) / denom
            T[:, N - 1] = d_[N - 1]
            for i in range(N - 2, -1, -1):
                T[:, i] = d_[i] - c_[i] * T[:, i + 1]
            sel = g['sel']
            self.T_s.index_copy_(0, sel, T[:, 0])
            t_ign = self.t_ign.index_select(0, sel)
            newly = (T[:, 0] >= g['T_ign']) & (t_ign > 1e8)
            self.t_ign.index_copy_(0, sel, torch.where(newly, m.t_t + dt, t_ign))
        # surface temperature of the solid nodes for the radiation (mean of their elements)
        node = torch.zeros_like(self.T_node).view(-1).index_add_(0, self.solid_idx, self.T_s)
        self.T_node.copy_(torch.where(self.node_weight.view(self.T_node.shape) > 0,
                                      node.view(self.T_node.shape) * self.node_weight.view(self.T_node.shape),
                                      torch.full_like(self.T_node, self.cfg.T_inf)))

    # ------------------------------------------------------------------ diagnostics (host)
    def diagnostics(self, hrr_field, cell_volume):
        """Per building: mean and max surface temperature (K), burning fraction of the surface, first ignition
        time (s, nan if none), fuel release (kg/s) and its heat (W), gas-phase heat release in the envelope (W),
        surface-integrated gauge and net heat fluxes (W, FDS GAUGE HEAT FLUX and NET HEAT FLUX), combustible
        mass left (kg)."""
        A = self.area
        flux = self.release_flux()
        out = {}
        bld = self.bld
        sumb = lambda v: torch.zeros(self.n, dtype=torch.float64, device=v.device).index_add_(0, bld, v.double())
        area_b = sumb(A)
        out['T_s_mean'] = (sumb(self.T_s * A) / area_b).cpu().numpy()
        Tmax = torch.full((self.n,), -np.inf, dtype=torch.float64, device=A.device)
        out['T_s_max'] = Tmax.scatter_reduce(0, bld, self.T_s.double(), 'amax').cpu().numpy()
        burning = (self.t_ign < 1e8).to(A.dtype)
        out['burning_fraction'] = (sumb(burning * A) / area_b).cpu().numpy()
        t_first = torch.full((self.n,), 1e9, dtype=torch.float64, device=A.device).scatter_reduce(
            0, bld, self.t_ign.double(), 'amin').cpu().numpy()
        out['t_ignition'] = np.where(t_first < 1e8, t_first, np.nan)
        out['release'] = sumb(flux * A).cpu().numpy()
        out['release_hrr'] = out['release'] * self.cfg.H_C
        out['gas_hrr'] = np.array([float((hrr_field * cell_volume)[env].double().sum()) for env in self.envelopes])
        out['gauge_heat_flux'] = sumb(self.q_gauge * A).cpu().numpy()
        out['net_heat_flux'] = sumb(self.q_net * A).cpu().numpy()
        out['mass_left'] = self.mass.double().cpu().numpy()
        out['area'] = area_b.cpu().numpy()
        return out
