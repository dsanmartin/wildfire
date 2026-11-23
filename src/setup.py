"""Grid, topography, initial conditions and immersed-boundary nodes (NumPy, float64)."""
import numpy as np


def gaussian(x, z, x0, z0, sx, sz):
    return np.exp(-((x - x0) ** 2 / sx ** 2 + (z - z0) ** 2 / sz ** 2))


def psi_m(zeta):
    """Integrated stability function for momentum, zeta = z / L. Unstable (zeta < 0): Paulson (1970)
    with the Businger-Dyer form phi_m = (1 - 16 zeta)^(-1/4); stable: psi_m = -5 zeta."""
    zeta = np.asarray(zeta, dtype=np.float64)
    x = (1 - 16 * np.minimum(zeta, 0.0)) ** 0.25
    unstable = 2 * np.log((1 + x) / 2) + np.log((1 + x ** 2) / 2) - 2 * np.arctan(x) + np.pi / 2
    return np.where(zeta < 0, unstable, -5 * zeta)


def monin_obukhov(z, u_ref, z_ref, z0, L, kappa=0.41):
    """Wind speed of Monin-Obukhov similarity theory,
        u(z) = u*/kappa [ln(z/z0) - psi_m(z/L) + psi_m(z0/L)],
    with the friction velocity u* chosen so that u(z_ref) = u_ref; zero below z0. L = None: neutral."""
    psi = (lambda zz: psi_m(np.asarray(zz) / L)) if L else (lambda zz: np.zeros_like(np.asarray(zz, dtype=np.float64)))
    shape = lambda zz: np.log(zz / z0) - psi(zz) + psi(z0)
    u_star = kappa * u_ref / shape(z_ref)
    zc = np.maximum(z, z0)
    return np.where(z > z0, u_star / kappa * shape(zc), 0.0), u_star


def topography(cfg):
    """Returns f(x) (2D) or f(x, y) (3D): terrain height."""
    shape = cfg.topography_shape
    if shape == 'flat':
        return lambda x, y=None: x * 0.0
    if shape == 'hill':
        if cfg.ndim == 2:
            return lambda x, y=None: cfg.hill_height * np.exp(-((x - cfg.hill_center_x) ** 2 / cfg.hill_length ** 2))
        return lambda x, y: cfg.hill_height * np.exp(-(((x - cfg.hill_center_x) / cfg.hill_length) ** 2
                                                       + ((y - cfg.hill_center_y) / cfg.hill_width) ** 2))
    if shape == 'slope':  # 10 degrees up to x = 30, down to x = 60
        m = 0.17632698070846498
        return lambda x, y=None: np.piecewise(x, [x < 0, x >= 0, x >= 30, x > 60],
                                              [0, lambda x: x * m, lambda x: (60 - x) * m, 0])
    raise ValueError(f"unknown topography '{shape}'")


def build(cfg):
    """Create the grid and the initial fields.

    Returns a dict of NumPy arrays. Field arrays have shape (Nz, Nx-1) or
    (Nz, Ny-1, Nx-1): the last node of each periodic axis is not stored.
    """
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.Nx)
    z = np.linspace(cfg.z_min, cfg.z_max, cfg.Nz)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.Nt + 1)
    dx, dz, dt = x[1] - x[0], z[1] - z[0], t[1] - t[0]
    if cfg.ndim == 2:
        y, dy = None, None
        Zm, Xm = np.meshgrid(z, x[:-1], indexing='ij')
        Ym = None
        spacings = (dx, dz)
    else:
        y = np.linspace(cfg.y_min, cfg.y_max, cfg.Ny)
        dy = y[1] - y[0]
        Zm, Ym, Xm = np.meshgrid(z, y[:-1], x[:-1], indexing='ij')
        spacings = (dx, dy, dz)
    topo = topography(cfg)
    ground = topo(Xm, Ym)

    # Wind: u as a function of the height above the terrain (above z_min in the original code, so that
    # the profile ran straight into a hill); other components zero
    H = Zm if cfg.legacy else np.clip(Zm - ground, 0.0, None)
    if cfg.initial_u_type == 'power law':
        u = cfg.u_r * (H / cfg.z_r) ** cfg.alpha_u
    elif cfg.initial_u_type == 'log':
        u = np.where(H > 0, cfg.u_ast / cfg.von_karman * np.log(np.where(H > 0, H - cfg.d, 1.0) / cfg.u_z_0), 0.0)
    elif cfg.initial_u_type == 'constant':
        u = cfg.u_r + Xm * 0
    elif cfg.initial_u_type == 'monin-obukhov':
        u, _ = monin_obukhov(H, cfg.u_r, cfg.z_r, cfg.u_z_0, cfg.L_obukhov, cfg.von_karman)
    else:
        raise ValueError(f"unknown wind profile '{cfg.initial_u_type}'")
    velocity = [u] + [np.zeros_like(u) for _ in range(cfg.ndim - 1)]

    # Fuel: 1 up to Y_h above the terrain, optionally limited to a horizontal rectangle
    Y = (Zm <= ground + cfg.Y_h).astype(np.float64)
    for coord, lo, hi in ((Xm, cfg.fuel_x_start, cfg.fuel_x_end), (Ym, cfg.fuel_y_start, cfg.fuel_y_end)):
        if coord is not None and lo is not None and hi is not None:
            Y *= ((coord >= lo) & (coord <= hi))

    # Initial temperature
    x_c = (cfg.T0_x_start + cfg.T0_x_end) / 2
    length = cfg.T0_x_end - cfg.T0_x_start
    height = cfg.T0_z_end - cfg.T0_z_start
    in_x = (Xm >= cfg.T0_x_start) & (Xm <= cfg.T0_x_end)
    in_z = (Zm >= cfg.T0_z_start) & (Zm <= cfg.T0_z_end)
    if cfg.ndim == 2:
        if cfg.T0_shape == 'plate':
            shape = (in_x & in_z).astype(np.float64)
        elif cfg.T0_shape == 'gaussian':
            if cfg.topography_shape == 'slope':  # rotate the ignition patch with the 10 degree slope
                c, s = np.cos(np.deg2rad(-10)), np.sin(np.deg2rad(-10))
                shape = gaussian(c * Xm - s * Zm, s * Xm + c * Zm, x_c, 0, length, height)
            else:
                shape = gaussian(Xm, Zm, x_c, 0, length, height)
        else:
            raise ValueError(f"unknown 2D initial temperature shape '{cfg.T0_shape}'")
    else:
        y_c = (cfg.T0_y_start + cfg.T0_y_end) / 2
        width = cfg.T0_y_end - cfg.T0_y_start
        in_y = (Ym >= cfg.T0_y_start) & (Ym <= cfg.T0_y_end)
        if cfg.T0_shape == 'plate':
            shape = (in_x & in_y & in_z).astype(np.float64)
        elif cfg.T0_shape == 'gaussian':
            shape = gaussian(Xm, Zm, x_c, 0, length, height) * np.exp(-((Ym - y_c) / width) ** 2)
        elif cfg.T0_shape == 'line':             # the 2D ignition extruded along y_start <= y <= y_end
            shape = gaussian(Xm, Zm, x_c, 0, length, height) * in_y
        else:
            raise ValueError(f"unknown 3D initial temperature shape '{cfg.T0_shape}'")
    T = cfg.T_inf + shape * (cfg.T_hot - cfg.T_inf)

    # Immersed boundary: cut nodes are the terrain surface, dead nodes lie below it
    cut = (Zm >= ground - dz) & (Zm <= ground)
    dead = Zm < ground - dz
    for field, value in zip(velocity + [T, Y], [0.0] * cfg.ndim + [cfg.T_inf, 0.0]):
        field[dead] = value
    # Buildings: boxes standing on the terrain. Solid nodes: no flow, no fuel, ambient temperature.
    solid = np.zeros_like(Zm, dtype=bool)
    for _, box in cfg.buildings:
        if cfg.ndim == 2:
            x0, x1, height = box
            inside = (Xm >= x0) & (Xm <= x1)
        else:
            x0, x1, y0, y1, height = box
            inside = (Xm >= x0) & (Xm <= x1) & (Ym >= y0) & (Ym <= y1)
        solid |= inside & (Zm <= ground + height)
    for field, value in zip(velocity + [T, Y], [0.0] * cfg.ndim + [cfg.T_inf, 0.0]):
        field[solid] = value
    fields = velocity + [T, Y]
    if cfg.solid_temperature:
        # The fuel starts at the gas temperature (heat-driven pyrolysis: at most T_pyr_end, where it is
        # already pyrolysing). Fuel inside the ignition patch (above boiling) is dry. No fuel gas yet.
        T_s = np.minimum(T, cfg.T_pyr_end) if cfg.pyrolysis == 'linear' else T.copy()
        W = cfg.moisture * Y * (T < cfg.T_boil)
        fields += [T_s, W, np.zeros_like(T), np.full_like(T, cfg.Y_O2_amb)]

    return {
        'x': x, 'y': y, 'z': z, 't': t, 'dt': dt, 'spacings': spacings,
        'Xm': Xm, 'Ym': Ym, 'Zm': Zm, 'ground': ground,
        'state': np.stack(fields),                   # (ndim + 2 [+ 2], Nz, [Ny-1,] Nx-1): u.., T, Y [, T_s, W, Y_F, Y_O2]
        'cut': cut, 'dead': dead, 'solid': solid,
        'source_mask': shape > 0.01,                 # nodes held at T0 while t <= t_source
    }
