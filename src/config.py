"""Model parameters: defaults (same values as src_legacy/parameters.py) plus the .ini reader.

Axis convention used by the whole package
-----------------------------------------
Arrays are stored as (z, x) in 2D and (z, y, x) in 3D, i.e. the vertical
(non-periodic) axis is always axis 0 and x is always the last axis. Saved output
adds time in front: (t, z, x) and (t, z, y, x).

In a 2D parameter file the vertical direction is called ``y`` (as in the published
2D model), so ``y_min, y_max, Ny`` of a 2D file are stored here as ``z_min, z_max, Nz``.
A file is 3D when it defines ``z_min``, ``z_max`` and ``Nz``.
"""
import configparser
import warnings
from dataclasses import dataclass, asdict

T_INF = 288.15


@dataclass
class Config:
    # Domain and grid (z is the vertical axis; y only exists in 3D)
    ndim: int = 2
    x_min: float = 0.0
    x_max: float = 200.0
    y_min: float = 0.0
    y_max: float = 200.0
    z_min: float = 0.0
    z_max: float = 20.0
    t_min: float = 0.0
    t_max: float = 1.0
    Nx: int = 128
    Ny: int = 128
    Nz: int = 64
    Nt: int = 1000
    NT: int = 100
    method: str = 'euler'           # euler, RK2, RK4
    experiment: str = 'fire'
    # Gas
    c_p: float = 1007.0
    rho_inf: float = 1.225
    mu: float = 1.802e-5
    kappa: float = 0.02476
    delta: float = 0.1              # optical path length
    sigma: float = 5.670374419e-8   # Stefan-Boltzmann
    T_inf: float = T_INF
    g: float = -9.807               # vertical component of gravity
    density_constant: bool = False
    # Turbulence
    turbulence: bool = True
    C_s: float = 0.2
    Pr: float = 0.7323
    # Fuel / reaction
    C_d: float = 0.15
    alpha_s: float = 0.002
    sigma_s: float = 4000.0
    T_pc: float = (573 + 473) / 2
    H_C: float = 21e6
    A: float = 1e9
    T_act: float = 18040.8533
    h_c: float = 1.42
    Y_f: float = 70.0
    Y_h: float = 0.51
    reaction: str = 'exact'         # exact: fuel decay integrated exactly over dt; explicit: original rate
    sink: str = 'ambient'           # ambient: natural convection fuel -> ambient air; legacy: original term
    fuel_x_start: float = None      # optional horizontal extent of the fuel bed
    fuel_x_end: float = None
    fuel_y_start: float = None
    fuel_y_end: float = None
    # Bounds
    bound: bool = True
    T_min: float = T_INF
    T_max: float = 2500.0
    Y_min: float = 0.0
    Y_max: float = 1.0
    # Initial temperature
    T_hot: float = T_INF + 500
    T0_shape: str = 'gaussian'      # gaussian, plate, line (3D: gaussian in x-z, plate in y)
    T0_x_start: float = 0.0
    T0_x_end: float = 6.0
    T0_y_start: float = 80.0
    T0_y_end: float = 120.0
    T0_z_start: float = 0.0
    T0_z_end: float = 1.0
    t_source: float = -1.0
    # Wind
    initial_u_type: str = 'power law'   # power law, log, constant
    u_r: float = 4.8
    z_r: float = 2.0
    alpha_u: float = 1.0 / 7.0
    u_z_0: float = 0.05
    d: float = 0.0
    u_ast: float = 0.1
    von_karman: float = 0.41
    # Topography
    topography_shape: str = 'flat'      # flat, hill, slope (2D only)
    hill_center_x: float = 40.0
    hill_center_y: float = 40.0
    hill_length: float = 20.0
    hill_width: float = 20.0
    hill_height: float = 2.5
    # Pressure solver
    tol: float = 1e-10
    max_iter: int = 50
    log_solver: bool = False
    poisson: str = 'auto'               # auto, thomas, eig
    # Reproduce the original src_legacy/ code, defects included (see README of this folder)
    legacy: bool = False
    # Run
    device: str = 'auto'
    dtype: str = 'float64'
    sim_name: str = 'sim'
    save_path: str = './data/output/sim/'

    def to_dict(self):
        return asdict(self)


# (section, key) -> (attribute, type). Keys of the original .ini files.
_KEYS = {
    ('domain', 'x_min'): ('x_min', float), ('domain', 'x_max'): ('x_max', float),
    ('domain', 'y_min'): ('y_min', float), ('domain', 'y_max'): ('y_max', float),
    ('domain', 'z_min'): ('z_min', float), ('domain', 'z_max'): ('z_max', float),
    ('domain', 't_min'): ('t_min', float), ('domain', 't_max'): ('t_max', float),
    ('numerics', 'Nx'): ('Nx', int), ('numerics', 'Ny'): ('Ny', int), ('numerics', 'Nz'): ('Nz', int),
    ('numerics', 'Nt'): ('Nt', int), ('numerics', 'NT'): ('NT', int),
    ('numerics', 'time_method'): ('method', str),
    ('experiment', 'name'): ('experiment', str),
    ('topography', 'shape'): ('topography_shape', str),
    ('fuel', 'C_d'): ('C_d', float), ('fuel', 'Y_h'): ('Y_h', float), ('fuel', 'T_act'): ('T_act', float),
    ('fuel', 'Y_f'): ('Y_f', float), ('fuel', 'A'): ('A', float), ('fuel', 'H_C'): ('H_C', float),
    ('fuel', 'sigma_s'): ('sigma_s', float), ('fuel', 'alpha_s'): ('alpha_s', float),
    ('fuel', 'reaction'): ('reaction', str), ('temperature', 'sink'): ('sink', str),
    ('temperature', 'T_inf'): ('T_inf', float), ('numerics', 'bound'): ('bound', bool),
    ('fuel', 'x_start'): ('fuel_x_start', float), ('fuel', 'x_end'): ('fuel_x_end', float),
    ('fuel', 'y_start'): ('fuel_y_start', float), ('fuel', 'y_end'): ('fuel_y_end', float),
    ('temperature', 'T_hot'): ('T_hot', float), ('temperature', 'T_min'): ('T_min', float),
    ('temperature', 'T_max'): ('T_max', float), ('temperature', 't_source'): ('t_source', float),
    ('temperature', 'shape'): ('T0_shape', str), ('temperature', 'delta'): ('delta', float),
    ('temperature', 'h_c'): ('h_c', float), ('temperature', 'T_pc'): ('T_pc', float),
    ('wind', 'u_r'): ('u_r', float), ('wind', 'shape'): ('initial_u_type', str),
    ('pressure', 'tol'): ('tol', float), ('pressure', 'max_iter'): ('max_iter', int),
    ('pressure', 'log_solver'): ('log_solver', bool), ('pressure', 'poisson'): ('poisson', str),
    ('density', 'constant'): ('density_constant', bool),
    ('turbulence', 'enabled'): ('turbulence', bool), ('turbulence', 'C_s'): ('C_s', float),
}
for _ax in 'xyz':  # ignition location: both 'x_start' (original reader) and 'T0_x_start' (used in some files)
    for _end in ('start', 'end'):
        _KEYS[('temperature', f'{_ax}_{_end}')] = (f'T0_{_ax}_{_end}', float)
        _KEYS[('temperature', f'T0_{_ax}_{_end}')] = (f'T0_{_ax}_{_end}', float)


def load_config(path: str, **overrides) -> Config:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path):
        raise FileNotFoundError(path)
    cfg = Config()
    given = set()
    for section in parser.sections():
        for key in parser[section]:
            if (section, key) not in _KEYS:
                warnings.warn(f"[{section}] {key} is not a known parameter and is ignored")
                continue
            attr, typ = _KEYS[(section, key)]
            value = parser.getboolean(section, key) if typ is bool else typ(parser.get(section, key))
            setattr(cfg, attr, value)
            given.add(attr)
    if 'T_inf' in given:  # values that default to the ambient temperature follow it
        if 'T_hot' not in given:
            cfg.T_hot = cfg.T_inf + 500
        if 'T_min' not in given:
            cfg.T_min = cfg.T_inf
    has = lambda key, sec: parser.has_option(sec, key)
    if has('z_min', 'domain') and has('z_max', 'domain') and has('Nz', 'numerics'):
        cfg.ndim = 3
    else:  # 2D file: its 'y' is the vertical axis
        cfg.ndim = 2
        cfg.z_min, cfg.z_max, cfg.Nz = cfg.y_min, cfg.y_max, cfg.Ny
    if cfg.experiment != 'fire':
        raise NotImplementedError(f"experiment '{cfg.experiment}' is not ported; only 'fire' is (use src_legacy/ for the others)")
    if cfg.topography_shape == 'slope' and cfg.ndim == 3:
        raise NotImplementedError("'slope' topography is 2D only")
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if cfg.legacy:  # everything as in src_legacy/
        cfg.reaction, cfg.sink = 'explicit', 'legacy'
    if cfg.reaction not in ('exact', 'explicit') or cfg.sink not in ('ambient', 'legacy'):
        raise ValueError("reaction must be 'exact' or 'explicit', sink must be 'ambient' or 'legacy'")
    return cfg
