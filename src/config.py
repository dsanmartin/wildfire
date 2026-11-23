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
from dataclasses import dataclass, asdict, field

T_INF = 288.15


@dataclass
class Config:
    # Domain and grid (z is the vertical axis; y only exists in 3D and in the plan-view model)
    model: str = 'vertical'         # vertical: 2D x-z or 3D coupled model; planview: 2D x-y fire-spread layer
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
    cfl: float = 0.4                # adaptive time step: target CFL (dt <= (t_max - t_min) / Nt); 0: fixed dt
    cuda_graph: str = 'auto'        # replay each step as CUDA graphs: auto (on with CUDA), on, off
    pcg_chunk: int = 4              # PCG iterations per replayed chunk (convergence checked between chunks)
    experiment: str = 'fire'
    # Gas
    c_p: float = 1007.0
    cp_model: str = 'mixture'       # constant (c_p); variable: c_p(T) of air; mixture: species-weighted c_p(T)
                                    # (needs the species: falls back to variable without solid_temperature)
    transport: str = 'sutherland'   # constant mu and kappa, or sutherland: mu(T), kappa(T) (Sutherland's law)
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
    filter_width: float = None      # Smagorinsky filter width in m; None: (dx dz)^(1/2) or (dx dy dz)^(1/3)
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
    # Solid fuel phase: its own temperature T_s, heated by convection and radiation, and moisture W
    solid_temperature: bool = True  # False: one temperature for gas and fuel (published model)
    rho_bulk: float = 0.6144        # dry fuel mass per bed volume, kg/m^3 (FDS Case_F19)
    c_s: float = 1500.0             # specific heat of the dry fuel, J/(kg K)
    H_pyr: float = 4.16e5           # heat absorbed by pyrolysis, J/kg (Mell et al. 2007)
    moisture: float = 0.058         # initial moisture, kg water per kg dry fuel (CSIRO F19, FDS Case_F19)
    c_w: float = 4184.0             # specific heat of liquid water, J/(kg K)
    L_v: float = 2.259e6            # latent heat of vaporisation of water, J/kg
    T_boil: float = 373.15          # temperature at which the fuel moisture evaporates, K
    pyrolysis: str = 'linear'       # linear: heat-driven between T_pyr and T_pyr_end (Mell et al. 2007);
                                    # arrhenius: dY/dt = -Y_f A exp(-T_act/T_s) Y for T_s > T_pc
    T_pyr: float = 400.0            # pyrolysis starts, K
    T_pyr_end: float = 500.0        # all net heat goes to pyrolysis above this, K
    # Gas-phase combustion of the pyrolysis gas (used with the solid fuel model)
    Y_O2_amb: float = 0.232         # oxygen mass fraction of the ambient air
    s_O2: float = 1.185             # kg of oxygen per kg of fuel gas (cellulose)
    tau_mix: float = None           # mixing time, s; None: sqrt(2 Delta / g) (FDS buoyancy time scale)
    # Radiation
    radiation: str = 'transport'    # transport: discrete ordinates; diffusion: 4 sigma delta T^3 conductivity
    kappa_gas: float = 0.1          # gray absorption coefficient of the gas, 1/m (to calibrate)
    n_polar: int = 12               # polar bands of the angular quadrature (even)
    n_azimuth: int = 16             # azimuths over the full circle (even; 2D uses half of them)
    rad_every: int = 5              # time steps between radiation solves
    sink: str = 'surface'           # surface: top fuel node <-> gas node above; legacy: original term
    fuel_threshold: float = 1e-3    # Y above which a node counts as fuel for the surface exchange
    fuel_slab: bool = True          # solid fuel terms and drag only on the rows that hold fuel
    T_advection: str = 'koren'      # koren (limited third order), upwind (second order), central (original)
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
    initial_u_type: str = 'monin-obukhov'   # monin-obukhov (neutral log law if L is unset), power law, log, constant
    L_obukhov: float = None             # Obukhov length, m (monin-obukhov; None: neutral). FDS Case_F19: -500
    forcing: str = 'profile'            # profile: nudge the horizontal-mean wind to the initial profile; none
    forcing_time: float = 1.0           # relaxation time of the wind forcing, s
    wall_model: str = 'log'             # log: wall stress from the log law; none: no-slip ground
    wall_z0: float = None               # roughness length of the wall model, m; None: the wind's z_0
    u_r: float = 4.8
    z_r: float = 2.0
    alpha_u: float = 1.0 / 7.0
    u_z_0: float = 0.03             # roughness length, m (open grassland; FDS Case_F19)
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
    poisson: str = 'auto'               # auto, thomas, eig (nodal projection)
    projection: str = 'face'            # face: discretely conservative face fluxes; nodal: original projection
    dilatation: str = 'consistent'      # consistent: div u = (1/T) DT/Dt; published: divided again by rho c_p
    # Reproduce the original src_legacy/ code, defects included (see README of this folder)
    legacy: bool = False
    # Buildings (immersed solids): boxes standing on the ground, (x0, x1, height) in 2D and
    # (x0, x1, y0, y1, height) in 3D, in metres. Defined in the [buildings] section, one per key.
    buildings: list = field(default_factory=list)
    wall_iterations: int = 0        # extra pressure corrections after the projection (see docs: not effective with the current collocated projection)
    # Plan-view model (model = planview): fuel-bed layer in x-y with prescribed wind
    wind_direction: float = 0.0     # direction the wind blows towards, degrees from +x (counter-clockwise)
    wind_factor: float = None       # mid-flame wind / u_r; None: power law at the bed height, (Y_h / z_r)^alpha_u
    heat_retained: float = 0.15     # fraction of the heat of combustion that stays in the bed (to calibrate)
    h_top: float = 20.0             # heat loss coefficient to the air above the bed, W/(m^2 K) (to calibrate)
    delta_bed: float = None         # radiative path length in the bed, m; None: 4 / (alpha_s sigma_s)
    # Output of 3D runs: planes saved at every output time (full fields only at the end)
    z_slice: float = 1.0            # height of the x-y plane, m
    full_interval: float = 0.0      # s between full-field snapshots (files in NAME/full/); 0: none
    full_dtype: str = 'float32'     # precision of the full-field snapshots
    y_slice: float = None           # position of the x-z plane, m; None: middle of the domain
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
    ('fuel', 'solid_temperature'): ('solid_temperature', bool), ('fuel', 'rho_bulk'): ('rho_bulk', float),
    ('fuel', 'c_s'): ('c_s', float), ('fuel', 'H_pyr'): ('H_pyr', float), ('fuel', 'moisture'): ('moisture', float),
    ('radiation', 'model'): ('radiation', str), ('radiation', 'kappa_gas'): ('kappa_gas', float),
    ('radiation', 'n_polar'): ('n_polar', int), ('radiation', 'n_azimuth'): ('n_azimuth', int),
    ('radiation', 'update_every'): ('rad_every', int),
    ('output', 'z_slice'): ('z_slice', float), ('output', 'y_slice'): ('y_slice', float),
    ('model', 'type'): ('model', str), ('wind', 'direction'): ('wind_direction', float),
    ('wind', 'factor'): ('wind_factor', float), ('planview', 'heat_retained'): ('heat_retained', float),
    ('planview', 'h_top'): ('h_top', float), ('planview', 'delta_bed'): ('delta_bed', float),
    ('output', 'full_interval'): ('full_interval', float), ('output', 'full_dtype'): ('full_dtype', str),
    ('fuel', 'pyrolysis'): ('pyrolysis', str), ('fuel', 'T_pyr'): ('T_pyr', float),
    ('fuel', 'slab'): ('fuel_slab', bool),
    ('fuel', 'T_pyr_end'): ('T_pyr_end', float), ('combustion', 's_O2'): ('s_O2', float),
    ('combustion', 'tau_mix'): ('tau_mix', float), ('combustion', 'Y_O2_amb'): ('Y_O2_amb', float),
    ('temperature', 'T_inf'): ('T_inf', float), ('numerics', 'bound'): ('bound', bool),
    ('gas', 'cp_model'): ('cp_model', str), ('gas', 'transport'): ('transport', str),
    ('numerics', 'T_advection'): ('T_advection', str), ('fuel', 'threshold'): ('fuel_threshold', float),
    ('numerics', 'cfl'): ('cfl', float), ('numerics', 'cuda_graph'): ('cuda_graph', str),
    ('numerics', 'pcg_chunk'): ('pcg_chunk', int),
    ('fuel', 'x_start'): ('fuel_x_start', float), ('fuel', 'x_end'): ('fuel_x_end', float),
    ('fuel', 'y_start'): ('fuel_y_start', float), ('fuel', 'y_end'): ('fuel_y_end', float),
    ('temperature', 'T_hot'): ('T_hot', float), ('temperature', 'T_min'): ('T_min', float),
    ('temperature', 'T_max'): ('T_max', float), ('temperature', 't_source'): ('t_source', float),
    ('temperature', 'shape'): ('T0_shape', str), ('temperature', 'delta'): ('delta', float),
    ('temperature', 'h_c'): ('h_c', float), ('temperature', 'T_pc'): ('T_pc', float),
    ('wind', 'u_r'): ('u_r', float), ('wind', 'shape'): ('initial_u_type', str),
    ('wind', 'z_r'): ('z_r', float), ('wind', 'z_0'): ('u_z_0', float), ('wind', 'L'): ('L_obukhov', float),
    ('wind', 'forcing'): ('forcing', str), ('wind', 'forcing_time'): ('forcing_time', float),
    ('wall', 'model'): ('wall_model', str), ('wall', 'z_0'): ('wall_z0', float),
    ('pressure', 'tol'): ('tol', float), ('pressure', 'max_iter'): ('max_iter', int),
    ('pressure', 'log_solver'): ('log_solver', bool), ('pressure', 'poisson'): ('poisson', str),
    ('pressure', 'wall_iterations'): ('wall_iterations', int), ('pressure', 'projection'): ('projection', str),
    ('pressure', 'dilatation'): ('dilatation', str),
    ('density', 'constant'): ('density_constant', bool),
    ('turbulence', 'enabled'): ('turbulence', bool), ('turbulence', 'C_s'): ('C_s', float),
    ('turbulence', 'filter_width'): ('filter_width', float),
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
        if section == 'buildings':   # any key: a box given by comma-separated numbers
            for key in parser[section]:
                box = tuple(float(v) for v in parser.get(section, key).split(','))
                cfg.buildings.append((key, box))
            continue
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
    if cfg.model == 'planview':   # 2D in x-y: y is horizontal
        cfg.ndim = 2
    elif has('z_min', 'domain') and has('z_max', 'domain') and has('Nz', 'numerics'):
        cfg.ndim = 3
    else:  # 2D file: its 'y' is the vertical axis
        cfg.ndim = 2
        cfg.z_min, cfg.z_max, cfg.Nz = cfg.y_min, cfg.y_max, cfg.Ny
    if cfg.experiment != 'fire':
        raise NotImplementedError(f"experiment '{cfg.experiment}' is not ported; only 'fire' is (use src_legacy/ for the others)")
    for name, box in cfg.buildings:
        if len(box) != (3 if cfg.ndim == 2 else 5):
            raise ValueError(f"[buildings] {name}: expected {'x0, x1, height' if cfg.ndim == 2 else 'x0, x1, y0, y1, height'}")
    if cfg.buildings and (cfg.model == 'planview' or cfg.legacy):
        raise NotImplementedError("buildings need the vertical model without --legacy")
    if cfg.topography_shape == 'slope' and cfg.ndim == 3:
        raise NotImplementedError("'slope' topography is 2D only")
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if cfg.legacy:  # everything as in src_legacy/
        cfg.reaction, cfg.sink, cfg.T_advection = 'explicit', 'legacy', 'central'
        cfg.solid_temperature, cfg.radiation, cfg.projection, cfg.cp_model = False, 'diffusion', 'nodal', 'constant'
        cfg.transport, cfg.cfl, cfg.forcing, cfg.wall_model = 'constant', 0.0, 'none', 'none'
        cfg.dilatation = 'published'
        if cfg.cuda_graph == 'auto':
            cfg.cuda_graph = 'off'
        for attr, value in (('initial_u_type', 'power law'), ('u_z_0', 0.05)):   # original defaults
            if attr not in given:
                setattr(cfg, attr, value)
    if cfg.cp_model == 'mixture' and not cfg.solid_temperature:
        if 'cp_model' in given:
            raise ValueError("cp_model = mixture needs the species of the extended model (solid_temperature = True)")
        cfg.cp_model = 'variable'                 # default without species: c_p(T) of air
    graph = str(cfg.cuda_graph).strip().lower()
    on_off = {'auto': 'auto', 'on': 'on', 'true': 'on', 'yes': 'on', '1': 'on',
              'off': 'off', 'false': 'off', 'no': 'off', '0': 'off'}
    if graph not in on_off:
        raise ValueError(f"cuda_graph must be auto, on or off, got '{cfg.cuda_graph}'")
    cfg.cuda_graph = on_off[graph]
    for key, allowed in (('reaction', ('exact', 'explicit')), ('sink', ('surface', 'legacy')),
                         ('T_advection', ('upwind', 'koren', 'central')),
                         ('radiation', ('transport', 'diffusion')), ('pyrolysis', ('linear', 'arrhenius')),
                         ('model', ('vertical', 'planview')), ('projection', ('face', 'nodal')),
                         ('cp_model', ('constant', 'variable', 'mixture')), ('transport', ('constant', 'sutherland')),
                         ('initial_u_type', ('power law', 'log', 'constant', 'monin-obukhov')),
                         ('forcing', ('none', 'profile')), ('wall_model', ('none', 'log')),
                         ('dilatation', ('consistent', 'published'))):
        if getattr(cfg, key) not in allowed:
            raise ValueError(f"{key} must be one of {allowed}, got '{getattr(cfg, key)}'")
    return cfg
