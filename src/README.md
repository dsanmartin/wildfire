# PyTorch implementation

One code path for 2D and 3D, running on CPU or GPU. The physics is the published 2D
model of `src_legacy/` (low Mach number, variable density, radiation, Smagorinsky turbulence);
the 3D case is the same equations with one more periodic direction.

```bash
python src/main.py data/input/2D/case_f19.ini -n my_run            # cuda if available, else cpu
python src/main.py data/input/3D/case_f19.ini --device cuda:0
python src/main.py data/input/2D/case_f19.ini --legacy              # reproduce src_legacy/ (see below)
```

Options: `--device {auto,cpu,cuda,cuda:N,mps}`, `--dtype {float64,float32}`,
`--poisson {auto,thomas,eig}`, `--threads N`, `-path DIR`, `-n NAME`.

## Axis order

Arrays are `(z, x)` in 2D and `(z, y, x)` in 3D: the vertical, non-periodic axis first
and x last. Saved fields add time in front: `(t, z, x)` and `(t, z, y, x)`.
In 2D files the vertical coordinate keeps its historical name `y`, so 2D output
(`data.npz` with `u, v, T, Y, p, x, y, t`) has the same layout as before.
3D output is `u, v, w, T, Y, p, x, y, z, t`; `w` is the vertical velocity.

## Parameter files

The `.ini` files of `data/input` work unchanged. A file is 3D when it has `z_min`,
`z_max` and `Nz`. Unknown keys produce a warning instead of being ignored silently.
New keys: `[temperature] shape = line` (3D: the 2D Gaussian ignition extruded along
`y_start..y_end`), `[fuel] x_start, x_end, y_start, y_end` (extent of the fuel bed),
`[turbulence] enabled`, `[pressure] poisson`.

Only `experiment = fire` is ported (flat, hill and slope terrain). The cavity and
cylinder experiments still need `src_legacy/`.

## Differences from `src_legacy/`

With `--legacy` the 2D results match `src_legacy/` to round-off (relative difference about
1e-14 after 3000 steps of case F19). Without it, the following are corrected.

Numerical defects:

1. The pressure solver scaled the x wavenumbers with `x_max` instead of `x_max - x_min`.
2. The second-order upwind stencil for the two top rows returned minus the derivative
   for downward flow.
3. Temperature was advected with central differences, which oscillates at the fire
   front. The default is now a third-order upwind-biased reconstruction with the Koren
   flux limiter (`[numerics] T_advection = koren`): it creates no new extrema, so the
   bounds never act, and it is less diffusive than second-order upwind. The other
   options are `upwind` (second order, as for the velocity; it still undershoots by 16
   to 44 K) and `central` (the original). The bounds (`[numerics] bound`) stay on by
   default as a safeguard.
   With the original central scheme and forward Euler the run is only stable because of
   the clip (without it the temperature falls to 81 K and the run fails within 4 s on
   the paper case).

Model terms:

4. Reaction (`[fuel] reaction = exact`, or `explicit` for the original). The Arrhenius
   rate is far larger than `1/dt` in the flame, so the explicit update overshot and the
   bounds on `T` and `Y` did the integration. The rate is now averaged exactly over the
   step: `Y` decays as `exp(-Y_f K H dt)` and the heat released equals
   `H_C / (c_p Y_f)` times the fuel burned in the step.
5. Convective exchange (`[temperature] sink = surface`, or `legacy`). The original term
   was zero inside the fuel bed and acted on the first gas node above it against a solid
   temperature of 0 K, with a strength that shrank like dz^(3/4) under grid refinement.
   Now the top fuel node of each column (highest node with `Y > [fuel] threshold`,
   default 1e-3) exchanges `Q = h (T_g - T_s)` W/m^2 with the gas node above it, with
   `h = h_c (|T_g - T_s| / Y_h)^(1/4)`. `Q / (rho c_p dz)` is added to the fuel node and
   subtracted from the gas node, so energy is conserved and the total is grid independent.

Terrain (hill, slope):

6. Nodes inside the terrain take the temperature of the surface node above them
   instead of `T_inf`. The artificial density jump made the pressure fixed point
   diverge (the original hill case ends in NaN after about 110 steps).
7. The initial wind profile and the turbulence wall damping use the height above the
   terrain instead of the height above `z_min`.

8. The fuel fraction has no boundary condition. The original extrapolated it at the ground
   like the temperature, `Y_0 = (4 Y_1 - Y_2) / 3`, every step: where the bed burned
   unevenly in z, ground fuel disappeared without releasing heat or went negative and
   was reset by the bounds.

Other changes: boundary conditions for a time derivative are homogeneous (the original
wrote the Dirichlet values into the derivative, which only matters when they are not
zero); a run whose `Nt` is not a multiple of `NT` saves its last step instead of
failing; `t_source` holds the nodes where the ignition shape is above 1 % rather than
every node whose temperature differs from `T_inf`; constant density works.
New keys: `[temperature] T_inf`, `[numerics] bound`.

Items 3 to 8 change results and the model has not been recalibrated for them. On the
2.35 m grid of the paper case, upwind temperature advection slows the fire through
numerical diffusion; at 1.17 m the front position matches the original again.

RK2 and RK4 cannot run the paper case at dt = 1e-3 s with `--legacy` (also true of
`src_legacy/`): the explicit reaction drives the fuel negative inside the stages and the
run fails at step 4. The default exact reaction does not have this limit.

## Physics added on the `more-physics` branch

All of it is on by default and can be switched off to recover the previous model
(`[fuel] solid_temperature = False`, `[radiation] model = diffusion`). `--legacy` turns all of it off.

* **Radiation transport** (`[radiation] model = transport`): gray discrete ordinates, solved by
  marching row by row in z with all directions at once (`src/radiation.py`). The gas absorbs and
  emits with `kappa_gas` (1/m, to calibrate), the fuel elements with `alpha_s sigma_s Y / 4`. The
  ground is black at the temperature of the lowest gas row, the top sees ambient radiation.
  Keys: `kappa_gas`, `n_polar` (even, default 12), `n_azimuth` (even, default 16), `update_every`
  (time steps between solves, default 5).
* **Solid fuel temperature** (`[fuel] solid_temperature = True`): new fields `Ts` (fuel temperature)
  and `W` (moisture, kg water per kg dry fuel). The fuel exchanges heat with the gas by convection
  (forced convection of a cylinder of diameter `4/sigma_s`, or natural convection
  `h_c |T - Ts|^(1/3)`, whichever is larger) and radiation. Keys: `rho_bulk` (dry fuel per bed
  volume, default 0.6144 kg/m^3 from FDS Case_F19), `c_s`, `H_pyr`, `moisture` (F19: 0.058).
* **Pyrolysis** (`[fuel] pyrolysis`): `linear` (default, after Mell et al. 2007): dry fuel between
  `T_pyr` (400 K) and `T_pyr_end` (500 K) converts a growing share of the heat it receives into
  pyrolysis at `H_pyr` J/kg; `arrhenius`: the published rate `Y_f A exp(-T_act/Ts)` at the fuel
  temperature. With `linear`, `Y_f`, `A`, `T_act` and `T_pc` are not used.
* **Gas-phase combustion**: new transported fields `YF` (pyrolysis gas) and `YO2` (oxygen). The
  pyrolysis gas is added mass (it dilutes the species and expands the gas through the low-Mach
  constraint) and burns at the mixing-limited rate `rho min(YF, YO2/s_O2) / tau_mix`, releasing
  `H_C`. Keys in `[combustion]`: `s_O2` (1.185, cellulose), `tau_mix` (default `sqrt(2 Delta/g)`),
  `Y_O2_amb`.

Why the gas phase is needed: the published model releases `rho_gas H_C / Y_f` per unit of fuel
fraction burned, about 86 times less than the physical fuel load (`rho_bulk H_C`, about 13 MJ/m^3).
Released locally, the physical amount would heat the gas by tens of thousands of kelvin; burning
the pyrolysis gas only where it meets oxygen keeps peak temperatures near 2200 K.

Status: verified for consistency (radiation against an analytic slab, energy budget, 2D = 3D for
y-invariant cases, legacy mode unchanged); not yet calibrated or validated against F19. On the
2.35 m grid of the paper case the first 3 s give a head-fire speed of about 2.4 m/s.

## Output and post-processing

Every run folder `data/output/NAME/` contains `input.ini` (the parameter file as given),
`config.json` (all resolved parameters), `run_info.json` (git branch and commit, uncommitted
changes, command, library versions, device), `code/` (the `src/*.py` that ran), `log.txt`,
`parameters.pkl` and `data.npz`.

Fields: the state (`u`, `v`[, `w`], `T`, `Y`, and with the solid model `Ts`, `W`, `YF`, `YO2`), `p`,
and derived fields: `HRR` (heat release rate, W/m^3, as FDS HRRPUV), `pyrolysis` (kg/(m^3 s)),
`Q_conv` (convective heating of the fuel, W/m^3), `G` (incident radiation, W/m^2), `q_rad_gas` and
`q_rad_fuel` (net radiative heating, W/m^3), `nu_t` (turbulent viscosity, m^2/s); each only when the
corresponding model is on. `E_*`: the energy budget over time.

`data.npz`, every NT steps: 2D runs, full fields; 3D runs, three planes of every field (`xz_*` at
`[output] y_slice`, `xy_*` at `z_slice`, `yz_*` through the head fire, at `yz_x`), the fuel left per
column (`fuel_column`) and the full fields of the last output (`final_*`).

Full-field snapshots, 2D or 3D: `[output] full_interval = 5` writes all fields every 5 s to
`NAME/full/t_XXXXX.XXX.npz` (`full_dtype`, default float32). A 3D F19 grid of 256 x 64 x 511 takes
about 0.6 GB per snapshot in float32.

```bash
python src/visualize.py data/output/NAME [--field T|Ts|YF|YO2|speed|...] [--times 30 60 120] [--mp4]
python src/firefront.py data/output/RUN1 data/output/RUN2 --obs f19_front.csv --window 20 120
```

`visualize.py` writes snapshots and a GIF (MP4 with `--mp4`, needs ffmpeg) to `NAME/figures/`.
`firefront.py` writes the head and back fire positions, the rate of spread and, with `--obs` (CSV of
`t [s], x [m]`), the comparison with observations.

## Turbulence filter width

By default the Smagorinsky filter width is `(dx dz)^(1/2)` in 2D (`(dx dy dz)^(1/3)` in
3D), so the subgrid mixing length `C_s * width` shrinks as the grid is refined and the
solution does not converge: in the paper case the spread rate rose by about a third per
halving of dx, and at dx = 0.29 m the fire went out. With a fixed width, set in metres
with `[turbulence] filter_width`, the spread rate changed by about 6% between dx = 1.17
and 0.29 m. The width is then a model parameter to calibrate, like `Y_f`.

## Energy budget

Every saved step the log reports the cumulative heat of the fuel burned and, when the
bounds are on, the energy they add or remove: `clip low` (temperatures raised to
`T_min`), `clip high` (lowered to `T_max`) and `fuel overshoot` (heat released from
fuel that went negative and was reset to 0). The same values, in J (per metre of depth
in 2D), are saved in `data.npz` as `E_combustion`, `E_clip_low`, `E_clip_high` and
`E_fuel_overshoot`. With the defaults (Koren advection, exact reaction, no fuel
boundary condition) both `clip low` and `fuel overshoot` should be near zero. With
`--legacy`, `fuel overshoot` also counts the ground-row fuel resets of item 8, which
release no heat, so it overstates the real excess.

## Performance notes

* Choose `Nx` (and `Ny`) as a power of two plus one: the FFT runs on `Nx - 1` points.
* `float32` is not recommended: the pressure fixed point cannot reach `tol = 1e-10`
  and always runs `max_iter` iterations.
* `--poisson eig` is fastest for 2D and moderate 3D grids, `thomas` for large 3D
  grids; `auto` picks by size. Try both on a new machine.
