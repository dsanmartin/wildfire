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
