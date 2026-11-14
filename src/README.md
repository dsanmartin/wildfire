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
   front (the `T_min` bound hid undershoots of more than 100 K). It now uses the same
   second-order upwind scheme as the velocity.

Model terms:

4. Reaction (`[fuel] reaction = exact`, or `explicit` for the original). The Arrhenius
   rate is far larger than `1/dt` in the flame, so the explicit update overshot and the
   bounds on `T` and `Y` did the integration. The rate is now averaged exactly over the
   step: `Y` decays as `exp(-Y_f K H dt)` and the heat released equals
   `H_C / (c_p Y_f)` times the fuel burned in the step.
5. Convective sink (`[temperature] sink = ambient`, or `legacy`). The original term was
   zero inside the fuel bed and acted in the node above it against 0 K. It is now
   natural convection from the fuel to the ambient air,
   `-h alpha_s sigma_s Y (T - T_inf) / (rho c_p)` with `h = h_c (|T - T_inf| / Y_h)^(1/4)`.

Terrain (hill, slope):

6. Nodes inside the terrain take the temperature of the surface node above them
   instead of `T_inf`. The artificial density jump made the pressure fixed point
   diverge (the original hill case ends in NaN after about 110 steps).
7. The initial wind profile and the turbulence wall damping use the height above the
   terrain instead of the height above `z_min`.

Other changes: boundary conditions for a time derivative are homogeneous (the original
wrote the Dirichlet values into the derivative, which only matters when they are not
zero); a run whose `Nt` is not a multiple of `NT` saves its last step instead of
failing; `t_source` holds the nodes where the ignition shape is above 1 % rather than
every node whose temperature differs from `T_inf`; constant density works.
New keys: `[temperature] T_inf`, `[numerics] bound`.

Items 3 to 7 change results and the model has not been recalibrated for them.

## Performance notes

* Choose `Nx` (and `Ny`) as a power of two plus one: the FFT runs on `Nx - 1` points.
* `float32` is not recommended: the pressure fixed point cannot reach `tol = 1e-10`
  and always runs `max_iter` iterations.
* `--poisson eig` is fastest for 2D and moderate 3D grids, `thomas` for large 3D
  grids; `auto` picks by size. Try both on a new machine.
