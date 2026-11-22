# Documentation

- [wildfire_model.tex](wildfire_model.tex) ([PDF](wildfire_model.pdf)): the model implemented in
  `src/`, i.e. the governing equations of the published single-temperature model and of the
  extended model (solid fuel temperature, moisture, pyrolysis, gas-phase combustion, radiation
  transport), the closures, the initial and boundary conditions (including the terrain immersed
  boundary), the numerical methods (finite differences, advection schemes, time integration,
  projection and pressure solver, radiation solver, safeguards and energy budget), all input
  parameters, and the known limitations.

Build the PDF with `latexmk -pdf wildfire_model.tex`.

Practical use (running, output files, plotting) is described in [src/README.md](../src/README.md).
