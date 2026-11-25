# Numerical simulations

## Current model (`src/`)

### CSIRO grassland fire F19, 3D (stretched vertical grid, 140 s)

Plan view, gas temperature at 1 m height (white: half of the fuel burned):

![F19 3D plan](./simulations/3D/f19_3d_plan.gif)

Side view through the center of the fire:

![F19 3D side](./simulations/3D/f19_3d_side.gif)

### C064 with four burning houses, 3D (FDS structures case, 120 s)

Gas temperature at 2 m height. The upwind houses ignite at 35 s (FDS: 32-34 s); the downwind ones ignite later
than in FDS, because the grass fire does not spread sideways behind the upwind houses.

![C064 houses](./simulations/3D/c064_houses_3d_plan.gif)

### Two houses 20 m apart, 2D (120 s)

The flame of the first house ignites the second one 39 s later; the grass in the gap burns afterwards from the
wake of the first house.

![Two houses](./simulations/2D/houses_2d_two.gif)

## Published 2D model (`src_legacy/`, `--legacy`)

### Flat terrain

#### Case F19

![Wind](./simulations/2D/case_f19.gif)

### Topography

#### Triangular hill

![Slope](./simulations/2D/triangular_hill.gif)

#### Gaussian hill

![Hill](./simulations/2D/gaussian_hill.gif)

### Validation

#### Rayleigh–Bénard convection

##### Temperature difference $30$ K
![Rayleigh-Benard-1](./simulations/2D/rb_1.gif)

##### Temperature difference $100$ K
![Rayleigh-Benard-2](./simulations/2D/rb_2.gif)