# Wildfire
Coupled Atmosphere-Wildfire Model Implementation

## Mathematical model
This code solves the following system of PDEs 

```math
\begin{split}
    \nabla\cdot\mathbf{u} &= 0 \\
    \dfrac{\partial \mathbf{u}}{\partial t} + \left(\mathbf{u}\cdot\nabla\right)\mathbf{u} &= -\dfrac{1}{p} + \nu\nabla^2\mathbf{u} + \mathbf{f}(\mathbf{u}, T) \\
    \dfrac{\partial T}{\partial t} + \mathbf{u}\cdot\nabla T &= k\nabla^2T + S(T, Y) \\
    \dfrac{\partial Y}{\partial t} &= -Y_{\text{f}}\,K(T) \\
    & + \text{Initial and boundary conditions},
\end{split}
```
to simulate the wildfires spreading.

## Examples

### Flat terrain
![Flat fire](./examples/simulations/2d/flat.gif)

### Simple hill
![Hill fire](./examples/simulations/2d/hill.gif)
