"""This file handles the default parameters of the model
"""

# Domain [x_min, x_max] \times [y_min, y_max] \times [t_min, t_max] 
x_min, x_max = -500, 700 # Distance interval [x_min, x_max] in m 
y_min, y_max = 0, 20 # Distance interval [y_min, y_max] in m
t_min, t_max = 0, 140 # Time interval [t_min, t_max] in s
# Numerical grid
Nx, Ny, Nt = 512, 256, 30001 # Number of nodes per axis
NT = 100 # Number of samples to store. The simulation stores each NT timesteps

# Time numerical method
method = 'RK4'

# Constants
R = 8.31446261815324 # Universal gas constant in J mol^{-1} K^{-1} or kg m^2 s^{-2} mol^{-1} K^{-1}
sigma = 5.670374419e-8 # Stefan-Boltzmann constant in W m^{-2} K^{-4} or kg s^{-3} K^{-4}

# Gas properties
C_p = 1005 # Specific heat capacity (constant pressure) in J kg^{-1} K^{-1} or m^2 s^{-2} K^{-1} (Air: 1005)
C_p = 1007 # Specific heat capacity (constant pressure) in J kg^{-1} K^{-1} or m^2 s^{-2} K^{-1} (Air: 1007 at 15 °C, NASA - Cengel 2018)
C_V =  717 # Specific heat capacity (constant volume) in J kg^{-1} K^{-1} or m^2 s^{-2} K^{-1} (Air: 717)
rho = 1.293 # Density in kg m^{-3} (Air: 1.293)
rho = 1.229 # Density in kg m^{-3} (Air: 1.229 at 15 °C - NASA)
rho = 1.225 # Density in kg m^{-3} (Air: 1.225 at 15 °C - Cengel 2018)
mu = 1.73e-5 # Dynamic viscosity in kg m^{-1} s^{-1} (Air: 1.73e-5)
mu = 1.802e-5 # Dynamic viscosity in kg m^{-1} s^{-1} (Air: 1.802e-5, Cengel 2018)

# Temperaure equation parameters
kappa = 0.024 # Thermal conductivity in W m^{-1} K^{-1} or kg m s^{-3} K ^{-1}  (Air: 0.024)
kappa = 0.02476 # Thermal conductivity in W m^{-1} K^{-1} or kg m s^{-3} K ^{-1}  (Air: 0.02476 at 15 °C, NASA - Cengel 2018)
k = 2.07e-5 # Thermal diffusivity in m^2 s^{-1} (Air: 2.07e-5) 
k = 2.009e-5 # Thermal diffusivity in m^2 s^{-1} (Air: 2.009e-5 at 15 °C - NASA)
k = kappa / (rho * C_p) # Thermal diffusivity in m^2 s^{-1}
delta = 1 # Optical path length in m

# Fluid equations parameters
nu = 1.5e-5 # Kinematic viscosity in m^2 s^{-1} (Air: 1.5e-5)
nu = mu / rho # Kinematic viscosity in m^2 s^{-1} (Air: 1.47e-5, Cengel 2018)
T_inf = 293.15 # Temperature of the environment in K (Ambient temperature: 20°C - 293.15 K)
T_inf = 288.15 # Temperature of the environment in K (Ambient temperature: 15°C - 288.15 K, NASA - Cengel 2018)
g = (0, -9.81) # Acceleration due to gravity in m s^{-2} (Typical value: 9.81 m s^{-2})
turb = True # Turbulence model
conser = False # Conservative equation for convection
C_s = 0.173 # Smagorinsky constant
C_s = 0.2 # Smagorinsky constant (McGrattan 2023)
Pr = nu / k # Prandtl number 1. / (Air: ~.74)
C_D = 1 # Drag coefficient "1 or near to unity according to works of Mell and Linn" 1
a_v = 5.508 #6000 #1 #  Contact area per unit volume between the gas and the solid in m

# Fuel and reaction parameters
T_pc = (573 + 473) / 2 # Temperature of solid-gas phase change in K. (473 - 573 K)
H_R = 21.2e6 # Heat energy per unit of mass (wood) in J kg^{-1} or m^2 s^{-2}. About 21.20e6 for wood according to https://en.wikipedia.org/wiki/Heat_of_combustion
# H_R = 15.6e6 # 15.6 (Mell 2007 - FDS)
# H_R = 19.4e6 # 19.4 (Dupuy 2011 - FIRETEC)
A = 1e9 # Pre-exponential factor in s^{-1}. (1e9, Asensio 2002)
n_arrhenius = 0 # Arrhenius-like parameter in 1. 1
E_A = 150e3 # Activation energy in J mol^{-1} or kg m^2 s^{-2} mol^{-1}. E_A = 20e3 cal mol^{-1} according to (Asensio 2002).
T_act = E_A / R # Activation temperature in K 
h = 1.147#3.3#18 # Convection coefficient in W m^{-2} K^{-1} or kg s^{-3} K^{-1}  (Air: 0.5-1000), (15.9 - 18.2, Maragkos 2021)
h_rad = 0*1e-7 #
Y_thr = 0.025 #.25 #.25 #.9 # Threshold to add solid fuel force
Y_f = 1e2 # Extra parameter to control the fuel consumption rate
Y_f = 100
T_hot = T_inf + 500 #500 #600 #450 #Temperature of fire in K
S_top = 3000 #3384 #S(800,1) ~ 3384 
S_bot = S_top
Sx = -1
include_source = True
source_filter = False
radiation = False
sutherland_law = False
debug_pde = False
bound = True
T_min, T_max = T_inf, 1500
# T_min, T_max = -10000, 10000
Y_min, Y_max = 0, 1
# Temperature source for t time
t_source = -1 # If t_source < 0, then the temperature source is not used

### Initial conditions parameters ###
# Wind
initial_u_type = 'power law' # 'power law' or 'log'
# Log wind profile
u_z_0 = 0.05 # Surface roughness in m
d = 0 #0.1 # Zero-plane displacement in m
u_ast = .1#.5 # .1 Friction velocity in ms^{-1}
kappa = 0.41 # Von Karman constant in 1
# Power law (used by FDS)
u_r = 4.8 # Reference speed in ms^{-1}
y_r = 2 # Reference height in m
alpha = 1 / 7 # Empirical constant in 1

# Temperature 
# Shape
T0_shape = 'half gaussian' # 'plate' or 'half gaussian'
# Location
# 'Width' of initial fire source in m
T0_x_start = 0
T0_x_end = T0_x_start + 6 #4#2 # PLATE -> x_start + 3.3 (FDS)
# T0_x_start = (x_max + x_min) / 2 - 2.5
# T0_x_end = (x_max + x_min) / 2 + 2.5
T0_x_center = (T0_x_start + T0_x_end) / 2
T0_width = (T0_x_end - T0_x_start)
# 'Height' of initial fire source in m
T0_y_start = 0
T0_y_end = 1 # PLATE -> .25
T0_height = (T0_y_end - T0_y_start)

# Fuel
fuel_height = .51 #.5 # Height of fuel in m

# Topography
topography_shape = 'flat' # 'flat' or 'hill'
hill_center = 100 / 2 # Center of hill in m
hill_height = 2.5 # Height of hill in m
hill_width = 20 # Width of hill in m

### Immerse boundary method parameterns ###
u_dead_nodes = 0
v_dead_nodes = 0
T_dead_nodes = T_inf
Y_dead_nodes = 1

# Sutherland's law parameters
S_T_0 = 273 # Reference temperature in K
S_k_0 = 0.024 # Thermal conductivity in W m^{-1} K^{-1} or kg m s^{-3} K ^{-1}  (Air: 0.024)
S_k = 194 # Sutherland's constant in K
