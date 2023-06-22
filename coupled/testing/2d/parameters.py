"""
This file handles the parameters of the model
"""

### DEFAULT VALUES ###

# Domain [x_min, x_max] \times [y_min, y_max] \times [t_min, t_max] 
x_min, x_max = -50, 250 # Distance interval [x_min, x_max] in m 
y_min, y_max = 0, 50 # Distance interval [y_min, y_max] in m
t_min, t_max = 0, 140 # Time interval [t_min, t_max] in s
# x_min, x_max = -200, 400 # Distance interval [x_min, x_max] in m 
y_min, y_max = 0, 20 # Distance interval [y_min, y_max] in m
t_min, t_max = 0, 10 # Time interval [t_min, t_max] in s

# Numerical grid
# Nx, Ny, Nt = 512, 512, 50001 # Number of nodes per axis
Nx, Ny, Nt = 512, 512, 10001 # Number of nodes per axis
NT = 100 # Number of samples to store. The simulation stores each NT timesteps

# Time numerical method
method = 'RK4'

# Temperaure equation parameters
kk = 0.024 # Thermal conductivity in W m^{-1} K^{-1} or kg m s^{-3} K ^{-1}  (Air: 0.024)
k = 2.07e-5 # 1e1 # Thermal diffusivity in m^2 s^{-1} (Air: 2.07e-5) # kk / (rho * C_p)

# Fluid equations parameters
nu = 1.5e-5 #1e0 # Viscosity in m^2 s^{-1} (Air: 1.5e-5)
rho = 1.293 # 1 Density in kg m^{-3} (Air: 1.293)
T_inf = 293 # Temperature of the environment in K
g = (0, -9.81) # Acceleration due to gravity in m s^{-2}
turb = True # Turbulence model
conser = False # Conservative equation for convection
C_s = 0.173 # Smagorinsky constant
Pr = nu / k # Pr = 1e1 # Prandtl number 1. / (Air: .74)
C_D = 1 # Drag coefficient "1 or near to unity according to works of Mell and Linn" 1
a_v = 1 #  Contact area per unit volume between the gas and the solid in m
C_p = 1005 # Specific heat capacity in J kg^{-1} K^{-1} (Air: 1005)
C_v =  717 # Specific heat capacity in J kg^{-1} K^{-1} (Air: 717)

# Fuel and reaction parameters
T_pc = 400 # Temperature of solid-gas phase change in K.
H = 21.20e6 #200 # Heat energy per unit of mass (wood) in J kg^{-1} or m^2 s^{-2}. About 21.20e6 for wood according to https://en.wikipedia.org/wiki/Heat_of_combustion
H_R = H / C_p 
H_R = 250000 #100000 
# R = 1.9872 # Universal gas constant in cal mol^{-1} K^{-1}. 1.9872 in (Asensio 2002) or 8.314 in J mol^{-1} K^{-1}. Included in 'B' parameter
A = 1 # Pre-exponential factor in s^{-1}. 1e9 According to Asensio 2002
E_A = 30000 # Activation energy in J mol^{-1} or kg m^2 s^{-2} mol^{-1}. E_A = 20e3 cal mol^{-1} according to (Asensio 2002).
R = 8.31446261815324 # Universal gas constant in J mol^{-1} K^{-1} or kg m^2 s^{-2} mol^{-1} K^{-1}
B = E_A / R
B = 300#100 # Activation energy and universal gas constant. 
h = 100#.5 # Convection coefficient in W m^{-2} K^{-1} or kg s^{-3} K^{-1}  (Air: 0.5-1000)
Y_thr = 0.5 # Threshold to add solid fuel force
Y_f = .025 #.075 #.05 0.025 # Extra parameter to control the fuel consumption
TA = T_inf + 500 # 300 Temperature of fire in K 300

### Initial conditions parameters ###
# Wind
# Log wind profile
z_0 = 0.05 # Surface roughness in m
d = 0#0.1 # Zero-plane displacement in m
u_ast = .1#.5 # .1 Friction velocity in ms^{-1}
kappa = 0.41 # Von Karman constant in 1
# Power law (used by FDS)
u_r = 4.9 # Reference speed in ms^{-1}
y_r = 2 # Reference height in m
alpha = 1 / 7 # Empirical constant in 1
# Temperature location
# 'Width' of initial fire source in m
x_start = 0
x_end = 2 # PLATE -> x_start + 3.3 (FDS)
# 'Height' of initial fire source in m
y_start = 0
y_end = 1 #1 # PLATE -> .25

### Immerse boundary method parameterns ###
u_dead_nodes = 0
v_dead_nodes = 0
T_dead_nodes = T_inf
Y_dead_nodes = 1
