"""
This file handles the parameters of the model
"""

### DEFAULT VALUES ###

# Domain [x_min, x_max] \times [y_min, y_max] \times [t_min, t_max] 
x_min, x_max = -50, 250 # Distance interval [x_min, x_max] in m 
y_min, y_max = 0, 50 # Distance interval [y_min, y_max] in m
t_min, t_max = 0, 140 # Time interval [t_min, t_max] in s
x_min, x_max = -500, 700 # Distance interval [x_min, x_max] in m 
# x_min, x_max = 0, 1000 # Distance interval [x_min, x_max] in m 
y_min, y_max = 0, 50 # Distance interval [y_min, y_max] in m
t_min, t_max = 0, 30 # Time interval [t_min, t_max] in s

# Numerical grid
# Nx, Ny, Nt = 512, 512, 50001 # Number of nodes per axis
Nx, Ny, Nt = 128, 128, 20001 # Number of nodes per axis
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
C_p = 1005 # Specific heat capacity (constant pressure) in J kg^{-1} K^{-1} or m^2 s^{-2} K^{-1} (Air: 1005)
C_v =  717 # Specific heat capacity (constant volume) in J kg^{-1} K^{-1} or m^2 s^{-2} K^{-1} (Air: 717)

# Fuel and reaction parameters
T_pc = (573 + 473) / 2 #500 # Temperature of solid-gas phase change in K. (473 - 573 K)
# H = 21.20e6 #200 # Heat energy per unit of mass (wood) in J kg^{-1} or m^2 s^{-2}. About 21.20e6 for wood according to https://en.wikipedia.org/wiki/Heat_of_combustion
# H_R = H #/ C_p 
H_R = 21.2e6 #250000 #100000 
# R = 1.9872 # Universal gas constant in cal mol^{-1} K^{-1}. 1.9872 in (Asensio 2002) or 8.314 in J mol^{-1} K^{-1}. Included in 'B' parameter
A = 1e9 #2e6#2.5e3#1e9 # Pre-exponential factor in s^{-1}. 1e9 According to Asensio 2002
A_T = 300 # A_T hard-coded value for A(T)
# Modificar
E_A = 150e3#85e3#85e3#83e3#83e3#115e3 # Activation energy in J mol^{-1} or kg m^2 s^{-2} mol^{-1}. E_A = 20e3 cal mol^{-1} according to (Asensio 2002).
R = 8.31446261815324 # Universal gas constant in J mol^{-1} K^{-1} or kg m^2 s^{-2} mol^{-1} K^{-1}
T_act = E_A / R
#B = 300#100 # Activation energy and universal gas constant. 
h = 18 #1000 #100#25#.5 # Convection coefficient in W m^{-2} K^{-1} or kg s^{-3} K^{-1}  (Air: 0.5-1000), (15.9 - 18.2 according to Maragkos)
Y_thr = 0.5 # Threshold to add solid fuel force
Y_f = .05 #0.05 / 2 #.0125 / .5 #.075 #.05 0.025 # Extra parameter to control the fuel consumption
T_hot = T_inf + 800# 1500 # 300 Temperature of fire in K 300
S_top = 250 #1000 / 2
debug_pde = False

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
x_start = .5
x_end = x_start + 4#2 # PLATE -> x_start + 3.3 (FDS)
# x_start = 100
# x_end = 105
# 'Height' of initial fire source in m
y_start = 0
y_end = 1 # PLATE -> .25
# Fuel
fuel_height = .5 # Height of fuel in m
# Topography
# if flat, none
hill_center = 50 # Center of hill in m
hill_height = 2.5 # Height of hill in m
sx = 20

### Immerse boundary method parameterns ###
u_dead_nodes = 0
v_dead_nodes = 0
T_dead_nodes = T_inf
Y_dead_nodes = 1

# Source/sink bounds
# S_top, S_bot, Sx = 500, 100, 1000 #
# S_top, S_bot, Sx = 500, 200, 400 #600 # 700
# S_top, S_bot, Sx = 500, 100, 400 # Nice for t=60s, Nt=5001
# S_top, S_bot, Sx = 500, 100, 50 # Nice for t=
# S_top, S_bot, Sx = 500, 100, 250 # Nice for t=
# S_top, S_bot, Sx = 10000, 200, 200
S_bot, Sx = -1, -1

# A(T) fitted parameters #
# # h=1000, E_A=85e3, filter=(Se >= 0) & (Se <= 500) & (T > T_pc)
# A_alpha = -16.777128 #-16.579513 #-16.756939 #-16.656926 #-13.558528 #-13.431608 
# B_tilde = 120.560762 #118.762667 #120.201488 #119.426307 #99.691432 #98.536655 
# # h=1000, E_A=85e3, filter=(Se >= 0) & (Se <= 100) & (T > T_pc) -- NO -- 
# A_alpha = -16.579513
# B_tilde = 118.762667
# # h=40, E_A=85e3, filter=(Se >= 0) & (Se <= 500) & (T > T_pc), 
# # With Y_f=10 - OK, but fire ends. Y_f=1 it explodes. Y_f=5 OK, Y_f=4 better. Y_f=4.5 pretty good (up to t=30s)
# A_alpha = -14.345368
# B_tilde = 104.674068
# # h=1000, E_A=85e3, filter=(Se >= 0) & (Se <= 250) & (T > T_pc). 
# # Y_f=1 Ok,
# A_alpha = -16.780480
# B_tilde = 120.282809
# # h=40, E_A=85e3, filter=(Se >= 0) & (Se <= 1000) & (T > T_pc)
# # A_alpha = -15.177128
# # B_tilde = 110.509900
# # # h=17, E_A=85e3, filter=(Se >= 0) & (Se <= 1000) & (T > T_pc) - IT DOES NOT WORK
# # A_alpha = -15.184551
# # B_tilde = 110.553633
# # h=40, E_A=85e3, filter=(Se >= 0) & (Se <= 250) & (T > T_pc)
# # Y_f=2 works up to t=30s
# # Y_f=2.5 it explodes
# # Y_f=3 works up to t=60s (but fire ends)
# # A_alpha = -13.161880
# # B_tilde = 96.678002
# # Y_f = 2
# # h=17, E_A=100e3, filter=(Se >= 0) & (Se <= 250) & (T > T_pc). Fire ends
# # A_alpha = -20.091293
# # B_tilde = 143.318524
# # h=17, E_A=85e3, filter=(Se >= -250) & (Se <= 250) & (T > T_pc) & (A > 1e4). It does not work
# # A_alpha = -13.070216
# # B_tilde = 96.084337