import numpy as np

H = 125
# Model bottom
A = 30
x0 = 200
y0 = 200
s = 5000
h = lambda x, y: A * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / s)
hx = lambda x, y: -2 * (x - x0) * h(x, y) / s
hy = lambda x, y: -2 * (y - y0) * h(x, y) / s
# h = lambda x, y: 0 * x
# hx = lambda x, y: 0 * x
# hy = lambda x, y: 0 * y

# Coordinate transformation
# Jacobian
G = lambda x, y, z: (H - h(x, y)) / H
G13 = lambda x, y, z: H * (z - H) * hx(x, y) / (H - h(x, y)) ** 2
G23 = lambda x, y, z: H * (z - H) * hy(x, y) / (H - h(x, y)) ** 2
Gx = lambda x, y, z: -hx(x, y) / H
Gy = lambda x, y, z: -hy(x, y) / H
Gz = lambda x, y, z: 0 * z

# Z change of variable
zc = lambda x, y, z: z * (H - h(x, y)) / H + h(x, y)