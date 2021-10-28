import numpy as np
import matplotlib.pyplot as plt



filename = 'output/test.npz'
data = np.load(filename)

X = data['X']
Y = data['Y']
U = data['U']
V = data['V']
R = data['R']
T = data['T']
t = data['t']

for n in range(len(t)):
#for n in [0]:
    plt.figure(figsize=(12,6))
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, T[n], cmap=plt.cm.jet, vmin=np.min(T), vmax=np.max(T))
    plt.quiver(X[::2,::2], Y[::2,::2], U[n,::2,::2], V[n,::2,::2], cmap=plt.cm.jet)
    plt.colorbar()
    plt.title('T')
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, R[n], cmap=plt.cm.Oranges, vmin=np.min(R), vmax=np.max(R))
    plt.colorbar()
    plt.title('R')
    plt.tight_layout()
    plt.show()