import numpy as np
import matplotlib.pyplot as plt

L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

dX = np.random.normal(size = [L, K-1])*np.sqrt(dt)
X = np.zeros([L, K])
X[:, 1:] = np.cumsum(dX, axis = 1)

K_strike = -0.5
G = np.fmax(X[:, K-1] - K_strike, 0)

np.savetxt("bm/bm_X.csv", X, delimiter = ";")
np.savetxt("bm/bm_G.csv", G, delimiter = ";")

plt.hist(X[:, -1])
plt.hist(G)