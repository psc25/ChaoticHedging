import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

d = 1

L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

dX = np.random.normal(size = [L, K-1, 1])*np.sqrt(dt)
X = np.concatenate([np.zeros([L, 1, 1]), np.cumsum(dX, axis = 1)], axis = 1)

K_strike = -0.5
G = np.fmax(X[:, K-1, 0] - K_strike, 0)

np.savetxt("bm/bm_X.csv", np.reshape(X, [L, -1]))
np.savetxt("bm/bm_G.csv", G)

plt.hist(X[:, -1, 0])
plt.hist(G)