import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

d = 1

L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

dB = np.random.normal(size = [L, K-1, 2])*np.sqrt(dt)
B = np.concatenate([np.zeros([L, 1, 2]), np.cumsum(dB, axis = 1)], axis = 1)

X0 = [100.0, 4.0]
kappa = 0.5
mu = 100.0
alpha = 1.0
beta = 5.0
xi = 1.0
rho = -0.7
K_strike = 101.0

Apar0 = np.zeros([2, 2])
Apar1 = np.zeros([2, 2])
Apar2 = np.reshape(np.array([1.0, xi*rho, xi*rho, xi**2]), [2, 2])
Apar = np.stack([Apar0, Apar1, Apar2])
np.savetxt("vs/vs_par.csv", np.reshape(Apar, [3, -1]))

X = np.zeros([L, K, 2])
X[:, 0, 0] = X0[0]
X[:, 0, 1] = X0[1]
for k in range(1, K):
    X[:, k, 0] = X[:, k-1, 0] + kappa*(mu - X[:, k-1, 0])*dt + np.sqrt(X[:, k-1, 1])*dB[:, k-1, 0]
    X[:, k, 1] = X[:, k-1, 1] + alpha*(beta - X[:, k-1, 1])*dt + xi*np.sqrt(X[:, k-1, 1])*(rho*dB[:, k-1, 0] + np.sqrt(1.0-rho**2)*dB[:, k-1, 1])

b = np.concatenate([kappa*(mu - X[:, :, 0:1]), alpha*(beta - X[:, :, 1:2])*dt], axis = -1)
sigma = np.reshape([1.0, 0.0, xi*rho, xi*np.sqrt(1-rho**2)], [1, 1, 2, 2])*np.expand_dims(np.sqrt(X[:, :, 1:2]), -1)
a = np.matmul(sigma, np.transpose(sigma, (0, 1, 3, 2)))
lam = np.einsum('mtij,mtj->mti', np.linalg.inv(a), b)

Z = np.zeros([L, K])
Z[:, 0] = 1.0
for k in range(1, K):
    Z[:, k] = Z[:, k-1]*(1.0 - np.sum(np.matmul(lam[:, (k-1):k], sigma[:, k-1])*dB[:, (k-1):k], axis = (1, 2)))

I = np.sum(X[:, :, 0], axis = 1)*dt/T
G = np.fmax(K_strike - I, 0)
print(np.mean(G))
print(np.mean(Z[:, -1]*G))

np.savetxt("vs/vs_X.csv", np.reshape(X[:, :, 0], [L, -1]))
np.savetxt("vs/vs_sigma.csv", np.reshape(X[:, :, 1], [L, -1]))
np.savetxt("vs/vs_G.csv", G)

plt.hist(X[:, -1, 0])
plt.hist(G)
plt.show()