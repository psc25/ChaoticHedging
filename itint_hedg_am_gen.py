import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

d = 10
n = 500
T = 1
dt = T/n
M = 50000
tt = np.linspace(0, T, n)

Bpar = [np.random.normal(size = [1, d], loc = 0.0, scale = np.sqrt(0.003))]
for k in range(d):
    B1 = np.random.normal(size = [1, d], loc = 0.0, scale = np.sqrt(0.0003))
    Bpar.append(B1)

Bpar2 = np.stack(Bpar, axis = 1)[0]
np.savetxt("itint_hedg_am_b.csv", Bpar2, delimiter = ";")

sigma0 = np.random.normal(size = [1, d], loc = 0.0, scale = np.sqrt(0.03))
Apar = [np.matmul(np.transpose(sigma0), sigma0)]
for k in range(d):
    sigma1 = np.random.normal(size = [1, d], loc = 0.0, scale = np.sqrt(0.003))
    Apar.append(np.matmul(np.transpose(sigma1), sigma1))

Apar2 = np.stack(Apar)
np.savetxt("itint_hedg_am_a.csv", np.reshape(Apar2, (d+1, np.square(d))), delimiter = ";")

dB = np.random.normal(size = [M, n-1, d])*np.sqrt(dt)
B = np.zeros([M, n, d])
B[:, 1:, :] = np.cumsum(dB, axis = 1)

X = np.zeros([M, n, d])
X0 = 10
X[:, 0, :] = X0
ones1 = np.ones([M, 1])
for t in range(1, n):
    print("Proceeding t = " + str(t+1) + "/" + str(n))
    X1 = np.concatenate((ones1, X[:, t-1]), axis = 1)
    drift = np.matmul(X1, Bpar2)
    diffu = np.linalg.cholesky(np.einsum('mk,kij->mij', X1, Apar2))
    X[:, t] = X[:, t-1] + drift*dt + np.matmul(diffu, np.expand_dims(dB[:, t-1], -1))[:, :, 0]

plt.plot(X[0])
plt.show()

print(np.sum(np.isnan(X)))
print(np.min(X))
print(np.max(X))

K = 4.0
w = np.reshape(np.array([1.0, -0.95, 0.9, -0.85, 0.8, -0.75, 0.7, -0.65, 0.6, -0.55]), (1, d))
Y = np.reshape(np.fmax(K - np.sum(w*X[:, n-1, :], axis = 1), 0), (-1, 1))

plt.hist(Y)
plt.show()

np.savetxt("itint_hedg_am_x.csv", np.reshape(X, (M, n*d)), delimiter = ";")
np.savetxt("itint_hedg_am_y.csv", Y, delimiter = ";")