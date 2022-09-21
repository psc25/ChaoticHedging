import numpy as np
import matplotlib.pyplot as plt

d = 1
n = 500
T = 1
dt = T/n
M = 10000
tt = np.linspace(0, T, n)

dB = np.random.normal(size = [M, n-1, d])*np.sqrt(dt)
B = np.zeros([M, n, d])
B[:, 1:, :] = np.cumsum(dB, axis = 1)

X0 = 0
X = B + X0

K = -0.5
Y = np.fmax(X[:, n-1, 0] - K, 0)
Y = np.reshape(Y, (-1, 1))

np.savetxt("itint_hedg_bm_x.csv", np.reshape(X, (M, n*d)), delimiter = ";")
np.savetxt("itint_hedg_bm_y.csv", Y, delimiter = ";")

plt.hist(X[:, -1, 0])
plt.hist(Y)