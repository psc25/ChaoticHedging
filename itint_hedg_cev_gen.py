import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

d = 1
n = 500
T = 1
dt = T/n
M = 10000
tt = np.linspace(0, T, n)

dB = np.random.normal(size = [M, n-1, d])*np.sqrt(dt)
B = np.zeros([M, n, d])
B[:, 1:, :] = np.cumsum(dB, axis = 1)

alpha = -0.02
sigma = 0.4

X = np.zeros([M, n, d])
X[:, 0] = 100
for t in range(1, n):
    X[:, t] = X[:, t-1] + alpha*X[:, t-1]*dt + sigma*np.sqrt(X[:, t-1])*dB[:, t-1]

lam = alpha/sigma*np.sqrt(X)
    
Z = np.zeros([M, n, d])
Z[:, 0] = 1
for t in range(1, n):
    Z[:, t] = Z[:, t-1]*(1 - lam[:, t-1]*dB[:, t-1])

K = 102
Y = np.fmax(K - np.sum(X, axis = 1)*dt/T, 0)
print(np.mean(Y))

print(np.mean(Z[:, -1]*(K > np.sum(X, axis = 1)*dt/T)))
print(np.mean(Z[:, -1]*np.sum(X, axis = 1)*dt/T*(K > np.sum(X, axis = 1)*dt/T)))

t1 = 400
K1 = 20
X1 = X - X[:, t1:(t1+1)] + 100
print(np.mean(Z[:, -1]*(K1 > np.sum(X1[:, t1:], axis = 1)*dt/T)))
print(np.mean(Z[:, -1]*np.sum(X1[:, t1:], axis = 1)*dt/T*(K1 > np.sum(X1[:, t1:], axis = 1)*dt/T)))

N = 2048
uMax = 200.0
u0 = 1e-8
jj = np.reshape(np.arange(N), (-1, 1))
du = (uMax+u0)/(N-1)
u = u0 + jj*du
u = u.flatten()

k = 10
print(np.mean(Z[:, -1]*np.exp(u[k]*1j*np.sum(X, axis = 1)*dt/T)))
print(np.mean(Z[:, -1]*np.sum(X, axis = 1)*dt/T*np.exp(u[k]*1j*np.sum(X, axis = 1)*dt/T)))
print(np.mean(Z[:, -1]*np.exp(u[k]*1j*np.sum(X1[:, t1:], axis = 1)*dt/T)))
print(np.mean(Z[:, -1]*np.sum(X1[:, t1:], axis = 1)*dt/T*np.exp(u[k]*1j*np.sum(X1[:, t1:], axis = 1)*dt/T)))

np.savetxt("itint_hedg_cev_x.csv", np.reshape(X, (M, n*d)), delimiter = ";")
np.savetxt("itint_hedg_cev_y.csv", Y, delimiter = ";")

plt.hist(X[:, -1, 0])
plt.hist(Y)
plt.show()