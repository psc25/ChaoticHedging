import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)

d = 2

L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

kappa = 1.0
mu = 10.0*np.ones([d, 1])
beta = d+1 # needs to be an integer and > d-1
U = 0.05*np.diag(np.random.normal(size = [d, 1])) + 0.02*np.random.normal(size = [d, d])
print(np.linalg.det(U)) # check that U is invertible
V = 0.1*np.diag(np.random.normal(size = [d, 1])) + 0.05*np.random.normal(size = [d, d])
rho = np.random.normal(size = [d, 1])
rho = rho/np.linalg.norm(rho)*np.random.uniform(low = 0.1, high = 0.9)

np.savetxt("ad/ad_beta.csv", np.array([beta]))
np.savetxt("ad/ad_U.csv", U)
np.savetxt("ad/ad_V.csv", V)
np.savetxt("ad/ad_rho.csv", rho)

X = np.zeros([L, K, d])
X[:, 0, :] = 10.0
Sigma = np.zeros([L, K, d, d])
S = np.zeros([L, K, d, beta])
S0 = 0.2*np.random.normal(size = [d, beta])
ind_diag = np.diag_indices(2)
S0[ind_diag] = S0[ind_diag] + 0.6*np.diag(np.random.normal(size = [d, 1]))
S[:, 0] = np.expand_dims(S0, 0)
Sigma[:, 0] = np.matmul(S[:, 0], np.transpose(S[:, 0], [0, 2, 1]))
np.savetxt("ad/ad_S0.csv", np.reshape(S[0, 0], [d, beta]))

Z = np.random.normal(size = [L, K-1, d, 1])*np.sqrt(dt)
E = np.random.normal(size = [L, K-1, d, beta])*np.sqrt(dt)
mu1 = np.transpose(mu)
rho1 = np.expand_dims(rho, 0)
rho2 = np.sqrt(1.0 - np.linalg.norm(rho)**2)
V1 = np.expand_dims(V, 0)
U1 = np.expand_dims(np.transpose(U), 0)
for k in range(1, K):
    print("Proceeding t = " + str(k+1) + "/" + str(K))
    S[:, k] = S[:, k-1] + np.matmul(V1, S[:, k-1])*dt + np.matmul(U1, E[:, k-1])
    Sigma[:, k] = np.matmul(S[:, k], np.transpose(S[:, k], [0, 2, 1]))
    drift = kappa*(mu1 - X[:, k-1])*dt
    diffu1 = np.matmul(np.matmul(S[:, k-1], np.transpose(E[:, k-1], [0, 2, 1])), rho1)[: , :, 0]
    diffu2 = rho2*np.matmul(np.linalg.cholesky(Sigma[:, k-1] + Sigma[:, k])/np.sqrt(2.0), Z[:, k-1])[: , :, 0]
    X[:, k] = X[:, k-1] + drift + diffu1 + diffu2
    
plt.plot(X[0])
plt.show()

print(np.sum(np.isnan(X)))
print(np.min(X))
print(np.max(X))

K_strike = 21.0
w = np.ones([1, d])
G = np.fmax(K_strike - np.sum(w*X[:, K-1], axis = 1), 0)

plt.hist(G)
plt.show()

np.savetxt("ad/ad_X.csv", np.reshape(X, [L, -1]))
for i in range(d):
    np.savetxt("ad/ad_Sigma_" + str(i+1) + ".csv", np.reshape(Sigma[:, :, :, i], [L, -1]))
    
np.savetxt("ad/ad_G.csv", G)