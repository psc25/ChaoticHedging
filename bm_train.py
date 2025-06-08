import numpy as np
import scipy.special as ssp
from sklearn import linear_model
import time

d = 1

L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K, dtype = np.float32)

X = np.reshape(np.loadtxt("bm/bm_X.csv", dtype = np.float32), [L, K, d])
G = np.loadtxt("bm/bm_G.csv", dtype = np.float32)

N = 6
mn = 50
val_split = 0.2
Ltrain = int(L*(1.0-val_split))

def dertanh(x):
    return 1.0/np.square(np.cosh(x))

begin = time.time()
A = np.random.normal(size = [1, 1, 2*N, mn, d]).astype(dtype = np.float32)
B = np.random.normal(size = [1, 1, 2*N, mn, d]).astype(dtype = np.float32)
phi = np.tanh(A[:, :, :N]*np.reshape(tt[:-1], (1, -1, 1, 1, 1)) + B[:, :, :N])
derphi = dertanh(A[:, :, :N]*np.reshape(tt[:-1], (1, -1, 1, 1, 1)) + B[:, :, :N])*A[:, :, :N]
psi = np.tanh(A[:, :, N:]*np.reshape(tt[:-1], (1, -1, 1, 1, 1)) + B[:, :, N:])
J1phi = np.zeros([L, K-1, N, mn], dtype = np.float32)
tms = np.zeros(N)
for n in range(N):
    b1 = time.time()
    WphiT = np.sum(phi[:, :, n]*np.expand_dims(X[:, :-1], axis = 2), axis = -1)
    Wphi0 = np.sum(phi[:, 0:1, n]*np.expand_dims(X[:, 0:1], axis = 2), axis = -1)
    WphiI = np.cumsum(np.sum(derphi[:, :, n]*np.expand_dims(X[:, :-1], axis = 2), axis = -1)*dt, axis = 1)
    J1phi[:, :, n] = np.power(WphiT-Wphi0-0.5*WphiI, n)/np.math.factorial(n)
    print("Iterated Stratonovich integrals prepared for n = " + str(n) + "/" + str(N))
    e1 = time.time()
    tms[n] = e1-b1
    
end = time.time()
print("Iterated Stratonovich integrals prepared in " + str(np.round(end-begin, 4)) + "s")
print("")
del A, B, WphiT, Wphi0, WphiI

for n in range(N+1):
    print("Chaotic hedging for N = " + str(n) + "/" + str(N))
    if n == 0:
        b1 = time.time()
        Gpred = np.mean(G)*np.ones_like(G)
        Hpred = np.zeros([L, K-1, d])
        e1 = time.time()
    else:
        b1 = time.time()
        theta = np.expand_dims(J1phi[:, :, :n], axis = -1)*psi[:, :, :n]
        gns = np.sum(theta*np.expand_dims(np.expand_dims(X[:, 1:] - X[:, :-1], axis = 2), axis = 2), axis = (1, 4))
        gns1 = np.reshape(gns, [L, -1])
        begin = time.time()
        regr = linear_model.LinearRegression()
        regr.fit(gns1[:Ltrain], G[:Ltrain])
        Y = np.reshape(regr.coef_, [1, n, mn, 1])
        end = time.time()
        print("Regression fitted in " + str(np.round(end-begin, 4)) + "s")
        Gpred = regr.intercept_ + np.sum(Y[:, :, :, 0]*gns, axis = (1, 2))
        del gns, gns1, regr
        
        begin = time.time()
        Hpred = np.sum(np.expand_dims(Y, axis = 0)*theta, axis = (2, 3))
        end = time.time()
        print("Computed hedging strategy in " + str(np.round(end-begin, 4)) + "s")
        e1 = time.time()
        del theta
        
    Gloss_train = np.sqrt(np.mean(np.square(Gpred[:Ltrain] - G[:Ltrain])))
    Gloss_test = np.sqrt(np.mean(np.square(Gpred[Ltrain:] - G[Ltrain:])))
    
    loss = np.array([Gloss_train, Gloss_test])
    comp = np.array([np.round(np.sum(tms[:n]) + e1-b1, 4), n*mn+1])
    
    print("Losses: " + str(np.round(loss[0], 4)) + " and " + str(np.round(loss[1], 4)))
    print("Performed in " + str(comp[0]) + "s")
    print("")
    
    np.savetxt("bm/bm_res_" + str(n) + ".csv", loss)
    np.savetxt("bm/bm_cmp_" + str(n) + ".csv", comp)
    
    np.savetxt("bm/bm_Gpr_" + str(n) + ".csv", Gpred)
    np.savetxt("bm/bm_Hpr_" + str(n) + ".csv", np.reshape(Hpred, [L, -1]))
    del Gpred, Hpred