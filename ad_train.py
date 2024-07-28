import numpy as np
import scipy.special as scsp
from sklearn import linear_model
import time

d = 2
L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K, dtype = np.float32)

X1 = np.loadtxt("ad/ad_X.csv", delimiter = ";", dtype = np.float32)
X = np.reshape(X1, [L, K, d])
del X1

Sigma = np.zeros([L, K, d, d])
for i in range(d):
    print("Loading Sigma for i = " + str(i+1) + "/" + str(d))
    Sigma1 = np.loadtxt("ad/ad_Sigma_" + str(i+1) + ".csv", delimiter = ";", dtype = np.float32)
    Sigma[:, :, :, i] = np.reshape(Sigma1, [L, K, d])
    
del Sigma1
print("")

G = np.loadtxt("ad/ad_G.csv", delimiter = ";", dtype = np.float32)

NN = 6
mn = 50
val_split = 0.2
Ltrain = int(L*(1.0-val_split))

for N in range(NN+1):
    print("Chaotic hedging for n <= " + str(N))
    if N == 0:
        b1 = time.time()
        Gpred = np.mean(G)*np.ones_like(G)
        Hpred = np.zeros([L, K-1, d])
        e1 = time.time()
    else:
        b1 = time.time()
        
        A = np.random.normal(size = [1, N, mn, d]).astype(dtype = np.float32)
        B = np.random.normal(size = [1, N, mn, d]).astype(dtype = np.float32)
        rnd_n = np.tanh(A*np.reshape(tt, (-1, 1, 1, 1)) + B)
        
        Hn = np.zeros([L, N, mn], dtype = np.float32)
        print("Data preparation for n = 0/" + str(N))
        begin = time.time()
        for n in range(N):
            print("Data preparation for n = " + str(n+1) + "/" + str(N))
            phi = np.expand_dims(rnd_n[:-1, n], 0)
            Wg = np.sum(np.expand_dims(X[:, 1:] - X[:, :-1], 2)*phi, axis = (1, 3))
            Qg = np.sum(np.matmul(phi, Sigma[:, :-1])*phi*dt, axis = (1, 3))
            Hn[:, n] = np.power(Qg, (n+1)/2)*scsp.eval_hermitenorm(n+1, Wg/np.sqrt(Qg))/np.math.factorial(n+1)
            
        end = time.time()
        print("Data prepared in " + str(np.round(end-begin, 4)) + "s")
        
        Hn2 = np.reshape(Hn, [L, -1])
        begin = time.time()
        regr = linear_model.LinearRegression()
        regr.fit(Hn2[:Ltrain], G[:Ltrain])
        Y = np.reshape(regr.coef_, [1, N, mn])
        end = time.time()
        print("Regression fitted in " + str(np.round(end-begin, 4)) + "s")
        
        Gpred = regr.intercept_ + np.sum(Y*Hn, axis = (1, 2))
        begin = time.time()
        Hpred = np.zeros([L, K-1, d])
        for n in range(N):
            print("Compute Hedge for n = " + str(n+1) + "/" + str(N))
            phi = np.expand_dims(rnd_n[:-1, n], 0)
            Wg = np.cumsum(np.sum(np.expand_dims(X[:, 1:] - X[:, :-1], 2)*phi, axis = -1), axis = 1)
            Qg = np.cumsum(np.sum(np.matmul(phi, Sigma[:, :-1])*phi*dt, axis = -1), axis = 1)
            Hn1 = np.expand_dims(Y[:, n:(n+1)]*np.power(Qg, n/2)*scsp.eval_hermitenorm(n, Wg/np.sqrt(Qg))/np.math.factorial(n), -1)
            Hpred = Hpred + np.sum(Hn1*phi, -2)
            
        end = time.time()
        print("Computed Hedge in " + str(np.round(end-begin, 4)) + "s")
        e1 = time.time()
        
    Gloss_train = np.mean(np.square(Gpred[:Ltrain] - G[:Ltrain]))
    Gloss_test = np.mean(np.square(Gpred[Ltrain:] - G[Ltrain:]))

    loss = np.array([Gloss_train, Gloss_test])
    comp = np.array([np.round(e1-b1, 4), N*mn+1])
    
    print("Losses: " + str(np.round(loss[0], 4)) + " and " + str(np.round(loss[1], 4)))
    print("Performed in " + str(np.round(e1-b1, 4)) + "s")
    print("")
    
    np.savetxt("ad/ad_res_" + str(N) + ".csv", loss, delimiter = ";")
    np.savetxt("ad/ad_cmp_" + str(N) + ".csv", comp, delimiter = ";")
    
    np.savetxt("ad/ad_Gpr_" + str(N) + ".csv", Gpred, delimiter = ";")
    np.savetxt("ad/ad_Hpr_" + str(N) + ".csv", np.reshape(Hpred, [L, (K-1)*d]), delimiter = ";")