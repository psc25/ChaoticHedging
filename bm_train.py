import numpy as np
import scipy.special as scsp
from sklearn import linear_model
import time

L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K, dtype = np.float32)

X = np.loadtxt("bm/bm_X.csv", delimiter = ";", dtype = np.float32)
G = np.loadtxt("bm/bm_G.csv", delimiter = ";", dtype = np.float32)

NN = 6
mn = 50
val_split = 0.2
Ltrain = int(L*(1.0-val_split))

for N in range(NN+1):
    print("Chaotic hedging for n <= " + str(N))
    if N == 0:
        b1 = time.time()
        Gpred = np.mean(G)*np.ones_like(G)
        Hpred = np.zeros([L, K-1])
        e1 = time.time()
    else:
        b1 = time.time()
        
        A = np.random.normal(size = [1, N, mn]).astype(dtype = np.float32)
        B = np.random.normal(size = [1, N, mn]).astype(dtype = np.float32)
        rnd_n = np.tanh(A*np.reshape(tt, (-1, 1, 1)) + B)
        
        Hn = np.zeros([L, N, mn], dtype = np.float32)
        print("Data preparation for n = 0/" + str(N))
        begin = time.time()
        for n in range(N):
            print("Data preparation for n = " + str(n+1) + "/" + str(N))
            Wg = np.sum(np.expand_dims(X[:, 1:] - X[:, :-1], -1)*np.expand_dims(rnd_n[:-1, n], 0), axis = 1)
            Qg = np.sum(np.expand_dims(np.square(rnd_n[:-1, n]), 0)*dt, axis = 1)
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
        Hpred = np.zeros([L, K-1])
        for n in range(N):
            print("Compute Hedge for n = " + str(n+1) + "/" + str(N))
            Wg = np.cumsum(np.expand_dims(X[:, 1:] - X[:, :-1], -1)*np.expand_dims(rnd_n[:-1, n], 0), axis = 1)
            Qg = np.cumsum(np.expand_dims(np.square(rnd_n[:-1, n]), 0)*dt, axis = 1)
            Hpred = Hpred + np.sum(Y[:, n]*np.power(Qg, n/2)*scsp.eval_hermitenorm(n, Wg/np.sqrt(Qg))/np.math.factorial(n)*np.expand_dims(rnd_n[:-1, n], 0), -1)
        
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
    
    np.savetxt("bm/bm_res_" + str(N) + ".csv", loss, delimiter = ";")
    np.savetxt("bm/bm_cmp_" + str(N) + ".csv", comp, delimiter = ";")
    
    np.savetxt("bm/bm_Gpr_" + str(N) + ".csv", Gpred, delimiter = ";")
    np.savetxt("bm/bm_Hpr_" + str(N) + ".csv", Hpred, delimiter = ";")