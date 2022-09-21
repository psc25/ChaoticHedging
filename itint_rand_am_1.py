# Chaos Expansion with Neural Networks
import numpy as np
import scipy.special as ssp
from sklearn import linear_model
import time

d = 10
n = 500
T = 1
dt = 1.0/500.0
M = 50000
tt = np.linspace(0, T, n, dtype = np.float32)

X = np.loadtxt("itint_hedg_am_x.csv", delimiter = ";", dtype = np.float32)
X = np.reshape(X, (M, n, d))
Y = np.loadtxt("itint_hedg_am_y.csv", delimiter = ";", dtype = np.float32)
Y = np.reshape(Y, (-1, 1))
Acoeff = np.loadtxt("itint_hedg_am_a.csv", delimiter = ";", dtype = np.float32)
Acoeff = np.reshape(Acoeff, (-1, d, d))

val_split = 0.2
Mtrain = int(M*(1-val_split))

NN = 1
L = 250

X_ex = np.concatenate((np.ones_like(X[:, :-1, :1]), X[:, :-1]), axis = -1)
adiff = np.einsum('kij,mtk->mtij', Acoeff, X_ex)*dt

def sigm(x):
    return 1/(1 + np.exp(-x))

if NN == 0:
    b1 = time.time()
    Ypred = np.mean(Y)*np.ones_like(Y)
    Hpred = np.zeros_like(X[:, :-1])
    e1 = time.time()
else:
    b1 = time.time()
    rng = np.random.default_rng()
    A = rng.standard_normal(size = [NN, 1, d, L],  dtype = np.float32)
    b = rng.standard_normal(size = [NN, 1, d, L],  dtype = np.float32)
    tt1 = np.reshape(tt, (1, -1, 1, 1))
    output1 = sigm(A*tt1 + b)
    
    Hn = np.zeros([NN, L, M], dtype = np.float32)
    begin = time.time()
    for N in range(NN):
        for m in range(M):
            if m % 1000 == 0:
                print("Data preparation for N = " + str(N+1) + " and m = " + str(m+1))
                
            Wg = np.sum(np.sum(np.expand_dims(X[m, 1:] - X[m, :-1], -1)*output1[N, :-1], axis = 1), axis = 0)
            Qg = np.sum(np.sum(output1[N, :-1]*np.einsum('tij,tjk->tik', adiff[m], output1[N, :-1]), axis = 1), axis = 0)
            xH = Wg/np.sqrt(Qg)
            xH[np.isnan(xH)] = 0.0
            Hn[N, :, m] = np.power(Qg, (N+1)/2)*ssp.eval_hermitenorm(N+1, xH)/np.math.factorial(N+1)
    
    end = time.time()
    print("")
    
    regr = linear_model.LinearRegression()
    
    Hn2 = np.transpose(np.reshape(Hn, (NN*L, M)))
    begin = time.time()
    regr.fit(Hn2[:Mtrain], Y[:Mtrain])
    end = time.time()
    
    w = np.reshape(regr.coef_, (NN, L))
    print("Regression fitting in " + str(np.round(end-begin, 4)) + "s")
    print(w)
    print("")
    
    Ypred = regr.intercept_[0]*np.ones([M, 1])
    Hpred = np.zeros([M, n-1, d])
    begin = time.time()
    for N in range(NN):
        for m in range(M):
            if m % 1000 == 0:
                print("Compute Option and Hedging Strategy for N = " + str(N+1) + " and m = " + str(m+1))
                
            Wg = np.cumsum(np.sum(np.expand_dims(X[m, 1:] - X[m, :-1], -1)*output1[N, :-1], axis = 1), axis = 0)
            Qg = np.cumsum(np.sum(output1[N, :-1]*np.einsum('tij,tjk->tik', adiff[m], output1[N, :-1]), axis = 1), axis = 0)
            xH = Wg/np.sqrt(Qg)
            xH[np.isnan(xH)] = 0.0
            Ypred[m] = Ypred[m] + np.sum(np.expand_dims(w[N], 0)*np.power(Qg[-1], (N+1)/2)*ssp.eval_hermitenorm(N+1, xH[-1])/np.math.factorial(N+1), axis = -1)
            Hpred[m] = Hpred[m] + np.sum(np.reshape(w[N], (1, 1, -1))*np.expand_dims(np.power(Qg, N/2)*ssp.eval_hermitenorm(N, xH), 1)/np.math.factorial(N)*output1[N, :-1], axis = -1)
    
    end = time.time()
    print("Computing in " + str(np.round(end-begin, 4)) + "s")
    e1 = time.time()
    print("Performed in " + str(np.round(end-begin, 4)) + "s")

Yloss_train = np.mean(np.square(Ypred[:Mtrain] - Y[:Mtrain]))
Yloss_test = np.mean(np.square(Ypred[Mtrain:] - Y[Mtrain:]))
Yloss = np.array([Yloss_train, Yloss_test])

Htrue = np.loadtxt("itint_hedg_am_strat.csv", delimiter = ";")
Htrue = np.reshape(Htrue, (M, n-1, d))
    
Hloss_train = np.sum(np.square(Hpred[:Mtrain] - Htrue[:Mtrain]))/Mtrain*dt
Hloss_test = np.sum(np.square(Hpred[Mtrain:] - Htrue[Mtrain:]))/(M-Mtrain)*dt
Hloss = np.array([Hloss_train, Hloss_test])

loss = np.array([[Yloss_train, Yloss_test], [Hloss_train, Hloss_test]])
comp = np.array([np.round(e1-b1, 4), NN*L+1])

np.savetxt("itint_rand_am_res_" + str(NN) + ".csv", loss, delimiter = ";")
np.savetxt("itint_rand_am_cmp_" + str(NN) + ".csv", comp, delimiter = ";")

np.savetxt("itint_rand_am_ypr_" + str(NN) + ".csv", Ypred[Mtrain:], delimiter = ";")
np.savetxt("itint_rand_am_hpr_" + str(NN) + ".csv", np.reshape(Hpred[Mtrain:], (M-Mtrain, (n-1)*d)), delimiter = ";")