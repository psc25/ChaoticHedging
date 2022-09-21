# Chaos Expansion with Neural Networks
import numpy as np
import scipy.special as ssp
from sklearn import linear_model
import time

d = 1
n = 500
T = 1
dt = 1.0/500.0
M = 10000
tt = np.linspace(0, T, n, dtype = np.float32)

X = np.loadtxt("itint_hedg_cev_x.csv", delimiter = ";", dtype = np.float32)
X = np.reshape(X, (M, n, d))
Y = np.loadtxt("itint_hedg_cev_y.csv", delimiter = ";", dtype = np.float32)
Y = np.reshape(Y, (-1, 1))
sigma = 0.4

val_split = 0.2
Mtrain = int(M*(1-val_split))

NN = 2
L = 50

def sigm(z):
    return 1/(1+np.exp(-z))

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
        print("Data preparation for N = " + str(N+1))
        Wg = np.sum(np.matmul(np.transpose(X[:, 1:] - X[:, :-1], (1, 0, 2)), output1[N, :-1]), axis = 0)
        Qg = np.sum(np.expand_dims(np.square(sigma*output1[N, :-1, 0]), 1)*np.transpose(X[:, :-1], (1, 0, 2))*dt, axis = 0)
        Hn[N] = np.transpose(np.power(Qg, (N+1)/2)*ssp.eval_hermitenorm(N+1, Wg/np.sqrt(Qg))/np.math.factorial(N+1))
        
    end = time.time()
    print("Data preparation in " + str(np.round(end-begin, 4)) + "s")
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
        print("Compute Option and Hedging Strategy for N = " + str(N+1))
        Wg = np.cumsum(np.matmul(np.transpose(X[:, 1:] - X[:, :-1], (1, 0, 2)), output1[N, :-1]), axis = 0)
        Qg = np.cumsum(np.expand_dims(np.square(sigma*output1[N, :-1, 0]), 1)*np.transpose(X[:, :-1], (1, 0, 2))*dt, axis = 0)
        Ypred = Ypred + np.sum(np.expand_dims(w[N], 0)*np.power(Qg[-1], (N+1)/2)*ssp.eval_hermitenorm(N+1, Wg[-1]/np.sqrt(Qg[-1]))/np.math.factorial(N+1), axis = -1, keepdims = True)
        Hpred = Hpred + np.transpose(np.sum(np.reshape(w[N], (1, 1, 1, -1))*np.expand_dims(np.power(Qg, N/2)*ssp.eval_hermitenorm(N, Wg/np.sqrt(Qg)), 2)/np.math.factorial(N)*np.expand_dims(output1[N, :-1], 1), axis = -1), (1, 0, 2))
    
    end = time.time()
    print("Computing in " + str(np.round(end-begin, 4)) + "s")
    e1 = time.time()
    print("Performed in " + str(np.round(e1-b1, 4)) + "s")

Yloss_train = np.mean(np.square(Ypred[:Mtrain] - Y[:Mtrain]))
Yloss_test = np.mean(np.square(Ypred[Mtrain:] - Y[Mtrain:]))
Yloss = np.array([Yloss_train, Yloss_test])

Htrue = np.loadtxt("itint_hedg_cev_strat.csv", delimiter = ";")
Htrue = np.reshape(Htrue, (M, n-1, d))
    
Hloss_train = np.sum(np.square(Hpred[:Mtrain] - Htrue[:Mtrain]))/Mtrain*dt
Hloss_test = np.sum(np.square(Hpred[Mtrain:] - Htrue[Mtrain:]))/(M-Mtrain)*dt
Hloss = np.array([Hloss_train, Hloss_test])

loss = np.array([[Yloss_train, Yloss_test], [Hloss_train, Hloss_test]])
comp = np.array([np.round(e1-b1, 4), NN*L+1])

np.savetxt("itint_rand_cev_res_" + str(NN) + ".csv", loss, delimiter = ";")
np.savetxt("itint_rand_cev_cmp_" + str(NN) + ".csv", comp, delimiter = ";")

np.savetxt("itint_rand_cev_ypr_" + str(NN) + ".csv", Ypred[Mtrain:], delimiter = ";")
np.savetxt("itint_rand_cev_hpr_" + str(NN) + ".csv", np.reshape(Hpred[Mtrain:], (M-Mtrain, (n-1)*d)), delimiter = ";")