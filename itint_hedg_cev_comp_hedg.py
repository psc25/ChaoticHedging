import numpy as np
import time

d = 1
n = 500
T = 1
dt = 1.0/500.0
M = 10000
tt = np.linspace(0, T, n)

sigma = 0.4
K = 102.0
    
X = np.loadtxt("itint_hedg_cev_x.csv", delimiter = ";")
X = np.reshape(X, (M, n, d))

N = 2048
uMax = 200.0
u0 = 1e-8
jj = np.reshape(np.arange(N), (-1, 1))
du = (uMax+u0)/(N-1)
u = u0 + jj*du
u1 = np.tile(np.reshape(u, (1, -1)), (M, 1))
dy = 2.0/(N-1)
alpha = du*dy/2.0/np.pi

def frft(x, a):
    e1 = np.tile(np.reshape(np.exp(-np.pi*1j*a*np.square(jj)), (1, -1)), (M, 1))
    e2 = np.tile(np.reshape(np.exp(np.pi*1j*a*np.square(np.flip(jj+1))), (1, -1)), (M, 1))
    z1 = np.concatenate((x*e1, np.zeros([M, N])), axis = -1)
    z2 = np.concatenate((1.0/e1, e2), axis = -1)
    fz1 = np.fft.fft(z1, axis = -1)
    fz2 = np.fft.fft(z2, axis = -1)
    ifz = np.fft.ifft(fz1*fz2, axis = -1)
    return e1*ifz[:, :N]

hedg_strat = np.zeros([M, n-1])
hedg_stratQ = np.zeros([M, n-1])
for t in range(n-1):
    begin = time.time()
    if t == 0:
        s = np.zeros([M, 1])
    else:
        s = np.sum(X[:, :t], axis = 1)*dt/T
        
    Y1 = K - s + np.transpose(jj)*dy

    psi = (1.0+1.0j)*np.sqrt(u)*np.tan((T-tt[t])*sigma*(1.0+1.0j)*np.sqrt(u)/2.0)/sigma
    dercharfct = np.exp(np.transpose(psi)*X[:, t])*np.transpose(psi)
    #charfct = np.exp(np.transpose(psi)*Xtest[:, t])
    z = dercharfct/u1*du*np.exp(-1.0j*np.tile(np.transpose(jj), (M, 1))*du*(K-s))
    z[:, 0] = z[:, 0]/2.0
    z[:, -1] = z[:, -1]/2.0
    z = frft(z, alpha)
    out0 = -np.imag(np.exp(-1.0j*u0*Y1)*z)/np.pi
    hedg1 = out0[:, 0]
    
    derpsi = psi/(2.0*u) + (T-tt[t])*1.0j/(2.0*np.square(np.cos((T-tt[t])*sigma*(1.0+1.0j)*np.sqrt(u)/2.0)))
    dercharfct1 = np.exp(np.transpose(psi)*X[:, t])*np.transpose(derpsi)*X[:, t]/1.0j*np.transpose(psi)
    dercharfct2 = np.exp(np.transpose(psi)*X[:, t])*np.transpose(derpsi)/1.0j
    #charfct1 = np.exp(np.transpose(psi)*Xtest[:, t])*np.transpose(derpsi)*Xtest[:, t]/1.0j
    z = (dercharfct1+dercharfct2)/u1*du*np.exp(-1.0j*np.tile(np.transpose(jj), (M, 1))*du*(K-s))
    z[:, 0] = z[:, 0]/2.0
    z[:, -1] = z[:, -1]/2.0
    z = frft(z, alpha)
    out0 = (T-tt[t])/(2.0*T)-np.imag(np.exp(-1.0j*u0*Y1)*z)/np.pi
    hedg2 = out0[:, 0]
    
    hedg_strat[:, t] = (K-s.flatten())*hedg1 - hedg2
    
    end = time.time()
    print("Step " + str(t+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg_strat[0:4, t], 3)))
    
np.savetxt("itint_hedg_cev_strat.csv", hedg_strat, delimiter = ";")