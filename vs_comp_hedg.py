import numpy as np
import time

d = 2
L = 10000
K = 500
T = 1
dt = T/K
tt = np.linspace(0, T, K)

xi = 1.0
rho = -0.7
K_strike = 101.0
Apar1 = np.loadtxt("vs/vs_par.csv")
Apar = np.reshape(Apar1, (d+1, d, d))
X = np.loadtxt("vs/vs_X.csv")
X = np.reshape(X, (L, K, d))
I = np.cumsum(np.concatenate([np.zeros([L, 1]), X[:, :, 0]], axis = 1), axis = 1)*dt/T

N = 512
uMax = 200.0
u0 = 1e-8
du = (uMax-u0)/(N-1)
jj = np.reshape(np.arange(N), (1, -1))
u = u0 + jj*du
dy = 2.0/(N-1)
alpha = du*dy/2.0/np.pi

def frft(x, a):
    e1 = np.tile(np.exp(-np.pi*1.0j*a*np.square(jj)), (L, 1))
    e2 = np.tile(np.exp(np.pi*1.0j*a*np.square(np.flip(jj+1))), (L, 1))
    z1 = np.concatenate((x*e1, np.zeros([L, N])), axis = -1)
    z2 = np.concatenate((1.0/e1, e2), axis = -1)
    fz1 = np.fft.fft(z1, axis = -1)
    fz2 = np.fft.fft(z2, axis = -1)
    ifz = np.fft.ifft(fz1*fz2, axis = -1)
    return e1*ifz[:, :N]

def ricatti(beta, B):
    res = np.zeros([2*(d+1), N], dtype = np.complex64)
    res[1] = 1.0j*u[0]
    res[2] = 0.5*np.sum(beta*np.matmul(Apar[2], beta), axis = 0)
    res[4] = 1.0j
    res[5] = np.sum(beta*np.matmul(Apar[2], B), axis = 0)
    return res

xirho1 = np.reshape(np.array([1.0, xi*rho]), [1, d, 1])
hedg = np.zeros([L, K-1])
for k in range(K-1):
    begin = time.time()
    psi = np.zeros([d+1, N], np.complex64)
    psi_ext = np.zeros([d+1, N], np.complex64)
    for l in range(K-k-1, 0, -1):
        KK1 = ricatti(psi[1:], psi_ext[1:])
        psi1 = psi[1:] + dt/2.0*KK1[1:(d+1)]
        psi_ext1 = psi_ext[1:] + dt/2.0*KK1[(d+2):]
        KK2 = ricatti(psi1, psi_ext1)
        psi2 = psi[1:] + dt/2.0*KK2[1:(d+1)]
        psi_ext2 = psi_ext[1:] + dt/2.0*KK2[(d+2):]
        KK3 = ricatti(psi2, psi_ext2)
        psi3 = psi[1:] + dt*KK3[1:(d+1)]
        psi_ext3 = psi_ext[1:] + dt*KK3[(d+2):]
        KK4 = ricatti(psi3, psi_ext3)
        psi = psi + (KK1[:(d+1)] + 2.0*KK2[:(d+1)] + 2.0*KK3[:(d+1)] + KK4[:(d+1)])/6.0*dt
        psi_ext = psi_ext + (KK1[(d+1):] + 2.0*KK2[(d+1):] + 2.0*KK3[(d+1):] + KK4[(d+1):])/6.0*dt
        
    KI = K_strike - np.expand_dims(I[:, k], -1)
    Y1 = KI + jj*dy
    
    charfct = np.exp(np.sum(np.expand_dims(psi[1:], 0)*np.expand_dims(X[:, k], -1), axis = 1))
    theta1 = charfct*np.sum(np.expand_dims(psi[1:], 0)*xirho1, axis = 1)
    z = theta1/(1.0j*np.pi*u)*du*np.exp(-1.0j*jj*du*KI)
    z[:, 0] = z[:, 0]/2.0
    z[:, -1] = z[:, -1]/2.0
    z = frft(z, alpha)
    out1 = np.real(np.exp(-1.0j*u0*Y1)*z)
    
    #charfct_ext = charfct*np.sum(np.expand_dims(psi_ext[1:], 0)*np.expand_dims(X[:, k], -1), axis = 1)
    theta2 = charfct*np.sum(np.expand_dims(psi_ext[1:], 0)*np.expand_dims(X[:, k], -1), axis = 1)*np.sum(np.expand_dims(psi[1:], 0)*xirho1, axis = 1)
    theta3 = charfct*np.sum(np.expand_dims(psi_ext[1:], 0)*xirho1, axis = 1)
    z = (theta2 + theta3)/(np.pi*u)*du*np.exp(-1.0j*jj*du*KI)
    z[:, 0] = z[:, 0]/2.0
    z[:, -1] = z[:, -1]/2.0
    z = frft(z, alpha)
    out2 = 0.5*(T-tt[k])/T + np.real(np.exp(-1.0j*u0*Y1)*z)
    
    hedg[:, k] = -KI[:, 0]*out1[:, 0] - out2[:, 0]
    end = time.time()
    print("Step " + str(k+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg[0, k], 4)) + ", min " + str(np.round(np.min(hedg[:, k]), 4)) + ", max " + str(np.round(np.max(hedg[:, k]), 4)))
    
np.savetxt("vs/vs_hedg.csv", hedg)