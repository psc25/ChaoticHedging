import numpy as np
import time

d = 10
n = 500
T = 1
dt = 1.0/500.0
M = 50000
tt = np.linspace(0, T, n)

Apar1 = np.loadtxt("itint_hedg_am_a.csv", delimiter = ";")
Apar = np.reshape(Apar1, (d+1, d, d))

X1 = np.loadtxt("itint_hedg_am_x.csv", delimiter = ";")
X = np.reshape(X1, (M, n, d))
hedg_strat = np.zeros([M, n-1, d])

K = 4.0
w = np.reshape(np.array([1.0, -0.95, 0.9, -0.85, 0.8, -0.75, 0.7, -0.65, 0.6, -0.55]), (d, 1))

N = 200
uMax = 100.0
u0 = 1e-8
jj = np.reshape(np.arange(N), (-1, 1))
du = (uMax+u0)/(N-1)
u = u0 + jj*du
u1 = np.tile(np.reshape(u, (1, 1, -1)), (M, d, 1))
dy = 2.0/(N-1)
Y = K + jj*dy
Y1 = np.tile(np.reshape(Y, (1, 1, -1)), (M, d, 1))
alpha = du*dy/2.0/np.pi

def frft(x, a):
    e1 = np.tile(np.reshape(np.exp(-np.pi*1j*a*np.square(jj)), (1, 1, -1)), (M, d, 1))
    e2 = np.tile(np.reshape(np.exp(np.pi*1j*a*np.square(np.flip(jj+1))), (1, 1, -1)), (M, d, 1))
    z1 = np.concatenate((x*e1, np.zeros([M, d, N])), axis = -1)
    z2 = np.concatenate((1.0/e1, e2), axis = -1)
    fz1 = np.fft.fft(z1, axis = -1)
    fz2 = np.fft.fft(z2, axis = -1)
    ifz = np.fft.ifft(fz1*fz2, axis = -1)
    return e1*ifz[:, :, :N]

def ricatti(beta):
    res = np.zeros([d+1, N], np.complex64)
    for i in range(d+1):
        res[i] = np.sum(beta*np.matmul(Apar[i], beta), axis = 0)/2.0
        
    return res

def ricatti2(beta, B):
    res = np.zeros([2*(d+1), N], np.complex64)
    for i in range(d+1):
        res[i] = np.sum(beta*np.matmul(Apar[i], beta), axis = 0)/2.0
        res[d+i+1] = np.sum(beta*np.matmul(Apar[i], B), axis = 0)/2.0

    return res

for t in range(n-1):
    begin = time.time()
    psi = np.zeros([n-t, d+1, N], np.complex64)
    psi[n-t-1, 1:] = 1j*w*np.transpose(u)
    for k in range(n-t-1, 0, -1):
        psi0 = psi[k, 1:]
        KK1 = ricatti(psi0)
        psi1 = psi0 + dt/2.0*KK1[1:]
        KK2 = ricatti(psi1)
        psi2 = psi0 + dt/2.0*KK2[1:]
        KK3 = ricatti(psi2)
        psi3 = psi0 + dt*KK3[1:]
        KK4 = ricatti(psi3)
        psi[k-1] = psi[k] + (KK1 + 2*KK2 + 2*KK3 + KK4)/6.0*dt

    charfct = np.exp(np.expand_dims(psi[0, 0], 0) + np.matmul(X[:, t], psi[0, 1:]))
    dercharfct = np.expand_dims(charfct, 1)*np.expand_dims(psi[0, 1:], 0)
    z = dercharfct/u1*du*np.exp(-1j*np.tile(np.reshape(jj, (1, 1, -1)), (M, d, 1))*du*K)
    z[:, :, 0] = z[:, :, 0]/2.0
    z[:, :, -1] = z[:, :, -1]/2.0
    z = frft(z, alpha)
    out0 = -np.imag(np.exp(-1j*u0*Y1)*z)/np.pi
    hedg1 = out0[:, :, 0]
    
    psi = np.zeros([n-t, d+1, N], np.complex64)
    psi[n-t-1, 1:] = 1j*w*np.transpose(u)
    A = np.zeros([n-t, d+1, N], np.complex64)
    A[n-t-1, 1:] = np.tile(w, (1, N))
    for k in range(n-t-1, 0, -1):
        KK1 = ricatti2(psi[k, 1:], A[k, 1:])
        psi1 = psi[k, 1:] + dt/2.0*KK1[1:(d+1)]
        A1 = A[k, 1:] + dt/2.0*KK1[(d+2):]
        KK2 = ricatti2(psi1, A1)
        psi2 = psi[k, 1:] + dt/2.0*KK2[1:(d+1)]
        A2 = A1 + dt/2.0*KK2[(d+2):]
        KK3 = ricatti2(psi2, A2)
        psi3 = psi[k, 1:] + dt*KK3[1:(d+1)]
        A3 = A1 + dt*KK3[(d+2):]
        KK4 = ricatti2(psi3, A3)
        psi[k-1] = psi[k] + (KK1[:(d+1)] + 2*KK2[:(d+1)] + 2*KK3[:(d+1)] + KK4[:(d+1)])/6.0*dt
        A[k-1] = A[k] + (KK1[(d+1):] + 2*KK2[(d+1):] + 2*KK3[(d+1):] + KK4[(d+1):])/6.0*dt
        
    charfct0 = np.exp(np.expand_dims(psi[0, 0], 0) + np.matmul(X[:, t], psi[0, 1:]))
    charfct1 = (np.expand_dims(A[0, 0], 0) + np.matmul(X[:, t], A[0, 1:]))*charfct0
    dercharfct = np.expand_dims(charfct1, 1)*np.expand_dims(psi[0, 1:], 0) + np.expand_dims(charfct0, 1)*np.expand_dims(A[0, 1:], 0)
    z = dercharfct/u1*du*np.exp(-1j*np.tile(np.reshape(jj, (1, 1, -1)), (M, d, 1))*du*K)
    z[:, :, 0] = z[:, :, 0]/2.0
    z[:, :, -1] = z[:, :, -1]/2.0
    z = frft(z, alpha)
    out0 = np.expand_dims(w, axis = 0)/2-np.imag(np.exp(-1j*u0*Y1)*z)/np.pi
    hedg2 = out0[:, :, 0]
    
    hedg_strat[:, t] = K*hedg1 - hedg2
    end = time.time()
    print("Step " + str(t) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg_strat[0, t], 3)))
    
hedg_strat1 = np.reshape(hedg_strat, (M, (n-1)*d))
np.savetxt("itint_hedg_am_strat.csv", hedg_strat1, delimiter = ";")