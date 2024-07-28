import numpy as np
import time

d = 2
L = 10000
K = 500
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

X1 = np.loadtxt("ad/ad_X.csv", delimiter = ";")
X = np.reshape(X1, (L, K, d))
del X1

Sigma = np.zeros([L, K, d, d])
for i in range(d):
    print("Loading Sigma for i = " + str(i+1) + "/" + str(d))
    Sigma1 = np.loadtxt("ad/ad_Sigma_" + str(i+1) + ".csv", delimiter = ";", dtype = np.float32)
    Sigma[:, :, :, i] = np.reshape(Sigma1, [L, K, d])
    
del Sigma1
print("")

beta = int(np.loadtxt("ad/ad_beta.csv", delimiter = ";").item())
U = np.loadtxt("ad/ad_U.csv", delimiter = ";")
V = np.loadtxt("ad/ad_V.csv", delimiter = ";")
rho = np.reshape(np.loadtxt("ad/ad_rho.csv", delimiter = ";"), [d, 1])

K_strike = 21.0
w = np.ones([d, 1])

N = 256
uMax = 150.0
u0 = 1e-8
du = (uMax-u0)/(N-1)
jj = np.reshape(np.arange(N), (1, 1, -1))
u = u0 + jj*du
dy = 2.0/(N-1)
Y1 = K_strike + jj*dy
alpha = du*dy/2.0/np.pi

def frft(x, a):
    e1 = np.tile(np.exp(-np.pi*1.0j*a*np.square(jj)), (L, d, 1))
    e2 = np.tile(np.exp(np.pi*1.0j*a*np.square(np.flip(jj+1))), (L, d, 1))
    z1 = np.concatenate((x*e1, np.zeros([L, d, N])), axis = -1)
    z2 = np.concatenate((1.0/e1, e2), axis = -1)
    fz1 = np.fft.fft(z1, axis = -1)
    fz2 = np.fft.fft(z2, axis = -1)
    ifz = np.fft.ifft(fz1*fz2, axis = -1)
    return e1*ifz[:, :, :N]

U1 = np.expand_dims(U, 0)
V1 = np.expand_dims(V, 0)
rho1 = np.expand_dims(rho, 0)
w1 = np.expand_dims(w, 0)
u1 = np.transpose(u, [2, 0, 1])
Urho = np.matmul(np.transpose(U), rho)
Qrhow = np.expand_dims(1.0j*np.matmul(np.transpose(U), np.matmul(rho, np.transpose(w))), 0)
Qrhowt = np.transpose(Qrhow, [0, 2, 1])

def ricatti(A, Au, c, cu):
    A1 = np.matmul(A, V1 + u1*Qrhow) + np.matmul(np.transpose(V1, [0, 2, 1]) + u1*Qrhowt, A) 
    A2 = 2.0*np.matmul(A, np.matmul(np.transpose(U1, [0, 2, 1]), np.matmul(U1, A)))
    A3 = -0.5*np.square(u1)*np.matmul(w1, np.transpose(w1, [0, 2, 1]))
    Ares = A1+A2+A3
    Au1 = np.matmul(Au, V1 + u1*Qrhow) + np.matmul(np.transpose(V1, [0, 2, 1]) + u1*Qrhowt, Au)
    Au2 = np.matmul(A, V1 + Qrhow) + np.matmul(np.transpose(V1, [0, 2, 1]) + Qrhowt, A)
    Au3 = 2.0*np.matmul(Au, np.matmul(np.transpose(U1, [0, 2, 1]), np.matmul(U1, A))) + 2.0*np.matmul(A, np.matmul(np.transpose(U1, [0, 2, 1]), np.matmul(U1, Au)))
    Au4 = -u1*np.matmul(w1, np.transpose(w1, [0, 2, 1]))
    Aures = Au1+Au2+Au3+Au4
    cres = beta*np.trace(np.matmul(np.transpose(U1, [0, 2, 1]), np.matmul(U1, A)), axis1 = 1, axis2 = 2)
    cures = beta*np.trace(np.matmul(np.transpose(U1, [0, 2, 1]), np.matmul(U1, Au)), axis1 = 1, axis2 = 2)
    return Ares, Aures, cres, cures
    
hedg = np.zeros([L, K-1, d])
for k in range(K-1):
    begin = time.time()
    A = np.zeros([N, d, d], np.complex64)
    Au = np.zeros([N, d, d], np.complex64)
    c = np.zeros(N, np.complex64)
    cu = np.zeros(N, np.complex64)
    for l in range(K-k-1, 0, -1):
        Ares1, Aures1, cres1, cures1 = ricatti(A, Au, c, cu)
        A1 = A + dt/2.0*Ares1
        Au1 = Au + dt/2.0*Aures1
        c1 = c + dt/2.0*cres1
        cu1 = cu + dt/2.0*cures1
        Ares2, Aures2, cres2, cures2 = ricatti(A1, Au1, c1, cu1)
        A2 = A + dt/2.0*Ares2
        Au2 = Au + dt/2.0*Aures2
        c2 = c + dt/2.0*cres2
        cu2 = cu + dt/2.0*cures2
        Ares3, Aures3, cres3, cures3 = ricatti(A2, Au2, c2, cu2)
        A3 = A + dt*Ares3
        Au3 = Au + dt*Aures3
        c3 = c + dt*cres3
        cu3 = cu + dt*cures3
        Ares4, Aures4, cres4, cures4 = ricatti(A3, Au3, c3, cu3)
        A = A + (Ares1 + 2.0*Ares2 + 2.0*Ares3 + Ares4)/6.0*dt
        Au = Au + (Aures1 + 2.0*Aures2 + 2.0*Aures3 + Aures4)/6.0*dt
        c = c + (cres1 + 2.0*cres2 + 2.0*cres3 + cres4)/6.0*dt
        cu = cu + (cures1 + 2.0*cures2 + 2.0*cures3 + cures4)/6.0*dt
    
    charfct0 = np.exp(np.einsum('nij,lji->ln', A, Sigma[:, k]) + np.einsum('nij,li->ln', 1.0j*u1*w1, X[:, k]) + np.expand_dims(c, 0))
    charfct_x = np.expand_dims(charfct0, 1)*1.0j*np.transpose(u1, [1, 2, 0])*w1
    charfct_s1 = np.expand_dims(charfct0, [1, 2])*np.transpose(A, [2, 1, 0])
    charfct_s2 = np.expand_dims(charfct0, [1, 2])*np.transpose(A, [1, 2, 0])
    intg0 = charfct_x + np.einsum('lijn,jk->lin', charfct_s1 + charfct_s2, Urho)
    
    z = intg0/(1.0j*np.pi*u)*du*np.exp(-1.0j*jj*du*K_strike)
    z[:, :, 0] = z[:, :, 0]/2.0
    z[:, :, -1] = z[:, :, -1]/2.0
    z = frft(z, alpha)
    out0 = np.real(np.exp(-1.0j*u0*Y1)*z)
    
    charfct1 = charfct0*(np.einsum('nij,lji->ln', Au, Sigma[:, k]) + np.einsum('nij,li->ln', 1.0j*w1, X[:, k]) + np.expand_dims(cu, 0))
    charfct1_x1 = np.expand_dims(charfct1, 1)*1.0j*np.transpose(u1, [1, 2, 0])*w1
    charfct1_x2 = np.expand_dims(charfct0, 1)*1.0j*w1
    charfct1_s1 = np.expand_dims(charfct1, [1, 2])*np.transpose(Au, [2, 1, 0])
    charfct1_s2 = np.expand_dims(charfct1, [1, 2])*np.transpose(Au, [1, 2, 0])
    charfct1_s3 = np.expand_dims(charfct0, [1, 2])*np.transpose(A, [1, 2, 0])
    intg1 = charfct1_x1 + charfct1_x2 + np.einsum('lijn,jk->lin', charfct1_s1 + charfct1_s2 + charfct1_s3, Urho)
    
    z = intg1/(np.pi*u)*du*np.exp(-1.0j*jj*du*K_strike)
    z[:, :, 0] = z[:, :, 0]/2.0
    z[:, :, -1] = z[:, :, -1]/2.0
    z = frft(z, alpha)
    out1 = np.real(np.exp(-1.0j*u0*Y1)*z)
    
    hedg[:, k] = -K_strike*out0[:, :, 0] - 0.5*np.transpose(w) - out1[:, :, 0]
    end = time.time()        
    print("Step " + str(k+1) + ", time " + str(round(end-begin, 1)) + "s, hedg " + str(np.round(hedg[0, k], 4)))
    
np.savetxt("ad/ad_strat.csv", np.reshape(hedg, [L, (K-1)*d]), delimiter = ";")