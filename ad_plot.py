import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])

d = 2

L = 10000
K = 500
N = 6
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

val_split = 0.2
Ltrain = int((1.0-val_split)*L)

loss = np.zeros([N+1, 2])
comp = np.zeros([N+1, 2])
Gpred = np.zeros([N+1, L])
Hpred = np.zeros([N+1, L, K-1, d])
for n in range(N+1):
    try:
        loss[n] = np.loadtxt("ad/ad_res_" + str(n) + ".csv")
        comp[n] = np.loadtxt("ad/ad_cmp_" + str(n) + ".csv")
        Gpred[n] = np.loadtxt("ad/ad_Gpr_" + str(n) + ".csv")
        Hpred[n] = np.reshape(np.loadtxt("ad/ad_Hpr_" + str(n) + ".csv"), [L, K-1, d])
        print("Tables for n = " + str(n) + " loaded")
    except:
        print("Tables for n = " + str(n) + " not available")

X = np.reshape(np.loadtxt("ad/ad_X.csv"), [L, K, d])
G = np.loadtxt("ad/ad_G.csv")

Htrue = np.reshape(np.loadtxt("ad/ad_hedg.csv"), [L, K-1, d])
hloss = np.zeros([N+1, 2])
for n in range(N+1):
    hloss[n, 0] = np.sqrt(T*np.mean(np.square(Hpred[n, :Ltrain] - Htrue[:Ltrain])))
    hloss[n, 1] = np.sqrt(T*np.mean(np.square(Hpred[n, Ltrain:] - Htrue[Ltrain:])))

# Plot Learning Curve
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
vmax1 = 1.2
vmax2 = 1.2
ax1.plot(np.arange(N+1), loss[:, 0], linestyle = "-", color = "black", alpha = 0.8)
ax2.plot(np.arange(N+1), hloss[:, 0], linestyle = ":", color = "black", alpha = 0.8)
for n in range(N+1):
    ax1.plot(n, loss[n, 1], linestyle = "None", marker = "<", color = cmap(n/N), markerfacecolor = cmap(n/N), markersize = 8, alpha = 0.8)
    ax2.plot(n, hloss[n, 1], linestyle = "None", marker = ">", color = cmap(n/N), markerfacecolor = cmap(n/N), markersize = 8, alpha = 0.8)
    ax1.plot(n, loss[n, 0], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    ax2.plot(n, hloss[n, 0], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    
plt.plot(np.nan, np.nan, linestyle = "-", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", label = "$L^2$-hedging error (Train)", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = ":", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", label = "$L^2$-strategy error (Train)", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = "None", marker = "<", markersize = 8, color = "black", markerfacecolor = "black", label = "$L^2$-hedging error (Test)", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = "None", marker = ">", markersize = 8, color = "black", markerfacecolor = "black", label = "$L^2$-strategy error (Test)", alpha = 0.8)
axes = plt.gca()
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_ylim([0.02, vmax1])
ax2.set_ylim([0.02, vmax2])
ax1.set_xlabel("N")
ax1.set_ylabel("$L^2$-hedging error")
ax2.set_ylabel("$L^2$-strategy error") 
plt.legend(loc = "upper right", ncol = 2, prop={'size': 8})
plt.savefig("ad/ad_train.png", bbox_inches = 'tight') 
plt.close(fig)

# Plot Distribution on Test Set
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for n in range(N+1):
    hist, bins = np.histogram(Gpred[n], bins = 8, range = (0, 3.0))
    hist = hist/np.sum(hist)
    x = (bins[:-1] + bins[1:])/2.0
    ax.bar3d(x, y = n, z = 0.0, dx = 0.09, dy = 0.09, dz = hist, color = cmap(n/N), alpha = 0.8, label = "N = " + str(n))

hist, bins = np.histogram(G[Ltrain:], bins = 8, range = (0, 3.0))
hist = hist/np.sum(hist)
ax.bar3d(x, y = N+1, z = 0.0, dx = 0.09, dy = 0.09, dz = hist, color = 'black', alpha = 0.8, label = "True")
ax.set_xlabel('G')
ax.set_yticks(np.arange(N+1))
ax.yaxis.set_ticklabels([' ', ' ', ' ', ' ', 'N', ' ', ' '])
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Frequency', rotation = 90)
ax.view_init(15, 273.5)
plt.legend(loc = "upper right", ncol = 4, prop={'size': 8})
plt.savefig("ad/ad_test_distr.png", bbox_inches = 'tight')
plt.close(fig)

# Plot Computation
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
vmax1 = 520
vmax2 = 350
ax1.plot(np.arange(N+1), comp[:, 0], linestyle = "-", color = "black", alpha = 0.8)
ax2.plot(np.arange(N+1), comp[:, 1], linestyle = ":", color = "black", alpha = 0.8)
for n in range(N+1):
    ax1.plot(n, comp[n, 0], linestyle = "None", marker = "o", color = cmap(n/N), markerfacecolor = cmap(n/N), markersize = 8, alpha = 0.8)
    ax2.plot(n, comp[n, 1], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
   
plt.plot(np.nan, np.nan, linestyle = "-", marker = "o", markersize = 8, color = "black", markerfacecolor = "black", alpha = 0.8, label = "Running time")
plt.plot(np.nan, np.nan, linestyle = ":", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", alpha = 0.8, label = "Nr. of Parameters")
axes = plt.gca()
ax1.set_ylim([0, vmax1])
ax2.set_ylim([0, vmax2])
axes = plt.gca()
ax1.set_xlabel("N")
ax1.set_ylabel("Running time in seconds") 
ax2.set_ylabel("Number of parameters") 
plt.legend(loc = "upper right", ncol = 2, prop={'size': 8})
plt.savefig("ad/ad_comp.png", bbox_inches = 'tight') 
plt.close(fig)

# Plot Hedging Strategy
Lplot = 2
indplot = np.random.choice(np.arange(Ltrain, L), Lplot)
fig = plt.figure()
plt.plot(tt[:-1], Htrue[indplot[0], :, 0], "k-")
for n in range(1, N+1):
    plt.plot(tt[:-1], Hpred[n, indplot[0], :, 0], linestyle = ":", color = cmap(n/N), label = "N = " + str(n))
    
for l in range(1, Lplot):
    plt.plot(tt[:-1], Htrue[indplot[l], :, 0], "k-")
    for n in range(1, N+1):
        plt.plot(tt[:-1], Hpred[n, indplot[l], :, 0], linestyle = ":", color = cmap(n/N))

axes = plt.gca()
plt.plot(np.zeros(2), [np.nan, np.nan], "k-", label = "True")
axes.set_ylim([-1.1, 0.2])
axes.set_xlabel("Time")
axes.set_ylabel("Number of shares")
axes.yaxis.set_label_position("right")
axes.yaxis.tick_right()
plt.legend(loc = "upper right", ncol = 4, prop={'size': 8})
plt.savefig("ad/ad_test_hedge.png", bbox_inches = 'tight') 
plt.close(fig)