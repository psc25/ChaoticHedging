import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])

d = 2
L = 10000
K = 500
NN = 6
T = 1.0
dt = T/K
tt = np.linspace(0, T, K)

val_split = 0.2
Ltrain = int((1.0-val_split)*L)

loss = np.zeros([NN+1, 2])
comp = np.zeros([NN+1, 2])
Gpred = np.zeros([NN+1, L])
Hpred = np.zeros([NN+1, L, K-1, d])
for N in range(NN+1):
    try:
        loss[N] = np.loadtxt("ad/ad_res_" + str(N) + ".csv", delimiter = ";")
        comp[N] = np.loadtxt("ad/ad_cmp_" + str(N) + ".csv", delimiter = ";")
        Gpred[N] = np.loadtxt("ad/ad_Gpr_" + str(N) + ".csv", delimiter = ";")
        Hpred[N] = np.reshape(np.loadtxt("ad/ad_Hpr_" + str(N) + ".csv", delimiter = ";"), [L, K-1, d])
        print("Tables for N = " + str(N) + " loaded")
    except:
        print("Tables for N = " + str(N) + " not available")

G = np.loadtxt("ad/ad_G.csv", delimiter = ";")

Htrue = np.reshape(np.loadtxt("ad/ad_strat.csv", delimiter = ";"), [L, K-1, d])
hloss = np.zeros([NN+1, 2])
for N in range(NN+1):
    hloss[N, 0] = np.mean(np.square(Hpred[N, :Ltrain] - Htrue[:Ltrain]))*T*d
    hloss[N, 1] = np.mean(np.square(Hpred[N, Ltrain:] - Htrue[Ltrain:]))*T*d
    
# Plot Learning Curve
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
vmax1 = 1.6
vmax2 = 1.6
ax1.plot(np.arange(NN+1), loss[:, 0], linestyle = "-", color = "black", alpha = 0.8)
ax2.plot(np.arange(NN+1), hloss[:, 0], linestyle = ":", color = "black", alpha = 0.8)
for N in range(NN+1):
    ax1.plot(N, loss[N, 1], linestyle = "None", marker = "<", color = cmap(N/NN), markerfacecolor = cmap(N/NN), markersize = 8, alpha = 0.8)
    ax2.plot(N, hloss[N, 1], linestyle = "None", marker = ">", color = cmap(N/NN), markerfacecolor = cmap(N/NN), markersize = 8, alpha = 0.8)
    ax1.plot(N, loss[N, 0], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    ax2.plot(N, hloss[N, 0], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    
plt.plot(np.nan, np.nan, linestyle = "-", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", label = "Option Train", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = ":", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", label = "Hedge Train", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = "None", marker = "<", markersize = 8, color = "black", markerfacecolor = "black", label = "Option Test", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = "None", marker = ">", markersize = 8, color = "black", markerfacecolor = "black", label = "Hedge Test", alpha = 0.8)
axes = plt.gca()
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_ylim([0.004, vmax1])
ax2.set_ylim([0.004, vmax2])
ax1.set_xlabel("N")
ax1.set_ylabel("MSE of option") 
ax2.set_ylabel("IMSE of strategy (dashed)") 
plt.legend(loc = "upper right", ncol = 2, prop={'size': 8})
plt.savefig("ad/ad_train.png", bbox_inches = 'tight') 
plt.close(fig)

# Plot Distribution on Test Set
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
for N in range(NN+1):
    hist, bins = np.histogram(Gpred[N], bins = 8, range = (0, 3.5))
    hist = hist/np.sum(hist)
    x = (bins[:-1] + bins[1:])/2.0
    ax.bar3d(x, y = N, z = 0.0, dx = 0.09, dy = 0.09, dz = hist, color = cmap(N/NN), alpha = 0.8, label = "N = " + str(N))

hist, bins = np.histogram(G[Ltrain:], bins = 8, range = (0, 3.5))
hist = hist/np.sum(hist)
ax.bar3d(x, y = NN+1, z = 0.0, dx = 0.09, dy = 0.09, dz = hist, color = 'black', alpha = 0.8, label = "True")
ax.set_xlabel('G')
ax.set_yticks(np.arange(NN+1))
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
vmax1 = 400
vmax2 = 350
ax1.plot(np.arange(NN+1), comp[:, 0], linestyle = "-", color = "black", alpha = 0.8)
ax2.plot(np.arange(NN+1), comp[:, 1], linestyle = ":", color = "black", alpha = 0.8)
for N in range(NN+1):
    ax1.plot(N, comp[N, 0], linestyle = "None", marker = "o", color = cmap(N/NN), markerfacecolor = cmap(N/NN), markersize = 8, alpha = 0.8)
    ax2.plot(N, comp[N, 1], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    
plt.plot(np.nan, np.nan, linestyle = "-", marker = "o", markersize = 8, color = "black", markerfacecolor = "black", alpha = 0.8, label = "Running time")
plt.plot(np.nan, np.nan, linestyle = ":", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", alpha = 0.8, label = "Nr. of parameters")
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
indplot = [8185, 8253]#np.random.choice(np.arange(Ltrain, L), Lplot)
fig = plt.figure()
plt.plot(tt[:-1], Htrue[indplot[0], :, 0], "k-")
for N in range(1, NN+1):
    plt.plot(tt[:-1], Hpred[N, indplot[0], :, 0], linestyle = ":", color = cmap(N/NN), label = "N = " + str(N))
    
for l in range(1, Lplot):
    plt.plot(tt[:-1], Htrue[indplot[l], :, 0], "k-")
    for N in range(1, NN+1):
        plt.plot(tt[:-1], Hpred[N, indplot[l], :, 0], linestyle = ":", color = cmap(N/NN))

axes = plt.gca()
plt.plot(np.zeros(2), [np.nan, np.nan], "k-", label = "True")
axes.set_ylim([-1.05, -0.3])
axes.set_xlabel("Time")
axes.set_ylabel("Number of shares")
axes.yaxis.set_label_position("right")
axes.yaxis.tick_right()
plt.legend(loc = "upper right", ncol = 4, prop={'size': 8})
plt.savefig("ad/ad_test_hedge.png", bbox_inches = 'tight') 
plt.close(fig)