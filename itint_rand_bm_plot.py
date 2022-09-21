# Chaos Expansion with Neural Networks
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as sst

cmap = LinearSegmentedColormap.from_list('mycmap', ["limegreen", "mediumturquoise", "blue"])

d = 1
NNN = 6
M = 10000
val_split = 0.2
eval_every = 100
Mtrain = np.int((1-val_split)*M)
n = 500
T = 1
tt = np.linspace(0, T, n)

loss = np.zeros([NNN+1, 2, 2])
comp = np.zeros([NNN+1, 2])
Ypred = np.zeros([NNN+1, M-Mtrain])
Hpred = np.zeros([NNN+1, M-Mtrain, n-1, d])
for NN in range(NNN+1):
    try:
        loss[NN] = np.loadtxt("itint_rand_bm_res_" + str(NN) + ".csv", delimiter = ";")
        comp[NN] = np.loadtxt("itint_rand_bm_cmp_" + str(NN) + ".csv", delimiter = ";")
        Ypred[NN] = np.loadtxt("itint_rand_bm_ypr_" + str(NN) + ".csv", delimiter = ";")
        hpred = np.loadtxt("itint_rand_bm_hpr_" + str(NN) + ".csv", delimiter = ";")
        Hpred[NN] = np.reshape(hpred, (M-Mtrain, n-1, d))
    except:
        print("Tables for NN = " + str(NN) + " not available yet")

X = np.loadtxt("itint_hedg_bm_x.csv", delimiter = ";")
X = np.reshape(X, (M, n, d))
Y = np.loadtxt("itint_hedg_bm_y.csv", delimiter = ";")
Y = np.reshape(Y, (-1, 1))

# Plot Learning Curve
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
vmax1 = 0.8
vmax2 = 0.6
ax1.plot(np.arange(NNN+1), loss[:, 0, 0], linestyle = "-", color = "black", alpha = 0.8)
ax2.plot(np.arange(NNN+1), loss[:, 1, 0], linestyle = ":", color = "black", alpha = 0.8)
for NN in range(NNN+1):
    ax1.plot(NN, loss[NN, 0, 1], linestyle = "None", marker = "<", color = cmap(NN/NNN), markerfacecolor = cmap(NN/NNN), markersize = 8, alpha = 0.8)
    ax2.plot(NN, loss[NN, 1, 1], linestyle = "None", marker = ">", color = cmap(NN/NNN), markerfacecolor = cmap(NN/NNN), markersize = 8, alpha = 0.8)
    ax1.plot(NN, loss[NN, 0, 0], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    ax2.plot(NN, loss[NN, 1, 0], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
    
plt.plot(np.nan, np.nan, linestyle = "-", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", label = "Option Train", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = ":", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", label = "Hedge Train", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = "None", marker = "<", markersize = 8, color = "black", markerfacecolor = "black", label = "Option Test", alpha = 0.8)
plt.plot(np.nan, np.nan, linestyle = "None", marker = ">", markersize = 8, color = "black", markerfacecolor = "black", label = "Hedge Test", alpha = 0.8)
axes = plt.gca()
ax1.set_yscale("log")
ax2.set_yscale("log")
ax1.set_ylim([0.0018, vmax1])
ax2.set_ylim([0.0018, vmax2])
ax1.set_xlabel("N")
ax1.set_ylabel("MSE of option") 
ax2.set_ylabel("IMSE of strategy (dashed)") 
plt.legend(loc = "upper right", ncol = 2)
plt.savefig("itint_rand_bm_training.png", bbox_inches = 'tight') 
plt.close(fig)

# Plot Distribution on Test Set
fig = plt.figure()
ax = Axes3D(fig)
for NN in range(NNN+1):
    hist, bins = np.histogram(Ypred[NN], bins = 8, range = (0, 3.0))
    hist = hist/np.sum(hist)
    xs = (bins[:-1] + bins[1:])/2
    ax.bar(xs, hist, zs = NN, zdir = 'y', width = 0.09, color = cmap(NN/NNN), alpha = 0.8, label = "N = " + str(NN))

hist, bins = np.histogram(Y[Mtrain:], bins = 8, range = (0, 3.0))
hist = hist/np.sum(hist)
ax.bar(xs, hist, zs = NNN+1, zdir = 'y', width = 0.09, color = 'black', alpha = 0.8, label = "True")
ax.set_xlabel('G')
ax.set_yticks(np.arange(NNN+1))
ax.yaxis.set_ticklabels([' ', ' ', ' ', ' ', 'N', ' ', ' '])
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('Frequency', rotation = 90)
ax.view_init(15, 278)
plt.legend(loc = "upper right", ncol = 4)
plt.savefig("itint_rand_bm_testdistr.png", bbox_inches = 'tight')
plt.close(fig)

# Plot Computation
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
vmax1 = 80
vmax2 = 350
ax1.plot(np.arange(NNN+1), comp[:, 0], linestyle = "-", color = "black", alpha = 0.8)
ax2.plot(np.arange(NNN+1), comp[:, 1], linestyle = ":", color = "black", alpha = 0.8)
for NN in range(NNN+1):
    ax1.plot(NN, comp[NN, 0], linestyle = "None", marker = "o", color = cmap(NN/NNN), markerfacecolor = cmap(NN/NNN), markersize = 8, alpha = 0.8)
    ax2.plot(NN, comp[NN, 1], linestyle = "None", marker = "x", color = "black", markerfacecolor = "black", markersize = 8, alpha = 0.8)
   
plt.plot(np.nan, np.nan, linestyle = "-", marker = "o", markersize = 8, color = "black", markerfacecolor = "black", alpha = 0.8, label = "Running time")
plt.plot(np.nan, np.nan, linestyle = ":", marker = "x", markersize = 8, color = "black", markerfacecolor = "black", alpha = 0.8, label = "#Parameters")
axes = plt.gca()
ax1.set_ylim([0, vmax1])
ax2.set_ylim([0, vmax2])
axes = plt.gca()
ax1.set_xlabel("N")
ax1.set_ylabel("Running time in seconds") 
ax2.set_ylabel("Number of est. parameters") 
plt.legend(loc = "upper right", ncol = 2)
plt.savefig("itint_rand_bm_comp.png", bbox_inches = 'tight') 
plt.close(fig)

# Plot Hedging Strategy
K = -0.5
Mplot = 2
Htrue = np.zeros([M-Mtrain, n-1, d])
for t in range(n-1):
    Htrue[:, t, :] = sst.norm.cdf((X[Mtrain:, t, :]-K)/np.sqrt(T-tt[t]))

indplot = np.random.choice(np.arange(M-Mtrain), Mplot)
fig = plt.figure()
plt.plot(tt[:-1], Htrue[indplot[0], :, 0], "k-")
for NN in range(1, NNN+1):
    plt.plot(tt[:-1], Hpred[NN, indplot[0], :, 0], linestyle = ":", color = cmap(NN/NNN), label = "N = " + str(NN))
    
for k in range(1, Mplot):
    plt.plot(tt[:-1], Htrue[indplot[k], :, 0], "k-")
    for NN in range(1, NNN+1):
        plt.plot(tt[:-1], Hpred[NN, indplot[k], :, 0], linestyle = ":", color = cmap(NN/NNN))

axes = plt.gca()
plt.plot(np.zeros(2), [np.nan, np.nan], "k-", label = "True")
axes.set_ylim([-0.2, 1.2])
axes.set_xlabel("Time")
axes.set_ylabel("Number of shares")
axes.yaxis.set_label_position("right")
axes.yaxis.tick_right()
plt.legend(loc = "upper right", ncol = 4)
plt.savefig("itint_rand_bm_hedgetest.png", bbox_inches = 'tight') 
plt.close(fig)