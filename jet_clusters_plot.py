import numpy as np
import matplotlib.pyplot as plt

variables = ['cluster pt (max energy)', 'cluster eta (max energy)', 'cluster dphi (max energy)', 'cluster energy (max)', 'cluster pt (sum max 3)', 'cluster energy (sum max 3)']

'''#plot signal and background separately for test files

hist_collector = np.genfromtxt("hist_collector.csv", delimiter=',')
bins_collector = np.genfromtxt("bins_collector.csv", delimiter=',')

for i in range(hist_collector.shape[0]):
    plt.bar(bins_collector[i,:][:-1], hist_collector[i,:], width=0.01)
    plt.savefig(str(i)+'.png')
    plt.clf()'''


# plot signal and background histograms together for production files
# input files has dimensions [variables x frequencies]
histo_sig = np.genfromtxt("histo_sig.csv", delimiter=',')
bino_sig = np.genfromtxt("bino_sig.csv", delimiter=',')
histo_bg = np.genfromtxt("histo_bg.csv", delimiter=',')
bino_bg = np.genfromtxt("bino_bg.csv", delimiter=',')

for i in range(histo_sig.shape[0]):
    pltSig = plt.plot(bino_sig[i,:][:-1], histo_sig[i,:], drawstyle='steps-post', color='blue', label='sig')
    pltBg = plt.plot(bino_bg[i,:][:-1], histo_bg[i,:], drawstyle='steps-post', color='red', label='bg')
    plt.title(variables[i])
    plt.legend(loc='upper right')
    plt.savefig(str(i)+'.png')
    plt.clf()
