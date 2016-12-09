import numpy as np
import matplotlib.pyplot as plt

Histo_Bg = np.genfromtxt("Histo_Bg.csv", delimiter=',')
Bino_Bg = np.genfromtxt("Bins_Bg.csv", delimiter=',')
Histo_Sig = np.genfromtxt("Histo_Sig.csv", delimiter=',')
Bino_Sig = np.genfromtxt("Bins_Sig.csv", delimiter=',')
Features = ["pt", "eta"]

fea_num = Histo_Sig.shape[1]

for i in range(fea_num):
    center = (Bino_Sig[:-1,i] + Bino_Sig[1:,i])/2
    width = 0.7 * (Bino_Sig[1,i] - Bino_Sig[0,i])
    plt.figure(i)
    plt.step(Bino_Bg[:-1,i], Histo_Bg[:,i], color='red', label='bg')
    plt.step(Bino_Sig[:-1,i], Histo_Sig[:,i], color='blue', label='sig')
    #plt.bar(Bino_Sig[:-1,i], Histo_Sig[:,i], color='blue', label='sig', width = 0.1)
    #plt.bar(center, Histo_Sig[:,i], align='center',width=width)
    plt.legend()
    plt.suptitle(Features[i])
    plt.xlim(-5,5)
    plt.savefig(Features[i]+'.png')
    #plt.show(block=True)pt
    plt.close
