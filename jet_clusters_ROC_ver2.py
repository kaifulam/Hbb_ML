import numpy as np
import matplotlib.pyplot as plt

variables = ['cluster pt (max energy)', 'cluster eta (max energy)', 'cluster dphi (max energy)', 'cluster energy (max)', 'cluster pt (sum of max 3)', 'cluster energy (sum of max 3)']

#variables = ['cluster pt (max energy)', 'cluster eta (max energy)', 'cluster dphi (max energy)', 'cluster energy (max)']

# input files has dimensions [variables x frequencies]
histo_sig = np.genfromtxt("histo_sig.csv", delimiter=',')
bino_sig = np.genfromtxt("bino_sig.csv", delimiter=',')
histo_bg = np.genfromtxt("histo_bg.csv", delimiter=',')
bino_bg = np.genfromtxt("bino_bg.csv", delimiter=',')

for i in range(histo_sig.shape[0]): # looping variables
    tpr = []
    fpr = []
    sigFreqSum = np.sum(histo_sig[i,:])
    #print i, 'sigFreqSum', sigFreqSum
    bgFreqSum = np.sum(histo_bg[i,:])
    #print i, 'bgFreqSum', bgFreqSum

    for j in range(histo_sig.shape[1]): # looping each histo_sig
        TP = np.sum(histo_sig[i,:j+1])
        FN = np.sum(histo_sig[i,:j+1])
        tpr.append(np.sum(histo_sig[i,j:]) / sigFreqSum)
        fpr.append(np.sum(histo_bg[i,j:]) / bgFreqSum)

    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    pltRoc = plt.plot(np.sort(fpr), np.sort(tpr), drawstyle='steps-post', color='blue')
    pltDiag = plt.plot([0,1],[0,1], 'r--')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title(variables[i])
    plt.savefig(str(i)+'_ROC'+'.png')
    plt.gcf().clear()
