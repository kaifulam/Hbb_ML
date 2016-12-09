import h5py
import numpy as np

bin_num = 100
Sig_Bg = ['Sig', 'Bg']
h5_Files = ['sig_jets.h5', 'bg_jets.h5']

for i in range(len(Sig_Bg)):
    f = h5py.File("/phys/users/kaifulam/MachineLearning/dguest/h5_files/" + h5_Files[i], 'r')
    ds = f['clusters'][:1000000]
    pt = ds['pt']
    eta = ds['eta']

    print("minmax.eta", min(eta), max(eta))
    #Gaussian Contrast Normalize pt and eta, across the full 2D array
    '''pt = (pt - np.mean(pt)) / np.std(pt)
    eta = (eta - np.mean(eta)) / np.std(eta)

    del f, ds

    #histrogram_Signal
    hist_collector = np.zeros((bin_num,2))
    bin_collector = np.zeros((bin_num+1,2))
    hist_collector[:,0], bin_collector[:,0] = np.histogram(pt.flatten(), bins = bin_num)
    hist_collector[:,1], bin_collector[:,1] = np.histogram(eta.flatten(), bins = bin_num)

    np.savetxt("Histo_" + Sig_Bg[i] + ".csv", hist_collector, delimiter=',')
    np.savetxt("Bins_" + Sig_Bg[i] + ".csv", bin_collector, delimiter=',')'''
