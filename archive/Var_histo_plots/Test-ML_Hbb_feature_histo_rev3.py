import h5py
import numpy as np

bin_num = 100
Sig_Bg = ['Sig', 'Bg']
h5_Files = ['sig_jets_clusters.h5', 'bg_jets_clusters.h5']
variables = ['pt', 'eta', 'dphi', 'energy', 'weight']
varCt = len(variables)

for i in range(len(Sig_Bg)):
    f = h5py.File("/phys/users/kaifulam/MachineLearning/dguest/h5_files/" + h5_Files[i], 'r')
    ds = f['clusters'][:1000000]
    hist_collector = np.zeros((bin_num, varCt))
    bin_collector = np.zeros((bin_num+1,varCt))

    for j in range(varCt):
        var = ds[variables[j]]
        print(j, variables[j])
        print(var.shape)

        print(variables[j]+".minmax", min(var.flatten()), max(var.flatten()))
        #Gaussian Contrast Normalize pt and eta, across the full 2D array
        var = (var - np.mean(var)) / np.std(var)

        #histrogram
        hist_collector[:,j], bin_collector[:,j] = np.histogram(var.flatten(), bins = bin_num, range=(-1,1))

        print(hist_collector.shape)
        np.savetxt("Histo_" + Sig_Bg[i] + ".csv", hist_collector, delimiter=',')
        np.savetxt("Bins_" + Sig_Bg[i] + ".csv", bin_collector, delimiter=',')
