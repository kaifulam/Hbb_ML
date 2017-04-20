#The goal of this code is to create the matrix Training_input

import h5py
import numpy as np

Sig_Bg = ['Sig', 'Bg']
h5_Files = ['sig_jets_clusters.h5', 'bg_jets_clusters.h5']

#create input matrix
variables = ['pt', 'eta', 'dphi', 'energy', 'weight']
varCt = len(variables)
Training_input = np.zeros((1))

for i in range(len(Sig_Bg)):
    file_Sig = h5py.File("/phys/users/kaifulam/MachineLearning/dguest/h5_files/" + h5_Files[i], 'r')
    ds_Sig = file_Sig['clusters'][:1000000]
    print(ds_Sig.dtype.names)
    merged = np.zeros((1))

    #create a ndarray "merged" that has a dimension  [1,000,000, 40, varCt]
    for j in range(varCt):
        var = ds_Sig[variables[j]]
        var = (var - np.mean(var)) / np.std(var)
        print(variables[j], var.shape)

        if len(merged.shape) == 1:
            merged = var
            print("merged.shape", merged.shape)
        elif len(merged.shape) == 2:
            merged = np.concatenate((merged[..., None], var[..., None]), axis = 2)
            print("merged.shape", merged.shape)
        else:
            merged = np.concatenate((merged ,var[..., None]), axis = 2)
            print("merged.shape", merged.shape)

    #Create Training_input variable that has a dimension [2,000,000, 40, varCt]
    #The first 1M rows are Signal, then the remaining 1M rows are background
    if len(Training_input.shape) == 1:
        Training_input = merged
        print("Training_input.shape", Training_input.shape)
    else:
        Training_input = np.concatenate((Training_input, merged), axis = 0)
        print("Training_input.shape", Training_input.shape)

    del file_Sig, ds_Sig

np.savetxt("Training_input.csv", Training_input.flatten(), delimiter=',')
