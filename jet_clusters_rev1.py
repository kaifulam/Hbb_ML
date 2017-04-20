'''
KEY
[jet pt, jet eta, weight,
[
[[track pt, track eta, dphi between jet and track], [d0, d0 significance, z0, z0 significance], [track chi2, track ndf, number of b-layer hits, b-layer hit expecte\
d, number of first pixel layer hits, number second pixel layer hits, number of pixel hits, number of sct hits]],
...
]
,
[[cluster pt, cluster eta, cluster dphi with respect to jet, cluster energy],
...
]
,
[tau21, c1, c2, c1_beta2, c2_beta2, d2, d2_beta2],
[[[jet pt, jet eta, dphi with respect to fat jet], [jet_jf_m, jet_jf_efc, jet_jf_deta, jet_jf_dphi, jet_jf_ntrkAtVx, jet_jf_nvtx, jet_jf_sig3d, jet_jf_nvtx1t, jet_\
jf_n2t, jet_jf_VTXsize], [jet_sv1_ntrkj, jet_sv1_ntrkv, jet_sv1_n2t, jet_sv1_m, jet_sv1_efc, jet_sv1_normdist, jet_sv1_Nvtx, jet_sv1_sig3d], [ip3d b likelihood, ip\
3d c likelihood, ip3d light likelihood], [jet_mv2c10, jet_mv2c20]], ... ]]
'''

import numpy as np

jc_collect = [] # collect jet cluster results (one entry per jet; modified by max or sum or etc.)
wt_collect = [] # collect Weights of jets

#filename = "/phys/groups/tev/scratch4/users/kaifulam/dguest/hbb/v1/signal.txt"
filename = "/phys/groups/tev/scratch4/users/kaifulam/dguest/hbb/v1/background_3M.txt"
#filename = "kaifu_one_jet.txt"
#filename = "kaifu_test_signal.txt"

f = open(filename,"r")

i=-1
for line in f.readlines():

    if i == 2500000:
        break
    i+=1
    if i % 100000 == 0:
        print "@ line# i = ", i

    #print "line length: ",len(line)
    #print "line: "
    #print line

    # Splitting: Jets
    split1 = line.split("[[[")
    #print "split1: ", len(split1)
    if len(split1) != 3: # To skip lines without tracks and high level variables
        continue
    jet=split1[0].replace("[","")
    jet = jet[:-2]
    #print "============================"
    #print "JET:"
    #print jet
    #print "============================"

    jv = jet.split(',')     #split jet variables
    jet_pt = jv[0]
    jet_eta = jv[1]
    weight = jv[2]

    jet_pt, jet_eta, weight = float(jet_pt), float(jet_eta), float(weight)

    #print "weight:", weight
    wt_collect.append(weight)

    # Spliting:  Tracks
    component="[[["+split1[1]
    split2 = component.split("]]]")
    tracks=split2[0]+"]]]"

    #print "============================"
    #print "TRACKS:"
    #print tracks
    #print "============================"

#Splitting: Clusters
    component1="[[["+split1[1]
    split2=component1.split("]]], ");
    #print "split2: "
    #print split2
    if len(split2) != 2: # To skip lines without tracks
        continue
    component2=split2[1]
    split3 = component2.split("]],")
    component3=split3[0]+"]]"
    clusters=component3

    #print "============================"
    #print "CLUSTERS:"
    #print clusters
    #print "============================"

    jet_clusters = []

    cs1 = clusters.split("[[")
    cs2 = cs1[1]
    cs3 = cs2.split("], ")

    j = 0
    for item in cs3:
        cs4 = item.split(", ")

        pt = cs4[0]
        eta = cs4[1]
        dphi = cs4[2]
        energy = cs4[3]

        if pt[0] == '[':
            pt = pt.split('[')[1]

        if energy[-1] == ("]"):
            energy = energy.split("]]")[0]

        pt, eta, dphi, energy = float(pt), float(eta), float(dphi), float(energy)

        jet_clusters.append([pt, eta, dphi, energy])
        j+=j

    #print "jet_clusters: ", jet_clusters

    #OPTION: SUM or MAX
    ## SUM
    #Sum All by column (variable) of each jet
    #jet_clusters = np.asarray(jet_clusters)
    #jet_clusters_sum = jet_clusters.sum(axis=0)
    #print jet_clusters_sum
    #jc_collect.append(jet_clusters_sum)

    ## MAX
    #Max entry for each jet by energy
    jet_clusters = np.asarray(jet_clusters)
    jet_clusters_max = jet_clusters[np.argmax(jet_clusters, axis=0)[3],:]

    #convert first 4 variable values to absolute values
    jet_clusters_max = np.absolute(jet_clusters_max)

    #find the lategest cluster pt and cluster energies and add them together
    cluster_pt = jet_clusters[:,0]
    #print 'cluster_pt', cluster_pt
    cluster_energy = jet_clusters[:,-1]
    #print 'cluster_energy', cluster_energy

    if cluster_pt.shape[0] > 3:
        # sort, find max 3
        #print 'if loop'
        cluster_pt = np.sort(cluster_pt)
        cluster_energy = np.sort(cluster_energy)
        #print 'cluster_pt_sort', cluster_pt
        #print 'cluster_energy_sort', cluster_energy
        cluster_pt[:] = cluster_pt[::-1]
        cluster_energy[:] = cluster_energy[::-1]
        #print 'cluster_pt_sorted', cluster_pt
        #print 'cluster_energy_sorted', cluster_energy
        top_3_pt = np.sum(cluster_pt[0:3])
        top_3_energy = np.sum(cluster_energy[0:3])
        #print 'top_3_pt', top_3_pt
        #print 'top_3_energy', top_3_energy
    else:
        # less than 3 clusters in jet, just sum all the cluster_pt and cluster_energy
        #print 'else loop'
        top_3_pt = np.sum(cluster_pt)
        top_3_energy = np.sum(cluster_energy)
        #print 'top_3_pt', top_3_pt
        #print 'top_3_energy', top_3_energy

    # attach cluster_pt and cluster_energy to the right of jet_cluster_max
    jet_clusters_result = np.concatenate((jet_clusters_max, np.array([top_3_pt, top_3_energy])))

    #print "jet_cluster_result", jet_clusters_result
    jc_collect.append(jet_clusters_result)

jc_collect = np.asarray(jc_collect)
wt_collect = np.asarray(wt_collect)
#print "jc_collect: ", jc_collect
#print "wt_collect:", wt_collect

#histogram
hist_collector = []
bins_collector = []

_binning = [[101,0,1000],[101,0,4],[101,0,1],[101,0,2000], [101,0,1000], [101,0,2000]]

# jet_clusters_rev1.py produce histogram data for 6 variables:
# cluster pt (max energy), cluster eta (max energy), cluster dphi (max energy),
# cluster energy (max energy), cluster pt (sum max 3), cluster energy (sum max 3)

for j in range(jc_collect.shape[1]):

    nbin, low, high = _binning[j]
    bins = np.linspace(low, high, nbin)
    hist, bins = np.histogram(jc_collect[:,j], weights = wt_collect, normed=True, bins=bins)
    hist_collector.append(hist)
    bins_collector.append(bins)

hist_collector = np.asarray(hist_collector)
bins_collector = np.asarray(bins_collector)
print "hist_c.shape", hist_collector.shape
print "bins_c.shape", bins_collector.shape

np.savetxt("hist_collector.csv", hist_collector, delimiter=',')
np.savetxt("bins_collector.csv", bins_collector, delimiter=',')
