import os
import sys

import pandas as pd

from attacks.kerasclassifier import Dense_siameseClassifier
from attacks.Linkability import Link
from utils.storage import saveKerasmodel, loadKerasmodel






epochs, regu, batchsize = 350,0.001,64,# int(sys.argv[1]), float(sys.argv[2]),  int(sys.argv[3]) #

weekend = True
aucfname = str(weekend) + "_"+ str(epochs) + "_" + str(regu) + "_" + str(batchsize) + "_aucs_small.csv"
arr, aucarr = [], []


for infile in os.listdir("../data/dzne/linkdataall/"):

    #infile= 'dzne_dsp'

    print ("infile", infile)

    link = Link(infile, weekends=weekend, in_datapath='../data/dzne/linkdataall/')

    from sklearn.utils import shuffle

    link.tr_pairs = shuffle(link.tr_pairs)

    #batchsize = 128


    clf = Dense_siameseClassifier(link.vecframe.shape[1]-2, regu, combi='l1')

    print ("model set up")


    clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                       link.tr_pairs.label,
                       batch_size=batchsize, epochs=epochs,
                       validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                       link.te_pairs.label), verbose=2)


    y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                    verbose=2)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(link.te_pairs.label, y_pred)

    print (auc)

    aucarr.append(auc)

    arr.append([epochs, regu,  batchsize, infile, auc])

    #aucs.append(auc)
    saveKerasmodel(clf.model, archFile = str(weekend) + "_"+ str(epochs) + "_" + str(regu) + "_" + str(batchsize) + infile +"arch.json",
                   weightsFile =  str(weekend) + "_"+ str(epochs) + "_" + str(regu) + "_" + str(batchsize) + infile + "wts.h5")

    del clf



print (aucarr)

aucs = pd.DataFrame(data=arr)    #names= epochs, regu,  batchsize, infile, auc

aucs.to_csv("../data/dzne/results/" + aucfname, mode='a', header=False, index=False)

print ("saved AUCs to ../data/dzne/results/" + aucfname)

#aucs [0.6132251445998222, 0.6647752043825985]
