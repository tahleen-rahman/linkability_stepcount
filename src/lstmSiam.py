# Created by rahman at 12:05 2020-02-19 using PyCharm


import os
import sys

import pandas as pd

from attacks.kerasclassifier import LSTMsiameseClassifier
from attacks.Linkability import Link
from utils.storage import saveKerasmodel, loadKerasmodel




exp = int(sys.argv[1])

expdict = { 0: (50, 0.001, 64, 'linkdata') ,
            1: (150, 0.001, 64, 'linkdata_med') ,
            2: (350, 0.001, 64, 'linkdataall')
            }
epochs, regu, batchsize, path = expdict[exp]


weekend = True
lstm_params=[[64, 0.2]]

aucfname =  str(epochs) + "_" + str(regu) + "_" + str(batchsize) + "_aucsLSTM_Linkability.csv"
arr, aucarr = [], []



for infile in os.listdir("../data/dzne/"+path):

    try:

        print ("infile", infile)

        link = Link(infile, weekends=weekend, in_datapath='../data/dzne/' + path + '/')

        from sklearn.utils import shuffle

        link.tr_pairs = shuffle(link.tr_pairs)

        clf = LSTMsiameseClassifier(link.vecframe.shape[1] - 2, combi='l1', lstm_params=lstm_params)

        print ("model set up")

        clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                           link.tr_pairs.label,
                           batch_size=batchsize, epochs=epochs,
                           validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                           link.te_pairs.label), verbose=0)


        y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]], verbose=0)

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(link.te_pairs.label, y_pred)

        print (auc)

        aucarr.append(auc)

        arr.append([epochs, regu,  batchsize, infile, auc])

        del clf

    except:

        print("infile skipped", infile)



print (aucarr)

aucs = pd.DataFrame(data=arr)    #names= epochs, regu,  batchsize, infile, auc

aucs.to_csv("../data/dzne/results/" + aucfname, mode='a', header=False, index=False)

print ("saved AUCs to ../data/dzne/results/" + aucfname)


