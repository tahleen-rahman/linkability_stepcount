import math
import os
import sys

import pandas as pd

from attacks.kerasclassifier import Dense_siameseClassifier, CNNsiameseClassifier
from attacks.Linkability import Link
from utils.storage import saveKerasmodel, loadKerasmodel


def add_padding(vf, padding):
    """
    takes a dataframe and the value whose multiple the padding needs to achieve
    :param vf: input vf
    :param padding: the padding process will add zeros to the front and back of vf so that it is a multiple of this value
    :return: padded vf
    """

    data = vf.iloc[:,2:]
    vec_len = data.shape[1]

    pad_len = vec_len - math.floor(vec_len / padding) * padding
    # self.vec_len += pad_len
    before = pad_len // 2
    after = pad_len - before
    data_pad= pd.np.pad(data, [(0, 0), (before, after)], 'constant')

    df = pd.DataFrame(data= data_pad)

    df.insert(0, 'user', vf['user'])
    df.insert(1, 'desc', vf['desc'])

    return df


epochs, regu, batchsize =  350,0.001,64#int(sys.argv[1]), float(sys.argv[2]),  int(sys.argv[3]) #

cnn_params = ((16, 6), (16, 6), 8, 1) # layer i (filt size, kernel size) , max poolsize , num_maxpools

num_maxpools = cnn_params[3]
weekend = True


aucfname = str(num_maxpools) + "maxplools_"+ str(epochs) + "_" + str(regu) + "_" + str(batchsize) + "_aucs_small_CNNsiam.csv"
arr, aucarr = [], []


for infile in os.listdir("../data/dzne/linkdataall/"):

    #infile= 'dzne_dsp'

    print ("infile", infile)

    link = Link(infile, weekends=weekend, in_datapath='../data/dzne/linkdataall/')

    link.vecframe = add_padding(link.vecframe, padding=cnn_params[2]**num_maxpools)

    from sklearn.utils import shuffle

    link.tr_pairs = shuffle(link.tr_pairs)

    clf = CNNsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi='l1', cnn_params=cnn_params)

    print ("model set up")


    clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                       link.tr_pairs.label,
                       batch_size=batchsize, epochs=epochs,
                       validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                       link.te_pairs.label), verbose=0)


    y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                    verbose=0)

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(link.te_pairs.label, y_pred)

    print (auc)

    aucarr.append(auc)

    arr.append([epochs, regu,  batchsize, infile, auc])

    #aucs.append(auc)
    #saveKerasmodel(clf.model, archFile = str(weekend) + "_"+ str(epochs) + "_" + str(regu) + "_" + str(batchsize) + infile +"arch.json",
               #    weightsFile =  str(weekend) + "_"+ str(epochs) + "_" + str(regu) + "_" + str(batchsize) + infile + "wts.h5")

    del clf



print (aucarr)

aucs = pd.DataFrame(data=arr)    #names= epochs, regu,  batchsize, infile, auc

aucs.to_csv("../data/dzne/results/" + aucfname, mode='a', header=False, index=False)

print ("saved AUCs to ../data/dzne/results/" + aucfname)

"""cnn cnn_params=((16, 6), (16, 6), 8), num_epoch=50, batch_size=50
cnn cnn_params=((16, 21), (16, 9), 2), num_epoch=100, batch_size=32
lstm num_layers=1, layer_params=[[64, 0.2]], num_epoch=100, batch_size=64
lstm num_layers=1, layer_params=[[32, 0.2]], num_epoch=100, batch_size=32"""


