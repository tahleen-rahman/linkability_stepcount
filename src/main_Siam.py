# Created by rahman at 19:14 2020-01-26 using PyCharm

import math
import os
import sys

import pandas as pd

from attacks.kerasclassifier2 import *
from attacks.Linkability import Link


def add_padding(vf, padding):
    """
    takes a dataframe and the value whose multiple the padding needs to achieve
    :param vf: input vf
    :param padding: the padding process will add zeros to the front and back of vf so that it is a multiple of this value
    :return: padded vf
    """

    data = vf.iloc[:,2:]
    vec_len = data.shape[1]

    if padding >= vec_len:
        pad_len = padding - vec_len
    else:
        pad_len = vec_len - math.floor(vec_len / padding) * padding
    # self.vec_len += pad_len
    before = pad_len // 2
    after = pad_len - before
    data_pad= pd.np.pad(data, [(0, 0), (before, after)], 'constant')

    df = pd.DataFrame(data= data_pad)

    df.insert(0, 'user', vf['user'])
    df.insert(1, 'desc', vf['desc'])

    return df


exp,  cl = int(sys.argv[1]),  sys.argv[2]

expdict = { 0: (50, 0.001, 64, 'l1', 'linkdata') ,
        1: (150, 0.001, 64,'l1', 'linkdata_med') ,
        2: (350, 0.001, 64,'l1', 'linkdataall')
        }

epochs, regu, batchsize, combi, path = expdict[exp]

clfdict = { 'lstm' : ([[64, 0.2]]),
            'cnn1' : ((16, 6), (16, 6), 8, 1), # layer i (filt size, kernel size) , max poolsize
            'dense' : [0.5, 0.25],
            'cnn2' : ((16, 6), (16, 6), 8, 2)
           }


params =  clfdict[cl]

weekend = True

aucfname = "clf_" + str(cl) + "exp_" + str(exp) + "_siam.csv"

arr, aucarr = [], []


for infile in os.listdir("../data/dzne/" + path):

    try:

        link = Link(infile, weekends=weekend, in_datapath='../data/dzne/'+ path+ '/')

        from sklearn.utils import shuffle

        link.tr_pairs = shuffle(link.tr_pairs)

        if cl == 'cnn1' or cl== 'cnn2':

            link.vecframe = add_padding(link.vecframe, padding=params[2]**params[3])

            clf = CNNsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, params, num_maxpools=params[3])

        elif cl == 'lstm':

            clf = LSTMsiameseClassifier(link.vecframe.shape[1] - 2, combi, lstm_params=params)

        elif cl == 'dense':

            clf = Dense_siameseClassifier(link.vecframe.shape[1]-2, regu, combi, params)

        clf.combine()

        auc = clf.fit_predict(link, batchsize, epochs,  verbose=2)

        print (infile, auc)

        aucarr.append(auc)

        arr.append([epochs, regu,  batchsize, infile, auc])

        del clf

    except:

        print ("infile skipped", infile)

print (aucarr)

aucs = pd.DataFrame(data=arr)# , names= epochs, regu,  batchsize, infile, auc

aucs.to_csv("../data/dzne/results/" + aucfname, mode='a', header=False, index=False)

print ("saved AUCs to ../data/dzne/results/" + aucfname)

