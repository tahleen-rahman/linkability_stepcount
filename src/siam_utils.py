# Created by rahman at 14:34 2020-04-06 using PyCharm

# Created by rahman at 19:14 2020-01-26 using PyCharm

import math
import os
import sys

import pandas as pd

from attacks.kerasclassifier2 import *
from attacks.Linkability_crossval import Link


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


def linkability_siam(epochs, regu, batchsize, combi, in_dir, params, exp, cl, datapath = "../data/dzne/"):
    """

    :param epochs:
    :param regu:
    :param batchsize:
    :param combi:
    :param in_dir:
    :param params:
    :param exp:
    :param cl:
    :param datapath :
    :return:
    """

    weekend = True

    aucfname = "clf_" + str(cl) + "_exp_" + str(exp) + "_cv_siam.csv"

     #aucarr = []

    for infile in os.listdir(datapath + in_dir):

        if 'vt' in infile and 'nor' in infile: #only use variance thresholded and normalized files

            print (infile)

            arr=[]

            for i in range(0, 5):

            #try:

                link = Link(i, infile, weekends=weekend, in_datapath=datapath + in_dir , out_datapath = datapath+'newfolds/')

                from sklearn.utils import shuffle
                link.tr_pairs = shuffle(link.tr_pairs)

                #first define the shared layers
                if cl == 'cnn1' or cl == 'cnn2':

                    link.vecframe = add_padding(link.vecframe, padding=params[2] ** params[3])

                    clf = CNNsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, params, num_maxpools=params[3])

                elif cl == 'lstm1':

                    clf = LSTMsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, lstm_params=params, fixed_units=False)

                elif cl == 'lstm2' or cl == 'lstm3':

                    clf = LSTMsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, lstm_params=params, fixed_units=True)

                elif cl == 'dense':

                    clf = Dense_siameseClassifier(link.vecframe.shape[1] - 2, regu, combi, params)

                #Next combine the layers
                clf.combine(plot=False)

                auc = clf.fit_predict(link, batchsize, epochs, verbose=2)

                print(infile, i, auc)

                #aucarr.append(auc)

                arr.append([epochs, regu, batchsize, i, infile, auc])

                del clf

            #except:

                #print("infile skipped", infile)

            aucs = pd.DataFrame(data=arr)  # , names= epochs, regu,  batchsize, i, infile, auc

            aucs.to_csv(datapath + "results/" + aucfname, mode='a', header=False, index=False)

            print("saved AUCs to " + datapath +" results/" + aucfname)
    #print(aucarr)


