# Created by rahman at 18:30 2020-02-26 using PyCharm

from attacks.kerasclassifier import LSTMsiameseClassifier, CuDNNLSTMsiameseClassifier
from attacks.Linkability import Link
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score


def train_lstm(infile, weekend, path, lstm_params, batchsize, epochs, regu):

    try:

        link = Link(infile, weekends=weekend, in_datapath='../data/dzne/' + path + '/')


        link.tr_pairs = shuffle(link.tr_pairs)

        clf = LSTMsiameseClassifier(link.vecframe.shape[1] - 2, combi='l1', lstm_params=lstm_params)

        print ("model set up")

        clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                           link.tr_pairs.label,
                           batch_size=batchsize, epochs=epochs,
                           validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                           link.te_pairs.label), verbose=0)


        y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]], verbose=0)


        auc = roc_auc_score(link.te_pairs.label, y_pred)

        print (epochs, regu,  batchsize,infile, auc)


        del clf

    except:

        print("infile skipped", infile)
