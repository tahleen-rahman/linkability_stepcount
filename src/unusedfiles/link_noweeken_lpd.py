import os
import sys

from attacks.kerasclassifier2 import InkensClassifier
from attacks.Linkability import Link
from utils.storage import saveKerasmodel, loadKerasmodel
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle


class Hyperparameters:

    def __init__(self, l2regularizer, layer_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy', metric = 'accuracy'):

        self.l2regularizer = l2regularizer
        self.layer_activation = layer_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric


epochs, batchsize, combi = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]


link = Link('dzne_dsp', weekends=False)
link.tr_pairs = shuffle(link.tr_pairs)



for regs in [ 0.01,  0.001]:

    archfile = 'weekends' + str(epochs) + "_" + str(batchsize)  +"_" + str(regs) +  "_" + combi + "_inks_binary_arch.json"

    weightsfile = 'weekends' + str(epochs) + "_" + str(batchsize) + "_" + str(regs) + "_" + combi + "_inks_binary_wts.h5"

    h = Hyperparameters(regs)

    clf = InkensClassifier(link.vecframe.shape[1]-2, h, combi )
    print ("model set up")


    clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                       link.tr_pairs.label,
                       batch_size=batchsize, epochs=epochs,
                       validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                       link.te_pairs.label), verbose=2)

    y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]], verbose=2)

    auc = roc_auc_score(link.te_pairs.label, y_pred)
    print("regu, combi, auc", regs, combi, auc)

    saveKerasmodel(clf.model, archFile = archfile,
                       weightsFile = weightsfile)

    del clf.model

    del clf


print ("No weekend")
print ("epochs, batchsize, combi, regs, auc ")



for regs in [0.01, 0.001]:

    archfile = 'weekends' + str(epochs) + "_" + str(batchsize)  +"_" + str(regs) +  "_" + combi + "_inks_binary_arch.json"

    weightsfile = 'weekends' + str(epochs) + "_" + str(batchsize) + "_" + str(regs) + "_" + combi + "_inks_binary_wts.h5"

    model = loadKerasmodel(archFile = archfile, weightsFile = weightsfile)

    y_pred = model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                    verbose=1)


    auc = roc_auc_score(link.te_pairs.label, y_pred)
    print (epochs, batchsize, combi, regs , auc)





