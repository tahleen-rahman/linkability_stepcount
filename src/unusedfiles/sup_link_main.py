import os
import sys

from attacks.kerasclassifier import InkensClassifier
from attacks.Linkability import Link
from utils.storage import saveKerasmodel, loadKerasmodel


class Hyperparameters:

    def __init__(self, l2regularizer, combi, layer_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy', metric = 'accuracy'):

        self.l2regularizer = l2regularizer
        self.layer_activation = layer_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric
        self.combi = combi


epochs, batchsize = 200, 128 #int(sys.argv[1]),˚˚ int(sys.argv[2])





link = Link('dzne_dsp', weekends=True)


from sklearn.utils import shuffle

link.tr_pairs = shuffle(link.tr_pairs)

for regs in  [0.01, 0.001]: #[0.1, 0.01,

    for combi in ['l1', 'avg', 'sql2', 'mul']:

        archfile = 'weekends' +  str(epochs) + "_" + str(batchsize) + "_" + str(regs) + "_" + combi + "_inks_binary_arch.json"
        weightsfile =  'weekends' + str(epochs) + "_" + str(batchsize) + "_" + str(regs)+ "_" + combi + "_inks_binary_wts.h5"

        h = Hyperparameters(regs, combi)

        clf = InkensClassifier(link.vecframe.shape[1]-2, h)
        print ("model set up")


        clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                       link.tr_pairs.label,
                       batch_size=batchsize, epochs=epochs,
                       validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                       link.te_pairs.label), verbose=2)





        y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]], verbose=2)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(link.te_pairs.label, y_pred)
        print ("regu, combi, auc", regs, combi, auc)


        saveKerasmodel(clf.model, archFile = archfile,
                       weightsFile = weightsfile)

        del h

        del clf





for regs in  [0.01, 0.001]: #[0.1, 0.01,

    for combi in ['sql2', 'mul', 'l1', 'avg']:

        archfile = 'weekends' + str(epochs) + "_" + str(batchsize) + "_" + str(regs) + "_" + combi + "_inks_binary_arch.json"
        weightsfile =  'weekends' + str(epochs) + "_" + str(batchsize) + "_" + str(regs)+ "_" + combi + "_inks_binary_wts.h5"

        model = loadKerasmodel(archFile = archfile,
                               weightsFile =weightsfile)

        y_pred = model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                    verbose=2)

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(link.te_pairs.label, y_pred)
        print ( regs, combi, auc)





