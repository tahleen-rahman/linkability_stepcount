from sklearn.metrics import roc_auc_score

from attacks.Linkability import Link
from attacks.kerasclassifier2 import InkensClassifier
from utils.storage import saveKerasmodel, loadKerasmodel


class Hyperparameters:

    def __init__(self, l2regularizer, layer_activation = 'relu', optimizer = 'adam', loss = 'binary_crossentropy', metric = 'accuracy'):

        self.l2regularizer = l2regularizer
        self.layer_activation = layer_activation
        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric


def siamese_link(in_vf, epochs, combi, regu, batchsize):

    link = Link(in_vf, weekends=True)
    from sklearn.utils import shuffle

    link.tr_pairs = shuffle(link.tr_pairs)

    archfile = 'weekends' + str(epochs) + "_" + str(batchsize)+"_" + str(regu)  + "_" + combi + "_inks_binary_arch.json"

    weightsfile = 'weekends' + str(epochs) + "_" + str(batchsize)+"_" + str(regu)  + "_" + combi + "_inks_binary_wts.h5"

    h = Hyperparameters(regu)

    clf = InkensClassifier(link.vecframe.shape[1] - 2, h, combi)
    print("model set up")

    clf.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                  link.tr_pairs.label,
                  batch_size=batchsize, epochs=epochs,
                  validation_data=(
                  [link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                  link.te_pairs.label), verbose=2)


    y_pred = clf.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                    verbose=1)


    auc = roc_auc_score(link.te_pairs.label, y_pred)

    saveKerasmodel(clf.model, archFile=archfile, weightsFile=weightsfile)
    del clf
    del h

    return auc



def test_siamese_link(in_vf, epochs, combi, regu, batchsize):

    link = Link(in_vf, weekends=True)

    archfile = 'weekends' + str(epochs) + "_" + str(batchsize)  +"_" + str(regu) +  "_" + combi + "_inks_binary_arch.json"

    weightsfile = 'weekends' + str(epochs) + "_" + str(batchsize) + "_" + str(regu) + "_" + combi + "_inks_binary_wts.h5"

    model = loadKerasmodel(archFile = archfile, weightsFile = weightsfile)

    y_pred = model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                    verbose=1)


    auc = roc_auc_score(link.te_pairs.label, y_pred)
    print (epochs, batchsize, combi, regu, auc)
    return auc
