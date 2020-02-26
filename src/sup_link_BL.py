# Created by rahman at 16:54 2019-10-20 using PyCharm

import os
import sys
from attacks.Linkability import Link
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from attacks import BinaryDNN
from sklearn import svm


folder = sys.argv[1]




names = ["RF", "LR",  "SVM", "LinearSVC", "1layerDense",
         "2layerDense", "3layerDense"]
classifiers =   [
                RandomForestClassifier(n_estimators=100, random_state=0)
                , LinearRegression()
                , svm.SVC(gamma='scale', decision_function_shape='ovo')
                , svm.LinearSVC(max_iter=2000) # May not converge if training data is not normalized
                , BinaryDNN(num_layers = 1, layer_params=[[0.25, 0.2]], num_epochs=5, batch_size=24, verbose=0)
                , BinaryDNN(num_layers = 2, layer_params=[[0.5, 0.2], [0.5, 0.2]], num_epochs=50, batch_size=64, verbose=0)
                , BinaryDNN(num_layers = 3, layer_params=[[0.25, 0.2], [0.25, 0.2], [0.25, 0.2]], num_epochs=5, batch_size=24, verbose=0)
                ]

weekend=True

for infile in os.listdir("../data/dzne/"+ folder ): # all  intervals for single stats + distributions 1, 6, 12 hrs

    #infile= 'dzne_dsp'

    #print ("infile", infile)

    link = Link(infile, weekends=weekend, in_datapath='../data/dzne/' + folder + "/")

    for combi in [ 'l1']: #'sql2', 'mul',

        link.tr_data_fp = link.out_datapath + infile + str(weekend) + 'weeknd_tr_data.csv'

        link.te_data_fp = link.out_datapath + infile + str(weekend) + 'weeknd_te_data.csv'

        if not (os.path.exists(link.tr_data_fp) and os.path.exists(link.te_data_fp)):

            link.prep_data(combi)


        for name, clf in zip( names, classifiers):

            print (infile, name, link.attack(clf))



