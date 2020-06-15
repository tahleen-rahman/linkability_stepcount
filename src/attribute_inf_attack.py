# Created by rahman at 15:54 2019-12-26 using PyCharm

import os
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from attacks import BinaryDNN, LSTMclassifier
from sklearn import svm

from attacks import AttributeInferenceCV



in_datapath = sys.argv[1]



classifiers =   [
                  RandomForestClassifier(n_estimators=100, random_state=0)
                , LinearRegression()
                , svm.SVC(gamma='scale', decision_function_shape='ovo')
                , svm.LinearSVC(max_iter=2000) # May not converge if training data is not normalized
                , BinaryDNN(num_layers = 1, layer_params=[[0.25, 0.2]], num_epochs=5, batch_size=24, verbose=0)
                , BinaryDNN(num_layers = 2, layer_params=[[0.5, 0.2], [0.25, 0]], num_epochs=150, batch_size=64, verbose=0)
                , BinaryDNN(num_layers = 3, layer_params=[[2, 0.2], [0.5, 0.2], [0.25, 0]], num_epochs=150, batch_size=64, verbose=0)
                , LSTMclassifier( num_layers = 1, layer_params=[[0.5, 0.2], [0.25, 0.2]], num_epochs=50 , batch_size=32, verbose=2)# (100, 64) and (50, 32)
                ]


for attribute in ['age_binary', 'sex', 'edu_binary']: #, 'age_multi',  'edu_multi']:

    for infile in os.listdir(in_datapath):

        if 'nor' in infile and 'dist' in infile:

            print (infile)

            inf = AttributeInferenceCV(infile[:-4], attribute, in_datapath)

            for i in range(0,5):

                if not (os.path.exists(inf.out_datapath + str(i) + '/' + inf.train_fname) and os.path.exists(inf.out_datapath + str(i) + '/' +  inf.test_fname)):

                    inf.makeDataset()


            for clf in classifiers:

                aucs = inf.fs_attack(clf, do_vt=True, do_rfe=True, verbose=True)

                print (aucs)

                #RFE for neural networks is not required, is done internally

                x,y = inf.attack(clf)

                print ("AUCS", x,y)
