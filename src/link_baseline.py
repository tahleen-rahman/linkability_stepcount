# Created by rahman at 16:54 2019-10-20 using PyCharm

import os
import sys
from attacks.Linkability import Link
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from attacks import BinaryDNN
from sklearn import svm
from link_utils import linkability_bl




expdict = { 0: (100, 'linkdata_0/', 0.005) ,
            1: (100, 'linkdata_1/', 0.001) ,
            2: (50,  'linkdata_2/', 0.0),
            3: (10,  'linkdata_3/', 0.0),
            4: (100, 'linkdata_dist/', 0.0)
          }



def link_baseline(exp,  cl, server, weekend):

    trees, in_dir, var_th = expdict[exp]

    clfdict = {'rf': RandomForestClassifier(n_estimators=trees, random_state=0),
               'lr': LinearRegression(),
               'svm': svm.SVC(gamma='scale', decision_function_shape='ovo'),
               'lsvc': svm.LinearSVC(max_iter=2000),  # May not converge if training data is not normalized
               'dense1': BinaryDNN(num_layers=1, layer_params=[[0.25, 0.2]], num_epochs=100, batch_size=64, verbose=0),
               'dense2': BinaryDNN(num_layers=2, layer_params=[[0.5, 0.2], [0.5, 0.2]], num_epochs=100, batch_size=64,
                                   verbose=0),
               'dense3': BinaryDNN(num_layers=3, layer_params=[[0.25, 0.2], [0.25, 0.2], [0.25, 0.2]], num_epochs=100,
                                   batch_size=64, verbose=0)
               }
    clf = clfdict[cl]


    if server:
        datapath="../../stepcount/data/dzne/"
    else:
        datapath="../data/dzne/"

    path = datapath + in_dir

    from prep_features import *

    #path = filter_mornings(path, f=0.25)
    in_path = variance_thresholding(path, th=var_th)


    linkability_bl(in_path, datapath, cl, clf, exp, weekend)



if __name__ == '__main__':

    exp,  cl, server, weekend = int(sys.argv[1]),  sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

    link_baseline(exp, cl, server, weekend)
