import os
from sklearn.ensemble import RandomForestClassifier

from attacks import AttributeInference

classifiers =   [
                  RandomForestClassifier(n_estimators=100, random_state=0)
                #, LinearRegression()
                #, svm.SVC(gamma='scale', decision_function_shape='ovo')
                #, svm.LinearSVC(max_iter=2000) # May not converge if training data is not normalized
                #, DenseNeuralNetwork(num_epochs=100, batch_size=24) # batch size should be a divisor of input size (997 is a prime number)
                ]

in_datapath = '../data/dzne/normalized/'

for attribute in ['age_binary', 'sex', 'edu_binary']: #, 'age_multi',  'edu_multi']:

    for infile in os.listdir(in_datapath):

        if 'nor' in infile:

            #print (infile)

            inf = AttributeInference(infile[:-4], attribute, in_datapath)

            if not (os.path.exists(inf.out_datapath + inf.train_fname) and os.path.exists(inf.out_datapath + inf.test_fname)):

                inf.makeDataset()


            for clf in classifiers:

                inf.fs_attack(clf, do_vt=True, do_rfe=True, verbose=True)

