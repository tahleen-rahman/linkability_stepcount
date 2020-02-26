# Created by rahman at 19:14 2020-02-26 using PyCharm


import os
import sys

TF_FORCE_GPU_ALLOW_GROWTH=1

from joblib import Parallel, delayed

from attacks.lstm_parallel_utils import  train_lstm

exp =  int(sys.argv[1])

expdict = { 3: (350, 0.001, 64, 'linkdatalstm3') ,
            1: (350, 0.001, 64, 'linkdatalstm') ,
            2: (350, 0.001, 64, 'linkdatalstm2'),
            4: (350, 0.001, 64, 'linkdatalstm_rem')
            }
epochs, regu, batchsize, path = expdict[exp]


weekend = True
lstm_params=[[64, 0.2]]

aucfname =  str(exp) + "_" + str(epochs) + "_" + str(regu) + "_" + str(batchsize) + "_aucsLSTM_Linkability.csv"


threads = len(os.listdir("../data/dzne/"+path) )

Parallel(n_jobs=threads)(delayed(train_lstm)(infile, weekend, path, lstm_params, batchsize, epochs, regu)
                         for infile in os.listdir("../data/dzne/"+path))


