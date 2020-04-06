from siam_utils import linkability_siam
import sys

exp,  cl = int(sys.argv[1]),  sys.argv[2]

expdict = { 0: (50, 0.001, 64, 'l1', 'linkdata') ,
        1: (150, 0.001, 64,'l1', 'linkdata_med') ,
        2: (350, 0.001, 64,'l1', 'linkdataall')
        }

epochs, regu, batchsize, combi, path = expdict[exp]

clfdict = { 'lstm' : ([[0.5, 0.2], [0.25, 0.2]]),  # list of size = num of lstm layers [lstm units as frac of inputsize, dropout]
            'cnn1' : ((16, 6), (16, 6), 8, 1), # layer i (filt size, kernel size) , max poolsize
            'dense' : [0.5, 0.25], #[frac of inputsize]
            'cnn2' : ((16, 6), (16, 6), 8, 2)
           }

params = clfdict[cl]

linkability_siam(epochs, regu, batchsize, combi, path, params, exp, cl)
