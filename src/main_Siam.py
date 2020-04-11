import os


os.environ['PYTHONHASHSEED'] = str(1)
from numpy.random import seed
seed(1)
import random
random.seed(2)
from tensorflow import set_random_seed
set_random_seed(3)


from siam_utils import linkability_siam
import sys



exp,  cl = int(sys.argv[1]),  sys.argv[2]

expdict = { 0: (200, 0.001, 64,'l1', 'linkdata_0', 0.025) , # run this on GPU only,
            1: (200, 0.001, 64,'l1', 'linkdata_1', 0.01) ,
            2: (100, 0.001, 64,'l1', 'linkdata_2', 0.0),
            2: (300, 0.001, 64,'l1', 'linkdata_3', 0.0)
        }

epochs, regu, batchsize, combi, in_dir, var_th = expdict[exp]

clfdict = { 'lstm_1' : ([[0.5, 0.2]]),  # list of size = num of lstm layers [lstm units as frac of inputsize, dropout]
            'lstm_2' : ([[16, 0.2]]),
            'lstm_3' : ([[10, 0.2]]),
            'cnn1'   : ((16, 6), (16, 6), 8, 1), # layer i (filt size, kernel size) , max poolsize
            'dense'  : [0.5, 0.25], #[frac of inputsize]
            'cnn2'   : ((16, 6), (16, 6), 8, 2)
           }

params = clfdict[cl]

from prep_features import variance_thresholding
variance_thresholding(in_dir, th=var_th)


linkability_siam(epochs, regu, batchsize, combi, in_dir, params, exp, cl, datapath="../../stepcount/data/dzne/")

