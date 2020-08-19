import os
os.environ['PYTHONHASHSEED'] = str(1)
from numpy.random import seed
seed(1)
import random
random.seed(2)
import tensorflow as tf
tf.random.set_seed(3)

from link_utils import linkability_siam
import sys
from prep_features import *

max_epochs, regu, batchsize, combi = 300, 0.001, 256, 'l1'


expdict = { 0: (10, 'linkdata_0/', 0.005), # run this on GPU only,
            1: (10, 'linkdata_1/', 0.001),
            2: (10, 'linkdata_2/', 0.0),
            3: (5,  'linkdata_3/', 0.0),
            4: (10, 'linkdata_dist/', 0.0)
          }



clfdict = { 'lstm88' : ([[8, 0.2], [8, 0.2]]),  # list of type [lstm units, dropout] of size = num of stacked lstm layers
            'lstm168' : ([[16, 0.2], [8, 0.2]]),
            'lstm1616' : ([[16, 0.2], [16, 0.2]]),
            'lstm2' : ([[16, 0.2]]), #for medium files
            'lstm3' : ([[8, 0.2]]),  #for the big files
            'cnn1'   : ((16, 6), (16, 6), 8, 1), # layer i (filt size, kernel size) , max poolsize
            'dense'  : [0.5, 0.25],  #[frac of inputsize]
            'cnn2'   : ((16, 6), (16, 6), 8, 2),
            'bilstm16': ([[16, 0.2]]),
            'bilstm8': ([[8, 0.2]]),
            'attntn': ([[8, 0.2]])
            }



def link_siamese(exp, cl, server, weekend, parallelize):

    patience, in_dir, var_th = expdict[exp]
    params = clfdict[cl]

    config = [max_epochs, patience, regu, batchsize, combi]

    if server:
        datapath="../../stepcount/data/dzne/"
    else:
        datapath="../data/dzne/"

    path = datapath + in_dir

    in_path = variance_thresholding(path, th=var_th)

    linkability_siam(config, in_path, params, exp, cl, weekend, datapath,  callback=True, parallelize=parallelize)



if __name__ == '__main__':

    exp, cl, server, weekend, parallelize = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])# 2, 'attntn', 0, 1, 0#

    link_siamese(exp, cl, server, weekend, parallelize)
