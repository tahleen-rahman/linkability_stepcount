import os
os.environ['PYTHONHASHSEED'] = str(1)
from numpy.random import seed
seed(1)
import random
random.seed(2)
import tensorflow as tf
tf.random.set_seed(3)

from link_utils import linkability_siam, core_siamese
import sys
from prep_features import *

max_epochs, regu, batchsize, combi = 300, 0.001, 64, 'l1'


clfdict = { 'lstm1' : ([[8, 0.2], [8, 0.2]]),  # list of type [lstm units, dropout] of size = num of stacked lstm layers
            'lstm2' : ([[16, 0.2], [8, 0.2]]),
            'lstm3' : ([[16, 0.2], [16, 0.2]]),
            'bilstm': ([[16, 0.2]]),
            'bilstm': ([[10, 0.2]])
            }




def link_siamese(cl, server):

    patience, in_dir, var_th = 10, 'linkdata_2/', 0.0 #expdict[exp]
    params = clfdict[cl]


    if server:
        datapath="../../stepcount/data/dzne/"
    else:
        datapath="../data/dzne/"

    path = datapath + in_dir

    in_datapath = variance_thresholding(path, th=var_th)


    #linkability_siam(config, in_path, params, exp, cl, weekend, datapath,  callback=True, parallelize=parallelize)

    aucfname = "weekend_" + "clf_" + str(cl)  + "_cv_siam.csv"

    infile = "stats_dzne_dsp_max120_nor_vt"

    core_siamese(infile, params, cl, datapath, in_datapath, True, aucfname, True, max_epochs, patience, regu,
             batchsize, combi)

if __name__ == '__main__':

    cl, server = int(sys.argv[1]), int(sys.argv[3])

    link_siamese(cl, server)


