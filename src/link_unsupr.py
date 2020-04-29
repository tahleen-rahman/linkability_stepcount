# Created by rahman at 13:18 2019-10-19 using PyCharm

import os
import sys
from attacks.Linkability import Link
from link_utils import linkability_unsup

metric, server, weekend = sys.argv[1], int(sys.argv[2]),  int(sys.argv[3])



if server:
    datapath = "../../stepcount/data/dzne/"
else:
    datapath = "../data/dzne/"



expdict = { 0: ('linkdata_0/', 0.005) , # run this on GPU only,
            1: ('linkdata_1/', 0.001) ,
            2: ('linkdata_2/', 0.0),
            3: ('linkdata_3/', 0.0),
            4: ('linkdata_dist/', 0.0)
          }

for exp in range(0, 5):

    in_dir, var_th = expdict[exp]

    path = datapath + in_dir

    from prep_features import *

    #path = filter_mornings(path, f=0.25)
    in_path = variance_thresholding(path, th=var_th)

    linkability_unsup(in_path, datapath, metric, exp, weekend)




