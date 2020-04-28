# Created by rahman at 13:18 2019-10-19 using PyCharm

import os
import sys
from attacks.Linkability import Link

datapath= sys.argv[1]




weekends = True

for combi in ['cosine', 'eucl']:

    for infile in os.listdir(datapath):

        try:

            link = Link(infile, weekends, in_datapath=datapath)

            link.unsup_data_fp = link.out_datapath + combi + str(weekends) + infile+ 'weeknd_unsup_data.csv'

            if not (os.path.exists(link.unsup_data_fp) ):
                link.prep_data_unsup(combi)

            print (infile, combi, link.unsup_attack())

        except:

            print (infile, "skipped")



