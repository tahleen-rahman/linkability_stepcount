# Created by rahman at 11:14 2019-12-19 using PyCharm

import os
import sys
from attacks import AttributeInferenceCV, LSTMclassifier

epochs, batch = int(sys.argv[1]), int(sys.argv[2])  # I recommend (epochs, batch)  = (100, 64) and (50, 32)





inf = AttributeInferenceCV('stats_dzne_dsp_sum24', attribute='edu_binary', in_datapath="../data/dzne/statistics/")

# LSTMs don’t perform well with more than 200-400 time steps based on what I have read.
if not (os.path.exists(inf.out_datapath + inf.train_fname) and os.path.exists(inf.out_datapath + inf.test_fname)):
    inf.makeDataset()

clf = LSTMclassifier( num_layers = 1, layer_params=[[0.5, 0.2], [0.25, 0.2]], num_epochs= epochs , batch_size= batch, verbose=2)
aucs = inf.attack(clf)
print (aucs)
