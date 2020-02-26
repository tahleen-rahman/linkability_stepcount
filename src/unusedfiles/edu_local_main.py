import os
from sklearn.ensemble import RandomForestClassifier

from attacks import AttributeInference
from compress import Statistics
from itertools import combinations, chain


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))



classifiers =   [
                RandomForestClassifier(n_estimators=100, random_state=0)
                # LinearRegression()
                #, DenseNeuralNetwork(num_epochs=100, batch_size=24) # batch size should be a divisor of input size (997 is a prime number)
                ]

print ("edu")

stats_arr = ['sum', 'max', 'std', 'medi', 'mean']
stats = list(powerset(stats_arr))
stats_datapath = "../data/dzne/statistics/"

"""

for hr in [0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 24]:

    for st in stats[1:]:  # beacuse 1st element is the null set  #( ['max'],['max', 'std'],['max', 'std', 'medi'], ['max', 'std', 'medi', 'mean']):

        ds = Statistics('dzne_dsp', st, window_size=int(hr * 4 * 60), bucket_size=4, data_path=stats_datapath)

        if not os.path.exists(stats_datapath + ds.out_name + ".ftr") :

            ds.compress_save()


        inf = AttributeInference(ds.out_name, 'edu_binary', in_datapath=stats_datapath)

        if not (os.path.exists(inf.out_datapath + inf.train_fname) and os.path.exists(inf.out_datapath + inf.test_fname)):

            inf.makeDataset()


        for clf in classifiers:

            inf.attack(clf)
"""


for hr in [ 1, 2, 3, 4, 6, 8, 12, 24, 24 * 7 ]:

    for st in stats[1:]:

        ds = Statistics('dzne_week',  st, window_size = hr * 4 * 60, bucket_size=4, data_path=stats_datapath)

        if not os.path.exists(stats_datapath + ds.out_name + ".ftr"):

            ds.compress_save()




        inf = AttributeInference(ds.out_name, 'edu_binary', in_datapath=stats_datapath)

        if not (os.path.exists(inf.out_datapath + inf.train_fname) and os.path.exists(inf.out_datapath + inf.test_fname)):

            inf.makeDataset()

        for clf in classifiers:

            inf.attack(clf)

"""   
embtyp = 'pca' #'agg'  # , 'pca', , 'mlp', '1dcnn'
embname_pref = "_emb"


by = 'day'
units = 7
agg_name = str(units) + by +"_emb"

aggregateTime('dzne', agg_name, by, units )

#daySplitter(step_name = agg_name) # creates  agg_name +'_dsp'

vf=load_frame(agg_name +'_dsp')
input_size = len(vf.columns)-2


for emb_size in [input_size/2, input_size/4, input_size/8]:

    if embtyp=='LSTM':

        epochs = 50
        ae = LSTM_AE(agg_name +'_dsp')
        ae.train_save(int(emb_size), epochs)
        #ae.compress_save(emb_size, epochs)


    elif embtyp=='pca':

        #emb_size = 10

        ae = PCA_AE('dzne_dsp')
        ae.compress_save(emb_size=128)

        emb_name = embtyp + embname_pref + str(emb_size)


ae = PCA_AE('dzne_dsp')
ae.compress_save(emb_size=128)

from utils.visualization import *
one_user_embeddings_distances(vec_name= emb_name)
one_user_TSNE(emb_name)
"""
"""
ds = DistributionsSplitter('dzne_dsp',  bucket_size=4, window_size=4*60)
ds.compress_save()

"""


