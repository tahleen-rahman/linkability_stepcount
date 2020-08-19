# Created by rahman at 15:14 2019-10-26 using PyCharm

import pandas as pd
from utils.plot_utils import *

"""
# Following commented blocks are used to generate the plots for preliminary analysis in the Dataset section of the paper 
# and can be uncommented according to requirement

### plot pairs of embbeddings for user ###
from utils.storage import load_frame
emb1 = load_frame('dzne_dsp')
emb2 = load_frame('1minute_emb_dsp_nor_LSTM_out_20_100')
user = 1000

plot_embpair(emb1.iloc[0,:], emb1.iloc[0,:], emb1.columns, step = 10)




### plot stepcounts per day for a week for 1 user as in dzne.csv ###

epochs = pd.read_csv( "../data/dzne/dzne.csv")
plotdays(epochs)




 
### plots distribution of users by age in our dataset ###
plot_age(data_path='../data/', plot_path = '../data/paper_plots/')



### plots for PCA by attributes for daily and weeklong stepcounts ###
from compress import PCA_AE


att =  pd.read_csv('../data/dzne/dzne_desc.csv')
att['user'] = pd.to_numeric(att['user'])

ae = PCA_AE('dzne_week')
pcadf = ae.compress_save(emb_size=2)
ae_day = PCA_AE('dzne_dsp')
pcadf_day = ae.compress_save(emb_size=2)


finalDf = pcadf.merge(att, on='user')
finalDf_day = pcadf.merge(att, on='user')"""

#finalDf.to_csv('../data/dzne/pca_week_attributes.csv', index=False)

finalDf_day = pd.read_csv('../data/dzne/pca_attributes.csv')
finalDf = pd.read_csv('../data/dzne/pca_week_attributes.csv')

plot_pca(finalDf, finalDf_day, plot_path = '../data/paper_plots/')


### plots for linkability evaluation ###

"""
from link_utils import merge_results
merge_results()


from utils.plot_utils import plot_link_dist, plot_link_stat
plot_link_dist(data_path='../data/plotdata/', plot_path = '../data/paper_plots/')

for stat in ['sum', 'max', 'mean', 'medi']:
    plot_link_stat(stat, data_path='../data/plotdata/', plot_path = '../data/paper_plots/')


plot_top_feats(data_path='../data/plotdata/',plot_path = '../data/paper_plots/')
"""
