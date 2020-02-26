import os
from sklearn.ensemble import RandomForestClassifier

from attacks import AttributeInference
from compress import Statistics
from itertools import combinations, chain


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))




stats_arr = ['sum', 'max', 'std', 'medi', 'mean']
stats = list(powerset(stats_arr))



stats_datapath = "../data/dzne/statistics/"



for minu in [1, 5, 10, 15, 30 ]:

    for st in (['max'], ['sum'], ['medi'], ['mean']):

        print (st)

        ds = Statistics('dzne_dsp', st, window_size=int(minu * 4), bucket_size=4, data_path=stats_datapath)

        if not os.path.exists(stats_datapath + ds.out_name + ".ftr"):

            ds.compress_save()

