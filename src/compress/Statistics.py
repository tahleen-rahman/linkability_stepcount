import itertools
from math import ceil
import pandas as pd

from compress import Compressor
from utils.storage import DATA_PATH, check_if_vecframe, dump_frame


class Statistics(Compressor):

    def __init__(self, vec_name,  stats, window_size=4*60, bucket_size=4, data_path=DATA_PATH):

        super().__init__(vec_name, data_path)

        self.bucket_size = bucket_size
        self.window_size = window_size
        self.stats=stats

        if len(stats)>1:
            name = '_'.join(stats)
        else:
            name = stats[0]
        name += str(self.window_size)

        self.out_name = 'stats_{}_{}'.format(vec_name, name)
        #print (self.out_name)


    def single_stat_batch(self, vals):

        ret = []

        if 'max' in self.stats:
            ret.append(vals.max())

        if 'std' in self.stats:
            ret.append(vals.std())

        if 'medi' in self.stats:
            ret.append(vals.median())

        if 'mean' in self.stats:
            ret.append(vals.mean())

        if 'sum' in self.stats:
            ret.append(vals.sum())

        return ret

    def compress_save(self):

        def all_distributions(vals):

            dist = []
            for left in range(0, len(vals), self.window_size):

                if len(self.stats)==1:
                    dist.append(self.single_stat_batch( vals[left:left+self.window_size])[0])
                else:
                    dist.extend(self.single_stat_batch( vals[left:left+self.window_size]))

            return dist

        data = self.vecframe.iloc[:, 2:].apply(all_distributions, axis=1)

        res = pd.DataFrame(list(data))

        #print (self.stats, res.shape)

        res.columns = [str(col) for col in res.columns]
        res['user'] = self.vecframe['user']
        res['desc'] = self.vecframe['desc']
        cols=list(res.columns)
        res = res[cols[-2:]+cols[:-2]]

        self.dump_vecframe(res)


