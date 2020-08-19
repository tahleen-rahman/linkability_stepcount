
from compress import Splitter
from utils.storage import DATA_PATH

class StatisticsSplitter(Splitter):
    """
    Splits each datapoint (day of each user) into windows (e.g. hours)
     and transforms them into statistics on steps.
    Output is a vector of concatenated steps statistics from each window.
    """

    def __init__(self, vec_name,  stats, window_size, out_path, data_path=DATA_PATH):

        super().__init__(vec_name, window_size, data_path, out_path)
        self.stats=stats
        if len(stats)>1:
            name = '_'.join(stats)
        else:
            name = stats[0]
        name += str(self.window_size)
        self.out_name = '{}_{}'.format(vec_name, name)

    def single_stat_window(self, vals):

        ret = []

        if 'max' in self.stats:
            ret.append(vals.max())

        if 'std' in self.stats:
            ret.append(vals.std())

        if 'medi' in self.stats:
            ret.append(vals.median())

        if 'mean' in self.stats:
            ret.append(vals.mean())

        return ret
'''
    def compress_save(self):

        def all_distributions(vals):

            dist = []
            for left in range(0, len(vals), self.window_size):

                if len(self.stats)==1:
                    dist.append(self.single_stat_batch( vals[left:left+self.window_size]))
                else:
                    dist.extend(self.single_stat_batch( vals[left:left+self.window_size]))

            return dist

        res = pd.DataFrame(list(self.vecframe.iloc[:,2:].apply(all_distributions, axis=1)))

        #print (self.stats, res.shape)

        res.columns = [str(col) for col in res.columns]
        res['user'] = self.vecframe['user']
        res['desc'] = self.vecframe['desc']
        cols=list(res.columns)
        res = res[cols[-2:]+cols[:-2]]

        return self.dump_vecframe(res, self.name , False)
'''
