import itertools
from math import ceil
import pandas as pd

from compress import Splitter
from utils.storage import DATA_PATH


class DistributionsSplitter(Splitter):
    """
    Splits each datapoint (day of each user) into windows (e.g. hours) and transforms them into distributions of steps.
    Output is a vector of concatenated step distributions from each window.
    """
    def __init__(self, vec_name, bucket_size=4, window_size=4*60,  data_path=DATA_PATH, out_path=DATA_PATH):
        """

        :param vec_name: filename
        :param bucket_size: distribution bucket size
        :param window_size: size of the distribution window, in 15s periods
        :param data_path: optional, if none utils.storage.DATA_PATH will be used
        """
        super().__init__(vec_name, window_size, data_path, out_path)
        self.bucket_size = bucket_size
        self.out_name = '{}_dist_{}_{}'.format(vec_name, bucket_size, window_size)
        self.max_steps = self.vecframe.max()[2:].max() # it's 44 for dzne data

    def single_stat_window(self, vals):
        """
        Calculatres a distribution, with buckets of size `self.bucket_size`.
        0 is always in it's own bucket, then 1 - `bucket_size` and so on, up to self.max_steps.
        :param vals:
        :return:
        """
        groups = [(i, len(list(j))) for i, j in itertools.groupby(sorted(vals), key=lambda x: ceil(x / self.bucket_size))]
        for i in range(ceil(self.max_steps / self.bucket_size) + 1):
            gr_names = list(zip(*groups))[0]
            if i not in gr_names:
                groups.append((i, 0))
        return list(zip(*sorted(groups)))[1]





