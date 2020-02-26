from abc import abstractmethod
import itertools
from math import ceil
import pandas as pd

from compress import Compressor
from utils.storage import DATA_PATH

class Splitter(Compressor):
    """
    Splits each datapoint (day of each user) into windows (e.g. hours)
    and applies `single_stat_batch` on each batch
    Output is a vector of concatenated results of `single_stat_batch` from each window.
    """
    def __init__(self, vec_name, window_size=4*60,  data_path=DATA_PATH, out_path=DATA_PATH):
        """

        :param vec_name: filename
        :param bucket_size: distribution bucket size
        :param window_size: size of the distribution window, in 15s periods
        :param data_path: optional, if none utils.storage.DATA_PATH will be used
        """
        super().__init__(vec_name, data_path, out_path)
        self.window_size = window_size

    @abstractmethod
    def single_stat_window(self, vals):
        """
        From given list of values from one window of `window_size`, calculate desired function.
        :param vals: list of values of length `self.window_size`
        :return: list of values
        """
        pass

    def all_stat_windows(self, row):
        """
        Splits row of data into windows of size `self.window_size`,
        applies `self.single_stat_window` on them
        and concatenates the resulting lists.
        :param row: list of values - full row from vecframe
        :return: concatenated lists of values
        """
        dist = []
        for left in range(0, len(row), self.window_size):
            dist += list(self.single_stat_window(row[left:left + self.window_size]))
        return dist

    def compress_save(self):

        res = pd.DataFrame(list(self.vecframe.iloc[:,2:].apply(self.all_stat_windows, axis=1)))
        res.columns = [str(col) for col in res.columns]
        res['user'] = self.vecframe['user']
        res['desc'] = self.vecframe['desc']
        cols=list(res.columns)
        res = res[cols[-2:]+cols[:-2]]
        return self.dump_vecframe(res)



