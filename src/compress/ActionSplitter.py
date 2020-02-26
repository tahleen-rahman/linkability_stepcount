import itertools
from math import ceil
import pandas as pd
import statistics

from compress import Compressor
from utils.storage import DATA_PATH


class ActionSplitter(Compressor):
    """
    Splits each datapoint (day or week of each user) into windows of continuous action with fixed pause allowed.
    In other words it cuts on pauses that are longer then max_pause_len
    """
    def __init__(self, vec_name, max_pause_len=8, bucket_size=4, data_path=DATA_PATH, out_path=DATA_PATH):
        """

        :param vec_name: filename
        :param bucket_size: distribution bucket size
        :param window_size: size of the distribution window, in 15s periods
        :param data_path: optional, if none utils.storage.DATA_PATH will be used
        """
        super().__init__(vec_name, data_path, out_path)
        self.bucket_size = bucket_size
        self.max_pause_len = max_pause_len
        self.out_appendix = 'act_{}'.format(max_pause_len)
        self.out_name = '{}_{}'.format(vec_name, self.out_appendix)
        self.max_steps = self.vecframe.max()[2:].max()  # it's 44 for dzne data

    def split_row_by_pause(self, vec):
        """
        Splits row by pauses longer then self.max_pause_len into actions
        :param vec: vector of actions
        :return: list of (timestamp of the beginning of action, action)
        """
        vec = list(vec)
        action_beg = 0
        action_end = -1
        pause_time = 0
        resulting_actions = []
        for i, val in enumerate(vec):
            if action_end == -1:  # looking for a beginning of action
                if val > 0:  # found beginning
                    action_beg = i
                    action_end = i
            else:  # are in action
                if val > 0:  # continuing action
                    action_end = i
                    pause_time = 0
                else:  # found pause
                    if pause_time < self.max_pause_len:  # allowed pause
                        pause_time += 1
                    else:  # not allowed pause
                        resulting_actions.append((action_beg, vec[action_beg: action_end + 1]))
                        action_end = -1
                        pause_time = 0
        if action_end != -1:
            resulting_actions.append((action_beg, vec[action_beg:action_end + 1]))
        return resulting_actions

    def compress_save(self):
        res = list(self.vecframe.iloc[:, 2:].apply(self.split_row_by_pause, axis=1))
        res_actions = []
        res_stats = []
        res_dists = []
        for user_desc, actions in zip((list(row) for _, row in self.vecframe.iloc[:, :2].iterrows()), res):
            res_actions += [user_desc + action for _, action in actions]
            res_stats += [user_desc + self.statistics(timestamp, action) for timestamp, action in actions]
            res_dists += [user_desc + self.distribution(timestamp, action) for timestamp, action in actions]
        df_actions = pd.DataFrame(res_actions)
        df_stats = pd.DataFrame(res_stats)
        df_dists = pd.DataFrame(res_dists)
        for df in (df_actions, df_stats, df_dists):
            df.columns = ['user', 'desc'] + [str(col) for col in df.columns][:-2]
        df_actions = df_actions.fillna(0)
        self.dump_vecframe(df_actions)
        self.dump_vecframe(df_stats, self.out_appendix + '_stats')
        self.dump_vecframe(df_dists, self.out_appendix + '_dist')
        return

    def statistics(self, timestamp, action):
        act_len = len(action)
        act_mean = statistics.mean(action)
        return [timestamp, act_len, max(action), min(action), sum(action),
                statistics.median(action), act_mean, statistics.stdev(action, act_mean) if act_len > 1 else 0]

    def distribution(self, timestamp, action):
        """
        Calculatres a distribution, with buckets of size `self.bucket_size`.
        0 is always in it's own bucket, then 1 - `bucket_size` and so on, up to self.max_steps.
        :param timestamp: timestamp of beginning of the action will be prepended to the distribution
        :param action: list of step values
        :return: list consisting of a timestamp and bucketed distribution
        """
        groups = [(i, len(list(j))) for i, j in
                  itertools.groupby(sorted(action), key=lambda x: ceil(x / self.bucket_size))]
        for i in range(ceil(self.max_steps / self.bucket_size) + 1):
            gr_names = list(zip(*groups))[0]
            if i not in gr_names:
                groups.append((i, 0))
        return [timestamp] + list(list(zip(*sorted(groups)))[1])

