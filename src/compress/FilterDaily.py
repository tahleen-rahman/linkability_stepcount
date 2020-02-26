
import pandas as pd

from compress import Compressor
from utils.storage import DATA_PATH, load_frame

class FilterDaily(Compressor):
    """
    Filters out every hour in which males and females behave similarly.
    """
    def __init__(self, vec_name, hourly_name='hourly_dzne_dsp', desc_name='dzne_desc', threshold=25, in_path=DATA_PATH, out_path=DATA_PATH):
        """

        :param vec_name: filename
        :param hourly_name: filename of hourly aggregated steps
        :param desc_name: filename of attribute descriptions
        :param in_path: optional, if none utils.storage.DATA_PATH will be used
        """
        super().__init__(vec_name, in_path, out_path)
        self.threshold = threshold
        hourly = load_frame(hourly_name)
        desc = load_frame(desc_name)
        hourly['sex'] = list(desc.sex) * 7
        m_average = []
        f_average = []
        for hour in range(24):
            m_average.append(hourly[(hourly.sex == 'm')][str(hour)].mean())
            f_average.append(hourly[(hourly.sex == 'f')][str(hour)].mean())
        self.dif = [abs(m - f) for m, f in zip(m_average, f_average)]

    def compress_save(self):
        filt = [i > self.threshold for i in self.dif]
        columns = [str(i) for i in range(4*60*24) if filt[i // (4*60)]]
        res_frame = pd.DataFrame()
        res_frame['user'] = self.vecframe['user']
        res_frame['desc'] = self.vecframe['desc']
        for i, col in enumerate(columns):
            res_frame[str(i)] = self.vecframe[col]
        self.dump_vecframe(res_frame, 'filter')




