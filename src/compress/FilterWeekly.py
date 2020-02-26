
import pandas as pd

from compress import Compressor
from utils.storage import DATA_PATH, load_frame

class FilterWeekly(Compressor):
    """
    Filters out every hour in which males and females behave similarly.
    """
    def __init__(self, vec_name, desc_name='dzne_desc', threshold=25, in_path=DATA_PATH, out_path=DATA_PATH):
        """

        :param vec_name: filename
        :param desc_name: filename of attribute descriptions
        :param in_path: optional, if none utils.storage.DATA_PATH will be used
        """
        super().__init__(vec_name, in_path, out_path)
        self.threshold = threshold
        desc = load_frame(desc_name)
        m_average = self.vecframe[desc.sex == 'm'].mean()[2:]
        f_average = self.vecframe[desc.sex == 'f'].mean()[2:]
        self.dif = [abs(m - f) for m, f in zip(m_average, f_average)]

    def compress_save(self):
        filt = [i > self.threshold for i in self.dif]
        columns = list(self.vecframe.columns[:2]) + list([col for col in self.vecframe.columns[2:] if filt[int(col)]])
        res_frame = self.vecframe[columns]
        self.dump_vecframe(res_frame, 'filter', in_csv=False)




