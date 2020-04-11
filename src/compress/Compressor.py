from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils.storage import load_frame, dump_frame, check_if_vecframe, DATA_PATH



class Compressor(ABC):
    """
    Abstract class for all the dimension compressors like: aggregators, LSTM, PCA, SVD, TSNE.
    """

    def __init__(self, vec_name, data_path=DATA_PATH, out_path=DATA_PATH):
        """
        Loads stepframe form file and checks for correct format
        :param step_name: filename
        :param data_path: optional, if none utils.storage.DATA_PATH will be used
        """
        super().__init__()
        self.vec_name = vec_name
        self.data_path = data_path
        self.out_path = out_path
        vecframe = load_frame(vec_name, data_path)
        check_if_vecframe(vecframe)
        self.vecframe = vecframe



    @abstractmethod
    def compress_save(self, out=''):
        """
        Compresses entries from loaded stepframe into smaller vectors and saves them in a vecframe.

        :param stepframe:
        :return:
        """
        pass



    def dump_vecframe(self, vf, appendix=None, in_csv=False):
        """
        Dumps resulting vectorframe
        :param appendix: added to the filename from which Compressor was initialized
        :return: name of the file created
        """
        check_if_vecframe(vf)
        if appendix == None:
            outname = self.out_name
        else:
            outname = self.vec_name + "_" + appendix

        dump_frame(vf, outname, self.out_path, in_csv=in_csv)
        return outname
