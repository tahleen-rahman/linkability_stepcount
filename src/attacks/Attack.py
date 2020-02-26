from abc import ABC, abstractmethod
import attacks.VecframeException as VecframeException
from utils.storage import load_frame, check_if_vecframe, DATA_PATH


class Attack(ABC):
    """
    Abstract class for all the dimention compressors like: aggregators, LSTM, PCA, SVD, TSNE.
    """

    def __init__(self, vf_fname, in_datapath=DATA_PATH):
        """
        Loads vecframe containing the compressed embeddings from file
        :param step_name: filename
        :param data_path: optional, if none, default from utils.storage.load_stepframe will be used
        """
        super().__init__()

        self.vf_fname = vf_fname

        self.in_datapath = in_datapath
        vecframe = load_frame(vf_fname, in_datapath)

        check_if_vecframe(vecframe)
        self.vecframe = vecframe


    @abstractmethod
    def attack(self):
        """
        Implements the attack

        :param vecframe:
        :return: AUC score of the attack
        """
        #if self.vecframe == None:
            #raise VecframeException("Vecframe not loaded, Create the vecframe by creating a Compressor object first")


