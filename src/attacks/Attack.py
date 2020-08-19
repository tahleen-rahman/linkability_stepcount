from abc import ABC, abstractmethod
import attacks.VecframeException as VecframeException
from utils.storage import load_frame, check_if_vecframe, DATA_PATH, load_frame_as_3d_nparray


class Attack(ABC):
    """
    Abstract class for all the dimention compressors like: aggregators, LSTM, PCA, SVD, TSNE.
    """

    def __init__(self, vf_fname, time_dim=0, in_datapath=DATA_PATH):
        """
        Loads vecframe containing the compressed embeddings from file
        :param step_name: filename
        :param data_path: optional, if none, default from utils.storage.load_stepframe will be used
        """
        super().__init__()

        self.vf_fname = vf_fname

        self.in_datapath = in_datapath

        if not time_dim:
            vecframe = load_frame(vf_fname, in_datapath)
            check_if_vecframe(vecframe)
            self.vecframe = vecframe

        else:
            self.vecframe = load_frame_as_3d_nparray(vf_fname, data_path=in_datapath)


    @abstractmethod
    def attack(self):
        """
        Implements the attack

        :param vecframe:
        :return: AUC score of the attack
        """
        #if self.vecframe == None:
            #raise VecframeException("Vecframe not loaded, Create the vecframe by creating a Compressor object first")


