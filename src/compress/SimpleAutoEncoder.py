import os.path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset


from compress import PytorchAE
from utils.storage import DATA_PATH

class SimpleAutoEncoder(PytorchAE):
    """
    Simple Auto Encoder
    """
    def __init__(self, vec_name, data_path=DATA_PATH, out_path=DATA_PATH, num_epochs = 10, batch_size = 100, learning_rate = 1e-3,
                 save_model=True, ae_params = [5760, 1024, 256, 64], par_name="64"):
        """

        :param vec_name: vecframe name, e.g., 'fake'
        :param data_path: path to dzne where vecframe is located
        :param num_epochs: number of epochs for network training
        :param batch_size: size of each batch for network training
        :param learning_rate: learning rate for network training
        :param ae_params: parameters for initializing the autoencoder, e.g. sizes of encoder layers
        :param par_name: name of parameters, used for naming the trained model file
        """
        print("initializing simpleAE")
        super().__init__(vec_name, data_path, out_path, num_epochs, batch_size, learning_rate, save_model)
        # these three need to be overwritten in child classes
        self.ae_class = Autoencoder
        self.AE_name = "simpleAE"
        self.par_name = par_name
        self.ae_params = self.make_params() if ae_params == 'auto' else ae_params

    def make_params(self):
        """
        automatically makes a three layer NN params.
        :return: ae_params
        """
        fst = self.vecframe.shape[1] - 2
        snd = max(1, min(2048, fst // 4 if fst > 255 else fst // 2))
        trd = max(1, snd // 4 if snd >> 127 else snd // 2)
        return [fst, snd, trd]


class Autoencoder(nn.Module):
    def __init__(self, layers_sizes):
        super(Autoencoder, self).__init__()
        encoder_structure = []
        for i in range(len(layers_sizes) - 1):
            encoder_structure.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:
                encoder_structure.append(nn.ReLU(True))
        layers_sizes = layers_sizes[::-1]
        decoder_structure = []
        for i in range(len(layers_sizes) - 1):
            decoder_structure.append(nn.Linear(layers_sizes[i], layers_sizes[i+1]))
            if i < len(layers_sizes) - 2:
                decoder_structure.append(nn.ReLU(True))
        decoder_structure.append(nn.Tanh())

        self.encoder = nn.Sequential(
            *encoder_structure)
        self.decoder = nn.Sequential(
            *decoder_structure)

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x
