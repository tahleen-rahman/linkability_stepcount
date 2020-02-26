import os.path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset


from compress import PytorchAE
from utils.storage import DATA_PATH

class CnnAE(PytorchAE):
    """
    CNN Auto Encoder
    """
    def __init__(self, vec_name, data_path=DATA_PATH, out_path=DATA_PATH, num_epochs=50, batch_size=64,
                 learning_rate=1e-3, save_model=False, ae_params=[21, 9, 4], par_name="21_9_4"):
        """

        :param vec_name: vecframe name, e.g., 'fake'
        :param data_path: path to dzne where vecframe is located
        :param num_epochs: number of epochs for network training
        :param batch_size: size of each batch for network training
        :param learning_rate: learning rate for network training
        :param ae_params: three values: kernel_size, kernel_size, embedding_size
                both kernel_sizes needs to be odd,
                max_pool needs to be a divisor of input vector size
        :param par_name: name of parameters, used for naming the trained model file, e.g. embedding_len as string
        """
        print("initializing CnnAE")
        super().__init__(vec_name, data_path, out_path, num_epochs, batch_size, learning_rate, save_model)
        # these three need to be overwritten in child classes
        self.ae_class = Autoencoder
        self.AE_name = "cnnAE"
        self.par_name = par_name
        # checks for autoencoder parameters
        ker1, ker2, max_pool = self.make_params() if ae_params == 'auto' else ae_params
        # if ker1 % 2 == 0 or ker2 % 2 == 0:
        #     raise Exception("kernel sizes need to be odd")
        vec_len = self.vecframe.shape[1] - 2
        if vec_len % max_pool != 0:
            raise Exception("max_pool needs to be a divisor of input vector, got {}".format(max_pool))
        self.ae_params = ker1, ker2, max_pool

    def make_params(self):
        """
        automatically makes a three layer NN params.
        :return: ae_params
        """
        size = self.vecframe.shape[1] - 2
        lst = 1
        for divisor in range(1, size):
            if size // divisor * divisor != size:
                continue
            if divisor > 8:
                break
            lst = divisor
        if lst == 1:
            pool_size = divisor
        else:
            pool_size = divisor if divisor - 8 <= 8 - lst else lst
        return [21, 9, pool_size]

    def reshape(self, data):
        nrows, ncols = data.shape
        return data.reshape(nrows, 1, ncols)

    def calculate_embeddings(self):
        return [list(key) + self.model.encoder(value.float().reshape(1, 1, len(value)))[0][0].tolist() for key, value in self.vecs.items()]

    def calculate_reconstructions(self):
        return [list(key) + self.model.forward(value.float().reshape(1, 1, len(value)))[0][0].tolist() for key, value in self.vecs.items()]

class Autoencoder(nn.Module):
    def __init__(self, layers_sizes):
        super(Autoencoder, self).__init__()
        self.ker1, self.ker2, self.max_pool = layers_sizes

        self.p1 = nn.ConstantPad1d((self.ker1//2, (self.ker1+1)//2-1), 0)
        self.c1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=self.ker1)
        self.m1 = nn.MaxPool1d(self.max_pool, return_indices=True)
        self.i1 = None
        self.c2 = nn.Conv1d(in_channels=8, out_channels=64, kernel_size=self.ker2, padding=self.ker2//2)

        self.d1 = nn.ConvTranspose1d(in_channels=64, out_channels=8, kernel_size=self.ker2, padding=self.ker2//2)
        self.u1 = nn.MaxUnpool1d(self.max_pool)
        self.d2 = nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=self.ker1)

    def encoder(self, x):
        _c1 = self.c1(self.p1(x))
        _m1, self.i1 = self.m1(_c1)
        return self.c2(_m1)

    def decoder(self, x):
        _d1 = self.d1(x)
        _u1 = self.u1(_d1, self.i1)
        _p1 = self.d2(_u1)
        _up1 = _p1[:, :, self.ker1//2:-((self.ker1+1)//2-1)]
        return _up1

        # encoder_structure = []
        # for i in range(len(layers_sizes) - 1):
        #     encoder_structure.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        #     if i < len(layers_sizes) - 2:
        #         encoder_structure.append(nn.ReLU(True))
        # layers_sizes = layers_sizes[::-1]
        # decoder_structure = []
        # for i in range(len(layers_sizes) - 1):
        #     decoder_structure.append(nn.Linear(layers_sizes[i], layers_sizes[i+1]))
        #     if i < len(layers_sizes) - 2:
        #         decoder_structure.append(nn.ReLU(True))
        # decoder_structure.append(nn.Tanh())
        #
        # self.encoder = nn.Sequential(
        #     *encoder_structure)
        # self.decoder = nn.Sequential(
        #     *decoder_structure)

    def forward(self, x):
        # nrows, ncols = x.shape
        # x = x.reshape(nrows, 1, ncols)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
