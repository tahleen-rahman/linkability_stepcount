import logging

from functools import reduce
from operator import mul
from torch import optim

import math
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F


class CdnnClassifier():
    def __init__(self, vec_len, cnn_params=[(21, 12), (9, 6)], dnn_params=[(0.5, 0.2)], num_epochs=100, batch_size=32,
                 padding='auto', learning_rate = 1e-5):
        '''

        :param vec_len: length of vectors in vecframe
        :param cnn_params: list of pairs (kernel_size, max_pool), will be applied one after another
        :param dnn_params: list of pairs (fraction_of_neurons, dropout), will be applied one after another
        :param num_epochs: number of training epochs
        :param batch_size: training batch size
        :param padding: vectors from vecframe will be padded up to a multiple of this value
                        if 'auto', then the product of max_pools will be taken
        '''
        super().__init__()
        self.vec_len = vec_len
        self.cnn_params = cnn_params
        self.dnn_params = dnn_params
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.padding = padding if padding != 'auto' else self.find_smallest_padding()
        self.learning_rate = learning_rate
        self.logger = logging.getLogger('CdnnClassifier')
        self.classifier_class = Classifier
        self.model = None

    def find_smallest_padding(self):
        """
        automatically finds the smallest padding, which is the product of max_pools
        :return: padding
        """
        return reduce(mul, (max_pool for (_, max_pool) in self.cnn_params), 1)

    def reshape(self, data):
        nrows, ncols = data.shape
        return data.reshape(nrows, 1, ncols)

    def add_padding(self, data):
        pad_len = self.vec_len - math.floor(self.vec_len / self.padding) * self.padding
        self.vec_len += pad_len
        before = pad_len // 2
        after = pad_len - before
        return np.pad(data, [(0, 0), (before, after)], 'constant')

    def shuffle_in_unison(self, a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b

    def fit(self, X_train, y_train):
        # padding
        assert X_train.shape[1] == self.vec_len
        X_train = self.add_padding(X_train)
        data_set = MyDataset(X_train, y_train)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=self.batch_size, shuffle=True)
        self.model = self.classifier_class(self.vec_len, self.cnn_params, self.dnn_params).cpu().double()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # , weight_decay=5e-5)

        for epoch in range(self.num_epochs):
            for local_X, local_y in data_loader:
                # possible reshaping
                local_y = local_y.double()
                local_X = self.reshape(local_X)
                # forward
                output = self.model(local_X)
                # print(output)
                # self.logger.debug(output)
                loss = criterion(output, local_y)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.logger.debug('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.num_epochs, loss.item()))

    def predict(self, X_test):
        # padding
        X_test = self.add_padding(X_test)
        assert X_test.shape[1] == self.vec_len
        X_test = torch.from_numpy(X_test).double()
        X_test = self.reshape(X_test)
        output = self.model(X_test)
        return output > 0.5


class Classifier(nn.Module):
    def __init__(self, vec_len, cnn_layer_params, dnn_layer_params):
        super().__init__()

        self.logger = logging.getLogger("CdnnClassifier")
        self.cnn_layers = []
        self.cnn_pools = []
        curr_size = vec_len
        # for kernel_size, max_pool in cnn_layer_params:
        #     self.cnn_layers.append(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
        #                                      padding= kernel_size//2))
        #     self.cnn_pools.append(nn.MaxPool1d(max_pool))
        #     curr_size //= max_pool
        self.dnn_layers = []
        self.dropout_layers = []
        for frac, drop in dnn_layer_params:
            next_size = int(curr_size * frac)
            self.dnn_layers.append(nn.Linear(curr_size, next_size).double())
            self.dropout_layers.append(nn.Dropout(drop).double())
            curr_size = next_size
        self.final_fc = nn.Linear(curr_size, 1)

    def forward(self, x):
        # for cnn, pool in zip(self.cnn_layers, self.cnn_pools):
        #     x = pool(F.relu(cnn(x)))
        for dnn, dropout in zip(self.dnn_layers, self.dropout_layers):
            x = dropout(F.relu(dnn(x)))
        x = torch.sigmoid(self.final_fc(x))
        x = x.squeeze()
        return x


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        super().__init__()
        self.data = data.astype('double')
        self.labels = labels.astype('double')

    def __len__(self):
        '''
        :return: Total number of samples
        '''
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.data[index], self.labels[index]
