import functools
import logging
import math
import numpy as np
import operator
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten
from keras import Sequential, utils


class CNNKerasClassifier:

    def __init__(self, vec_len, cnn_params=[(32, 6), (32, 6), 2], dnn_params=[(0.5, 0.2)], num_epochs=100, batch_size=32,
                 verbose=1, padding='auto'):
        '''

        :param vec_len: length of vectors in vecframe
        :param cnn_params: list of pairs (kernel_size, max_pool), will be applied one after another
        :param dnn_params: list of pairs (fraction_of_neurons, dropout), will be applied one after another
        :param num_epochs: number of training epochs
        :param batch_size: training batch size
        :param padding: vectors from vecframe will be padded up to a multiple of this value
                        if 'auto', then the product of max_pools will be taken
        '''

        self.vec_len = vec_len
        self.cnn_params = cnn_params
        self.dnn_params = dnn_params
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.padding = padding if padding != 'auto' else cnn_params[2]

        self.logger = logging.getLogger("DnnClassifier")

    def find_smallest_padding(self):
        """
        automatically finds the smallest padding, which is the product of max_pools
        :return: padding
        """
        return functools.reduce(operator.mul, (max_pool for (_, max_pool) in self.cnn_params), 1)

    def add_padding(self, data):
        pad_len = self.vec_len - math.floor(self.vec_len / self.padding) * self.padding
        # self.vec_len += pad_len
        before = pad_len // 2
        after = pad_len - before
        return np.pad(data, [(0, 0), (before, after)], 'constant')

    def reshape(self, data):
        nrows, ncols = data.shape
        return data.reshape(nrows, ncols, 1)

    def fit(self, X_train, y_train):
        (filt1, ker1), (filt2, ker2), pool = self.cnn_params

        X_train = self.add_padding(X_train)
        X_train = self.reshape(X_train)
        dim = X_train.shape[1]
        self.model = Sequential()
        self.model.add(Conv1D(filters=filt1, kernel_size=ker1, activation='relu', input_shape=[dim, 1], padding='same'))
        self.model.add(Conv1D(filters=filt2, kernel_size=ker2, activation='relu', padding='same'))
        self.model.add(Dropout(0.5))
        self.model.add(MaxPooling1D(pool_size=pool))
        self.model.add(Flatten())
        # for kernel_size, max_pool in self.cnn_params:
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        # for factor, dropout in self.dnn_params:
        #
        #     self.model.add(Dense(int(dim * factor if dim * factor > 1 else 1), input_dim=dim, activation='relu'))
        #     self.model.add(Dropout(dropout))
        #
        #
        # self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

        if self.verbose!=0:

            self.model.summary()

        # y_keras= utils.to_categorical(y_train, 2)
        self.model.fit(X_train, y_train, epochs = self.num_epochs, batch_size = self.batch_size, verbose=self.verbose)


    def predict(self, X_test):
        X_test = self.add_padding(X_test)
        X_test = self.reshape(X_test)
        y_pred = self.model.predict(X_test, verbose=True)

        return (y_pred > 0.5)

