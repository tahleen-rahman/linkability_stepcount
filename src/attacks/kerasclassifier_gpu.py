# Created by rahman at 11:06 2020-02-22 using PyCharm


import logging
import math

import keras
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Reshape, LSTM
from keras import Sequential, utils, Input, Model
from keras.utils.vis_utils import plot_model
#if tf.__version__ != '2.1.0':
    #import keras.layers.CuDNNLSTM

class BinaryDNN:

    def __init__(self, num_layers = 2, layer_params=[[0.5, 0.2], [0.25, 0]], num_epochs=100, batch_size=24, verbose=1):
        """

        :param num_layers: just for a sanity check
        :param layer_params: list of size 2 lists, containing [layer size factor, dropout] for each layer
        :param num_epochs:
        :param batch_size:
        """

        assert(num_layers == len(layer_params))

        self.num_layers = num_layers
        self.layer_params = layer_params
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose

        self.logger = logging.getLogger("DnnClassifier")


    def fit(self, X_train, y_train):

        dim = X_train.shape[1]

        self.model = Sequential()


        for factor, dropout in self.layer_params:

            self.model.add(Dense(int(dim * factor if dim * factor > 1 else 1), input_dim=dim, activation='relu'))
            self.model.add(Dropout(dropout))


        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])

        if self.verbose!=0:

            self.model.summary()


        #y_keras= utils.to_categorical(y_train, 2)

        self.model.fit(X_train, y_train, epochs = self.num_epochs, batch_size = self.batch_size, verbose=self.verbose)


    def predict(self, X_test):

        y_pred = self.model.predict(X_test, verbose=True)

        return (y_pred > 0.5)







class siameseClassifier:

    def __init__(self, num_features, regu, combi):


        self.sample_a = Input(shape=(num_features,))
        self.sample_b = Input(shape=(num_features,))

        self.regu =  keras.regularizers.l2(regu)
        self.combi = combi

        #self.l_a and self.l_b will be defined in the child constructors



    def combine(self, plot=False):


        if self.combi=='l1':

            difference = keras.layers.subtract([self.l_a, self.l_b])
            combined = keras.layers.Lambda(tensorabs)(difference)  # in general: absolute value: abs(<tensor>)

        elif self.combi=='mul':

            combined = keras.layers.multiply([self.l_a, self.l_b])
            #combined = keras.layers.Lambda(tensorabs)(difference)  # in general: absolute value: abs(<tensor>)

        elif self.combi == 'avg':

            combined = keras.layers.average([self.l_a, self.l_b])
            #combined = keras.layers.Lambda(tensorabs)(difference)  # in general: absolute value: abs(<tensor>)

        elif self.combi == 'sql2':

            #combined = keras.layers.average([l_a2, l_b2])
            combined = keras.layers.Lambda(lambda x: (x[0] - x[1]) ** 2 )([self.l_a, self.l_b])



        predictions = Dense(1, activation='sigmoid')(combined)  # logistic regression according to docu
        # example does not add regularization to prediction layer - so don't do it here as well

        self.model = Model(inputs=[self.sample_a, self.sample_b], outputs=predictions)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        if plot:
            plot_model(self.model, to_file='siam.png', show_shapes=True, show_layer_names=True)



    def fit_predict(self, link, batchsize, epochs, verbose=0):


        self.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                           link.tr_pairs.label,
                           batch_size=batchsize, epochs=epochs,
                           validation_data=([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                           link.te_pairs.label), verbose=verbose)


        y_pred = self.model.predict([link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                                        verbose=verbose)

        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(link.te_pairs.label, y_pred)

        return auc




class Dense_siameseClassifier(siameseClassifier):

    def __init__(self, num_features, regu, combi, dense_params):

        super().__init__(num_features, regu, combi)

        half_num_features = int(math.floor(num_features * dense_params[0]) if num_features>=2 else 1)
        shared_dense1 = Dense(half_num_features, input_shape=(num_features,),
                              activation='relu')  # compresses the input to half the size


        if self.regu != None:
            shared_dense1.kernel_regularizer = self.regu

        l_a = shared_dense1(self.sample_a)
        l_b = shared_dense1(self.sample_b)

        quater_num_features = int(math.floor(num_features * dense_params[1])if num_features>=4 else 1)
        shared_dense2 = Dense(quater_num_features, input_shape=(half_num_features,),
                              activation='relu')  # second layer: again, compresses the input to half the size

        if self.regu != None:
            shared_dense2.kernel_regularizer = self.regu

        l_a2 = shared_dense2(l_a)
        l_b2 = shared_dense2(l_b)

        self.l_a = l_a2
        self.l_b = l_b2




class LSTMsiameseClassifier(siameseClassifier):

    def __init__(self, num_features, regu, combi, lstm_params):
        super().__init__(num_features, regu, combi)

        shared_nn = Sequential()

        shared_nn.add(Reshape((num_features, 1), input_shape=(num_features,)))

        for param in lstm_params:

            units = int(math.floor(num_features * param[0]) if num_features >= 4 else 1)

            shared_nn.add(LSTM(units, input_shape=(num_features, 1)))
            shared_nn.add(Dropout(param[1]))

        self.l_a = shared_nn(self.sample_a)
        self.l_b = shared_nn(self.sample_b)


def tensorabs(t):
    return abs(t)


class CNNsiameseClassifier(siameseClassifier):

    def __init__(self, num_features, regu, combi, cnn_params, num_maxpools):

        super().__init__(num_features, regu, combi)


        (filt1, ker1), (filt2, ker2), pool = cnn_params

        assert (num_maxpools in [1, 2])


        shared_conv1 = Sequential()

        shared_conv1.add(Reshape((num_features, 1), input_shape=(num_features,)))

        shared_conv1.add(Conv1D(filters=filt1, kernel_size=ker1, activation='relu', kernel_regularizer=self.regu, padding='same'))

        if num_maxpools==2:

            shared_conv1.add(MaxPooling1D(pool_size=pool))

        shared_conv1.add(Conv1D(filters=filt2, kernel_size=ker2, activation='relu', kernel_regularizer=self.regu, padding='same'))

        shared_conv1.add(MaxPooling1D(pool_size=pool))


        shared_conv1.add(Flatten())

        shared_conv1.add(Dense(100, activation='relu'))

        self.l_a = shared_conv1(self.sample_a)
        self.l_b = shared_conv1(self.sample_b)





