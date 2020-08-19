# Created by rahman at 11:04 2019-12-19 using PyCharm

import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Dense, LSTM, Bidirectional

from attacks import BinaryDNN

class LSTMclassifier(BinaryDNN):

    def __init__(self,  num_layers = 1, layer_params=[[8, 0.2]], num_epochs=100, batch_size=16, verbose=1, patience=10):

        super().__init__(num_layers, layer_params, num_epochs, batch_size, verbose)
        self.patience = patience


    def fit(self, X_train, y_train):

        model = Sequential()

        trainX = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        """the input part of your training data (X) must be a three-dimensional array 
        with the dimensions [samples][timesteps][features], and you must have at least one sample, one time step and one feature.
        """
        for units, dropout in self.layer_params:

            model.add(LSTM(units, input_shape= (X_train.shape[1], X_train.shape[2]))) # (shape = timesteps, features)
            model.add(Dropout(dropout))


        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=self.patience, restore_best_weights=True)
        model.fit(trainX, y_train, epochs=self.num_epochs, batch_size=self.batch_size, callbacks=[es])
        self.model = model

    def predict(self, X_test):

        testX= X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        y_pred = self.model.predict(testX, verbose=True)

        return (y_pred)

class biLSTMclassifier(BinaryDNN):

    def __init__(self,  num_layers = 1, layer_params=[[8, 0.2]], num_epochs=100, batch_size=16, verbose=1, patience=10):

        super().__init__(num_layers, layer_params, num_epochs, batch_size, verbose)
        self.patience = patience


    def fit(self, X_train, y_train):

        model = Sequential()

        trainX = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        """the input part of your training data (X) must be a three-dimensional array 
        with the dimensions [samples][timesteps][features], and you must have at least one sample, one time step and one feature.
        """
        for units, dropout in self.layer_params:

            model.add(Bidirectional(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))) # (shape = timesteps, features)
            model.add(Dropout(dropout))


        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=self.patience, restore_best_weights=True)
        model.fit(trainX, y_train, epochs=self.num_epochs, batch_size=self.batch_size, callbacks=[es])
        self.model = model

    def predict(self, X_test):

        testX= X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        y_pred = self.model.predict(testX, verbose=True)

        return (y_pred)


class AttentionbiLSTMclassifier(BinaryDNN):

    def __init__(self,  num_layers = 1, layer_params=[[8, 0.2]], num_epochs=100, batch_size=16, verbose=1, patience=10):

        super().__init__(num_layers, layer_params, num_epochs, batch_size, verbose)
        self.patience = patience


    def fit(self, X_train, y_train):


        trainX = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        """the input part of your training data (X) must be a three-dimensional array 
        with the dimensions [samples][timesteps][features], and you must have at least one sample, one time step and one feature.
        """
        for units, dropout in self.layer_params:

            lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(
                LSTM(units, input_shape=(self.num_timesteps, self.num_features), \
                     return_sequences=True,
                     return_state=True))(trainX)

            state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
            # state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

            # Query-value attention of shape [batch_size, Tq, filters].
            query_value_attention_seq = tf.keras.layers.Attention()([state_h, lstm])

            query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
                query_value_attention_seq)


        predictions = Dense(1, activation='sigmoid')(query_value_attention)
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=self.patience, restore_best_weights=True)
        self.model = tf.keras.Model(inputs=trainX, outputs=predictions)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print(self.model.summary())

        self.model.fit(trainX, y_train, epochs=self.num_epochs, batch_size=self.batch_size, callbacks=[es])



    def predict(self, X_test):

        testX= X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        y_pred = self.model.predict(testX, verbose=True)

        return (y_pred)


