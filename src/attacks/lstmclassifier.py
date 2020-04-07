# Created by rahman at 11:04 2019-12-19 using PyCharm


from tensorflow import keras
from keras import Sequential
from keras.layers import Dropout, Dense, LSTM

from attacks import BinaryDNN

class LSTMclassifier(BinaryDNN):

    def __init__(self,  num_layers, layer_params, num_epochs=100, batch_size=16, verbose=1):

        super().__init__(num_layers, layer_params, num_epochs, batch_size, verbose)


    def fit(self, X_train, y_train):

        model = Sequential()

        trainX= X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        """the input part of your training data (X) must be a three-dimensional array 
        with the dimensions [samples][timesteps][features], and you must have at least one sample, one time step and one feature.
        """

        if len(self.layer_params)==1:

            units, dropout = self.layer_params[0][0], self.layer_params[0][1]
            model.add(LSTM(units, input_shape = (X_train.shape[1], 1))) # (shape = timesteps, features)
            model.add(Dropout(dropout))

        elif len(self.layer_params)==2:

            units, dropout = self.layer_params[0][0], self.layer_params[0][1]
            model.add(LSTM(units, return_sequences=True, input_shape = (X_train.shape[1], 1))) # (shape = timesteps, features)
            model.add(Dropout(dropout))

            units, dropout = self.layer_params[1][0], self.layer_params[1][1]
            model.add(LSTM(units)) # (shape = timesteps, features)
            model.add(Dropout(dropout))


        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        model.fit(trainX, y_train, epochs=self.num_epochs, batch_size=self.batch_size)
        self.model = model

    def predict(self, X_test):

        testX= X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        y_pred = self.model.predict(testX, verbose=True)

        return (y_pred)

