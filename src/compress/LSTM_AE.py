# Created by rahman at 13:14 2019-09-30 using PyCharm

import pandas as pd

from compress import Compressor
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

from utils.storage import saveKerasmodel, loadKerasmodel, load_frame, check_if_vecframe, dump_frame

from sklearn import preprocessing


class LSTM_AE(Compressor):

    def train_save(self, emb_size, epochs=50, sample_size=None, n_in=None):

        if sample_size==None and n_in==None:

            sample_size, n_in = self.vecframe.shape[0] , self.vecframe.shape[1] -2

        Xtrain=self.vecframe.iloc[:sample_size, 2:n_in+2].values
        Xtrain = preprocessing.MinMaxScaler().fit_transform(Xtrain)


        # reshape input into [samples, timesteps, features]
        Xtrain = Xtrain.reshape((sample_size, n_in , 1))  ## 1 feature(stepcount only)
        print(Xtrain.shape)

        # define model
        model = Sequential()

        model.add(LSTM(emb_size, activation='relu', input_shape=(n_in, 1)))
        ### 1st argument above is the length of fixed-sized vector ie the internal representation of the input sequence.

        model.add(RepeatVector(n_in))
        # This layer simply repeats the provided 2D input multiple times to create a 3D output.
        # output of the encoder  is 2D and 3D input to the decoder is required.

        model.add(LSTM(emb_size, activation='relu', return_sequences=True))
        ### One or more LSTM layers can also be used to implement the decoder model.
        ### This model reads from the fixed sized output from the encoder model.

        model.add(TimeDistributed(Dense(1)))
        ### a Dense layer is used as the output for the network.
        # The same weights can be used to output each time step in the output sequence
        # by wrapping the Dense layer in a TimeDistributed wrapper.

        model.compile(optimizer='adam', loss='mse')
        model.fit(Xtrain, Xtrain, epochs=epochs, verbose=2)

        saveKerasmodel(model, archFile = str(epochs) + "_" + str(emb_size) + "lstm_arch.json", weightsFile =  str(epochs) + "_" + str(emb_size)+ "lstm.h5")
        del model


    def compress_save(self,  emb_size=10,  epochs=100, out_name='LSTM_emb',sample_size=None, n_in=None):


        if sample_size==None and n_in==None:

            sample_size, n_in = self.vecframe.shape[0] , self.vecframe.shape[1]

        in_vecframe = self.vecframe.iloc[:sample_size, 2:n_in]

        super().compress_save()

        model = loadKerasmodel(archFile =  str(epochs) + "_" + str(emb_size) + "lstm_arch.json", weightsFile =  str(epochs) + "_" + str(emb_size)+ "lstm.h5")

        ## connect the encoder LSTM as the output layer
        model = Model(inputs=model.inputs, outputs=model.layers[0].output)
        ## This gives us the latent layer embeddings of our desired size as the predict output instead of the reconstructions
        # create a new model that has the same inputs as our original model,
        # and outputs directly from the end of encoder model, before the RepeatVector layer.


        reshaped=in_vecframe.values.reshape((sample_size, n_in -2 , 1))
        emb = model.predict(reshaped)  # get the feature vector for the input sequence because we modified the output layer earlier

        print(emb.shape)
        assert(emb.shape[1]== emb_size)
        assert(emb.shape[0]== sample_size)

        emb_frame = pd.DataFrame(emb)
        emb_frame['user']=self.vecframe.iloc[:sample_size, 0]
        emb_frame['desc']=self.vecframe.iloc[:sample_size, 1]
        emb_frame.columns = emb_frame.columns.map(str)


        cols=emb_frame.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        emb_frame=emb_frame[cols]

        print("saving emb_frame in", self.vec_name + out_name  + str(epochs) + "_" + str(emb_size))
        dump_frame(emb_frame, self.vec_name + out_name +  str(epochs) + "_" + str(emb_size) , in_csv=True)

        return emb_frame
