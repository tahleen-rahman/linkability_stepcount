# Created by rahman at 15:14 2019-09-21 using PyCharm

import sys
from numpy import array
from tensorflow import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from keras.models import model_from_json
import pandas as pd

DATAPATH = sys.argv[1]


def saveKerasmodel(model, data_path='../dzne/', archFile="model.json", weightsFile="model.h5"):
    # serialize model to JSON
    model_json = model.to_json()
    with open(data_path+ archFile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(data_path+ weightsFile)
    print("Saved model to disk, deleting now")


def loadKerasmodel(data_path='../dzne/', archFile="model.json", weightsFile="model.h5"):
    # load json and create model
    json_file = open(data_path+ archFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(data_path+ weightsFile)
    print("Loaded model from disk from", data_path+ weightsFile)

    return loaded_model


def load_frame(name, data_path='../dzne/'):
    """
    Reads dataframe from the disk from feather format.

    :param name: filename without extentions, e.g., "fake"
    :param data_path:  path to dzne file, probably should always be '../dzne/'
    :return: pandas dataframe of stepcounts for all users
    """
    return pd.read_feather('{}{}.ftr'.format(data_path, name))



def check_if_vecframe(vecframe):
    """
    Checks format of vecframe.
    Can be overloaded for additional checks.

    :param vecframe:
    :return:
    """
    assert isinstance(vecframe, pd.DataFrame)
    assert vecframe.shape[1] > 2
    assert vecframe.shape[0] > 1
    assert 'desc' in vecframe.columns
    assert 'user' in vecframe.columns

vecframe = load_frame(name='fake_dsp', data_path=DATAPATH)

check_if_vecframe(vecframe)
vecframe=vecframe.iloc[0:700, 2:22]

print(vecframe.shape)
sequence= vecframe.values
# reshape input into [samples, timesteps, features]
n_in = vecframe.shape[1]
n_samples=vecframe.shape[0]
sequence = sequence.reshape((n_samples, n_in, 1)) ## because 7 days for 1 user
print(sequence.shape)

# define model
model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(n_in,1)))
### 1st argument above is the length of fixed-sized vector ie the internal representation of the input sequence.

model.add(RepeatVector(n_in))
#This layer simply repeats the provided 2D input multiple times to create a 3D output.
# output of the encoder is 2D and 3D input to the decoder is required.


model.add(LSTM(10, activation='relu', return_sequences=True))
### One or more LSTM layers can also be used to implement the decoder model.
### This model reads from the fixed sized output from the encoder model.

model.add(TimeDistributed(Dense(1)))
### a Dense layer is used as the output for the network.
# The same weights can be used to output each time step in the output sequence
# by wrapping the Dense layer in a TimeDistributed wrapper.

model.compile(optimizer='adam', loss='mse')
model.fit(sequence, sequence, epochs=100, verbose=2)

saveKerasmodel(model, DATAPATH)
del model


model=loadKerasmodel(DATAPATH)

## connect the encoder LSTM as the output layer
model = Model(inputs=model.inputs, outputs=model.layers[0].output)
## This gives us the latent layer embeddings of our desired size as the predict output instead of the reconstructions
# create a new model that has the same inputs as our original model,
# and outputs directly from the end of encoder model, before the RepeatVector layer.


#plot_model(model, show_shapes=True, to_file='lstm_encoder.png')


print(vecframe.iloc[0:7].values)
yhat = model.predict(vecframe.iloc[0:7].values.reshape(7, n_in, 1)) # get the feature vector for the input sequence because we modified the output layer earlier
print(yhat.shape)
print(yhat[0:7,:,0])
