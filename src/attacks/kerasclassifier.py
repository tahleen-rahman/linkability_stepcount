# Created by rahman at 11:06 2020-02-22 using PyCharm
import tensorflow as tf
tf.random.set_seed(3)

#if tf.test.is_built_with_cuda or tf.__version__ == '2.1.0':
from tensorflow import keras

k_init = tf.keras.initializers.Constant(value=0.1)
b_init = keras.initializers.Constant(value=0)
r_init = keras.initializers.Constant(value=0.1)

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten , LSTM, Bidirectional, Reshape


from tensorflow.keras import Sequential, utils, Input, Model

#if tf.__version__ != '2.1.0':
    #import keras.layers.CuDNNLSTM

"""import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Reshape, LSTM
from keras import Sequential, utils, Input, Model"""

from keras.utils.vis_utils import plot_model




import logging
import math

class BinaryDNN:

    def __init__(self, num_layers = 2, layer_params=[[0.5, 0.2], [0.25, 0]], num_epochs=100, batch_size=64, verbose=0):
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
        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=30, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs = self.num_epochs, batch_size = self.batch_size, verbose=self.verbose, callbacks=[es])


    def predict(self, X_test):

        y_pred = self.model.predict(X_test, verbose=True)

        return (y_pred > 0.5)







class SiameseClassifier:

    def __init__(self, shape, regu, combi, time_dim=0):

        self.time_dim = time_dim
        self.regu =  keras.regularizers.l2(regu)
        self.combi = combi

        #self.l_a and self.l_b will be defined in the child constructors
        if len(shape)==3:

            assert(self.time_dim==1)
            self.num_features = shape[2]
            self.num_timesteps = shape[1]

            self.sample_a = Input(shape=(self.num_timesteps, self.num_features))
            self.sample_b = Input(shape=(self.num_timesteps, self.num_features))

        elif len(shape)==2: #it is an old style vecframe

            self.num_features = shape[1]-2

            self.sample_a = Input(shape=(self.num_features,))
            self.sample_b = Input(shape=(self.num_features,))


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



        predictions = Dense(1, activation='sigmoid')(combined)

        self.model = tf.keras.Model(inputs=[self.sample_a, self.sample_b], outputs=predictions)

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

        print(self.model.summary())

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



    def fit_predict_callback(self, link, batchsize, epochs, patience, verbose=0):

        es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=patience, restore_best_weights=True)
        #mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        print (self.model.layers)

        if self.time_dim: #link.vecframe is actually just a 3d arrary, not dataframe
            self.model.fit([link.vecframe[link.tr_pairs.i][:, :], link.vecframe[link.tr_pairs.j][:, :]],
                           link.tr_pairs.label,
                           batch_size=batchsize, epochs=epochs,
                           validation_data=(
                           [link.vecframe[link.te_pairs.i][:, :], link.vecframe[link.te_pairs.j][:, :]],
                           link.te_pairs.label), verbose=verbose,
                           callbacks=[es]) #, mc
            y_pred = self.model.predict(
                [link.vecframe[link.te_pairs.i][:, :], link.vecframe[link.te_pairs.j][:, :]],
                verbose=verbose)

        else:

            self.model.fit([link.vecframe.loc[link.tr_pairs.i].iloc[:, 2:], link.vecframe.loc[link.tr_pairs.j].iloc[:, 2:]],
                       link.tr_pairs.label,
                       batch_size=batchsize, epochs=epochs,
                       validation_data=(
                       [link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                       link.te_pairs.label), verbose=verbose,
                       callbacks=[es]) #, mc
            y_pred = self.model.predict(
                [link.vecframe.loc[link.te_pairs.i].iloc[:, 2:], link.vecframe.loc[link.te_pairs.j].iloc[:, 2:]],
                verbose=verbose)




        #layer_name = 'attention' its layer 8
        """for layer in self.model.layers[2:]:  # all layer outputs

            print(layer)

            intermediate_layer_model = keras.Model(inputs=self.model.input,
                                               outputs=layer.output)

            test_a = tf.ones((1,12))
            test_b = tf.ones((1,12))
            intermediate_output = intermediate_layer_model.predict([test_a, test_b])

            print (intermediate_output[0])"""



        #saved_model = load_model('best_model.h5')





        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(link.te_pairs.label, y_pred)

        return auc



class Dense_siameseClassifier(SiameseClassifier):

    def __init__(self, shape, regu, combi, dense_params):

        super().__init__(shape, regu, combi)

        half_num_features = int(math.floor(self.num_features * dense_params[0]) if self.num_features>=2 else 1)
        shared_dense1 = Dense(half_num_features, input_shape=(self.num_features,),
                              activation='relu') #, kernel_initializer=k_init, bias_initializer=b_init


        if self.regu != None:
            shared_dense1.kernel_regularizer = self.regu

        l_a = shared_dense1(self.sample_a)
        l_b = shared_dense1(self.sample_b)

        quater_num_features = int(math.floor(self.num_features * dense_params[1]) if self.num_features>=4 else 1)
        shared_dense2 = Dense(quater_num_features, input_shape=(half_num_features,),
                              activation='relu') #, kernel_initializer=k_init, bias_initializer=b_init

        if self.regu != None:
            shared_dense2.kernel_regularizer = self.regu

        l_a2 = shared_dense2(l_a)
        l_b2 = shared_dense2(l_b)

        self.l_a = l_a2
        self.l_b = l_b2




class LSTMsiameseClassifier(SiameseClassifier):

    def __init__(self, num_timesteps, regu, combi, time_dim, lstm_params, fixed_units=True):
        super().__init__(num_timesteps, regu, combi, time_dim)

        shared_nn = Sequential()

        shared_nn.add(Reshape((num_timesteps, 1), input_shape=(num_timesteps,)))

        if len(lstm_params)==2:

            param_0 = lstm_params[0]

            units = param_0[0]

            shared_nn.add(LSTM(units, return_sequences=True, input_shape=(num_timesteps, 1),\
                               kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
            shared_nn.add(Dropout(param_0[1]))

            param_1 = lstm_params[1]
            units = param_1[0]

            shared_nn.add(LSTM(units, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
            shared_nn.add(Dropout(param_1[1]))

        elif len(lstm_params)==1:

            param_0 = lstm_params[0]

            units = param_0[0] if fixed_units else (int(math.floor(num_timesteps * param_0[0])) if num_timesteps >= 4 else 1)

            shared_nn.add(LSTM(units,input_shape=(num_timesteps, 1), \
                               kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))
            shared_nn.add(Dropout(param_0[1]))


        self.l_a = shared_nn(self.sample_a)
        self.l_b = shared_nn(self.sample_b)




class BiLSTMsiameseClassifier(SiameseClassifier):

    def __init__(self, num_timesteps, regu, combi, lstm_params, fixed_units=True):
        super().__init__(num_timesteps, regu, combi)

        shared_nn = Sequential()

        shared_nn.add(Reshape((num_timesteps, 1), input_shape=(num_timesteps,)))

        if len(lstm_params)==2:

            param_0 = lstm_params[0]

            units = param_0[0]


            shared_nn.add(Bidirectional(LSTM(units, return_sequences=True, input_shape=(num_timesteps, 1),\
                               kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)))
            shared_nn.add(Dropout(param_0[1]))

            param_1 = lstm_params[1]
            units = param_1[0]

            shared_nn.add(Bidirectional(LSTM(units, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)))
            shared_nn.add(Dropout(param_1[1]))

        elif len(lstm_params)==1:

            param_0 = lstm_params[0]

            units = param_0[0] if fixed_units else (int(math.floor(num_timesteps * param_0[0])) if num_timesteps >= 4 else 1)

            shared_nn.add(Bidirectional(LSTM(units,input_shape=(num_timesteps, 1), \
                               kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)))
            shared_nn.add(Dropout(param_0[1]))


        self.l_a = shared_nn(self.sample_a)
        self.l_b = shared_nn(self.sample_b)



"""class Attention(tf.keras.Model):

    def __init__(self, units):
        super(Attention, self).__init__()

        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):

        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

max_len = 200
rnn_cell_size = 128
vocab_size=250"""

#https://medium.com/@sanjivgautamofficial/lstm-in-keras-56a59264c0b2
#https://www.coursera.org/lecture/nlp-sequence-models/attention-model-lSwVa
#The original idea of attention uses the output of the decoder as h_t, representing "current decoding state".
# If you think of the "many-to-one" problem as a special case of the "many-to-many" problem, h_t becomes the last hidden state of the encoder.
#https://github.com/tensorflow/tensorflow/issues/39332
"""class AttentionBiLSTMClassifier(SiameseClassifier):

    def __init__(self, num_timesteps, regu, combi, lstm_params, fixed_units=True):
        super().__init__(num_timesteps, regu, combi)

        units, dropout = lstm_params[0][0],  lstm_params[0][1]
        shared = []

        for sample in [self.sample_a, self.sample_b]:

            sample = tf.keras.layers.Reshape((num_timesteps, 1), input_shape=(num_timesteps,))(sample)

            lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional \
                (tf.keras.layers.LSTM
                 (units,
                  dropout= dropout,
                  return_sequences=True,
                  return_state=True,
                  recurrent_activation='relu',
                  recurrent_initializer='glorot_uniform'))(sample)

            state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
            #state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

            # Query-value attention of shape [batch_size, Tq, filters].
            query_value_attention_seq = tf.keras.layers.Attention()([lstm, state_h])

            query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
                query_value_attention_seq)

            #output = keras.layers.Dense(1, activation='sigmoid')(context_vector)

            #model = keras.Model(inputs=sequence_input, outputs=output)

            # summarize layers
            #print(model.summary())
            shared.append(query_value_attention)


        self.l_a = shared[0]
        self.l_b = shared[1]
"""

class AttentionBiLSTMClassifier(SiameseClassifier):

    def __init__(self, shape, regu, combi, time_dim, lstm_params, fixed_units=True):
        super().__init__(shape, regu, combi, time_dim)


        units, dropout = lstm_params[0][0],  lstm_params[0][1]

        shared = []

        for sample in [self.sample_a, self.sample_b]:

            sample = tf.keras.layers.Reshape((self.num_timesteps, self.num_features), input_shape=(self.num_timesteps, self.num_features))(sample)

            lstm, forward_h, forward_c, backward_h, backward_c = tf.keras.layers.Bidirectional(LSTM(units,input_shape=(self.num_timesteps, self.num_features), \
                               kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init, return_sequences=True, return_state=True))(sample)

            state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
            # state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

            # Query-value attention of shape [batch_size, Tq, filters].
            query_value_attention_seq = tf.keras.layers.Attention()([state_h, lstm])

            query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
                query_value_attention_seq)

            shared.append(query_value_attention)

        self.l_a = shared[0]
        self.l_b = shared[1]

def GlobalAttention(hidden_states):
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention
    # Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = tf.keras.layers.Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(
        hidden_states)
    score = tf.keras.layers.Dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = tf.keras.layers.Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = tf.keras.layers.Dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = tf.keras.layers.Concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh',
                             name='attention_vector')(
        pre_activation)
    return attention_vector


class GlobalAttentionBiLSTMClassifier(SiameseClassifier):

    def __init__(self, num_timesteps, regu, combi, lstm_params, fixed_units=True):
        super().__init__(num_timesteps, regu, combi)

        units, dropout = lstm_params[0][0],  lstm_params[0][1]

        shared = []

        for sample in [self.sample_a, self.sample_b]:

            sample = tf.keras.layers.Reshape((num_timesteps, 1), input_shape=(num_timesteps,))(sample)

            lstm_out = tf.keras.layers.Bidirectional(LSTM(units,input_shape=(num_timesteps, 1), \
                               kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init, return_sequences=True))(sample)

            # Query-value attention of shape [batch_size, Tq, filters].
            attention = tf.keras.layers.GlobalAttention(lstm_out)

            shared.append(attention)

        self.l_a = shared[0]
        self.l_b = shared[1]

def tensorabs(t):
    return abs(t)


class CNNsiameseClassifier(SiameseClassifier):

    def __init__(self, shape, regu, combi, cnn_params):

        super().__init__(shape, regu, combi)


        (filt1, ker1), (filt2, ker2), pool, num_maxpools = cnn_params

        assert (num_maxpools in [1, 2])


        shared_conv1 = Sequential()

        shared_conv1.add(Reshape((self.num_features, 1), input_shape=(self.num_features,)))

        shared_conv1.add(Conv1D(filters=filt1, kernel_size=ker1, activation='relu', kernel_regularizer=self.regu, padding='same'))

        if num_maxpools==2:

            shared_conv1.add(MaxPooling1D(pool_size=pool))

        shared_conv1.add(Conv1D(filters=filt2, kernel_size=ker2, activation='relu', kernel_regularizer=self.regu, padding='same'))

        shared_conv1.add(MaxPooling1D(pool_size=pool))


        shared_conv1.add(Flatten())

        shared_conv1.add(Dense(100, activation='relu'))

        self.l_a = shared_conv1(self.sample_a)
        self.l_b = shared_conv1(self.sample_b)





