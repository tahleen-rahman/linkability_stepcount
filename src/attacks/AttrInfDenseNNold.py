import numpy as np
import logging
from keras.layers import Dropout, Dense
from keras import Sequential
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from attacks import AttributeInf
from utils.storage import DATA_PATH


class DenseNN(AttributeInf):

    def __init__(self, emb_name, attribute, num_epochs=100, batch_size=24, desc_file='dzne_desc', data_path=DATA_PATH):
        # this already loads the vecframe from vec_name into a dataframe self.vecframe
        super().__init__(emb_name, attribute, desc_file, data_path)

        self.logger = logging.getLogger("AttrInfDNN")
        self.num_epochs = num_epochs
        self.batch_size = batch_size


    def attack(self):
        super().attack()

        # embedding is already in self.vecframe

        # choose the first week
        # self.emb = self.emb[0:997]
        #self.embObj = self.emb.set_index('user')
        # choose only embedding part
        #emb = self.vecframe.iloc[:, 2:]

        # all of this is already done by AttrInference
        # self.att = load_frame("dzne_desc") #pd.read_csv(DATA_PATH + 'dzne_desc.csv')
        # self.att = self.att.loc[:,['user','age', 'edu', 'sex']]
        # self.att['user'] = pd.to_numeric(self.att['user'])
        self.attObj = self.att.set_index('user')

        # merge attributes columns to embedding with index "User ID"
        # self.merged_emb = self.embObj.merge(self.attObj, on='user')
        # self.merged_emb = pd.concat([self.embObj, self.attObj], axis=1, join_axes=[self.embObj.index])
        # add gender_vec as one-hot encode
        self.merged_emb["gender_vec"] = self.merged_emb["sex"].map(lambda x: [0, 1] if x == "m" else [1, 0])
        self.labels = np.stack(self.merged_emb["gender_vec"].values)

        # class_labels is gender label class
        self.class_labels = np.unique(self.merged_emb['sex'])
        classifier = DnnClassifier(self.class_labels)
        classifier.data_split(self.merged_emb.iloc[:, 2:-4], self.labels)
        classifier.construct_network(self.vecframe.shape[1]-2)
        classifier.train_model(self.num_epochs, self.batch_size)
        classifier.predict_model()
        auc = classifier.evaluate_model()
        for gender, auc_value in auc.items():
            self.logger.info('the auc for %s is %s' % (gender, auc_value))
        return auc['m']

class DnnClassifier:
    def __init__(self, class_labels):
        self.logger = logging.getLogger("DnnClassifier")

        self.class_labels = class_labels

    def data_split(self, embedding, labels):
        self.logger.info('embedding size is %s, class_label size is %s' % (embedding.shape, labels.shape))
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(embedding, labels,
                                                                                test_size=0.25,
                                                                                random_state=2018
                                                                                )

    def construct_network(self, dim):
        self.model = Sequential()
        self.model.add(Dense(512, input_dim=dim, activation='relu'))
        self.model.add(Dense(128))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(len(self.class_labels), activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy',
                                         metrics=['accuracy', 'mae'])
        self.model.summary()

    def train_model(self, num_epoches, batch_size):
        self.model.fit(self.train_x,
                       self.train_y,
                       epochs=num_epoches,
                       batch_size=batch_size
                       )

    def predict_model(self):
        self.pred_y = self.model.predict(self.test_x, batch_size=32, verbose=True)

    def evaluate_model(self):
        class_auc = {}

        for (idx, c_label) in enumerate(self.class_labels):
            fpr, tpr, thresholds = roc_curve(self.test_y[:, idx].astype(int), self.pred_y[:, idx])
            class_auc[c_label] = auc(fpr, tpr)

        return class_auc
