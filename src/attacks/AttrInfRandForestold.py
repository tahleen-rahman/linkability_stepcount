from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize, LabelBinarizer

import numpy as np
import pandas as pd
import logging

from attacks import AttributeInf#, DnnClassifier
from utils.storage import DATA_PATH, load_frame

class AttrInfRandForest(AttributeInf):

    def __init__(self, vf_fname, attribute, desc_file='dzne_desc', data_path=DATA_PATH):
        # this already loads the vecframe from vec_name into a dataframe self.vecframe
        super().__init__(vf_fname, attribute, desc_file, data_path)

        self.logger = logging.getLogger("AttrInfRF")


    def attack(self):

        super().attack()

        #self.vecframe['user'] = pd.to_numeric(self.vecframe['user'])

        clf = RandomForestClassifier(n_estimators=100, random_state=0)

        # att = load_frame("dzne_desc") #pd.read_csv(DATA_PATH + 'dzne_desc.csv')## make column  gender
        # att = self.attr.loc[:,['user','age', 'edu', 'sex']]
        # att['user'] = pd.to_numeric(att['user'])

        if self.attribute=='gender':

            self.att.loc[self.att['sex'] == 'm', 'sex'] = 0
            self.att.loc[self.att['sex'] == 'f', 'sex'] = 1

            df = self.vecframe.merge(self.att, on = 'user' )

            males = df[df.sex == 0]
            male_users = males.user.unique()
            male_u_tr = male_users[:int(0.8 * len(male_users))]
            #m_train = males.iloc[: int(0.8 * len(males)), :]

            m_train = males[males.user.isin(male_u_tr)]
            m_test = males.drop(m_train.index)

            females = df[df.sex == 1]
            fem_users = females.user.unique()
            fem_u_tr = fem_users[:int(0.8*len(fem_users))]

            f_train = females[females.user.isin(fem_u_tr)]
            f_test = females.drop(f_train.index)

            """
            m_train = df[df.sex == 0].sample(frac=0.8)
            m_test = df[df.sex == 0].drop(m_train.index)

            f_train = df[df.sex == 1].sample(frac=0.8)
            f_test = df[df.sex == 1].drop(f_train.index)
            """

            train_ = m_train.append(f_train)
            train_.to_csv(DATA_PATH + 'train_.csv')

            test_ = m_test.append(f_test)
            test_.to_csv(DATA_PATH + 'test_.csv')

            X_train, y_train = train_.iloc[:, 2:-3].values, train_.sex.values
            X_test, y_test = test_.iloc[:, 2:-3].values, test_.sex.values

            clf.fit(X_train, y_train)

            pred_proba = clf.predict_proba(X_test)

            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_test, pred_proba[:, 1])
            print(self.vf_fname, auc)
            return auc


        elif self.attribute=='edu':

            #print ("WARNING! USER WISE Split not yet implemented for edu")

            self.att.loc[self.att['edu'] == 'high', 'edu'] = 2
            self.att.loc[self.att['edu'] == 'middle', 'edu'] = 1
            self.att.loc[self.att['edu'] == 'low', 'edu'] = 0

            df = self.vecframe.merge(self.att, on='user')

            low = df[df.edu == 0]
            low_u = low.user.unique()
            low_u_tr = low_u[:int(0.8 * len(low_u))]

            l_train = low[low.user.isin(low_u_tr)]
            l_test = low.drop(l_train.index)


            med = df[df.edu == 1]
            med_u = med.user.unique()
            med_u_tr = med_u[:int(0.8 * len(med_u))]

            m_train = med[med.user.isin(med_u_tr)]
            m_test = med.drop(m_train.index)

            high = df[df.edu == 2]
            high_u = high.user.unique()
            high_u_tr = high_u[:int(0.8 * len(high_u))]

            h_train = high[high.user.isin(high_u_tr)]
            h_test = high.drop(h_train.index)



            """l_train = df[df.edu == 0].sample(frac=0.8)
            l_test = df[df.edu == 0].drop(l_train.index)

            m_train = df[df.edu == 1].sample(frac=0.8)
            m_test = df[df.edu == 1].drop(m_train.index)

            h_train = df[df.edu == 2].sample(frac=0.8)
            h_test = df[df.edu == 2].drop(h_train.index)"""

            train_ = h_train.append(m_train)
            train_ = train_.append(l_train)
            train_.to_csv(DATA_PATH + 'train_.csv')

            test_ = h_test.append(m_test)
            test_ = test_.append(l_test)
            test_.to_csv(DATA_PATH + 'test_.csv')

            X_train, y_train = train_.iloc[:, 2:-3].values, train_.edu.values.tolist()
            X_test, y_test = test_.iloc[:, 2:-3].values, test_.edu.values.tolist()

            y_multi = label_binarize(y_train, classes=[0, 1, 2])

            clf.fit(X_train, y_multi)

            y_multi_test = label_binarize(y_test, classes=[0, 1, 2])
            pred = clf.predict(X_test)

            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_multi_test, pred)
            self.logger.debug('{}: {}'.format(self.vf_fname, auc))
# use macro micro accuracy **
            return auc
