# Created by rahman at 11:14 2020-01-25 using PyCharm


import os

import pandas as pd
from sklearn.model_selection import GroupKFold
from utils.storage import DATA_PATH, load_frame
from attacks import Attack
from sklearn.feature_selection import VarianceThreshold, RFECV

from sklearn.metrics import roc_auc_score


class AttributeInferenceCV(Attack):

    def __init__(self, vf_fname, attribute, in_datapath=DATA_PATH,  out_datapath = DATA_PATH):

        """

        :param vf_fname: filename of vecframe of input features
        :param attribute: gender or age or edu
        :param in_datapath: location of input features
        :param out_datapath: location where train and test sets will be saved
        """
        # this constructor sets the attribute, out_datapath, train and test fnames, merges with the input features in vecframe,

        # the parent class constructor loads the vecframe from vf_fname into a dataframe self.vecframe and sets in_datapath
        super().__init__(vf_fname, in_datapath)

        assert (attribute in ['sex', 'edu_binary', 'age_binary'])#, 'age_multi', 'edu_multi'

        self.attribute = attribute

        self.att = load_frame("dzne_desc")  # pd.read_csv(DATA_PATH + 'dzne_desc.csv')
        self.att = self.att.loc[:, ['user', 'age', 'edu', 'sex']]
        self.att['user'] = pd.to_numeric(self.att['user'])

        self.merged = self.vecframe.merge(self.att, on='user')


        self.train_fname = self.attribute + '_train_' + self.vf_fname + '.csv'
        self.test_fname = self.attribute + '_test_' + self.vf_fname + '.csv'

        self.out_datapath = out_datapath + self.attribute + '/'

        for i in range(0,5):

            if not os.path.exists(self.out_datapath + str(i) + '/'):
                os.makedirs(self.out_datapath + str(i) + '/')



    def makeDataset(self):

        df = self.merged

        for i in range(0,5):

            if self.attribute=='sex':

                df.loc[df['sex'] == 'm', 'sex'] = 0
                df.loc[df['sex'] == 'f', 'sex'] = 1

                males = df[df.sex == 0]
                male_users = males.user.unique()
                male_u_te = male_users[i * int(0.2 * len(male_users)): (i + 1) * int(0.2 * len(male_users))]
                #m_train = males.iloc[: int(0.8 * len(males)), :]

                m_test = males[males.user.isin(male_u_te)]
                m_train = males.drop(m_test.index)

                females = df[df.sex == 1]
                fem_users = females.user.unique()
                fem_u_test = fem_users[i * int(0.2 * len(fem_users)): (i + 1) * int(0.2 * len(fem_users))]

                f_test = females[females.user.isin(fem_u_test)]
                f_train = females.drop(f_test.index)

                """
                m_train = df[df.sex == 0].sample(frac=0.8)
                m_test = df[df.sex == 0].drop(m_train.index)
    
                f_train = df[df.sex == 1].sample(frac=0.8)
                f_test = df[df.sex == 1].drop(f_train.index)
                """

                train_ = m_train.append(f_train)
                train_.to_csv(self.out_datapath +  str(i) + '/'+ self.train_fname)

                test_ = m_test.append(f_test)
                test_.to_csv(self.out_datapath  +  str(i) + '/'+ self.test_fname)




            if self.attribute=='age_binary':

                median = df.age.median()

                df.loc[df['age']< median, 'age_binary'] = 0
                df.loc[df['age']>= median, 'age_binary'] = 1

                df.drop(['age'], axis=1, inplace=True)


                youngs = df[df.age_binary == 0]
                young_users = youngs.user.unique()
                young_u_te = young_users[i * int(0.2*len(young_users)) : (i + 1) * int(0.2 * len(young_users))]

                young_test = youngs[youngs.user.isin(young_u_te)]
                young_train = youngs.drop(young_test.index)

                olds = df[df.age_binary == 1]
                old_users = olds.user.unique()
                old_u_te = old_users[i * int(0.2*len(old_users)) : (i + 1) * int(0.2 * len(old_users)) ]

                old_test = olds[olds.user.isin(old_u_te)]
                old_train = olds.drop(old_test.index)


                train_ = young_train.append(old_train)
                train_.to_csv(self.out_datapath + str(i) + '/' + self.train_fname)

                test_ = young_test.append(old_test)
                test_.to_csv(self.out_datapath  + str(i) + '/' + self.test_fname)


            if self.attribute == 'age_multi':

                print ("WARNING! age_multiclass not implemented!!")



            if self.attribute == 'edu_binary':

                df.loc[df['edu'] == 'high', 'edu_binary'] = 1
                df.loc[df['edu'] == 'middle', 'edu_binary'] = 0

                df.drop(['edu'], axis=1, inplace=True)


                #df.drop(df[df['edu'] == 'low'].index, inplace=True)
                #df.drop(df[df['edu'] < 0].index, inplace=True)



                med = df[df.edu_binary == 0]
                med_u = med.user.unique()
                med_u_te = med_u[i * int(0.2 * len(med_u)) : (i + 1) * int(0.2 * len(med_u)) ]

                m_test = med[med.user.isin(med_u_te)]
                m_train = med.drop(m_test.index)

                high = df[df.edu_binary == 1]
                high_u = high.user.unique()
                high_u_te = high_u[i * int(0.2 * len(high_u)) : (i + 1) * int(0.2 * len(high_u))]

                h_test = high[high.user.isin(high_u_te)]
                h_train = high.drop(h_test.index)

                train_ = h_train.append(m_train)
                train_.to_csv(self.out_datapath + str(i) + '/' + self.train_fname)

                test_ = h_test.append(m_test)
                test_.to_csv(self.out_datapath + str(i) + '/' + self.test_fname)


            if self.attribute == 'edu_multi':

                print ("WARNING!  multiclass attack is not implemented!")

                """df.loc[df['edu_multi'] == 'high', 'edu'] = 2
                df.loc[df['edu_multi'] == 'middle', 'edu'] = 1
                df.loc[df['edu_multi'] == 'low', 'edu'] = 0
    
    
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
    
                train_ = h_train.append(m_train)
                train_ = train_.append(l_train)
                train_.to_csv(self.out_datapath + self.train_fname)
    
                test_ = h_test.append(m_test)
                test_ = test_.append(l_test)
                test_.to_csv(self.out_datapath + self.test_fname)"""





    def fs_attack(self, clf, do_vt = None, do_rfe = None, verbose = None):
        """
        :param clf: classifier
        :param do_vt: do variance thresholding
        :param do_rfe: do recursive feature selection
        :return: [auc, auc_lv, auc_rfe] always 3 values. if no features were removed, the regular auc repeats.

        """

        retarr=[]

        train_ = pd.read_csv(self.out_datapath + self.train_fname, index_col=0)
        test_ =  pd.read_csv(self.out_datapath + self.test_fname, index_col=0)


        X_train, y_train = train_.iloc[:, 2:-3].values, train_[self.attribute].values
        X_test, y_test = test_.iloc[:, 2:-3].values, test_[self.attribute].values


        clf.fit(X_train, y_train)
        pred_ = clf.predict(X_test)
        auc = roc_auc_score(y_test, pred_)

        if auc >= 0.5:
            print(self.vf_fname + ',', auc)
        else:
            print(self.vf_fname + ',', 1 - auc)

        retarr.append(auc)

        if do_vt:

            sel = VarianceThreshold()
            sel.fit(X_train)

            #print (sel.variances_)
            X_train_lv = sel.transform(X_train)
            #print(sel.get_support(indices=True))


            if (X_train.shape[1] > X_train_lv.shape[1]):

                if verbose:
                    print("X_train.shape[1], X_train_lv.shape[1]", X_train.shape[1], X_train_lv.shape[1])  # , X_test_lv.shape)

                X_test_lv = sel.transform(X_test)
                clf.fit(X_train_lv, y_train)
                pred_ = clf.predict(X_test_lv)
                auc_lv = roc_auc_score(y_test, pred_)

                if auc_lv >= 0.5:
                    print(self.vf_fname + '_lv,', auc_lv)
                else:
                    print(self.vf_fname + '_lv,', 1 - auc_lv)


                X_train = X_train_lv
                X_test = X_test_lv

                retarr.append(auc_lv)

            else:
                retarr.append(retarr[-1])



        if do_rfe:

            if not hasattr(clf, 'score'):

                print ("WARNING! The classifier passed should have a 'score' method for RFE! You are probably using BinaryDNN! RFE will be skipped!")
                retarr.append(retarr[-1])

            else:

                if X_train.shape[1] <= 14 : # too few features
                    if verbose:
                        print ("too few features, skipping RFE")
                    retarr.append(retarr[-1])

                else:
                    selector = RFECV(clf, step=1, cv=5,  n_jobs=-2)
                    selector.fit(X_train, y_train)

                    if (selector.n_features_ < X_train.shape[1]):

                        if verbose:
                            print(selector.n_features_, " feats selected out of", X_train.shape[1])

                        X_train_fe = selector.transform(X_train)
                        X_test_fe = selector.transform(X_test)


                        clf.fit(X_train_fe, y_train)
                        pred_ = clf.predict(X_test_fe)
                        auc_fe = roc_auc_score(y_test, pred_)

                        if auc_fe >= 0.5:
                            print(self.vf_fname + '_lv_fe,', auc_fe)
                        else:
                            print(self.vf_fname + '_lv_fe,', 1 - auc_fe)

                        retarr.append(auc_fe)

                    else: # if nothing was removed

                        retarr.append(retarr[-1])

        return retarr




        """
        else:

            clf.fit(X_train, y_train)
            pred_ = clf.predict(X_test)
            auc = roc_auc_score(y_test, pred_)

            if auc >= 0.5:
                print(self.vf_fname +',' , auc)
            else:
                print(self.vf_fname +',' , 1 - auc)

            return auc, auc
        """



    def attack(self, clf):

        aucarr=[]

        for i in range(0,5):

            train_ = pd.read_csv(self.out_datapath +  str(i) + '/' + self.train_fname, index_col=0)
            test_ =  pd.read_csv(self.out_datapath  + str(i) + '/' + self.test_fname, index_col=0)


            X_train, y_train = train_.iloc[:, 2:-3].values, train_[self.attribute].values
            X_test, y_test = test_.iloc[:, 2:-3].values, test_[self.attribute].values


            clf.fit(X_train, y_train)

            pred_ = clf.predict(X_test)



            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(y_test, pred_)

            if auc >= 0.5:
                print(self.vf_fname +',' , auc)

                aucarr.append(auc)
            else:
                print(self.vf_fname +',' , 1 - auc)

                aucarr.append(1-auc)

        return aucarr



    def attack_activities(self, clf,  th = 0.5):
        """

        :param clf: classifier object
        :param th: how much irrelevant activities to filter out
        :return:    auc_vote_bin:  AUC between true label and majority voted label after binarizing the positive class probabilities post filtering
                    auc_proba1_bin: AUC between true label and binarized average of the positive class probabilities  post filtering
        """


        arr_vote, arr_proba_1 = [], []

        for i in range(0,5):

            train_ = pd.read_csv(self.out_datapath + str(i) + '/'+ self.train_fname, index_col=0)
            test_ =  pd.read_csv(self.out_datapath + str(i) + '/'+ self.test_fname, index_col=0)


            X_train, y_train = train_.iloc[:, 2:-3].values, train_[self.attribute].values
            X_test = test_.iloc[:, 2:-3].values


            df = test_[['user', self.attribute]]

            clf.fit(X_train, y_train)

            if not hasattr(clf, 'predict_proba'):
                print( "WARNING! The classifier should support class probabilities! Use Softmax activation for NNs ")
                pred_proba = clf.predict(X_test)
                df['proba_1'] = pred_proba
            else:
                pred_proba = clf.predict_proba(X_test)
                df['proba_1']= pred_proba[:, 1]


            df['abs_diff'] = df.proba_1.apply(lambda x: abs(0.5 - x))

            df = df.groupby('user').apply(lambda grp: grp.nlargest( n = int(th*len(grp)) , columns='abs_diff'))

            df = df.drop(columns='abs_diff')

            df = df.reset_index(drop=True)

            #meaned = df.groupby('user', as_index=False).mean()

            df['vote'] = df.proba_1.apply(lambda x:int(x > 0.5 ))

            meaned = df.groupby('user', as_index=False).mean()

            meaned['vote_bin'] = meaned.vote.apply(lambda x:int(x > 0.5 ))

            meaned['proba_1_bin'] = meaned.proba_1.apply(lambda x: int(x > 0.5))



            # auc_vote_bin = roc_auc_score(meaned[self.attribute], meaned['vote_bin'])
            #
            # auc_proba_1_bin = roc_auc_score(meaned[self.attribute], meaned['proba_1_bin'])


            auc_vote_bin = roc_auc_score(meaned[self.attribute], meaned['vote'])

            auc_proba_1_bin = roc_auc_score(meaned[self.attribute], meaned['proba_1'])


            arr_vote.append(auc_vote_bin)
            arr_proba_1.append(auc_proba_1_bin)

            print ("split", i, auc_vote_bin, auc_proba_1_bin)


        return pd.np.mean(arr_vote), pd.np.mean(arr_proba_1)



