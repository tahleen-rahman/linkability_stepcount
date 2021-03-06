import os

from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine, euclidean

from attacks import Attack
import pandas as pd


class Link(Attack):


    def __init__(self, vf_fname, weekends, in_datapath = '../data/dzne/',  out_datapath = '../data/dzne/'):
        """
        creates or loads self.tr_pairs and  self.te_pairs
        :param vf_fname: filename name of input vecframe
        :param weekends: whether to remove weekends or not
        :param in_datapath:
        :param out_datapath:

        """

        super().__init__(vf_fname, in_datapath)

        assert('dsp' in self.vf_fname)
        assert('week' not in self.vf_fname)

        self.out_datapath =  out_datapath
        self.weekends = weekends

        if weekends:

            self.pairsfilepath = self.out_datapath + "pairs.csv"
            self.tr_pairsfilepath = self.out_datapath + "tr_pairs.csv"
            self.te_pairsfilepath = self.out_datapath + "te_pairs.csv"

        else:

            self.pairsfilepath = self.out_datapath + "noweekend_pairs.csv"
            self.tr_pairsfilepath = self.out_datapath + "noweekend_tr_pairs.csv"
            self.te_pairsfilepath = self.out_datapath + "noweekend_te_pairs.csv"


        if not (os.path.exists(self.tr_pairsfilepath) and os.path.exists(self.te_pairsfilepath)):

            print ('making', self.tr_pairsfilepath , 'and', self.te_pairsfilepath)
            self.makePairs()

        self.tr_pairs = pd.read_csv(self.tr_pairsfilepath)
        self.te_pairs = pd.read_csv(self.te_pairsfilepath)
        self.pairs    =  pd.read_csv(self.pairsfilepath)



    def makePairs(self):
        """
        make umsymmetric pairs and saves indices
        :return:
        """

        import itertools
        import numpy as np

        users = self.vecframe.index.unique()
        df = self.vecframe

        true, false = [], []

        if self.weekends:

            for user in users:

                user_ind = df[df.user == user].index
                true_pairs_ = list(itertools.combinations(user_ind, 2))
                true_pairs = [list(x) for x in true_pairs_]

                false_ind = df[df.user != user].index
                false_pairs_ = list(np.random.choice(false_ind, size=(len(true_pairs), 2), replace=False))
                false_pairs = [list(x) for x in false_pairs_]

                true += true_pairs
                false += false_pairs

        else: # remove weekends

            for user in users:

                user_ind = df[df.user == user][df.desc >1].index
                true_pairs_ = list(itertools.combinations(user_ind, 2))
                true_pairs = [list(x) for x in true_pairs_]

                false_ind = df[df.user != user][df.desc >1].index
                false_pairs_ = list(np.random.choice(false_ind, size=(len(true_pairs), 2), replace=False))
                false_pairs = [list(x) for x in false_pairs_]

                true += true_pairs
                false += false_pairs


        true_df = pd.DataFrame(data=true, columns=['i', 'j'])
        true_df['label'] = 1

        false_df = pd.DataFrame(data=false, columns=['i', 'j'])
        false_df['label'] = 0

        pairs = true_df.append(false_df)
        pairs.to_csv(self.pairsfilepath, index=False)
        print (len(pairs), "pairs made")



        tr_tru = true_df.sample(frac=0.8)
        te_tru = true_df.drop(tr_tru.index)

        tr_fal = false_df.sample(frac=0.8)
        te_fal = false_df.drop(tr_fal.index)

        tr_pairs = tr_tru.append(tr_fal)
        te_pairs = te_tru.append(te_fal)

        tr_pairs.to_csv(self.tr_pairsfilepath, index=False)
        te_pairs.to_csv(self.te_pairsfilepath, index=False)



    def prep_data(self, combi):

        def combine(u,v):

            if combi == 'avg':

                combined = pd.np.mean([u, v], axis=0)

            elif combi == 'concat':

                combined = pd.concat([u, v], ignore_index=True)

            elif combi=='l1':

                combined = pd.np.absolute(pd.np.subtract(u, v))

            elif combi=='mul':

                combined = pd.np.multiply(u,v)

            elif combi == 'l2':

                combined = [i ** 2 for i in pd.np.subtract(u, v)]

            if combi == 'cosine':

                combined = cosine(u, v)

            return combined



        u = self.vecframe.loc[self.tr_pairs.i].iloc[:, 2:].values
        v = self.vecframe.loc[self.tr_pairs.j].iloc[:, 2:].values

        tr_data = pd.DataFrame(data = combine(u,v))
        #tr_data[['i', 'j', 'label']] = self.tr_pairs
        self.tr_data = self.tr_pairs.join(tr_data)



        u_ = self.vecframe.loc[self.te_pairs.i].iloc[:, 2:].values
        v_ = self.vecframe.loc[self.te_pairs.j].iloc[:, 2:].values

        te_data = pd.DataFrame(data=combine(u_, v_))
        self.te_data = self.te_pairs.join(te_data)


        self.tr_data.to_csv(self.tr_data_fp)
        self.te_data.to_csv(self.te_data_fp)



    def prep_data_unsup(self, combi):

        def combine(u, v):

            if combi == 'cosine':
                combined = [cosine(u[i], v[i]) for i in range(len(u))]

            if combi == 'eucl':
                combined = [euclidean(u[i], v[i]) for i in range(len(u))]

            return combined



        u = self.vecframe.loc[self.pairs.i].iloc[:, 2:].values
        v = self.vecframe.loc[self.pairs.j].iloc[:, 2:].values

        unsup_data = pd.DataFrame(data=combine(u, v))

        self.unsup_data = self.pairs.join(unsup_data)

        self.unsup_data.to_csv(self.unsup_data_fp)



    def attack(self, clf):

        train_ = pd.read_csv(self.tr_data_fp, index_col=0)
        test_ = pd.read_csv(self.te_data_fp, index_col=0)

        X_train, y_train = train_.iloc[:, 3:].values, train_.label.values
        X_test, y_test = test_.iloc[:, 3:].values, test_.label.values

        clf.fit(X_train, y_train)
        pred_ = clf.predict(X_test)
        auc = roc_auc_score(y_test, pred_)

        return auc


    def unsup_attack(self,):

        train_ = pd.read_csv(self.unsup_data_fp, index_col=0)

        train_.dropna(inplace=True)

        auc = roc_auc_score(train_.label.values, train_.iloc[:, 3:].values)

        return 1-auc

