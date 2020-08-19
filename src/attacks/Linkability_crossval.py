import os

from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cosine, euclidean

from attacks import Attack
import pandas as pd


class Link(Attack):


    def __init__(self, i, vf_fname, time_dim, weekends, in_datapath,  out_datapath, unsup=False):
        """
        creates or loads self.tr_pairs and  self.te_pairs
        :param i: cross validation fold
        time_dim: consider time as a separate dimension or not, used for RNNs
        :param vf_fname: filename name of input vecframe
        :param weekends: whether to remove weekends or not
        :param in_datapath:
        :param out_datapath:

        """

        super().__init__(vf_fname, time_dim, in_datapath)

        assert('dsp' in self.vf_fname)
        assert('week' not in self.vf_fname)


        self.out_datapath = out_datapath + str(i) + '/'
        if not os.path.exists(self.out_datapath):
            os.makedirs(self.out_datapath)

        self.weekends = weekends


        if weekends:

            self.pairsfilepath = out_datapath + "pairs.csv"
            self.tr_pairsfilepath = self.out_datapath + "tr_pairs.csv"
            self.te_pairsfilepath = self.out_datapath + "te_pairs.csv"

        else:

            self.pairsfilepath = out_datapath + "noweekend_pairs.csv"
            self.tr_pairsfilepath = self.out_datapath + "noweekend_tr_pairs.csv"
            self.te_pairsfilepath = self.out_datapath + "noweekend_te_pairs.csv"


        if not (os.path.exists(self.pairsfilepath)):
            print ('making', self.pairsfilepath)
            self.pairs = self.makePairs()

        else:
            self.pairs = pd.read_csv(self.pairsfilepath)


        if not unsup:

            if not (os.path.exists(self.tr_pairsfilepath) and os.path.exists(self.te_pairsfilepath)):

                print ('making', self.tr_pairsfilepath , 'and', self.te_pairsfilepath)
                self.train_test_folds(i, self.pairs)


            self.tr_pairs = pd.read_csv(self.tr_pairsfilepath)
            self.te_pairs = pd.read_csv(self.te_pairsfilepath)



    def makePairs(self):
        """
        make umsymmetric pairs and saves indices of pairs
        :return:
        """

        import random
        import itertools
        import numpy as np
        def is_same_user(i, j):
            return int(df.loc[i].user == df.loc[j].user)

        df = self.vecframe
        users = df.user.unique()


        true, false = [], []

        if self.weekends:

            for user in users:

                user_ind = df[df.user == user].index.values
                true_pairs_ = list(itertools.combinations(user_ind, 2))
                true_pairs = [list(x) for x in true_pairs_]

                other_ind = df[df.user != user].index.values.tolist()

                others = random.sample(other_ind, len(true_pairs))
                false_pairs = [[random.choice(user_ind), other] for other in others]

                """
                #oversample  twice as many other pairs as true pairs
                other_pairs_ = list(np.random.choice(other_ind, size=(len(true_pairs)*2, 2), replace=False))
                other_pairs = [list(x) for x in other_pairs_]

                # remove index pairs might actually belong to the same user
                count = 0
                for i, j in other_pairs:

                    if df.loc[i].user == df.loc[j].user:
                        print("same user ind", i, j)
                        other_pairs.remove([i, j])
                        count+=1

                false_pairs = other_pairs[:len(true_pairs)]
                """

                true += true_pairs
                false += false_pairs

        else: # remove weekends

            for user in users:

                user_ind = df[df.user == user][df.desc >1].index.values.tolist()
                true_pairs_ = list(itertools.combinations(user_ind, 2))
                true_pairs = [list(x) for x in true_pairs_]


                other_ind = df[df.user != user][df.desc>1].index.values.tolist()

                others = random.sample(other_ind, len(true_pairs))
                false_pairs = [[random.choice(user_ind), other] for other in others]

                """
                #oversample  twice as many other pairs as true pairs
                other_pairs_ = list(np.random.choice(other_ind, size=(len(true_pairs)*2, 2), replace=False))
                other_pairs = [list(x) for x in other_pairs_]

                # remove index pairs might actually belong to the same user
                count = 0
                for i, j in other_pairs:

                    if df.loc[i].user == df.loc[j].user:
                        print("same user ind", i, j)
                        other_pairs.remove([i, j])
                        count+=1

                false_pairs = other_pairs[:len(true_pairs)]
                """



                true += true_pairs
                false += false_pairs


        true_df = pd.DataFrame(data=true, columns=['i', 'j'])
        true_df['label'] = 1

        false_df = pd.DataFrame(data=false, columns=['i', 'j'])
        #false_df['label'] = 0
        false_df['label'] = [is_same_user(row[0], row[1]) for row in false_df[['i', 'j']].values]


        pairs = true_df.append(false_df)
        pairs.to_csv(self.pairsfilepath, index=False)
        print (len(pairs), "pairs made")

        return pairs


    def train_test_folds(self, fold, pairs):
        """
        split pairs for cross val into train and test sets
        :param fold:
        :param true_df:
        :param false_df:
        :return:
        """

        true_df = pairs[pairs.label==1]
        false_df = pairs[pairs.label==0]

        te_tru = true_df[fold * int(0.2 * len(true_df)) : (fold+ 1) * int(0.2 * len(true_df))]
        tr_tru = true_df.drop(te_tru.index)

        te_fal = false_df[fold * int(0.2 * len(false_df)) : (fold+ 1) * int(0.2 * len(false_df))]
        tr_fal = false_df.drop(te_fal.index)

        tr_pairs = tr_tru.append(tr_fal)
        te_pairs = te_tru.append(te_fal)

        tr_pairs.to_csv(self.tr_pairsfilepath, index=False)
        te_pairs.to_csv(self.te_pairsfilepath, index=False)



    def prep_data(self, combi):
        """
        for the random forest baseline attack, prepare features and save in the filepaths tr_data_fp and te_data_fp
        :param combi:
        :return:
        """

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
        """
        for the unsupevised attack
        :param combi:
        :return:
        """

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
        """
        random forest bl attack
        :param clf:
        :return:
        """

        train_ = pd.read_csv(self.tr_data_fp, index_col=0)
        test_ = pd.read_csv(self.te_data_fp, index_col=0)

        X_train, y_train = train_.iloc[:, 3:].values, train_.label.values
        X_test, y_test = test_.iloc[:, 3:].values, test_.label.values

        clf.fit(X_train, y_train)
        pred_ = clf.predict_proba(X_test)
        auc = roc_auc_score(y_test, pred_[:,1])

        return auc


    def unsup_attack(self,):
        """
        for the unsupevised attack

        :return:
        """

        train_ = pd.read_csv(self.unsup_data_fp, index_col=0)

        train_.dropna(inplace=True)

        auc = roc_auc_score(train_.label.values, train_.iloc[:, 3:].values)

        return 1-auc

