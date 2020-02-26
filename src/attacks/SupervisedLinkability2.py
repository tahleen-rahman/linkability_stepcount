import os
from attacks import Attack
import pandas as pd


class Link(Attack):


    def __init__(self, vf_fname,  in_datapath = '../data/dzne/',  out_datapath = '../data/dzne/'):
        """
        creates or loads self.tr_pairs and  self.te_pairs
        :param vf_fname:
        :param in_datapath:
        :param out_datapath:

        """

        super().__init__(vf_fname, in_datapath)

        assert('dsp' in self.vf_fname)
        assert('week' not in self.vf_fname)

        self.out_datapath =  out_datapath
        self.pairsfilepath = self.out_datapath + "pairs.csv"
        self.tr_pairsfilepath = self.out_datapath + "tr_pairs.csv"
        self.te_pairsfilepath = self.out_datapath + "te_pairs.csv"


        if not (os.path.exists(self.tr_pairsfilepath) and os.path.exists(self.te_pairsfilepath)):
            print ('making', self.tr_pairsfilepath , 'and', self.te_pairsfilepath)
            self.makePairs()

        self.tr_pairs = pd.read_csv(self.tr_pairsfilepath)
        self.te_pairs = pd.read_csv(self.te_pairsfilepath)



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

        for user in users:

            user_ind = df[df.user == user].index
            true_pairs_ = list(itertools.combinations(user_ind, 2))
            true_pairs = [list(x) for x in true_pairs_]

            false_ind = df[df.user != user].index
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


    def attack(self, clf):

        clf.fit(self)

        clf.predict(self)

