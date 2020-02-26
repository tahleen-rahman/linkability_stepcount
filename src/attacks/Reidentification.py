# Created by rahman at 17:18 2019-05-10 using PyCharm


import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_distances

from attacks import Attack

class Reidentification(Attack):
    """
    1-N linkability
    """


    def picker(self, targetuser, targetday, groupday, VERBOSE=True):
        #super().picker()

        df=self.vecframe

        if groupday == targetday and VERBOSE:
            print('Target sample is from the same day as the given/known samples, Should be an exact match')

        dftarget = df[(df.desc == targetday) & (df.user== targetuser)].iloc[:, 2:]
        dfknown = df[df.desc == groupday].iloc[:, 2:]
        assert (len(dftarget)==1)

        return dftarget, dfknown

    # def reidentify(df, targetuser, targetday, groupday, VERBOSE=True):

    # print(dfknown.shape, dftarget.shape)


    def attack(self, dftarget, dfknown):

        super().attack()

        distances = dfknown.apply(lambda row: cosine(row, dftarget), axis=1)
        top3 = distances.reset_index(drop=True).sort_values().index.values[:3]

        return top3


    def reidentify_user(self, targetuser, VERBOSE=True):

        df=self.vecframe

        arr = []
        for targetday in df.desc.unique():  # 0-6

            topsuccess, top3success = [], []

            dftarget = df[(df.desc == targetday) & (df.user == targetuser)].iloc[:, 2:]
            assert (len(dftarget) == 1)

            for groupday in df.desc.unique():  # 0-6

                if groupday!=targetday:

                    #dftarget, dfknown = self.picker(targetuser, targetday, groupday, VERBOSE=False)

                    dfknown = df[df.desc == groupday].iloc[:, 2:]

                    top3 = self.attack(dftarget, dfknown)

                    topsuccess.append(top3[0] == targetuser)
                    top3success.append(targetuser in top3)

            arr.append([targetuser, targetday, sum(topsuccess), sum(top3success)])#, topsuccess, top3success])

        userarr = pd.DataFrame(data=arr, columns=['targetuser', 'targetday', 'success_count', 'top3success_count']) #, 'topsuccess', 'top3success'])

        tot_topsuccess, tot_top3success = userarr.success_count.sum(), userarr.top3success_count.sum()

        if VERBOSE:
            #print(arr)

            print("user", targetuser, ', top matches:', tot_topsuccess, ', top3 matches:', tot_top3success)

        return [targetuser, tot_topsuccess, tot_top3success]


    def reidentify_all(self):

        df=self.vecframe

        users = df.user.unique()

        arr = []
        for targetuser in users:

            arr.append(self.reidentify_user(targetuser, VERBOSE=False))

            #if targetuser % 10 ==0:
                #print (arr[-10:])

            if targetuser % 100 ==0:

                df_res = pd.DataFrame(data=arr, columns=['targetuser', 'success_count', 'top3success_count'])

                print(targetuser, 'users done,   matches: top', df_res.success_count.sum(), ', top3', df_res.top3success_count.sum() ,', out of', (targetuser+1)*(42) ) # 49 - 7


        df_res = pd.DataFrame(data=arr, columns=['targetuser', 'success_count', 'top3success_count'])

        print('total matches: top', df_res.success_count.sum(), ', top3', df_res.top3success_count.sum())

        return df_res


    #def evaluate(self):
        #super().evaluate()
