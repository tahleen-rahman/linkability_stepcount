import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from munkres import Munkres
from attacks import Attack

class Matching(Attack):
    """
    N-N linkability
    """

    def picker(self, day):
        #super().picker()

        df=self.vecframe
        res = df[df.desc == day].iloc[:, 2:]

        return res


    def attack(self, df1, df2):
        super().attack()

        costs = cosine_distances(df1, df2)
        m = Munkres()
        indexes = m.compute(costs)

        return indexes, costs


    def evaluate(self, indexes, costs):
        #super().evaluate()

        total, mismatches=0, 0

        #print ('u1, u2, Dist')
        for row, column in indexes:
            value = costs[row][column]
            total += value
            if row!=column:
                mismatches+=1
                #print (row, column, value)

        print ('total dist of optimal assignment',  total)
        print ('number of mismatches', mismatches)
        return total, mismatches

