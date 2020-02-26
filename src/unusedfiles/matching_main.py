import os
from attacks import Matching
from utils.storage import DATA_PATH



def match(vf_fname):

    matcher = Matching(vf_fname)
    #print ("day1, day2, mismatches, optimal assignment dist")

    arr = []
    for day1 in range(7):

        df1 = matcher.picker(day1)

        for day2 in range(7):

            if day1!=day2:

                df2 = matcher.picker(day2)

                indexes, costs = matcher.attack(df1, df2)
                total, mismatches =  matcher.evaluate(indexes, costs)
                arr.append([day1, day2, mismatches, total])

                print (arr)

    import pandas as pd

    df=pd.DataFrame(data=arr, columns=['day1', 'day2', 'mismatches', 'dist'])

    df.to_csv(DATA_PATH + "results/" + vf_fname + "_Match.csv", index=False)





in_datapath = '../data/dzne/normalized/'

for infile in os.listdir(in_datapath):

    if 'nor' in infile and 'dsp' in infile:

        match(infile, 3)



"""
TODO: 
1. sample few users

2. splits:

    (i) random prior: the adversary scraping information on target user u randomly
from various social media sources.
    (ii) chrono prior: adversary has historical data on the targeted user, such as
from a previously de-identified account


3. supervised attacks:
    1. reid: per user 4 known weekday samples, 1 target weekday sample
        1. baseline random guess
        2. unsupervised cosine dist reid for the same target day, prior avged over others
        3. multi-class classifier to classify users.
        4. K-NN 
        5. MLP
        6. 
        
    2. pair-matching: 
        1. baseline random guess
        2. unsup cosine dist 
    
"""
