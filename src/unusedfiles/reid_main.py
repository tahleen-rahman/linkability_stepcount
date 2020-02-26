import os
from attacks import Reidentification
from utils.storage import DATA_PATH


def reidentify(vf_fname, variant=3, targetuser=0, targetday=0, groupday=1):

    reidentification = Reidentification(vf_fname)

    if variant == 1:
        # variant 1  compare targetuser from targetday with database from groupday

        dftarget, dfknown = reidentification.picker(targetuser, targetday, groupday)

        top3 = reidentification.attack(dftarget, dfknown)

        print("top 3 matches on day:", groupday, " to user:", targetuser, "from targetday:",
              targetday, "are:", top3)

        # assert (top3[0] == targetuser)

    elif variant == 2:
        # variant 2  compare compare targetuser from all days with database from all days except targetday

        arr = reidentification.reidentify_user(targetuser)

    if variant == 3:
        # variant 3  compare all users with all databases

        df = reidentification.reidentify_all()

        df.to_csv(DATA_PATH + "results/" + vf_fname + "_reId.csv", index=False)





in_datapath = '../data/dzne/normalized/'

for infile in os.listdir(in_datapath):

    if 'nor' in infile and 'dsp' in infile:

        reidentify(infile, 3)


"""
TODO: 
1. sample few users
"""
