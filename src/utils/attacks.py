import pandas as pd
from munkres import Munkres
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine


def match(df, day1=0, day2=1):

    df1 = df[df.day == day1 ].iloc[:, 4:].T
    df2 = df[df.day == day2].iloc[:, 4:].T
    costs = cosine_distances(df1, df2)
    m = Munkres()
    indexes = m.compute(costs)
    total, mismatches=0, 0

    print ('u1, u2, Dist')
    for row, column in indexes:
        value = costs[row][column]
        total += value
        if row!=column:
            mismatches+=1
            print (row, column, value)

    print ('total dist of optimal assignment',  total)
    print ('number of mismatches', mismatches)
    return total, mismatches



def reidentify(df, targetuser, targetday, groupday, VERBOSE=True):

    if groupday==targetday and VERBOSE:
        print('Target sample is from the same day as the given/known samples, Should be an exact match')

    dftarget= df[df.day == targetday].loc[:,str(targetuser)]
    dfknown=  df[df.day == groupday].iloc[:, 4:].T
    print(dfknown.shape, dftarget.shape)
    distances=dfknown.apply(lambda row: cosine(row, dftarget),axis= 1)
    top3= (distances.sort_values().index.values[:3])

    if VERBOSE:
        print ("top 3 matches among a group in day:", groupday," to user:", targetuser, "from targetday:", targetday,  "are:", top3)

    return top3


def reidentify_user(df, targetuser, VERBOSE=True):

    arr = []
    for targetday in range(df.day.unique()):  # 0-6

        topsuccess, top3success = [], []
        for groupday in range(df.day.unique()):  # 0-6

            top3 = reidentify(df, targetuser, targetday, groupday, VERBOSE=False)

            topsuccess.append(top3[0] == targetuser)
            top3success.append(targetuser.isin(top3))

        arr.append([targetuser, targetday, topsuccess.sum(), top3success.sum(), topsuccess, top3success])

    userarr = pd.DataFrame(data=arr, columns=['targetuser','targetday', 'success_count', 'top3success_count','topsuccess, top3success'])

    tot_topsuccess, tot_top3success = userarr.success_count.sum()

    if VERBOSE:
        print(arr)
        print("for user", targetuser, 'total top matches:', tot_topsuccess, 'top 3 matches:', tot_top3success, 'out of', len(df.day.unique()) ** 2)
        print ("Exact matches to be substracted:", len(df.day.unique()))

    return [targetuser, tot_topsuccess, tot_top3success]


def reidentify_all(df):

    users=df.columns[4:].values

    arr = []
    for targetuser in users:

        arr.append(reidentify_user(df, targetuser, VERBOSE=False))

    df = pd.DataFrame(data=arr, columns=['targetuser', 'success_count', 'top3success_count'])

    print('total top matches:', df.success_count.sum(), 'top 3 matches:',df.top3success_count.sum(), 'out of', len(users)* (len(df.day.unique()) ** 2))
    print("Exact matches to be substracted:",  len(users)* len(df.day.unique()))



def cosine_link(df, entry1, entry2, threshold=0.01):
    """
    Temporal linkability attack.
    Checks whether cosine similarity of two entries is below `threshold` and if so return a match.
    :param df: StepFrame of all the people's steps
    :param entry1: first entry to compare
    :param entry2: second entry to compare
    :param threshold: threshold below each two users are considered to be the same
    :return: bool, whether two entries come from the same user
    """
    vec1 = list(df[str(entry1)])
    vec2 = list(df[str(entry2)])
    return cosine_distances([vec1], [vec2])[0][0] < threshold
