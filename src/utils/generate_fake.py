import pandas as pd
import numpy as np

from utils.storage import dump_frame

def draw_random_steps(mean, std_dev=1.):
    """
    Draws a number from normal distribution,
     converts it to int and
     transforms all the negative values to zeros.

    :param mean: mean of the distribution
    :param std_dev: standard deviation of the distribution
    :return: a random number of steps - an integer
    """
    return max(round(np.random.normal(mean, std_dev)), 0)

def get_original_simple_df(path='../dzne/epochs.csv'):
    df_all = pd.read_csv(path)
    return df_all[['step_count', 'day', 'hour', 'minute']]

def perturb_epoch_df(epochs, path='../dzne/fake/', times=1000):
    """
    Given `epochs` add noise to step_count and return perturbed epochs.
    Do it `times` times to get that many fake entries
    :param epochs: dataframe with 'step_count_per', 'day', 'hour', 'minute'
    :param path: folder in which to save fake dzne
    :param times: number of resulting fake dzne entries
    :return:
    """
    for i in range(1000):
        if i%10 == 0:
            print('{} / {}'.format(i, 1000))
        epochs['step_count_per'] = epochs['step_count'].apply(draw_random_steps)
        epochs.to_csv('{}{}.csv'.format(path, i),
                      columns=['step_count_per', 'day', 'hour', 'minute'],
                      header=['step_count', 'day', 'hour', 'minute'],
                      index=False)


def merge_all_to_one_file():
    """
    Take all 1000 files and put them into a single csv and a single feather files.

    :return:
    """

    path = '../dzne/fake/{}.csv'
    db = pd.read_csv(path.format(0))
    # db = db.rename(columns={'step_count': 0})
    db['quarter'] = [0, 1, 2, 3] * (len(db) // 4)
    db['0'] = db['step_count']
    del db['step_count']
    for i in range(1, 1000):
        df = pd.read_csv(path.format(i))
        db[str(i)] = df.step_count

    dump_frame(db, 'fake')

def run_all():
    # you need to create ../dzne/fake/ folder
    epochs = get_original_simple_df()
    perturb_epoch_df(epochs)
    merge_all_to_one_file()
    # you probably want to remove ../dzne/fake/ folder
