import json
import pandas as pd
import numpy as np
from keras.models import model_from_json

DATA_PATH = '../data/dzne/'


def saveKerasmodel(model,  archFile="model.json", weightsFile="model.h5",  data_path=DATA_PATH):
    # serialize model to JSON
    model_json = model.to_json()
    with open(data_path + "models/" + archFile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(data_path + "models/" +  weightsFile)
    print("Saved model"+ data_path  + "models/" +  weightsFile +" to disk, deleting now")


def loadKerasmodel(archFile="model.json", weightsFile="model.h5", data_path=DATA_PATH):
    # load json and create model
    json_file = open(data_path   + "models/" +  archFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(data_path + "models/" +  weightsFile)
    print("Loaded model from", data_path  + "models/" +  weightsFile)

    return loaded_model


def dump_frame(frame, name, data_path=DATA_PATH, in_csv=False):
    """
    Saves dataframe on the disk in both a csv and a feather file.

    :param frame: pandas dataframe with step_counts for all users
    :param name: filename without extentions, e.g., "fake"
    :param data_path: path to dzne file, probably should always be '../dzne/'
    :param in_csv: whether we should save a csv as well
    :return:
    """
    frame = frame.reset_index(drop=True)
    frame.to_feather('{}{}.ftr'.format(data_path, name))
    if in_csv:
        frame.to_csv('{}{}.csv'.format(data_path, name), index=False)

def load_frame(name, data_path=DATA_PATH):
    """
    Reads dataframe from the disk from feather format.

    :param name: filename without extentions, e.g., "fake"
    :param data_path:  path to dzne file, probably should always be '../dzne/'
    :return: pandas dataframe of stepcounts for all users
    """

    if "ftr" not in name:

        frame = pd.read_feather('{}{}.ftr'.format(data_path, name))
        if 'user' in frame.columns and 'desc' in frame.columns:
            frame['user'] = frame['user'].apply(np.int64)
            frame['desc'] = frame['desc'].apply(np.int64)
    else:

        frame = pd.read_feather('{}{}'.format(data_path, name))
        if 'user' in frame.columns and 'desc' in frame.columns:
            frame['user'] = frame['user'].apply(np.int64)
            frame['desc'] = frame['desc'].apply(np.int64)

    return frame




"""
def second_to_hour(old_name, new_name, time_slot, data_path=DATA_PATH)
    
    Change dataframe to any time period
    :param old_name: filename without extentions, e.g., "dzne_data"
    :param new_name: filename without extentions, e.g., "hourly_dzne_data"
    :param time_slot: hourly=1, half_hourly=0.5, quarter_hourly=0.25
    :param data_path: path to dzne file probably should always be '../dzne/'
    :return: pandas dataframe of stepcounts for all users
    
    df = pd.read_feather('{}{}.ftr'.format(data_path, old_name))
    df_hourly = pd.DataFrame()
    for i in range(time_slot * 24):
        new_col = df.iloc[:, time_slot * 240 * i + 2:time_slot * 240 * (i + 1) + 2].sum(axis=1)
        df_hourly.insert(loc=i, column=str(i), value=new_col)
    df_hourly.insert(loc=0, column='user', value=df['user'])
    df_hourly.insert(loc=1, column='desc', value=df['desc'])

    frame.to_feather('{}{}.ftr'.format(data_path, new_name))
    if in_csv:
        frame.to_csv('{}{}.csv'.format(data_path, new_name), index=False)
"""

def check_if_stepframe(stepframe):
    """
    Checks format of stepframe.
    Can be overloaded for additional checks.

    :param stepframe:
    :return:
    """
    assert isinstance(stepframe, pd.DataFrame)
    assert stepframe.shape[1] > 4
    assert stepframe.shape[0] > 1
    assert 'day' in stepframe.columns
    assert 'hour' in stepframe.columns
    #assert 'minute' in stepframe.columns
    #assert 'quarter' in stepframe.columns

def check_if_vecframe(vecframe):
    """
    Checks format of vecframe.
    Can be overloaded for additional checks.

    :param vecframe:
    :return:
    """
    assert isinstance(vecframe, pd.DataFrame)
    assert vecframe.shape[1] > 2
    assert vecframe.shape[0] > 1
    assert 'user' in vecframe.columns[:2]
    assert 'desc' in vecframe.columns[:2]
    assert vecframe.user.dtype == 'int64'
    assert vecframe.desc.dtype == 'int64'


def json_dump_stepseries(stepseries, user_id, intervals=15, dataset='fake', data_path=DATA_PATH):
    """
    @Deprecated
    Save stepseries on disk.

    :param stepseries: list or array of stepcounts in equal intervals
    :param user_id: id of a user from which the dzne was collected
    :param intervals: length of each epoch in seconds
    :param dataset: dataset name, like 'fake', 'test'
    :param data_path: path to dzne file, probably should always be '../dzne/'
    :return:
    """
    path = '{}{}/{}_{}.json'.format(data_path, dataset, user_id, intervals)
    open(path, 'w').write(json.dumps(stepseries))

def json_load_stepseries(user_id, intervals=15, dataset='fake', data_path=DATA_PATH):
    """
    @Deprecated
    Load stepseries from the disk.

    :param user_id: id of a user from which the dzne was collected
    :param intervals: length of each epoch
    :param dataset: dataset name, like 'fake', 'test'
    :param data_path: path to dzne file, probably should always be '../dzne/'
    :return: stepseries as list
    """
    path = '{}{}/{}_{}.json'.format(data_path, dataset, user_id, intervals)
    return json.loads(open(path, 'r').read())

