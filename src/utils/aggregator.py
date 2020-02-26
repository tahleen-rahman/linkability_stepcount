import pandas as pd

from utils.storage import load_frame, dump_frame, DATA_PATH, check_if_stepframe, check_if_vecframe



def processTime(epochsfile, step_name , data_path=DATA_PATH):
    epochs=pd.read_csv(data_path+ epochsfile)
    epochs = epochs[['step_count']]
    epochs['day']=epochs.index.to_series().apply(lambda x: x/(4*60*24))
    epochs['hour']=epochs.index.to_series().apply(lambda x: (x/(4*60))  % 24 )
    epochs['minute']=epochs.index.to_series().apply(lambda x: (x/4)% 60 )
    dump_frame(epochs, '{}_epochs'.format(step_name))
    #return epochs



def aggregateTime(step_name, emb_name, by, units=1, data_path=DATA_PATH):

    df=load_frame(step_name, data_path)

    print ("WARNING! This func supports EQUAL BUCKETS ONLY")

    if by=='day':
        groupeddf = df.groupby(['day']).apply(lambda x: x.iloc[:,4:].sum()).reset_index()

    if by=='hour':
        groupeddf = df.groupby(['day', 'hour']).apply(lambda x: x.iloc[:,4:].sum()).reset_index()


    elif by=='minute':
        groupeddf = df.groupby(['day', 'hour', 'minute']).apply(lambda x: x.iloc[:,4:].sum()).reset_index()

    if units >1:

        ret_df = groupeddf.groupby(groupeddf.index // units).sum()

        ret_df['day'] = ret_df.day.apply(lambda x: int(x / units))

        if 'hour' in ret_df.columns:
            ret_df['hour'] = ret_df.hour.apply(lambda x: int(x / units))

        if 'minute' in ret_df.columns:
            ret_df['minute'] = ret_df.minute.apply(lambda x: int(x / units))

    else:
        ret_df= groupeddf

    dump_frame(ret_df, name=emb_name)


def get_stats(group):
    return {'min': group.min(), 'max': group.max(), 'median': group.median(), 'mean': group.mean(), 'std': group.std()}

def statistics(step_name, emb_name, by, units=1, data_path=DATA_PATH):

    df=load_frame(step_name, data_path)

    print ("WARNING! This func supports EQUAL BUCKETS ONLY")

    if by=='hour':
        groupeddf = df.groupby(['day', 'hour']).apply(get_stats).reset_index()


    elif by=='minute':
        groupeddf = df.groupby(['day', 'hour', 'minute']).apply(get_stats).reset_index()

    if units >1:

        ret_df = groupeddf.groupby(groupeddf.index // units).sum()

        ret_df['day'] = ret_df.day.apply(lambda x: int(x / units))

        if 'hour' in ret_df.columns:
            ret_df['hour'] = ret_df.hour.apply(lambda x: int(x / units))

        if 'minute' in ret_df.columns:
            ret_df['minute'] = ret_df.minute.apply(lambda x: int(x / units))

    else:
        ret_df= groupeddf

    dump_frame(ret_df, name=emb_name)




def daySplitter(step_name, data_path=DATA_PATH):
    """
    Splits entries into days and saves results as vecframe.
    """

    stepframe = load_frame(step_name, data_path)
    check_if_stepframe(stepframe)
    vec_len = stepframe.loc[stepframe.day == 0].shape[0]
    columns = ['user', 'desc'] + list(range(vec_len))
    vfs = []
    for day in stepframe.day.unique():
        vf = stepframe[stepframe.day == day].iloc[:, 4:999+4].T.astype('int32')
        vf.columns = list(range(vec_len))
        vf['user'] = vf.index.to_numpy(dtype=pd.np.int)
        vf['desc'] = day
        vfs.append(vf)
    vecframe = pd.concat(vfs, sort=False, ignore_index=True)
    vecframe = vecframe[columns]
    vecframe.columns = vecframe.columns.astype(str)
    check_if_vecframe(vecframe)
    dump_frame(vecframe, '{}_dsp'.format(step_name))


def make_weekly_vecframe(step_name, vec_name='{}_week', data_path=DATA_PATH):
    '''
    Transforms a stepframe into a vecframe without splittling the data.
    'desc' will always be 0.

    :param step_name: name of the stepframe
    :param vec_name: name under which vecframe will be saved
    :param data_path: optional, path to data folder
    :return:
    '''
    stepframe = load_frame(step_name, DATA_PATH)
    vecframe = stepframe.loc[:, '0':].transpose()
    vecframe.columns = [str(col) for col in vecframe.columns]
    vecframe['user'] = vecframe.index
    vecframe['user'] = vecframe['user'].apply(int)
    vecframe['desc'] = [0] * vecframe.shape[0]
    cols = list(vecframe.columns)
    vecframe = vecframe[cols[-2:] + cols[:-2]]
    # vecframe = vecframe.reset_index(drop=True)
    check_if_vecframe(vecframe)
    dump_frame(vecframe, vec_name.format(step_name), data_path)











