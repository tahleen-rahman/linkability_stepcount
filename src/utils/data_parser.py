import pandas as pd

from utils.storage import *




def csv_to_step_vector(filename, path=DATA_PATH):
    '''
    From a given file extracts stepcount vector, trims it to 40320 records and returns it
    '''
    df = pd.read_csv(path + filename)
    return df['StepCount'].head(40320)



def parse_all_csv_to_stepframe(path=DATA_PATH, desc_file='YangCispa1_plus_files.csv', subfolder='files/'):
    '''
    Parses DZNE provided epoch files and the `desc_file` into a stepframe and saves them as 'dzne' and 'dzne_desc'
    All data is trimmed to size 40320 and users without as many epochs get discarded
    :param path: path to data folder
    :param desc_file: csv file with descriptions (attributes) of users
    :param subfolder: subfolder in the data folder
    :return:
    '''
    desc_raw = pd.read_csv('{}{}'.format(path, desc_file))
    db = pd.DataFrame()
    fake_db = load_frame('../fake')
    for col in ['day', 'hour', 'minute', 'quarter']:
        db[col] = fake_db[col]
    desc_list = []
    #desc = pd.DataFrame(columns=['user', 'sex', 'age', 'edu'])
    for index, row in desc_raw.iterrows():
        vec = csv_to_step_vector(row['file']+'-VANE-CL08090336-15sEpochs.csv', path + subfolder)
        if vec.size == 40320:
            db[str(index)] = vec
            desc_dict = {'user': str(index), 'sex': row['BASE_SEX_R1'], 'age': row['BASE_AGE1_R1'], 'edu': row['QUEST_SES_EDU_R1']}
            desc_list.append(desc_dict)
        else:

            print('user {} dzne is corrupted')

    desc = pd.DataFrame(desc_list)
    dump_frame(db, 'dzne', path)
    dump_frame(desc, 'dzne_desc', path)


def normalize_vecframe_by_col(vecframe, in_path=DATA_PATH, out_path=DATA_PATH):
    def normalize_one_col(col):
        m = max(col)
        if m == 0:
            return [0.0 for i in col]
        return [i / m for i in col]
    vf = load_frame(vecframe, in_path)
    check_if_vecframe(vf)
    res = vf.apply(normalize_one_col)
    res['user'] = vf['user']
    res['desc'] = vf['desc']
    dump_frame(res, vecframe + '_nor', out_path, False)


def normalize_vecframe_by_row(vecframe, in_path=DATA_PATH, out_path=DATA_PATH):
    def normalize_one_row(row):
        m = max(row)
        if m == 0:
            return [m for _ in row]
        return [i / m for i in row]

    vf = load_frame(vecframe, in_path)
    check_if_vecframe(vf)
    res = pd.DataFrame(list(vf.iloc[:,2:].apply(normalize_one_row, axis=1)))
    res.columns = [str(col) for col in res.columns]
    res['user'] = vf['user']
    res['desc'] = vf['desc']
    cols=list(res.columns)
    res = res[cols[-2:]+cols[:-2]]
    dump_frame(res, vecframe + '_nou', out_path, False)


def probabilize_vecframe_by_row(vecframe, in_path=DATA_PATH, out_path=DATA_PATH):
    def normalize_one_row(row):
        m = sum(row)
        if m == 0:
            return [m for _ in row]
        return [i / m for i in row]

    vf = load_frame(vecframe, in_path)
    check_if_vecframe(vf)
    res = pd.DataFrame(list(vf.iloc[:, 2:].apply(normalize_one_row, axis=1)))
    res.columns = [str(col) for col in res.columns]
    res['user'] = vf['user']
    res['desc'] = vf['desc']
    cols = list(res.columns)
    res = res[cols[-2:] + cols[:-2]]
    dump_frame(res, vecframe + '_pro', out_path, False)
