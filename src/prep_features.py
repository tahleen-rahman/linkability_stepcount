import os

from sklearn.feature_selection import VarianceThreshold

from compress import Statistics
from itertools import combinations, chain
from utils.data_parser import normalize_vecframe_by_col, load_frame, check_if_vecframe, dump_frame



def make_stats(stats_datapath, combine_stats):

    if combine_stats:

        def powerset(iterable):

            s = list(iterable)
            return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

        stats_arr = ['sum', 'max', 'std', 'medi', 'mean']
        stats = list(powerset(stats_arr))

    else:
        stats =  (['max'], ['sum'], ['medi'], ['mean'])


    for minu in [1, 5, 10, 15, 30, 60, 120, 180, 360]:  #  from 60mins as hrs: 1, 2, 3, 6, 12, 24

        for st in stats:

            print(st)

            ds = Statistics('dzne_dsp', st, window_size=int(minu * 4), bucket_size=4, data_path=stats_datapath)

            if not os.path.exists(stats_datapath + ds.out_name + ".ftr"):

                ds.compress_save()


def filter_mornings(in_path, f):

    out_path = in_path + "filt" + str(f) + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for infile in os.listdir(in_path):

        if 'nor.ftr' in infile: #for normalized files only

            if not os.path.exists(in_path + "filt" + str(f) + "/" + infile[:-4] + "filt" + str(f) + ".ftr"):

                df = load_frame(infile, in_path)

                df_filt = df.iloc[:, 2:]
                df_filt = df_filt.iloc[:, int(len(df_filt.columns) * f):]

                df_filt['user'] = df.user
                df_filt['desc'] = df.desc

                cols = list(df_filt.columns)
                df_filt = df_filt[cols[-2:] + cols[:-2]]

                try:
                    check_if_vecframe(df_filt)
                    dump_frame(df_filt, infile[:-4] + '_filt' + str(f), out_path, in_csv=False)
                except:
                    "skipping file", infile

    return out_path



def normalize(inpath, norm_path):

    for infile in os.listdir(inpath):

        normalize_vecframe_by_col(infile, in_path=inpath, out_path=norm_path)



def variance_thresholding(in_path, th=0.0):

    out_path = in_path + str(th) + "vt/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    sel = VarianceThreshold(th)


    for infile in os.listdir(in_path):

        if 'nor' in infile: #for normalized files only

            if not os.path.exists(out_path  +  infile[:-4] + "_vt" + str(th) +".ftr"):

                vf = load_frame(infile, in_path)
                check_if_vecframe(vf)

                # user and desc columns always have high variance so no need to remove here
                sel.fit(vf)

                vt = vf[vf.columns[sel.get_support(indices=True)]]

                try:
                    check_if_vecframe(vf)
                    dump_frame(vt, infile[:-4] + "_vt" + str(th), out_path, False)
                    print(infile, vf.shape, vt.shape)
                except:
                    "skipping file", infile


    return out_path


def prep_all(DATA_PATH):
    """
    :param DATA_PATH: path where unprocessed data is located
    :return: the path where the features ready for attack are located
    """

    stats_path = DATA_PATH + "statistics/"

    make_stats(stats_path, combine_stats=False)

    path = DATA_PATH + "normalized/"

    normalize(stats_path,  path)

    #path = filter_mornings(path, f=0.25)

    path = variance_thresholding(path, th=0.001)

    return path


if __name__ == '__main__':

    print (prep_all(DATA_PATH='../data/dzne/'))
