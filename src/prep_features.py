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



def normalize(inpath, norm_path):

    for infile in os.listdir(inpath):

        normalize_vecframe_by_col(infile, in_path=inpath, out_path=norm_path)


def variance_thresholding(DATA_PATH, in_dir, th=0.0, outpath=None):

    sel = VarianceThreshold(th)

    in_path = DATA_PATH + in_dir

    for infile in os.listdir(in_path):

        if 'nor' in infile: #for normalized files only

            if not os.path.exists(in_path + infile[:-4] + "_vt" + str(th)):

                vf = load_frame(infile, in_path)
                check_if_vecframe(vf)

                # user and desc columns always have high variance so no need to remove here
                sel.fit(vf)

                vt = vf[vf.columns[sel.get_support(indices=True)]]

                if outpath == None:
                    outpath = in_path

                dump_frame(vt, infile[:-4] + "_vt" + str(th), outpath, False)
                print (infile, vf.shape, vt.shape)





def prep_all(DATA_PATH):

    stats_datapath = DATA_PATH +  "statistics/"

    make_stats(stats_datapath, combine_stats=False)

    norm_dir =  "normalized/"

    normalize(stats_datapath, DATA_PATH + norm_dir)

    #outpath = DATA_PATH + "var_th_norm/"

    #variance_thresholding(norm_path, outpath, th=0.01)
    variance_thresholding(norm_dir,  th=0.01)

    #return outpath


if __name__ == '__main__':

    prep_all(DATA_PATH='../data/dzne/')
