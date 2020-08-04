# Created by rahman at 14:34 2020-04-06 using PyCharm

# Created by rahman at 19:14 2020-01-26 using PyCharm

import math
import os
import sys

import pandas as pd
from joblib import Parallel, delayed

from attacks.kerasclassifier import *
from attacks.Linkability_crossval import Link


def add_padding(vf, padding):
    """
    takes a dataframe and the value whose multiple the padding needs to achieve
    :param vf: input vf
    :param padding: the padding process will add zeros to the front and back of vf so that it is a multiple of this value
    :return: padded vf
    """

    data = vf.iloc[:,2:]
    vec_len = data.shape[1]

    if padding >= vec_len:
        pad_len = padding - vec_len
    else:
        pad_len = vec_len - math.floor(vec_len / padding) * padding
    # self.vec_len += pad_len
    before = pad_len // 2
    after = pad_len - before
    data_pad= pd.np.pad(data, [(0, 0), (before, after)], 'constant')

    df = pd.DataFrame(data= data_pad)

    df.insert(0, 'user', vf['user'])
    df.insert(1, 'desc', vf['desc'])

    return df


def core_siamese(infile,params, cl, datapath, in_datapath,callback,aucfname, weekend, max_epochs, patience, regu, batchsize, combi):
    """
    core of the siamese model creation for the linkability attacks
    :param infile:
    :param params:
    :param cl:
    :param datapath:
    :param in_datapath:
    :param callback:
    :param aucfname:
    :param weekend:
    :param max_epochs:
    :param patience:
    :param regu:
    :param batchsize:
    :param combi:
    :return:
    """

    if 'vt' in infile and 'nor' in infile:  # only use variance thresholded and normalized files

        print(infile)

        arr = []

        for i in range(0, 5):

            # try:

            link = Link(i, infile, weekend, in_datapath, out_datapath=datapath + 'cv_folds/')

            from sklearn.utils import shuffle
            link.tr_pairs = shuffle(link.tr_pairs)

            # first define the shared layers
            if 'cnn' in cl:

                link.vecframe = add_padding(link.vecframe, padding=params[2] ** params[3])

                clf = CNNsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, params)

            elif 'lstm' in cl:

                if 'bilstm' in cl:

                    clf = BiLSTMsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, lstm_params=params, fixed_units=True)

                else:

                    clf = LSTMsiameseClassifier(link.vecframe.shape[1] - 2, regu, combi, lstm_params=params, fixed_units=True)


            elif cl == 'dense':

                clf = Dense_siameseClassifier(link.vecframe.shape[1] - 2, regu, combi, params)

            # Next combine the layers
            clf.combine(plot=False)

            if callback:
                auc = clf.fit_predict_callback(link, batchsize, max_epochs, patience, verbose=2)
            else:
                auc = clf.fit_predict(link, batchsize, max_epochs, verbose=2)

            print(infile, i, auc)

            # aucarr.append(auc)

            arr.append([patience, regu, batchsize, i, infile, auc])

            del clf

        # except:

        # print("infile skipped", infile)

        aucs = pd.DataFrame(data=arr)  # , names= epochs, regu,  batchsize, i, infile, auc

        aucs.to_csv(datapath + "results/" + aucfname, mode='a', header=False, index=False)

        print("saved AUCs to " + datapath + "results/" + aucfname)






def linkability_siam(config, in_datapath, params, exp, cl, weekend, datapath, callback=True, parallelize=False):
    """

    :param config: epochs, regu, batchsize, combi
    :param in_datapath:
    :param params: model layer params
    :param exp: reqd for results filename
    :param cl: reqd for results filename
    :param datapath:
    :param callback: use modelcheckpoint and early stopping
    :return:
    """

    # unpack config
    max_epochs, patience, regu, batchsize, combi = config

    if not weekend:
        aucfname = "noweekend_" + "clf_" + str(cl) + "_exp_" + str(exp) + "_cv_siam.csv"
    else:
        aucfname = "weekend_" + "clf_" + str(cl) + "_exp_" + str(exp) + "_cv_siam.csv"

    if parallelize:

        threads = len(os.listdir(in_datapath))
        Parallel(n_jobs=threads)(delayed(core_siamese)(infile,params, cl, datapath, in_datapath,callback,aucfname, weekend, max_epochs, patience, regu, batchsize, combi)
                                 for infile in os.listdir(in_datapath))


    else:

        for infile in os.listdir(in_datapath):

            core_siamese(infile,params, cl, datapath, in_datapath,callback,aucfname, weekend, max_epochs, patience, regu, batchsize, combi)



def linkability_bl(in_path, datapath, cl, clf, exp, weekend):
    """
    baseline attacks
    :param in_path:
    :param cl:
    :param clf:
    :param weekend:
    :return:
    """

    if not weekend:
        aucfname = "noweekend_" + "clf_" + str(cl) + "_exp_" + str(exp) + "_cv_BL.csv"
    else:
        aucfname = "weekend_" + "clf_" + str(cl) + "_exp_" + str(exp) + "_cv_BL.csv"


    for infile in os.listdir(in_path):  # all  intervals for single stats + distributions 1, 6, 12 hrs

        if 'vt' in infile and 'nor' in infile: #only use variance thresholded and normalized files

            print (infile)

            arr=[]

            for i in range(0, 5):

                link = Link(i, infile, weekend, in_path , out_datapath = datapath + 'cv_folds/')


                for combi in ['l1']:  # 'sql2', 'mul',

                    link.tr_data_fp = link.out_datapath + infile[:-4] + combi + "_" + str(weekend) + 'weekend_tr_data.csv'

                    link.te_data_fp = link.out_datapath + infile[:-4] + combi + "_" + str(weekend) + 'weekend_te_data.csv'

                    if not (os.path.exists(link.tr_data_fp) and os.path.exists(link.te_data_fp)):
                        link.prep_data(combi)

                    auc = link.attack(clf)
                    print (auc)

                    arr.append([i, infile, auc])
                #print("infile skipped", infile)

            aucs = pd.DataFrame(data=arr)  # , names = i, infile, auc

            aucs.to_csv(datapath + "results/" + aucfname, mode='a', header=False, index=False)

            print("saved AUCs to " + datapath + "results/" + aucfname)


def linkability_unsup(in_path, datapath, metric, exp, weekend):
    """

    :param in_path:
    :param datapath:
    :param metric:
    :param exp:
    :param weekend:
    :return:
    """

    if not weekend:
        aucfname = "noweekend_" + metric + "_cv_BL.csv"
    else:
        aucfname = "weekend_" + metric + "_cv_BL.csv"


    for infile in os.listdir(in_path):  # all  intervals for single stats + distributions 1, 6, 12 hrs

        if 'vt' in infile and 'nor' in infile: #only use variance thresholded and normalized files

            print (infile)

            arr=[]

            for i in range(0, 1): # no cross val for unsupervised,

                try:

                    link = Link(i, infile, weekend, in_path , out_datapath = datapath + 'cv_folds/', unsup=True)

                    link.unsup_data_fp = link.out_datapath + infile[:-4] + metric + str(weekend)  + 'weekend_unsup_data.csv'

                    if not (os.path.exists(link.unsup_data_fp)):
                        link.prep_data_unsup(metric)

                    auc = link.unsup_attack()

                    print(auc)

                    arr.append([infile, auc])

                except Exception as e:

                    print (e)
                    print (infile, "skipped")
                    continue

            aucs = pd.DataFrame(data=arr)  # , names = infile, auc

            aucs.to_csv(datapath + "results/" + aucfname, mode='a', header=False, index=False)

            print("saved AUCs to " + datapath + "results/" + aucfname)

def convert(val):
    return 1-val if val<0.5 else val


def merge_results():

    cnn1 = pd.read_csv('../data/dzne/results/link_cnn1.csv')#, names = ['fold', 'infile', 'auc'])
    lstm = pd.read_csv('../data/dzne/results/link_lstm.csv')#, names = ['fold', 'infile', 'auc'])

    rf = pd.read_csv('../data/dzne/results/link_rf.csv')#, names = ['fold', 'infile', 'auc'])
    dense = pd.read_csv('../data/dzne/results/siam_dense.csv')#, names = ['epochs', 'regu',  'batchsize', 'fold', 'infile', 'auc'])
    cos = pd.read_csv('../data/dzne/results/link_cos.csv')#,names = ['infile', 'cos_auc'])
    eucl = pd.read_csv('../data/dzne/results/link_eucl.csv')#, names = ['infile', 'eucl_auc'])

    ## ensure all aucs are >0.5
    cos['cos_auc'] = cos.cos_auc.apply(convert)
    dense['auc'] = dense.auc.apply(convert)
    cnn1['auc'] = cnn1.auc.apply(convert)
    lstm['auc'] = lstm.auc.apply(convert)
    rf['auc'] = rf.auc.apply(convert)
    eucl['eucl_auc'] = eucl.eucl_auc.apply(convert)

    rf_res = rf.groupby('infile').auc.agg([pd.np.mean, pd.np.std]).rename(columns={'mean': 'rf_mean', 'std': 'rf_std'})
    dense_res = dense.groupby('infile').auc.agg([pd.np.mean, pd.np.std]).rename(columns={'mean': 'dense_mean', 'std': 'dense_std'})
    cnn1_res = cnn1.groupby('infile').auc.agg([pd.np.mean, pd.np.std]).rename(columns={'mean': 'cnn1_mean', 'std': 'cnn1_std'})
    lstm_res = lstm.groupby('infile').auc.agg([pd.np.mean, pd.np.std]).rename(columns={'mean': 'lstm_mean', 'std': 'lstm_std'})


    dense_rf = dense_res.merge(rf_res, on='infile', how='outer')
    dense_rf_cos = dense_rf.merge(cos, on='infile', how='outer')
    dense_rf_cos_eucl = dense_rf_cos.merge(eucl, on='infile', how='outer')
    dense_rf_cos_eucl_cnn1 = dense_rf_cos_eucl.merge(cnn1_res, on='infile', how='outer')
    dense_rf_cos_eucl_cnn1_lstm = dense_rf_cos_eucl_cnn1.merge(lstm_res, on='infile', how='outer')


    merged_fp = '../data/plotdata/link_merged.csv'
    dense_rf_cos_eucl_cnn1_lstm.to_csv(merged_fp)

    return merged_fp
