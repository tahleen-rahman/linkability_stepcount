import pandas as pd
import re

import matplotlib.pyplot as plt

from utils.storage import *
from utils.plot_utils import *

def average_gender_steps(frame='hourly_dzne_dsp'):
    hourly = load_frame(frame)
    desc = load_frame('dzne_desc')
    hourly['sex'] = list(desc.sex) * (7 if hourly.shape[0] == 6979 else 1)
    m_average = list(hourly.loc[hourly['sex'] == 'm'].mean())[2:]
    m_std = list(hourly.loc[hourly['sex'] == 'm'].std())[2:]
    f_average = list(hourly.loc[hourly['sex'] == 'f'].mean())[2:]
    f_std = list(hourly.loc[hourly['sex'] == 'f'].std())[2:]
    fig = plt.figure()
    plt.errorbar(range(len(m_average)), m_average, m_std, label='m', alpha=0.7)
    plt.errorbar(range(len(f_average)), f_average, f_std, label='f', alpha=0.7)
    plt.legend(loc='lower right')
    plt.savefig('{}plots/{}_sex_average'.format(DATA_PATH, frame))


def plot_stat_window_analysis():
    all_aucs = pd.read_csv(DATA_PATH + 'results/all_aucs.csv', index_col=0)

    # prepare all windows
    stat_week_aucs = all_aucs.loc[
        (idx for idx in all_aucs.index
            if (all(key in idx for key in ['stats', 'week', 'nor']))
            and all(key not in idx for key in ['AE'])),
        all_aucs.columns.str.contains('age')
        ]
    best_stat_week = stat_week_aucs.max(axis=1)
    result = pd.DataFrame()
    regex = re.compile(r'stats_dzne_week_([a-z_]+)(\d+)')
    for idx, val in best_stat_week.items():
        stats, window = regex.match(idx).groups()
        result.loc[stats, window] = val
    result.columns = pd.to_numeric(result.columns)
    result = result.loc[:,sorted(result.columns)]
    result.loc['average',:] = result.mean(axis=0)
    result.loc['highest',:] = result.max(axis=0)

    # plot all windows
    xtickslabels = [str(col) for col in result.columns]
    plot_frame = result.loc[['max', 'mean', 'medi', 'std', 'sum', 'average', 'highest'], :].transpose()
    plot_frame = plot_frame.reset_index(drop=True)
    plot_frame.plot.line()
    plt.xticks(list(range(len(xtickslabels))), xtickslabels)
    plt.tight_layout()
    plt.savefig('{}plots/stats_window_choosing.pdf'.format(DATA_PATH), format='pdf')
    #plt.show()

    # prepare all stats for window 720
    all_aucs = pd.read_csv(DATA_PATH + 'results/all_aucs.csv', index_col=0)
    window = '720'
    wind_all_aucs = all_aucs.loc[
        (idx for idx in all_aucs.index
            if (all(key in idx for key in ['stats', 'week', window, 'nor'])
            and all(key not in idx for key in ['AE', 'filter', 'hourly']))),
        :]
    attributes = ['sex', 'age', 'edu']
    for attr in attributes:
        wind_all_aucs.loc[:, attr] = wind_all_aucs.loc[:, wind_all_aucs.columns.str.contains(attr)].max(axis=1)
    best_attr_week = wind_all_aucs.loc[:, attributes]
    best_attr_week.index = [idx[16:-7] for idx in best_attr_week.index]
    best_attr_week = best_attr_week.sort_values(by='age', axis=0, ascending=False)

    # plot all stats for window 720
    best_attr_week.plot.line()
    xtickslabels = best_attr_week.index
    plt.xticks(list(range(len(xtickslabels))), xtickslabels, rotation=90)
    plt.tight_layout()
    plt.savefig('{}plots/stats_subset_choosing.pdf'.format(DATA_PATH), format='pdf')
    #plt.show()

    best_stats = ['max_medi', 'max', 'sum_mean']

    '''
    best_attr_week.sort_values(by='sex', axis=0, ascending=False).head(3)
    Out[129]: 
                            sex       age       edu
    max_medi           0.612013  0.778697  0.618254
    sum_std_medi_mean  0.611100  0.688722  0.614286
    max                0.595779  0.742857  0.613333
    best_attr_week.sort_values(by='age', axis=0, ascending=False).head(3)
    Out[130]: 
                       sex       age       edu
    max_medi      0.612013  0.778697  0.618254
    max           0.595779  0.742857  0.613333
    sum_max_mean  0.551136  0.741153  0.612910
    best_attr_week.sort_values(by='edu', axis=0, ascending=False).head(3)
    Out[131]: 
                            sex       age       edu
    sum_mean           0.552760  0.674887  0.655556
    sum_max_std_medi   0.567979  0.703008  0.642857
    max_std_medi_mean  0.579748  0.719799  0.642857
    '''


def plot_dist_param_analysis():
    all_aucs = pd.read_csv(DATA_PATH + 'results/all_aucs.csv', index_col=0)

    fig, axarr = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
    axarr[0].set_xlabel('age', fontsize=15)
    axarr[1].set_xlabel('gender', fontsize=15)
    axarr[2].set_xlabel('education', fontsize=15)

    for i, attr in enumerate(['age', 'sex', 'edu']):
        # prepare all windows
        dist_week_aucs = all_aucs.loc[
            (idx for idx in all_aucs.index
             if (all(key in idx for key in ['dist', 'week', 'nor']))
             and all(key not in idx for key in ['AE', 'filter', 'hourly'])),
            all_aucs.columns.str.contains(attr)
        ]
        best_dist_week = dist_week_aucs.max(axis=1)
        result = pd.DataFrame()
        regex = re.compile(r'dzne_week_dist_([\d]+)_(\d+)')
        for idx, val in best_dist_week.items():
            bucket, window = regex.match(idx).groups()
            result.loc[bucket, window] = val
        result.columns = pd.to_numeric(result.columns)
        result = result.loc[:, sorted(result.columns)]

        # plot all windows
        xtickslabels = [str(col) for col in result.columns]
        plot_frame = result.transpose()
        plot_frame = plot_frame.reset_index(drop=True)
        plt.setp(axarr[i], xticks=list(range(len(xtickslabels))), xticklabels=xtickslabels)
        axarr[i].plot(plot_frame)
    plt.tight_layout()

    #plt.show()
    plt.savefig('{}plots/dists_param_choosing.pdf'.format(DATA_PATH), format='pdf')

    # window = 240, bucket = 2 age
    # window = 720, bucket = 8 edu
    # window = 1440, bucket = 2 sex


def plot_AE_sum_windows():
    all_aucs = pd.read_csv(DATA_PATH + 'results/all_aucs.csv', index_col=0)

    # fig, axarr = plt.subplots(1, 3, figsize=(8, 3), sharey=True)
    # axarr[0].set_xlabel('age', fontsize=15)
    # axarr[1].set_xlabel('gender', fontsize=15)
    # axarr[2].set_xlabel('education', fontsize=15)

    # for i, attr in enumerate(['age', 'sex', 'edu']):
    # prepare all windows
    dist_week_aucs = all_aucs.loc[
        (idx for idx in all_aucs.index
         if (all(key in idx for key in ['AE', 'dsp', 'nor']))
         and all(key not in idx for key in ['dist', 'max', 'medi', 'mean', 'std', 'filter', 'hourly'])),
        all_aucs.columns.str.contains('sex')
    ]
    best_dist_week = dist_week_aucs.max(axis=1)
    result = pd.DataFrame()
    regex = re.compile(r'dzne_week_dist_([\d]+)_(\d+)')
    for idx, val in best_dist_week.items():
        bucket, window = regex.match(idx).groups()
        result.loc[bucket, window] = val
    result.columns = pd.to_numeric(result.columns)
    result = result.loc[:, sorted(result.columns)]

    # plot all windows
    xtickslabels = [str(col) for col in result.columns]
    plot_frame = result.transpose()
    plot_frame = plot_frame.reset_index(drop=True)
    plt.setp(axarr[i], xticks=list(range(len(xtickslabels))), xticklabels=xtickslabels)
    axarr[i].plot(plot_frame)
    plt.tight_layout()

    # plt.show()
    plt.savefig('{}plots/dists_param_choosing.pdf'.format(DATA_PATH), format='pdf')