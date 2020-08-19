# Created by rahman at 10:14 2019-06-20 using PyCharm

import pandas as pd
import matplotlib.pyplot as plt

plot_path = '../../data/paper_plots/'

plt.rcParams["figure.figsize"] = (10,7)
plt.rcParams.update({'font.size': 20, 'lines.linewidth': 2})
#plt.rcParams['text.usetex'] = True



def plot_pca(finalDf, finalDf_day, plot_path):
    """
    plots for PCA by attributes for daily and weeklong stepcounts
    :param finalDf: weeklong df
    :param finalDf_day: daywise df
    :return:
    """

    fig, axarr = plt.subplots(2, 3,  sharey=True, sharex=True, figsize=(10,7))
    # plt.subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.98, wspace=0.05)

    axarr[1,0].set_xlabel('gender', fontsize=17)
    axarr[0,0].set_ylabel('week', fontsize=17)
    axarr[1,0].set_ylabel('day',fontsize=17)
    # axarr[0].set_title('2 component PCA', fontsize = 20)
    targets = ['m', 'f']
    colors = ['b', 'r']

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['sex'] == target
        axarr[0,0].scatter(finalDf.loc[indicesToKeep, '0']
                         , finalDf.loc[indicesToKeep, '1']
                         , c=color
                         , s=50
                         , alpha=0.5)

        indicesToKeep = finalDf_day['sex'] == target
        axarr[1,0].scatter(finalDf_day.loc[indicesToKeep, '0']
                         , finalDf_day.loc[indicesToKeep, '1']
                         , c=color
                         , s=50
                         , alpha=0.5)

    axarr[0,0].legend(['male', 'female'], loc='lower center', bbox_to_anchor= (0.5, -0.2), ncol=2)


    axarr[1,1].set_xlabel('edu', fontsize=17)
    #axarr[1,1].set_xlabel('day, edu', fontsize=15)
    #axarr[0].set_ylabel('Principal Component 2', fontsize=15)
    # axarr[0].set_title('2 component PCA', fontsize = 20)
    targets = ['high', 'middle']
    colors = ['b', 'r']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['edu'] == target
        axarr[0,1].scatter(finalDf.loc[indicesToKeep, '0']
                         , finalDf.loc[indicesToKeep, '1']
                         , c=color
                         , s=50
                         , alpha=0.5)

        indicesToKeep = finalDf_day['edu'] == target

        axarr[1,1].scatter(finalDf_day.loc[indicesToKeep, '0']
                         , finalDf_day.loc[indicesToKeep, '1']
                         , c=color
                         , s=50
                         , alpha=0.5)

    axarr[0,1].legend(targets, loc='lower center', bbox_to_anchor= (0.5, -0.2), ncol=2)


    axarr[1,2].set_xlabel('age', fontsize=17)
    #axarr[1,2].set_xlabel('day, age', fontsize=15)
    #axarr[0].set_ylabel('Principal Component 2', fontsize=15)
    # axarr[0].set_title('2 component PCA', fontsize = 20)
    targets = ['young', 'old']

    indicesToKeep = finalDf['age'] < 55
    axarr[0,2].scatter(finalDf.loc[indicesToKeep, '0']
                         , finalDf.loc[indicesToKeep, '1']
                         , c='b'
                         , s=50
                         , alpha=0.5)

    indicesToKeep = finalDf['age'] >= 55
    axarr[0,2].scatter(finalDf.loc[indicesToKeep, '0']
                         , finalDf.loc[indicesToKeep, '1']
                         , c='r'
                         , s=50
                         , alpha=0.5)


    indicesToKeep = finalDf_day['age'] < 55
    axarr[1,2].scatter(finalDf_day.loc[indicesToKeep, '0']
                         , finalDf_day.loc[indicesToKeep, '1']
                         , c='b'
                         , s=50
                         , alpha=0.5)
    indicesToKeep = finalDf_day['age'] >= 55
    axarr[1,2].scatter(finalDf_day.loc[indicesToKeep, '0']
                         , finalDf_day.loc[indicesToKeep, '1']
                         , c='r'
                         , s=50
                         , alpha=0.5)


    axarr[0,2].legend(targets, loc='lower center', bbox_to_anchor= (0.5, -0.2), ncol=2)


    plt.tight_layout()

    #plt.subplots_adjust(hspace=0.05, wspace=0.05, right=0.99, top=0.99, bottom=0.13)

    plt.savefig(plot_path + 'both_pca_all.pdf', format='pdf')
    plt.show()



def plot_age(data_path, plot_path):
    """
    plots distribution of users by age in our dataset
    :return:
    """

    desc = pd.read_csv('/Users/tahleen/Desktop/githubrepos/linkability_stepcount/data/dzne/dzne_desc.csv')

    desc.hist(column='age', bins=len(desc.age.unique()))
    plt.xlabel('age of users')
    plt.ylabel('number of users')
    plt.tight_layout()
    plt.savefig(plot_path + 'age.pdf', format='pdf')
    plt.show()



def plot_link_dist(data_path, plot_path):

    merged = pd.read_csv(data_path + 'link_merged.csv', index_col=0)

    dist = merged[merged.infile.str.contains('dist')]

    dist['binsize'] = pd.to_numeric(dist.infile.str.extract('(\d+)[_][\d+]')[0].values)

    dist['winsize'] = pd.to_numeric(dist.infile.str.extract('[\d+][_](\d+)[_]')[0].values)

    dist = dist.set_index(['binsize', 'winsize']).sort_index()

    dist["dense_mean"].plot( marker="o", label=r'$\mathtt{Dense\_siamese}$',  yerr=dist['dense_std'])

    dist["rf_mean"].plot( marker="v", label=r'$\mathtt{RF\_standard}$',  yerr=dist['rf_std'])

    dist["cnn1_mean"].plot( marker="s", label=r'$\mathtt{CNN\_siamese}$',  yerr=dist['cnn1_std'])

    dist["lstm_mean"].plot( marker="x", label=r'$\mathtt{LSTM\_siamese}$',  yerr=dist['lstm_std'])

    dist["cos_auc"].plot( marker="^", label=r'$\mathtt{Cosine}$', linestyle=':')

    dist["eucl_auc"].plot( marker="d", label=r'$\mathtt{Euclidean}$',  linestyle=':')

    xtickslabels = dist.infile.str.extract('(\d+[_]\d+)')[0].values

    plt.xticks(pd.np.arange(0, len(dist)), xtickslabels, rotation=45)

    plt.grid(True)
    plt.xlabel('bin-size_window-size')
    plt.ylabel('AUC')
    plt.title("feature: s_dist")
    plt.legend(ncol=2)
    plt.tight_layout()
    #plt.subplots_adjust(right=0.97, top=0.97, bottom=0.1, left=0.05)

    plt.savefig(plot_path + 'dist_link.pdf', format='pdf')
    plt.show()




def plot_link_stat(stat, data_path, plot_path):
    """
    takes there result dataframes and makes plots for linkability evaluation

    :param stat: which statstic to plot
    :return:
    """

    merged = pd.read_csv(data_path + 'link_merged.csv', index_col=0)

    stat_df = merged[merged.infile.str.contains(stat)]

    window = pd.to_numeric(stat_df.infile.str.extract('(\d+)')[0].values)

    stat_df = stat_df.set_index(pd.to_numeric(window))
    stat_df.sort_index(inplace=True)
    stat_df.reset_index(drop=True, inplace=True)


    stat_df["dense_mean"].plot( marker="o", label=r'$\mathtt{Dense\_siamese}$', yerr=stat_df['dense_std'])

    stat_df["rf_mean"].plot( marker="v", label=r'$\mathtt{RF\_standard}$',  yerr=stat_df['rf_std'])

    stat_df["cnn1_mean"].plot( marker="s", label=r'$\mathtt{CNN\_siamese}$',  yerr=stat_df['cnn1_std'])

    stat_df["lstm_mean"].plot( marker="x", label=r'$\mathtt{LSTM\_siamese}$',  yerr=stat_df['lstm_std'])

    stat_df["cos_auc"].plot( marker="^", label=r'$\mathtt{Cosine}$', linestyle=':')

    stat_df["eucl_auc"].plot( marker="d", label=r'$\mathtt{Euclidean}$',  linestyle=':')

    plt.xticks(stat_df.index.values, sorted(window) , rotation=45)
    plt.grid(True)
    plt.xlabel('window size')
    plt.ylabel('AUC')
    plt.title("feature: s_"+stat)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path + stat +'_link.pdf', format='pdf')
    plt.show()



def label_bar(ax, bars, type, infile, **kwargs):
    """
    Attach a text label to each bar displaying its y value
    """

    for i in range(len(bars)):

        bar = bars[i]

        text_x = bar.get_x() + bar.get_width() / 2

        text_y1 = 0.15 #bar.get_height() -

        text = type + " " + infile[i]

        ax.text(text_x, text_y1, text, ha='center', va='bottom', color='white', rotation =90, **kwargs)

        text_y2 = bar.get_height() - 0.06 + 1/(10+i)

        ax.text(text_x, text_y2, "{:.2}".format(bar.get_height()), ha='center', va='bottom', color='black', fontsize=13 , **kwargs)



def plot_top_feats(data_path, plot_path):
    """

    :return:
    """
    fig, ax = plt.subplots()

    merged = pd.read_csv(data_path + 'link_merged.csv', index_col=0)

    ind = pd.np.arange(1, 4)

    for type in  ['dist', 'sum', 'max', 'mean', 'medi']:

        df = merged[merged.infile.str.contains(type)]

        if type == 'dist':
            top = df.nlargest(3, ['dense_mean'])[['infile', 'dense_mean', 'dense_std']]

            ind = ind + 4

            bars = ax.bar(ind.tolist(), top.dense_mean, yerr=top.dense_std, zorder=3, label=type)

        else:
            top = df.nlargest(3, ['cnn1_mean'])[['infile', 'cnn1_mean', 'cnn1_std']]

            ind = ind + 4

            bars = ax.bar(ind.tolist(), top.cnn1_mean, yerr=top.cnn1_std, zorder=3, label=type)


        label_bar(ax, bars, type, top.infile.str.extract('[_][a-z]+(\d+|_\d_\d+)[_]')[0].values.tolist())
        #top.infile.str.extract('[_](([a-z]+\d+)|(\d_\d+))[_]')[0].values.tolist()

    plt.ylim(top=0.9)
    plt.yticks()
    plt.ylabel('AUC')
    plt.xlabel("feature extraction methods")
    ax.axes.xaxis.set_ticklabels([])

    ax.yaxis.grid(zorder=0)

    #ax.legend(ncol=5 , loc='upper center')
    plt.tight_layout()
    plt.savefig(plot_path + "top_feats.pdf", bbox_inches="tight", format='pdf')
    plt.show()






def plot_embpair(emb1, emb2, cols, step):

    """plot pairs of embbeddings for a user
    :param emb1: 1 full row of vecframe
    :param emb2: 1 full row of vecframe
    :param cols: column names (cane be time points or embedding features)
    :step: how often we want xticks
    :return:
    """


    plt.rcParams["figure.figsize"] = (40,4)

    #print (cols[2:])
    plt.plot(cols[2:], emb1[2:] ,'r', label='emb1')

    plt.plot(cols[2:], emb2[2:], 'b', label='emb2' )

    plt.xticks(range(0, len(emb1) -2,  step ), range(0, len(emb1) -2, step ), rotation=90)

    plt.xlabel('embedding feature OR time point', fontsize=10)


    plt.ylabel("   step count or some embedding value",  fontsize=10)
    plt.legend()
    plt.tight_layout()

    #plt.subplots_adjust(right=0.99, top=0.99, bottom=0.05, left=0.05)
    plt.savefig(plot_path +  "embpairs.pdf", format='pdf')
    print (len(emb1) - 2)


    print  (emb1[2:])
    print(emb1[2:])
    plt.show()


def plotdays(epochs):
    """
    plot stepcounts per day for a week for 1 user as in dzne.csv
    :param epochs:
    :return:
    """

    uid = '0'

    grouped=epochs.groupby('day')
    ### Plot STEPCOUNT per day

    f, axarr = plt.subplots(7, sharex=True, figsize=(20, 15))

    for day, group in grouped:
        axarr[day].plot(epochs.index[:len(group)], group[uid], label ='day '+str(day))
        axarr[day].set_title('day '+str(day))


    hours=epochs.hour.apply(lambda x: int(x)).unique()
    plt.xticks(range(len(hours)), hours)
    plt.xlabel('hour of each day', fontsize=20)
    plt.ylabel("   step count ",  fontsize=20)
    plt.subplots_adjust(hspace=0.2, wspace=0.05, right=0.99, top=0.97, bottom=0.05, left=0.05)
    plt.savefig(plot_path + uid + "stepcount.pdf", format='pdf')
    plt.show()



### 1. plot all days dzne, diff colors for each day



"""for day, group in grouped:
    group.act_score.plot()
#plt.savefig("activity_time.pdf", format='pdf')
plt.show()


### 2. Plot ACTI per day

f, axarr = plt.subplots(7, sharex=True, figsize=(20, 15))

for day, group in grouped:
    axarr[day].plot(group.hours, group.act_score, label ='day '+str(day))
    axarr[day].set_title('day '+str(day))

hours=epochs.hours.apply(lambda x: int(x)).unique()
plt.xticks(range(len(hours)), hours)
plt.xlabel('hour of each day', fontsize=20)
plt.ylabel("                Activity score ",  fontsize=20)
plt.subplots_adjust(hspace=0.2, wspace=0.05, right=0.99, top=0.97, bottom=0.05, left=0.05)
plt.savefig("activity_per_day.pdf", format='pdf')
plt.show()

"""

