# Created by rahman at 10:14 2019-06-20 using PyCharm

import pandas as pd
import matplotlib.pyplot as plt
datapath="../data/dzne/plots/"


plt.rcParams["figure.figsize"] = (9,7)
#plt.rcParams.update({'font.size': 31, 'lines.linewidth': 4})
plt.rcParams['text.usetex'] = True



def plot_pca(finalDf, finalDf_day):
    """
    plots for PCA by attributes for daily and weeklong stepcounts
    :param finalDf: weeklong df
    :param finalDf_day: daywise df
    :return:
    """

    fig, axarr = plt.subplots(2, 3,  sharey=True, sharex=True)
    # plt.subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.98, wspace=0.05)

    axarr[1,0].set_xlabel('gender', fontsize=15)
    axarr[0,0].set_ylabel('week', fontsize=15)
    axarr[1,0].set_ylabel('day', fontsize=15)
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


    axarr[1,1].set_xlabel('edu', fontsize=15)
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


    axarr[1,2].set_xlabel('age', fontsize=15)
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

    plt.savefig('paper_plots/both_pca_all.pdf', format='pdf')
    plt.show()



def plot_age():
    """
    plots distribution of users by age in our dataset
    :return:
    """

    desc = pd.read_csv('../../data/dzne/dzne_desc.csv')

    desc.hist(column='age', bins=len(desc.age.unique()))
    plt.xlabel('age of users')
    plt.ylabel('number of users')
    plt.tight_layout()
    plt.savefig('paper_plots/age.pdf', format='pdf')
    plt.show()


def plot_link(dense, cnn, rf, eucl, cos, lstm, stat):
    """
    takes there result dataframes and makes plots for linkability evaluation
    :param dense:
    :param cnn:
    :param rf:
    :param eucl:
    :param cos:
    :param lstm:
    :param stat: which statstic to plot
    :return:
    """

    cnn = cnn.drop(["epochs", "regu", "batchsize"], axis=1)
    lstm = lstm.drop(["epochs", "regu", "batchsize"], axis=1)
    dense = dense.drop(["epochs", "regu", "batchsize"], axis=1)
    cos = cos.drop(["clf"], axis=1)
    eucl = eucl.drop(["clf"], axis=1)
    #rf = rf.drop(["clf"], axis=1)

    dense = dense[dense.infile.str.contains(stat)]
    cnn = cnn[cnn.infile.str.contains(stat)]
    lstm = lstm[lstm.infile.str.contains(stat)]
    eucl = eucl[eucl.infile.str.contains(stat)]
    cos = cos[cos.infile.str.contains(stat)]
    rf = rf[rf.infile.str.contains(stat)]

    joined = dense.merge(eucl, on='infile')
    joined = joined.merge(cos, on='infile')
    joined = joined.merge(cnn, on='infile')
    #joined = joined.merge(lstm, on='infile')
    joined = joined.merge(rf, on='infile')

    norms = joined[joined.infile.str.contains("nor")]
    joined = joined[~joined.infile.str.contains("nor")]

    xtickslabelss = sorted(pd.to_numeric(joined.infile.str.extract('(\d+)')[0].values))

    joined = joined.set_index(pd.to_numeric(xtickslabelss))
    joined.sort_index(inplace=True)
    joined.reset_index(drop=True, inplace=True)

    xticks_ = pd.to_numeric(norms.infile.str.extract('(\d+)')[0].values)
    norms = norms.set_index(pd.to_numeric(xticks_))
    norms.sort_index(inplace=True)
    norms.reset_index(drop=True, inplace=True)


    joined["dense_auc"].plot( marker="o", label=r'$\mathtt{Dense\_siamese}$', color= 'r')
    norms["dense_auc"].plot(marker="o", label=r'$\mathtt{Dense\_siamese\_norm}$', linestyle=':', color='r')

    joined["eucl_auc"].plot(marker="d", label=r'$\mathtt{Euclidean}$', color= 'g')
    norms["eucl_auc"].plot(marker="d", label=r'$\mathtt{Euclidean\_norm}$', linestyle=':', color= 'g')

    joined["cos_auc"].plot( marker="^", label=r'$\mathtt{Cosine}$', color='b')
    norms["cos_auc"].plot( marker="^", label=r'$\mathtt{Cosine\_norm}$', linestyle=':', color= 'b')

    joined["rf_auc"].plot( marker="v", label=r'$\mathtt{RF\_standard}$', color= 'c')
    norms["rf_auc"].plot( marker="v", label=r'$\mathtt{RF\_standard\_norm}$',linestyle=':', color= 'c')

    joined["cnn_auc"].plot( marker="*", label=r'$\mathtt{CNN\_siamese\_norm}$', color='#A2142F')
    norms["cnn_auc"].plot( marker="*", label=r'$\mathtt{CNN\_siamese\_norm}$', linestyle=':',color='#A2142F')

    #joined["lstm_auc"].plot(marker="s", label=r'$\mathtt{LSTM\_siamese$', color='m')
    #norms["lstm_auc"].plot(marker="s", label=r'$\mathtt{CNN\_siamese}$', linestyle=':', color='m')

    plt.xticks(joined.index.values,xtickslabelss)
    plt.grid(True)
    plt.xlabel('window size')
    plt.ylabel('AUC')
    plt.title("feature: s\_"+stat)
    plt.legend()
    plt.tight_layout()
    plt.savefig('paper_plots/'+ stat +'_link.pdf', format='pdf')
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
    plt.savefig(datapath +  "embpairs.pdf", format='pdf')
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
    plt.savefig(datapath + uid + "stepcount.pdf", format='pdf')
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

