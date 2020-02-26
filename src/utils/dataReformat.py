
#aggDf.to_csv("aggregated.csv", index=False)


"""
epochs['day']=epochs.time.apply(lambda x: int(x))-int(epochs.time.min())
epochs_small=epochs[['day', 'act_score', 'step_count']]
epochs_small.loc[:, 'time_day']=epochs.time.apply(lambda x: x%1)

grouped=epochs_small.groupby('day')
for day, group in grouped:
    print len(group), group.time.min()

for d in epochs_small.day.unique():
    epochs_small.loc[epochs_small.day==d, 'time_id']=int(range(5760))

epochs['time']=epochs.time_day.apply(lambda x: x*1000000)


epochs_small.to_csv("epochs_plot.csv")    


"""
