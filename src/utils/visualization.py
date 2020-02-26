import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import TSNE



from utils.storage import load_frame, DATA_PATH

def one_user_embeddings_distances(vec_name='fake_dsp_sae_emb', user=0, data_path=DATA_PATH):
    vecframe = load_frame(vec_name, data_path)
    embeddings = vecframe.loc[vecframe.user == user, '0':]
    distances = euclidean_distances(embeddings, embeddings)
    sns_plot = sns.heatmap(distances, annot=True)
    sns_plot.get_figure().savefig("../results/{}_dist_{}.png".format(vec_name, user))
    plt.clf()


def one_user_TSNE(vec_name='fake_dsp_sae_emb', data_path=DATA_PATH):
    vecframe = load_frame(vec_name, data_path)
    embeddings = vecframe.loc[vecframe.user == 0, '0':]
    tsne = TSNE().fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(*zip(*tsne), c=list(vecframe.desc.to_numpy()))#, s=4)
    for day, point in enumerate(tsne):
        ax.annotate(day, point)
    plt.savefig("../results/{}_TSNE.png".format(vec_name))
    plt.clf()
