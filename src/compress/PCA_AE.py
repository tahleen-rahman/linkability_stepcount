# Created by rahman at 15:00 2019-10-17 using PyCharm

import pandas as pd
from compress import Compressor
from sklearn.decomposition import PCA
from utils.storage import dump_frame


class PCA_AE(Compressor):

    def compress_save(self,  emb_size = 10, out_name = 'pca_emb'):
        super().compress_save()

        pca = PCA(n_components=emb_size)
        principalComponents = pca.fit_transform(self.vecframe.iloc[:, 2:].values)
        principalDf = pd.DataFrame(data=principalComponents)

        principalDf['user']=self.vecframe.user
        principalDf['desc']=self.vecframe.desc
        principalDf.columns = principalDf.columns.map(str)

        cols=principalDf.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        principalDf=principalDf[cols]

        outfile = out_name + str(emb_size)
        print("saving PCAframe in", outfile)

        dump_frame(principalDf, outfile, in_csv=True)

        return principalDf
