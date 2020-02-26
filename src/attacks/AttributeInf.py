import pandas as pd

from utils.storage import DATA_PATH, load_frame
from attacks import Attack

class AttributeInf(Attack):

    def __init__(self, vec_name, attribute, desc_file='dzne_desc', data_path=DATA_PATH):
        # this already loads the vecframe from vec_name into a dataframe self.vecframe
        super().__init__(vec_name, data_path)

        self.emb_name = vec_name

        # attribute = gender only right now
        self.attribute = attribute

        self.att = load_frame("dzne_desc")  # pd.read_csv(DATA_PATH + 'dzne_desc.csv')
        self.att = self.att.loc[:, ['user', 'age', 'edu', 'sex']]
        self.att['user'] = pd.to_numeric(self.att['user'])
        self.merged_emb = self.vecframe.merge(self.att, on='user')
