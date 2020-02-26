import logging

from attacks import Reidentification, Matching, AttributeInf, AttrInfRandForest
from utils.storage import DATA_PATH

for emb_name in ['pca_emb20', 'pca_emb50', 'pca_emb100']:#'dzne_dsp_max480']: #'dzne_dsp_dist_4_5760']:#, 'hourly_week','1minute_emb_dsp', '5minute_emb_dsp', '10minute_emb_dsp', '12minute_emb_dsp', 'hourly_emb',  ]:# 'simple_emb64', 'cnn_emb128',

    inf = AttrInfRandForest( emb_name, attribute='gender')
    inf.attack()

"""   
    
def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create file handlers
    # fh1 = logging.FileHandler(config.LOG_PATH + "warning.log", 'w')
    # fh1.setLevel(logging.WARNING)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    ch.setFormatter(formatter)
    # fh1.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    # logger.addHandler(fh1)

def main():
	for emb_name in ['dzne_dsp_dist_4_240']:#, 'hourly_week','1minute_emb_dsp', '5minute_emb_dsp', '10minute_emb_dsp', '12minute_emb_dsp', 'hourly_emb', 'pca_emb10', 'pca_emb150' ]:# 'simple_emb64', 'cnn_emb128',

            inf = AttrInfRandForest(emb_name, attribute='edu')
            inf.attack()

            # Use DNN for attribute attack, if choose RF Attribute attack, comment next 2 lines
            # dim = 128
            # inf.DNN_attack(emb_name, dim)

if __name__ == '__main__':
    config_logger()
    main()
"""
