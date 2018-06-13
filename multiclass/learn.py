import os
import pwd
import sys

global cur_user_name

cur_user_name = pwd.getpwuid(os.getuid()).pw_name

if cur_user_name == "liu121":
    sys.path.append('/home/liu121/emnlp_baseline')
elif cur_user_name == "che313":
    sys.path.append('/home/che313/emnlp_baseline')

import argparse
from pathlib import Path

from multiclass.datafeed import DataFeed
from multiclass.classifier import Classifier

def main(nn_config,data_config):
    df = DataFeed(data_config)
    nn_config['source_NETypes_num']=df.source_NETypes_num
    nn_config['target_NETypes_num']=df.target_NETypes_num
    print('source_NETypes_num: ',str(nn_config['source_NETypes_num']))
    print('target_NETypes_num: ',str(nn_config['target_NETypes_num']))
    cl = Classifier(nn_config,df)
    cl.train()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int)
    args = parser.parse_args()
    nn_configs = [
        {'lstm_cell_size': 150,
         'batch_size': 50,
         'vocabulary_size': 2981402,
         'feature_dim': 200,
         'lr': 0.03,
         'reg_rate': 0.000003,
         'source_NETypes_num': None,
         'target_NETypes_num': None,
         'pad_index': 1,
         'reg_linear_rate':0.000003}
    ]

    # which configuration will be chosen
    nn_config = nn_configs[args.num]

    # decide after how many epochs, the program will print out the test results.
    nn_config['mod'] = 50

    data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                   'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_bbn_kn.pkl',
                   'k_instances': '16.0'}
    # maximal length of a sentence.
    nn_config['words_num'] = 100
    # epoch stands for number of batch. there are at most 16*42 sentences in the training dataset.
    nn_config['epoch'] = 10001
    report = '/datastore/liu121/nosqldb2/multiclass/report_bbn_kn/report_'

    report = report + str(args.num)
    nn_config['report'] = report
    print(str(nn_config))
    main(nn_config, data_config)