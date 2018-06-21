import os
import pwd
import sys

global cur_user_name

cur_user_name = pwd.getpwuid(os.getuid()).pw_name

if cur_user_name == "liu121":
    sys.path.append('/home/liu121/emnlp_baseline')

import argparse
from pathlib import Path

from crf_target.datafeed import DataFeed
from crf_target.classifier import Classifier
from emnlp_baseline.metrics import Metrics


def main(nn_config,data_config):
    df = DataFeed(data_config)
    nn_config['source_NETypes_num']=df.source_NETypes_num
    nn_config['target_NETypes_num']=df.target_NETypes_num
    print('source_NETypes_num: ',str(nn_config['source_NETypes_num']))
    print('target_NETypes_num: ',str(nn_config['target_NETypes_num']))
    mt = Metrics(data_config)
    cl = Classifier(nn_config,df,data_config,mt)
    cl.train()
    # true_labels, pred_labels, X_data = cl.train()
    # id2label_dic = df.source_id2label_generator()
    # I = mt.word_id2txt(X_data, true_labels, pred_labels, id2label_dic)
    # print('output')
    # mt.conll_eval_file(I)
    # print('finish output')

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--num',type=int)
    args = parser.parse_args()

    nn_configs = [{'lstm_cell_size': 150,
                 'vocabulary_size': 2981402,
                 'feature_dim': 200,
                 'lr': 0.003,
                 'reg_rate': 0.00003,
                 'source_NETypes_num': None,
                 'target_NETypes_num': None,
                 'pad_index': 1,
                 'epoch_stage2':200,
                 'epoch_stage3':150},
                  {'lstm_cell_size': 150,
                   'vocabulary_size': 2981402,
                   'feature_dim': 200,
                   'lr': 0.0003,
                   'reg_rate': 0.00003,
                   'source_NETypes_num': None,
                   'target_NETypes_num': None,
                   'pad_index': 1,
                   'epoch_stage2': 200,
                   'epoch_stage3': 150},
                  {'lstm_cell_size': 150,
                   'vocabulary_size': 2981402,
                   'feature_dim': 200,
                   'lr': 0.00003,
                   'reg_rate': 0.00003,
                   'source_NETypes_num': None,
                   'target_NETypes_num': None,
                   'pad_index': 1,
                   'epoch_stage2': 200,
                   'epoch_stage3': 150},
                  {'lstm_cell_size': 150,
                   'vocabulary_size': 2981402,
                   'feature_dim': 200,
                   'lr': 0.000003,
                   'reg_rate': 0.00003,
                   'source_NETypes_num': None,
                   'target_NETypes_num': None,
                   'pad_index': 1,
                   'epoch_stage2': 200,
                   'epoch_stage3': 150},
                  {'lstm_cell_size': 150,
                   'vocabulary_size': 2981402,
                   'feature_dim': 200,
                   'lr': 0.0000003,
                   'reg_rate': 0.00003,
                   'source_NETypes_num': None,
                   'target_NETypes_num': None,
                   'pad_index': 1,
                   'epoch_stage2': 200,
                   'epoch_stage3': 150},
                  ]
    nn_config=nn_configs[args.num]
    nn_config['mod'] = 1
    nn_config['epoch'] = 3

    # k_shot = [['1.0', '1.1', '1.2', '1.3', '1.4', ],
    #           ['2.0', '2.1', '2.2', '2.3', '2.4', ],
    #           ['4.0', '4.1', '4.2', '4.3', '4.4', ],
    #           ['8.0', '8.1', '8.2', '8.3', '8.4', ],
    #           # ['16.0', '16.1', '16.2', '16.3', '16.4']
    #           ]
    k_shot = [['1.0',],
              ['2.0',],
              ['4.0',],
              ['8.0',],
              ['16.0',]]
    k_groups = k_shot[args.num]
    for k in k_groups:
        # BBN
        data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                       'pkl_filePath': '/datastore/liu121/nosqldb2/crf_target/data/data_conll_bbn_kn.pkl',
                       'k_instances': k,
                       'batch_size':50,
                       'conlleval_filePath': '/datastore/liu121/nosqldb2/crf_target/conlleval'+str(args.num)}
        report = '/datastore/liu121/nosqldb2/crf_target/report/'
        model = '/datastore/liu121/nosqldb2/crf_target/model/'
        nn_config['words_num'] = 100

        model = model + 'model' + str(args.num)
        path = Path(model)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        path = Path(report)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        model = model + '/model' + str(args.num) + '.ckpt'
        nn_config['model'] = model
        report = report + 'report_' + str(args.num)
        nn_config['report'] = report
        main(nn_config, data_config)
