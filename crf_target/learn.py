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

print('======================import DataFeed======================')
from crf_target.datafeed import DataFeed
print('======================import Classifier======================')
from crf_target.classifier import Classifier

def main(nn_config,data_config):
    print('======================load_data======================')
    df = DataFeed(data_config)
    print('======================load_over======================')
    nn_config['source_NETypes_num']=df.source_NETypes_num
    nn_config['target_NETypes_num']=df.target_NETypes_num
    print('source_NETypes_num: ',str(nn_config['source_NETypes_num']))
    print('target_NETypes_num: ',str(nn_config['target_NETypes_num']))
    cl = Classifier(nn_config,df)
    cl.train()

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--num',type=int)
    parser.add_argument('--dn',type=str)
    args = parser.parse_args()

    nn_config = {'lstm_cell_size': 300,
                 'vocabulary_size': 2981402,
                 'feature_dim': 200,
                 'lr': 0.000003,
                 'reg_rate': 0.0003,
                 'source_NETypes_num': None,
                 'target_NETypes_num': None,
                 'pad_index': 1,
                 'epoch_stage2':200,
                 'epoch_stage3':1000}
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
        if args.dn == "bbn_bbn_kn":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_bbn_kn.pkl',
                           'k_instances': k,
                           'batch_size':50,
                           'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_bbn_kn/conlleval' + str(k)}
            nn_config['report'] = '/datastore/liu121/nosqldb2/crf_target/report_'+k
            nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/model0/model0.ckpt.meta'
            nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/model0/model0.ckpt'
            nn_config['words_num'] = 100
        # elif args.dn=="bbn_cadec":
        #     # args.dn == "cadec"
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec/conlleval' + str(k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/report_'+k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 200
        # elif args.dn=="bbn_nvd":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_nvd.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_nvd/conlleval' + str(
        #                        k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/report_' + k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 100
        # elif args.dn=="bbn_cadec_simple":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec_simple.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec_simple/conlleval' + str(
        #                        k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/report_' + k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 200
        # # CONLL
        # elif args.dn=="conll_bbn_kn":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_bbn_kn.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_bbn_kn/conlleval' + str(
        #                        k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/report_' + k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 100
        # elif args.dn=="conll_cadec":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec/conlleval' + str(
        #                        k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/report_' + k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 200
        # elif args.dn == "conll_cadec_simple":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec_simple.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec_simple/conlleval' + str(
        #                        k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/report_' + k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 200
        # elif args.dn=="conll_nvd":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_nvd.pkl',
        #                    'k_instances': k,
        #                    'batch_size': 50,
        #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_nvd/conlleval' + str(
        #                        k)}
        #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/report/report_' + k
        #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/model0/model0.ckpt.meta'
        #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/model0/model0.ckpt'
        #     nn_config['words_num'] = 100
        main(nn_config,data_config)
