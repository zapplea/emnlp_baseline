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

from emnlp_baseline.datafeed import DataFeed
from emnlp_baseline.classifier import Classifier
from emnlp_baseline.metrics import Metrics

def main(nn_config,data_config):
    df = DataFeed(data_config)
    nn_config['source_NETypes_num']=df.source_NETypes_num
    nn_config['target_NETypes_num']=df.target_NETypes_num
    print('source_NETypes_num: ',str(nn_config['source_NETypes_num']))
    print('target_NETypes_num: ',str(nn_config['target_NETypes_num']))
    cl = Classifier(nn_config,df)
    if nn_config['stage1'] == "True":
        cl.train()
    else:
        mt = Metrics(data_config)
        true_labels, pred_labels, X_data = cl.train()
        id2label_dic = df.id2label_generator()
        I = mt.word_id2txt(X_data,true_labels,pred_labels,id2label_dic)
        print('output')
        mt.conll_eval_file(I)
        print('finish output')

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--num',type=int)
    parser.add_argument('--stage1',type=str)
    parser.add_argument('--dn',type=str)
    args = parser.parse_args()

    # Train the relation model and target crf model
    if args.stage1=='False1' or args.stage1=='False2' or args.stage1=='False0' or args.stage1=='False4' or args.stage1=='False5' or args.stage1=='False6':

        nn_config = {'lstm_cell_size': 150,
                     'vocabulary_size': 2981402,
                     'feature_dim': 200,
                     'lr': 0.000003,
                     'reg_rate': 0.03,
                     'source_NETypes_num': None,
                     'target_NETypes_num': None,
                     'pad_index': 1,
                     'epoch_stage2':200,
                     'epoch_stage3':100}
        nn_config['stage1'] = args.stage1
        nn_config['mod'] = 1
        nn_config['epoch'] = 3

        k_shot = [['1.0', '1.1', '1.2', '1.3', '1.4', ],
                  ['2.0', '2.1', '2.2', '2.3', '2.4', ],
                  ['4.0', '4.1', '4.2', '4.3', '4.4', ],
                  ['8.0', '8.1', '8.2', '8.3', '8.4', ],
                  # ['16.0', '16.1', '16.2', '16.3', '16.4']
                  ]
        # k_shot = [['1.0',],
        #           ['2.0',],
        #           ['4.0',],
        #           ['8.0',],
        #           ['16.0',]]
        k_groups = k_shot[args.num]
        for k in k_groups:
            # BBN
            if args.dn == "bbn_bbn_kn":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_bbn_kn.pkl',
                               'k_instances': k,
                               'batch_size':50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_bbn_kn/conlleval' + str(k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/report/report_'+k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/model0/model0.ckpt'
                nn_config['words_num'] = 100
            elif args.dn=="bbn_cadec":
                # args.dn == "cadec"
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec/conlleval' + str(k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/report_'+k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn=="bbn_nvd":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_nvd.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_nvd/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/report_' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/model0/model0.ckpt'
                nn_config['words_num'] = 100
            elif args.dn=="bbn_cadec_simple":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec_simple.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec_simple/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/report_' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/model0/model0.ckpt'
                nn_config['words_num'] = 200
            # CONLL
            elif args.dn=="conll_bbn_kn":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_bbn_kn.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_bbn_kn/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/report_' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/model0/model0.ckpt'
                nn_config['words_num'] = 100
            elif args.dn=="conll_cadec":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/report_' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn == "conll_cadec_simple":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec_simple.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec_simple/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/report_' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn=="conll_nvd":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_nvd.pkl',
                               'k_instances': k,
                               'batch_size': 50,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_nvd/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/report/report_' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/model0/model0.ckpt'
                nn_config['words_num'] = 100
            main(nn_config,data_config)

    # train crf source model and store it
    if args.stage1 == 'True':
        nn_configs = [
            {'lstm_cell_size': 150,
             'vocabulary_size': 2981402,
             'feature_dim': 200,
             'lr': 0.03,
             'reg_rate': 0.00003,
             'source_NETypes_num': None,
             'target_NETypes_num': None,
             'pad_index': 1,
             'epoch_stage1':150,}
        ]
        nn_config = nn_configs[args.num]
        nn_config['stage1'] = args.stage1
        # BBN
        if args.dn == "bbn_bbn_kn":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_bbn_kn.pkl',
                           'batch_size':50}
            nn_config['words_num'] =100
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/'
        elif args.dn =="bbn_cadec":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 200
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/'
        elif args.dn == "bbn_cadec_simple":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec_simple.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 200
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/'
        elif args.dn == "bbn_nvd":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_nvd.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 100
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/'
        # CONLL
        elif args.dn == "conll_bbn_kn":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_bbn_kn.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 100 # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/'
        elif args.dn == "conll_cadec":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 200  # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/'
        elif args.dn == "conll_cadec_simple":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec_simple.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 200  # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/'
        elif args.dn == "conll_nvd":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_nvd.pkl',
                           'batch_size': 50}
            nn_config['words_num'] = 100  # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_nvd/model/'




        model = model +'model'+str(args.num)
        path = Path(model)
        if not path.exists():
            path.mkdir(parents=True,exist_ok=True)
        path = Path(report)
        if not path.exists():
            path.mkdir(parents=True,exist_ok=True)


        model = model +'/model'+str(args.num)+'.ckpt'
        nn_config['model'] = model
        report = report +'report_' +str(args.num)
        nn_config['report'] = report
        main(nn_config, data_config)