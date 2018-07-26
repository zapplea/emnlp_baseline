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
    mt = Metrics(data_config)
    cl = Classifier(nn_config, df, data_config, mt)
    cl.train()

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--num',type=int)
    parser.add_argument('--stage1',type=str)
    parser.add_argument('--dn',type=str)
    args = parser.parse_args()

    epoch_stage1=100
    epoch_stage2=200
    epoch_stage3=200

    # Train the relation model and target crf model
    if args.stage1=='False1' or args.stage1=='False2' or args.stage1=='False0' or args.stage1=='False4' or args.stage1=='False5' or args.stage1=='False6':

        nn_config = {'lstm_cell_size': 150,
                     'vocabulary_size': 2981402,
                     'feature_dim': 200,
                     'source_NETypes_num': None,
                     'target_NETypes_num': None,
                     'pad_index': 1,
                     'epoch_stage2':epoch_stage2,
                     'epoch_stage3':epoch_stage3,
                     'stage1':args.stage1,
                     'dropout':0.5,
                     'bilstm_num_layers':1,
                     'early_stop':100,
                     }

        lr = [0.003,]
        reg_rate=[0.00003,]

        nn_config['lr']=lr[args.num]
        nn_config['reg_rate']=reg_rate[args.num]

        data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                       'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_%s.pkl'%args.dn,
                       'batch_size': 50, }

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
        k_groups = ['1.0','2.0','4.0','8.0','16.0']
        for k in k_groups:
            data_config['k_instances']=k
            data_config['conlleval_filePath'] = '/datastore/liu121/nosqldb2/emnlp_baseline/CoNLLEval/conlleval_%s%s_%s' % (args.dn, k, str(args.num))
            nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/%s/report/report_%s'%(args.dn,k)
            nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/%s/model/model0/model0.ckpt.meta'%(args.dn)
            nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/%s/model/model0/model0.ckpt'%(args.dn)
            # BBN
            if args.dn == "bbn_bbn_kn":
                nn_config['words_num'] = 100
            elif args.dn=="bbn_cadec":
                nn_config['words_num'] = 200
            elif args.dn=="bbn_nvd":
                nn_config['words_num'] = 100
            elif args.dn=="bbn_cadec_simple":
                nn_config['words_num'] = 200
            # CONLL
            elif args.dn=="conll_bbn_kn":
                nn_config['words_num'] = 100
            elif args.dn=="conll_cadec":
                nn_config['words_num'] = 200
            elif args.dn == "conll_cadec_simple":
                nn_config['words_num'] = 200
            elif args.dn=="conll_nvd":
                nn_config['words_num'] = 100

            main(nn_config,data_config)

    # train crf source model and store it
    if args.stage1 == 'True':
        # fixed variables
        nn_config ={'lstm_cell_size': 150,
                     'vocabulary_size': 2981402,
                     'feature_dim': 200,
                     'source_NETypes_num': None,
                     'target_NETypes_num': None,
                     'pad_index': 1,
                     'epoch_stage1':epoch_stage1,
                     'dropout':0.5,
                     'bilstm_num_layers':1,
                     'stage1':args.stage1,
                     'early_stop': 5,
                    }
        # flesible
        lr = [0.003,]
        reg_rate = [0.00003,]

        nn_config['lr']=lr[args.num]
        nn_config['reg_rate']=reg_rate[args.num]

        data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                       'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_%s.pkl' % args.dn,
                       'batch_size': 50,
                       'conlleval_filePath':'/datastore/liu121/nosqldb2/emnlp_baseline/CoNLLEval/conlleval_%s_%s' % (args.dn, str(args.num))}
        # BBN
        if args.dn == "bbn_bbn_kn":
            nn_config['words_num'] =100
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_bbn_kn/model/'
        elif args.dn =="bbn_cadec":
            nn_config['words_num'] = 200
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec/model/'
        elif args.dn == "bbn_cadec_simple":
            nn_config['words_num'] = 200
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_cadec_simple/model/'
        elif args.dn == "bbn_nvd":
            nn_config['words_num'] = 100
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/bbn_nvd/model/'
        # CONLL
        elif args.dn == "conll_bbn_kn":
            nn_config['words_num'] = 100 # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_bbn_kn/model/'
        elif args.dn == "conll_cadec":
            nn_config['words_num'] = 200  # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec/model/'
        elif args.dn == "conll_cadec_simple":
            nn_config['words_num'] = 200  # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/report/'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/conll_cadec_simple/model/'
        elif args.dn == "conll_nvd":
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