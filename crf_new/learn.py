import os
import pwd
import sys

global cur_user_name

cur_user_name = pwd.getpwuid(os.getuid()).pw_name
if cur_user_name == "liu121":
    sys.path.append('/home/liu121/emnlp_baseline')

import argparse
from pathlib import Path

from crf_new.datafeed import DataFeed
from crf_new.classifier import Classifier
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
    # I = mt.word_id2txt(X_data,true_labels,pred_labels,id2label_dic)
    # print('output')
    # mt.conll_eval_file(I)
    # print('finish output')

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--num',type=int)
    args = parser.parse_args()

    nn_configs = [
        {'lstm_cell_size': 150,
         'vocabulary_size': 2981402,
         'feature_dim': 200,
         'lr': 0.003,
         'reg_rate': 0.00003,
         'source_NETypes_num': None,
         'target_NETypes_num': None,
         'pad_index': 1,
         'epoch_stage1':150,},
        {'lstm_cell_size': 150,
         'vocabulary_size': 2981402,
         'feature_dim': 200,
         'lr': 0.0003,
         'reg_rate': 0.00003,
         'source_NETypes_num': None,
         'target_NETypes_num': None,
         'pad_index': 1,
         'epoch_stage1': 150, },
        {'lstm_cell_size': 150,
         'vocabulary_size': 2981402,
         'feature_dim': 200,
         'lr': 0.00003,
         'reg_rate': 0.00003,
         'source_NETypes_num': None,
         'target_NETypes_num': None,
         'pad_index': 1,
         'epoch_stage1': 150, },
        {'lstm_cell_size': 150,
         'vocabulary_size': 2981402,
         'feature_dim': 200,
         'lr': 0.000003,
         'reg_rate': 0.00003,
         'source_NETypes_num': None,
         'target_NETypes_num': None,
         'pad_index': 1,
         'epoch_stage1': 150, },
        {'lstm_cell_size': 150,
         'vocabulary_size': 2981402,
         'feature_dim': 200,
         'lr': 0.0000003,
         'reg_rate': 0.00003,
         'source_NETypes_num': None,
         'target_NETypes_num': None,
         'pad_index': 1,
         'epoch_stage1': 150, },
    ]
    nn_config = nn_configs[args.num]
    nn_config['tfb_filePath'] = '/datastore/liu121/tfb/emnlp_baseline/crf_new/summ_'+args.num
    nn_config['words_num'] = 100
    nn_config['dropout']=0.5
    nn_config['bilstm_num_layers']=1
    # BBN
    data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                   'pkl_filePath': '/datastore/liu121/nosqldb2/crf_new/data/data_conll_bbn_kn.pkl',
                   'batch_size':50,
                   'conlleval_filePath': '/datastore/liu121/nosqldb2/crf_new/conlleval'+str(args.num)}


    report = '/datastore/liu121/nosqldb2/crf_new/report/'
    model = '/datastore/liu121/nosqldb2/crf_new/model/'

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