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

# def word_id2txt(X_data,true_labels,pred_labels,id2word,id2label):
#     I = []
#     for i in range(len(X_data)):
#         instance= X_data[i]
#         txt = []
#         true_length = 0
#         for id in instance:
#             if id !=1:
#                 word = id2word[id]
#                 txt.append(word)
#                 true_length+=1
#
#         tlabel=true_labels[i]
#         tlabel_txt = []
#         cur_length = 0
#         for id in tlabel:
#             type_txt = id2label[id]
#             type_txt = type_txt.replace('I-','')
#             tlabel_txt.append(type_txt)
#             cur_length+=1
#             if cur_length>=true_length:
#                 break
#
#         plabel = pred_labels[i]
#         plabel_txt = []
#         cur_length = 0
#         for id in plabel:
#             type_txt = id2label[id]
#             type_txt = type_txt.replace('I-', '')
#             plabel_txt.append(type_txt)
#             cur_length+=1
#             if cur_length>=true_length:
#                 break
#
#         I.append((plabel_txt,tlabel_txt,txt))
#     return I
#
# def conll_eval_file(I,data_config):
#     with open(data_config['conlleval_filePath'],'w+') as f:
#         for t in I:
#             pred_labels_txt = t[0]
#             true_labels_txt = t[1]
#             txt = t[2]
#             for i in range(len(txt)):
#                 f.write(txt[i]+'\t')
#                 f.write(true_labels_txt[i]+'\t')
#                 f.write(pred_labels_txt[i]+'\n')
#                 f.flush()
#             f.write('\n')
#             f.flush()

def main(nn_config,data_config):
    df = DataFeed(data_config)
    nn_config['source_NETypes_num']=df.source_NETypes_num
    nn_config['target_NETypes_num']=df.target_NETypes_num
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

        nn_config = {'lstm_cell_size': 300,
                     'batch_size': 30,
                     'vocabulary_size': 2981402,
                     'feature_dim': 200,
                     'lr': 0.000003,
                     'reg_rate': 0.03,
                     'source_NETypes_num': None,
                     'target_NETypes_num': None,
                     'pad_index': 1, }
        nn_config['stage1'] = args.stage1
        nn_config['mod'] = 1
        nn_config['epoch'] = 3

        k_shot = [['1.0', '1.1', '1.2', '1.3', '1.4', ],
                  ['2.0', '2.1', '2.2', '2.3', '2.4', ],
                  ['4.0', '4.1', '4.2', '4.3', '4.4', ],
                  ['8.0', '8.1', '8.2', '8.3', '8.4', ],
                  ['16.0', '16.1', '16.2', '16.3', '16.4']]
        # k_shot = [['1.0',],
        #           ['2.0',],
        #           ['4.0',],
        #           ['8.0',],
        #           ['16.0',]]
        k_groups = k_shot[args.num]
        for k in k_groups:
            print(k)
            # if args.dn == "bbn_bbn_unk":
            #
            #
            #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
            #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_unk.pkl',
            #                    'k_instances': k,
            #                    'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_bbn_unk/conlleval' + str(k)}
            #     nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_unk/report_bbn'+k
            #     nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_unk/model4/model4.ckpt.meta'
            #     nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_unk/model4/model4.ckpt'
            #     nn_config['words_num'] = 822 #275
            if args.dn == "bbn_bbn_kn":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_kn.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_bbn_kn/conlleval' + str(k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_kn/report_bbn'+k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn/model0/model0.ckpt'
                nn_config['words_num'] = 100
            elif args.dn=="bbn_cadec":
                # args.dn == "cadec"
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_cadec.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec/conlleval' + str(k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_cadec/report_bbn'+k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn200/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn200/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn=="bbn_nvd":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_nvd.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_nvd/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_nvd/report_bbn' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn/model0/model0.ckpt'
                nn_config['words_num'] = 100
            elif args.dn=="bbn_cadec_simple":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_cadec_simple.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_bbn_cadec_simple/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_cadec_simple/report_bbn' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn200/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn200/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn=="conll_bbn_kn":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_kn.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_bbn_kn/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_kn/report_conll' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll/model0/model0.ckpt'
                nn_config['words_num'] = 100
            elif args.dn=="conll_cadec":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_cadec.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_cadec/report_conll' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll200/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll200/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn == "conll_cadec_simple":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_cadec_simple.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_cadec_simple/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_cadec_simple/report_conll' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll200/model0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll200/model0/model0.ckpt'
                nn_config['words_num'] = 200
            elif args.dn=="conll_nvd":
                data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                               'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_nvd.pkl',
                               'k_instances': k,
                               'conlleval_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/conlleval_conll_nvd/conlleval' + str(
                                   k)}
                nn_config['report'] = '/datastore/liu121/nosqldb2/emnlp_baseline/report_nvd/report_conll' + k
                nn_config['model'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll/mode0/model0.ckpt.meta'
                nn_config['model_sess'] = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll/model0/model0.ckpt'
                nn_config['words_num'] = 100
            print('in the main')
            print('epoch:',nn_config['epoch'])
            main(nn_config,data_config)

    # train crf source model and store it
    if args.stage1 == 'True':
        nn_configs = [
            # adjust reg_rate
            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.003,
            #  'reg_rate': 0.0003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },

            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.003,
            #  'reg_rate': 0.00003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },

            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.003,
            #  'reg_rate': 0.000003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },
            #
            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.003,
            #  'reg_rate': 0.0000003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },
            #
            # adjust lr_rate
            {'lstm_cell_size': 150,
             'batch_size': 50,
             'vocabulary_size': 2981402,
             'feature_dim': 200,
             'lr': 0.03,
             'reg_rate': 0.00003,
             'source_NETypes_num': None,
             'target_NETypes_num': None,
             'pad_index': 1, },

            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.003,
            #  'reg_rate': 0.00003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },
            #
            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.0003,
            #  'reg_rate': 0.00003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },

            # {'lstm_cell_size': 150,
            #  'batch_size': 50,
            #  'vocabulary_size': 2981402,
            #  'feature_dim': 200,
            #  'lr': 0.00003,
            #  'reg_rate': 0.00003,
            #  'source_NETypes_num': None,
            #  'target_NETypes_num': None,
            #  'pad_index': 1, },
        ]
        nn_config = nn_configs[args.num]
        nn_config['stage1'] = args.stage1
        nn_config['mod'] = 1000
        nn_config['epoch'] = 1001

        # if args.dn == "bbn_unk":
        #     data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #                    'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_unk.pkl',
        #                    'k_instances': '16.0'}
        #     nn_config['words_num'] =822 #275
        #     report = '/datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_unk/report'
        #     model = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_unk/'
        # elif args.dn == "cadec":
        #     # # args.dn == "cadec"
        #     # data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
        #     #                'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_cadec.pkl',
        #     #                'k_instances': '16.0'}
        #     # nn_config['words_num'] = None
        #     # report = '/datastore/liu121/nosqldb2/emnlp_baseline/report_cadec/report'
        #     # model = '/datastore/liu121/nosqldb2/emnlp_baseline/model_cadec/'
        #     pass
        if args.dn == "bbn_kn":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_kn.pkl',
                           'k_instances': '16.0'}
            nn_config['words_num'] =100
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_kn/report_'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn/'
        elif args.dn =="bbn_kn200":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_kn200.pkl',
                           'k_instances': '16.0'}
            nn_config['words_num'] = 200
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/report_bbn_kn/report200_'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/model_bbn_kn200/'
        elif args.dn == "conll":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll.pkl',
                           'k_instances': '16.0'}
            nn_config['words_num'] = 100 # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/report_conll/report_'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll/'
        elif args.dn == "conll200":
            data_config = {'table_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/table.pkl',
                           'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll200.pkl',
                           'k_instances': '16.0'}
            nn_config['words_num'] = 200  # 113
            report = '/datastore/liu121/nosqldb2/emnlp_baseline/report_conll/report200_'
            model = '/datastore/liu121/nosqldb2/emnlp_baseline/model_conll200/'




        model = model +'/model'+str(args.num)
        path = Path(model)
        if not path.exists():
            path.mkdir()
        model = model +'/model'+str(args.num)+'.ckpt'
        nn_config['model'] = model
        report = report + str(args.num)
        nn_config['report'] = report
        print(str(nn_config))
        main(nn_config, data_config)