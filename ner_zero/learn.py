import sys
sys.path.append('/home/liu121/dlnlp')

#from pathlib import Path
import argparse

from nerd.ner_zero.datafeed import DataFeed
from nerd.ner_zero.ner_zero import NerZero
from nerd.ner_zero.metrics import Metrics

def main(nn_config,data_config):
    df = DataFeed(data_config)
    nn = NerZero(nn_config,df=df)
    if nn_config['stage1'] == "True":
        nn.train()
    else:
        true_labels, pred_labels, X_data = nn.train()
        id2label = df.id2label_generator()
        dictionary = df.dictionary_generator()
        mt = Metrics(data_config)
        mt.conlleval(true_labels,pred_labels,X_data,dictionary,id2label)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num',type=int)
    parser.add_argument('--stage1', type=str)
    parser.add_argument('--dn', type=str)
    args = parser.parse_args()

    # path = Path('/datastore/che313/yibing_data/nosqldb/coling_baseline/report')
    # if not path.exists():
    #     path.mkdir()
    if args.stage1 == "False":
        nn_config = {'feature_vec_dim': 1000,
                     'words_num': 9,  # 10
                     'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
                     'is_B_trainable': False,
                     'batch_size': 18,
                     'lr': 0.003,
                     'lambda': 0.00003,
                     'pred_threshold': 0,}
        nn_config['stage1'] = args.stage1
        nn_config['mod'] = 100
        nn_config['epoch'] = 1001
        nn_config['gpu'] = '/gpu:0'


        k_shot =[['1.0','1.1','1.2','1.3','1.4'],
                 ['2.0','2.1','2.2','2.3','2.4'],
                 ['4.0','4.1','4.2','4.3','4.4'],
                 ['8.0','8.1','8.2','8.3','8.4'],
                 ['16.0','16.1','16.2','16.3','16.4']]
        k_groups = k_shot[args.num]
        for k in k_groups:

            if args.dn == "bbn_unk":
                # data_config = {
                #     'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/train_unk'+k+'.pkl',
                #     'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/test_unk.pkl',
                #     'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/Bp_bbn_unk'+k+'.pkl',
                #     'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                #     'with_padding': nn_config['words_num'],
                #     'labels_num': nn_config['labels_num'],
                #     #'t2id': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/t2id_bbn_unk.pkl',
                #     'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_bbn_unk/conlleval'+k}
                # nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_bbn_unk/report' + k
                # nn_config['labels_num'] = 20
                pass
            elif args.dn == "bbn_bbn_kn":
                data_config = {
                    'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/train_kn'+k+'.pkl',
                    'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/test_kn.pkl',
                    'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/Bp_bbn_kn'+k+'.pkl',
                    'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                    'with_padding': nn_config['words_num'],
                    'labels_num': nn_config['labels_num'],
                    #'t2id': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/t2id_bbn_kn.pkl',
                    'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_bbn_kn/conlleval'+k}
                nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_bbn_kn/report' + k
                nn_config['labels_num'] = 20
            elif args.dn == "bbn_cadec":
                # cadec
                data_config = {
                    'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/train'+k+'.pkl',
                    'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/test.pkl',
                    'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/Bp_cadec'+k+'pkl',
                    'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                    'with_padding': nn_config['words_num'],
                    'labels_num': nn_config['labels_num'],
                    #'t2id': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/t2id_cadec.pkl',
                    'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_cadec/conlleval'+k}
                nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_cadec/report' + k
                nn_config['labels_num'] = 6
            elif args.dn == "bbn_cadec_simple":
                # cadec_simple
                data_config = {
                    'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/train'+k+'.pkl',
                    'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/test.pkl',
                    'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/Bp_cadec'+k+'pkl',
                    'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                    'with_padding': nn_config['words_num'],
                    'labels_num': nn_config['labels_num'],
                    #'t2id': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/t2id_cadec.pkl',
                    'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_cadec/conlleval'+k}
                nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_cadec/report' + k
                nn_config['labels_num'] = 6
            elif args.dn == "bbn_nvd":
                # nvd
                data_config = {
                    'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/train'+k+'.pkl',
                    'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/test.pkl',
                    'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/Bp_cadec'+k+'pkl',
                    'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                    'with_padding': nn_config['words_num'],
                    'labels_num': nn_config['labels_num'],
                    #'t2id': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/t2id_cadec.pkl',
                    'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_cadec/conlleval'+k}
                nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_cadec/report' + k
                nn_config['labels_num'] = 6

            main(nn_config,data_config)


    # Choose parameter
    if args.stage1 == "True":
        nn_configs = [
            # lr
            {'feature_vec_dim': 1000,
             'words_num': 9,  # 9
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.03,
             'lambda': 0.00003,
             'pred_threshold': 0, },

            {'feature_vec_dim': 1000,
             'words_num': 9,  # 10
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.003,
             'lambda': 0.00003,
             'pred_threshold': 0, },

            {'feature_vec_dim': 1000,
             'words_num': 9,  # 11
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.0003,
             'lambda': 0.00003,
             'pred_threshold': 0, },

            {'feature_vec_dim': 1000,
             'words_num': 9,  # 11
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.00003,
             'lambda': 0.00003,
             'pred_threshold': 0, },

            # lambda
            {'feature_vec_dim': 1000,
             'words_num': 9,  # 10
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.003,
             'lambda': 0.0003,
             'pred_threshold': 0, },

            {'feature_vec_dim': 1000,
             'words_num': 9,  # 10
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.003,
             'lambda': 0.00003,
             'pred_threshold': 0, },

            {'feature_vec_dim': 1000,
             'words_num': 9,  # 10
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.003,
             'lambda': 0.000003,
             'pred_threshold': 0, },

            {'feature_vec_dim': 1000,
             'words_num': 9,  # 10
             'label_embed_dim': 200,  # W = (A^(d*M))^T * B^d*N; M: feature vec dim; N: number of type
             'is_B_trainable': False,
             'batch_size': 18,
             'lr': 0.003,
             'lambda': 0.0000003,
             'pred_threshold': 0, },
        ]
        nn_config = nn_configs[args.num]
        nn_config['stage1'] = args.stage1
        nn_config['epoch'] = 1001
        nn_config['mod'] = 100
        nn_config['gpu'] = '/gpu:0'


        if args.dn == "bbn_unk":
            # data_config ={
            #                 'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/train_unk16.0.pkl',
            #                 'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/test_unk.pkl',
            #                 'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/Bp_bbn_unk16.0.pkl',
            #                 'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
            #                 'with_padding': nn_config['words_num'],
            #                 'labels_num': nn_config['labels_num'],
            #                 't2id': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/t2id_bbn_unk.pkl',
            #                 'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_bbn_unk/conlleval16.0'}
            # nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_bbn_unk/report' + str(args.num)
            # nn_config['labels_num'] = 20
            pass
        elif args.dn == "bbn_kn":
            data_config ={
                            'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/train_kn16.0.pkl',
                            'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/test_kn.pkl',
                            'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/Bp_bbn_kn16.0.pkl',
                            'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                            'with_padding': nn_config['words_num'],
                            'labels_num': nn_config['labels_num'],
                            't2id': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/t2id_bbn_kn.pkl',
                            'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_bbn_kn/conlleval16.0'}
            nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_bbn_kn/report' + str(args.num)
            nn_config['labels_num'] = 20
        elif args.dn == "cadec":
            # # cadec
            # data_config = {
            #                 'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/train16.0.pkl',
            #                 'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/test.pkl',
            #                 'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/Bp_cadec16.0.pkl',
            #                 'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
            #                 'with_padding': nn_config['words_num'],
            #                 'labels_num': nn_config['labels_num'],
            #                 't2id': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/t2id_cadec.pkl',
            #                 'conlleval_filePath': '/datastore/liu121/nosqldb2/ner_zero/conlleval_cadec/conlleval16.0'}
            # nn_config['report_path'] = '/datastore/che313/yibing_data/nosqldb/coling_baseline/report_cadec/report' + str(args.num)
            # nn_config['labels_num'] = None
            pass
        elif args.dn == "conll":
            pass
        main(nn_config, data_config)