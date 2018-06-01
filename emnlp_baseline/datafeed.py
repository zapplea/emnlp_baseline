import gensim
import numpy as np
from sklearn.utils import check_array
import pickle
import random

class DataFeed:
    def __init__(self,data_config):
        self.data_config = data_config
        self.table = self.table_load()

        f = open(self.data_config['pkl_filePath'],'rb')
        data_dic = pickle.load(f)
        self.source_data = data_dic['source_data']
        # random.shuffle(self.source_data)
        self.source_NETypes_num = data_dic['source_NETypes_num']
        self.target_train_data = data_dic['target_train_data']
        self.target_test_data =data_dic['target_test_data']
        self.target_NETypes_num = data_dic['target_NETypes_num']
        self.id2label_dic= data_dic['id2label_dic']

    def table_load(self):
        # vecfpath = self.data_config['table_filePath']
        # word_embed = gensim.models.KeyedVectors.load_word2vec_format(vecfpath, binary=False, datatype=np.float32)
        # embed_mat = word_embed.syn0
        # embed_mat = check_array(embed_mat, dtype='float32', order='C')
        f = open(self.data_config['table_filePath'],'rb')
        dictionary = pickle.load(f)
        del dictionary
        embed_mat = pickle.load(f)
        return embed_mat

    def table_generator(self):
        return self.table

    def source_data_generator(self,mode,**kwargs):
        if mode == 'train':
            batch_num = kwargs['batch_num']
            batch_size = kwargs['batch_size']
            data_temp = self.source_data[:]
        elif mode == 'test':
            data_temp = self.source_data[-1000:]

        if mode == 'train':
            train_size = len(data_temp)
            start = batch_num * batch_size % train_size
            end = (batch_num * batch_size + batch_size) % train_size
            if start < end:
                batch = data_temp[start:end]
            elif start >= end:
                batch = data_temp[start:]
                batch.extend(data_temp[0:end])
        else:
            batch = data_temp
        X = []
        Y_ = []
        # for instance in batch:
        #     X.append(np.array(instance[0], dtype='int32'))
        #     Y_.append(np.array(instance[1], dtype='int32'))

        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        # print('X len: ',str(len(X)))
        # print('Y_ len: ',str(len(Y_)))
        # print('X: ')
        # for x in X:
        #     print('type: ',type(x),' len: ',str(len(x)),'\n')
        # np.array(X)
        # print('Y_: ')
        # for y in Y_:
        #     print('type: ', type(y), ' len: ', str(len(y)), '\n')
        # np.array(Y_)
        # print('====================')
        return (np.array(X,dtype='int32'), np.array(Y_,dtype='int32'))

    def target_data_generator(self,mode,**kwargs):
        if mode == 'train':
            batch_num = kwargs['batch_num']
            batch_size= kwargs['batch_size']
            data_temp = self.target_train_data[self.data_config['k_instances']]
        else:
            data_temp = self.target_test_data

        if mode == 'train':
            train_size = len(data_temp)
            start = batch_num * batch_size % train_size
            end = (batch_num * batch_size + batch_size) % train_size
            if start < end:
                batch = data_temp[start:end]
            elif start >= end:
                batch = data_temp[start:]
                batch.extend(data_temp[0:end])
        else:
            batch = data_temp
        X = []
        Y_ = []
        for instance in batch:
            X.append(np.array(instance[0], dtype='int32'))
            Y_.append(np.array(instance[1], dtype='int32'))

        # during validation and test, to avoid errors are counted repeatedly,
        # we need to avoid the same data sended back repeately
        return (np.array(X, dtype='int32'), np.array(Y_, dtype='int32'))

    def id2label_generator(self):
        return self.id2label_dic


class DataFeedTest:
    def __init__(self):
        pass