import gensim
import numpy as np
from sklearn.utils import check_array
import pickle
import random

class Dataset:
    def __init__(self,dataset,**kwargs):
        if len(kwargs)==0:
            self.batch_size=len(dataset)
        else:
            self.batch_size=kwargs['batch_size']
        self.dataset=dataset
        self.count=0

    def __iter__(self):
        return self

    def __next__(self):
        if self.count<len(self.dataset):
            if self.count+self.batch_size<len(self.dataset):
                batch=self.dataset[self.count:self.count+self.batch_size]
            else:
                batch=self.dataset[self.count:]
            self.count+=self.batch_size
        else:
            raise StopIteration
        X=[]
        Y_=[]
        for instance in batch:
            X.append(instance[0])
            Y_.append(instance[1])
        # print(len(X))
        # print(len(Y_))
        for i in [1000,5005,10000]:
            print(i)
            X_new=np.array(X[:i],dtype='int32')
        # exit()
        # Y_=np.array(Y_,dtype='int32')
        print('successfull')
        exit()
        return X,Y_

class DataFeed:
    def __init__(self,data_config):
        self.data_config = data_config
        self.table = self.table_load()

        f = open(self.data_config['pkl_filePath'],'rb')
        data_dic = pickle.load(f)
        self.source_data = data_dic['source_data']
        # random.shuffle(self.source_data)
        self.source_NETypes_num = data_dic['source_NETypes_num']
        self.source_id2label_dic=data_dic['source_id2label_dic']

        self.target_train_data = data_dic['target_train_data']
        self.target_test_data =data_dic['target_test_data']
        self.target_NETypes_num = data_dic['target_NETypes_num']
        self.target_id2label_dic= data_dic['target_id2label_dic']

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

    def source_data_generator(self,mode):
        if mode == 'train':
            dataset = Dataset(self.source_data[:], batch_size=self.data_config['batch_size'])
        else:
            dataset = Dataset(self.source_data[-1000:])
        return dataset

    def target_data_generator(self,mode):
        if mode == 'train':
            dataset=Dataset(self.target_train_data[self.data_config['k_instances']],batch_size=self.data_config['batch_size'])
        else:
            #self.check(self.target_train_data[self.data_config['k_instances']])
            dataset = Dataset(self.target_test_data)
        return dataset

    def check(self,dataset):
        print('type of dataset: ',type(dataset))
        for txt,ty in dataset:
            print(len(txt),':',len(ty))
        print('====================')
        exit()

    def source_id2label_generator(self):
        return self.source_id2label_dic

    def target_id2label_generator(self):
        return self.target_id2label_dic


class DataFeedTest:
    def __init__(self):
        pass