import numpy as np
import pickle

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
        for i in [1040,1042,1045,1048,1050]:
            print(i)
            X_new=np.array(X[:i],dtype='int32')
        # exit()
        # Y_=np.array(Y_,dtype='int32')
        print('successfull')
        exit()
        return X,Y_

class Check:
    def __init__(self,data_config):
        self.data_config = data_config
        f = open(self.data_config['pkl_filePath'], 'rb')
        data_dic = pickle.load(f)
        self.target_test_data = data_dic['target_test_data']

    def target_data_generator(self):

        dataset = Dataset(self.target_test_data)

        return dataset

if __name__ == "__main__":
    data_config = {'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_bbn_kn.pkl'}
    ch = Check(data_config)
    dataset = ch.target_data_generator()
    for X,Y in dataset:
        print('successful')