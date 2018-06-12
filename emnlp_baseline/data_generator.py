import sys
sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader

import gensim
import numpy as np
from sklearn.utils import check_array
import pickle
import random
import json

class DataGenerator:
    def __init__(self,data_config,table,dictionary):
        self.data_config = data_config
        self.table= table
        self.dictionary = dictionary

    def conll_data_reader(self,filePath):
        data = BBNDataReader.readFile(filePath=filePath)
        return data

    def nn_data_generator(self,data,labels_dic):
        data_len=len(data.text)
        # train_data = [[text,labels], ...]
        nn_data = []
        labels_num = 1
        for i in range(data_len):
            text = data.text[i]
            labels = data.labels[i]
            if len(text)>self.data_config['max_len']:
                continue
            if len(text)>self.data_config['max_len']:
                exit()
            x = []
            for word in text:
                if word not in self.dictionary:
                    x.append(self.dictionary['#UNK#'])
                else:
                    x.append(self.dictionary[word])
            while len(x) < self.data_config['max_len']:
                x.append(self.dictionary['#PAD#'])
            y_ = []
            for label in labels:
                if label == 'OTHER':
                    label='O'
                if label not in labels_dic:
                    labels_dic[label] = labels_num
                    labels_num += 1
                #y_.append(labels_dic[label])
                y_.append(label)
            while len(y_) < self.data_config['max_len']:
                y_.append('O')
            # print('labels: \n',labels_dic)
            # print('x:\n',x)
            # print('y_:\n',y_)
            # exit()
            nn_data.append((x, y_))
        return nn_data,labels_num,labels_dic

    def target_nn_data_generator(self,data,labels_dic,labels_num):
        data_len=len(data.text)
        # train_data = [[text,labels], ...]
        nn_data = []
        for i in range(data_len):
            text = data.text[i]
            labels = data.labels[i]
            if len(text)>self.data_config['max_len']:
                continue
            if len(text)>self.data_config['max_len']:
                exit()
            x = []
            for word in text:
                if word not in self.dictionary:
                    x.append(self.dictionary['#UNK#'])
                else:
                    x.append(self.dictionary[word])
            while len(x) < self.data_config['max_len']:
                x.append(self.dictionary['#PAD#'])
            y_ = []
            for label in labels:
                if label == 'OTHER':
                    label='O'
                if label not in labels_dic:
                    labels_dic[label] = labels_num
                    labels_num += 1
                # y_.append(labels_dic[label])
                y_.append(label)
            while len(y_) < self.data_config['max_len']:
                y_.append('O')
            # print('labels: \n',labels_dic)
            # print('x:\n',x)
            # print('y_:\n',y_)
            # exit()
            nn_data.append((x, y_))
        return nn_data,labels_num,labels_dic

    def source_data_generator(self):
        data = self.conll_data_reader(self.data_config['source_Conll_filePath'])
        labels_dic = {'O':0}
        source_data, source_labels_num, labels_dic = self.nn_data_generator(data,labels_dic)
        random.shuffle(source_data)
        return source_data,source_labels_num,labels_dic

    def check(self,data, label_dic):
        print('check: \n')
        ids=set()
        for label in label_dic:
            ids.add(label_dic[label])
        for instance in data:
            y=instance[1]
            for label in y:
                if label not in ids:
                    print(str(label))

    def target_data_gnerator(self):
        target_draw_data = self.conll_data_reader(self.data_config['target_train_Conll_filePath'])
        target_eval_data = self.conll_data_reader(self.data_config['target_test_Conll_filePath'])

        labels_dic = {'O':0}
        labels_num=1
        target_train_data, labels_num, labels_dic = self.target_nn_data_generator(target_draw_data,labels_dic,labels_num)
        #self.check(target_train_data,labels_dic)
        #print('target_train_data length:{}\n'.format(str(len(target_train_data))))
        target_test_data, labels_num, labels_dic = self.target_nn_data_generator(target_eval_data,labels_dic,labels_num)
        #self.check(target_test_data, labels_dic)

        return target_train_data,target_test_data,labels_num, labels_dic

    # TODO: there are two kindes of target data: draw and eval. draw is used to train, and eval to predict.
    def target_data_split(self,data):
        """
        
        :param data: [[text_ids,labels_id],...]
        :return: 
        """
        sample = {}
        train_conductor = json.load(open(self.data_config['target_train_jsonPath']))
        for i in range(self.data_config['groups_num']):
            for j in self.data_config['instances_num']:
                affix = str(j) + '.' + str(i)
                sample[affix]=[]
                t2indexes = train_conductor[str(j)][str(i)]
                for key in t2indexes:
                    for instance in t2indexes[key]:
                        id = instance[0]
                        if id>=len(data):
                            # print(affix,' ',str(id))
                            continue
                        instance = data[id]
                        sample[affix].append(instance)
        # type2instances={}
        # data_len = len(data)
        # for i in range(data_len):
        #     x,y_ = data[i]
        #     labels = set(y_)
        #     for label in labels:
        #         if label not in type2instances:
        #             type2instances[label] = [(x,y_)]
        #         else:
        #             type2instances[label].append((x,y_))
        #         random.shuffle(type2instances[label])
        # sample = {}
        # for k in [1,2,4,8,16]:
        #     sample[k]=[]
        #     for label in type2instances:
        #         # [(x,y_),...]
        #         instances = type2instances[label]
        #         for i in range(k):
        #             sample[k].append(instances[i])
        return sample

    def id2label(self,labels_dic):
        """
        
        :param labels_dic: {label txt: id} 
        :return: (id : label txt)
        """
        id2label_dic = {}
        for label in labels_dic:
            id = labels_dic[label]
            id2label_dic[id] = label
        return id2label_dic

    def write(self,data):
        with open(self.data_config['pkl_filePath'],'wb') as f:
            pickle.dump(data,f)
            f.flush()

    def check2(self,id2labels,data):
        print('\ncheck target train:')
        # for instance in data:
        #     print(str(instance)+'\n')
        print('id2label_dic: \n', str(id2labels))
        for group in data:
            instances = data[group]
            for instance in instances:
                y = instance[1]
                for label in y:
                    if label not in id2labels:
                        print('label: ',str(label),' type:',type(label))
                        print('instance_text:\n ', instance[0])
                        print('instance_label:\n ',y)

    def check3(self,id2labels,data):
        print('\ncheck target test')
        print('id2label_dic: \n', str(id2labels))
        for instance in data:
            y = instance[1]
            for label in y:
                if label not in id2labels:
                    print('label: ', str(label), ' type:', type(label))
                    print('instance_text:\n ', instance[0])
                    print('instance_label:\n ', y)


    def labels_share_dic(self,source_labels_dic,target_labels_dic):
        new_target_labels_dic={}
        new_source_labels_dic={}
        labels_num=0
        for key in source_labels_dic:
            if key in target_labels_dic:
                new_target_labels_dic[key]=labels_num
                new_source_labels_dic[key]=labels_num
                labels_num+=1

        labels_num=len(new_source_labels_dic)
        for key in source_labels_dic:
            if key not in new_source_labels_dic:
                new_source_labels_dic[key]=labels_num
                labels_num+=1

        labels_num=len(new_target_labels_dic)
        for key in target_labels_dic:
            if key not in new_target_labels_dic:
                new_target_labels_dic[key]=labels_num
                labels_num+=1
        return new_source_labels_dic,new_target_labels_dic

    def labels_map(self,data,labels_dic):
        id_data=[]
        for instance in data:
            labels = instance[1]
            text = instance[0]
            labels_id=[]
            for label in labels:
                labels_id.append(labels_dic[label])
            id_data.append((text,labels_id))
        return id_data

    def main(self):
        # data = BBNDataReader.readFile(filePath=self.data_config['target_train_Conll_filePath'])
        # print('test_draw_length:{}\n'.format(str(len(data.text))))
        data={}
        source_data, source_labels_num,source_labels_dic = self.source_data_generator()


        target_train_data, target_test_data, target_labels_num, target_labels_dic = self.target_data_gnerator()

        source_labels_dic,target_labels_dic=self.labels_share_dic(source_labels_dic,target_labels_dic)
        print('source:\n ',str(source_labels_dic))
        print('target:\n ',str(target_labels_dic))
        # labels to id
        source_data = self.labels_map(source_data,source_labels_dic)
        data['source_data'] = source_data
        data['source_NETypes_num'] = source_labels_num

        target_train_data = self.labels_map(target_train_data,target_labels_dic)
        target_test_data = self.labels_map(target_test_data, target_labels_dic)
        # print(labels_dic)
        id2label_dic = self.id2label(target_labels_dic)
        data['id2label_dic'] = id2label_dic
        data['target_NETypes_num'] = target_labels_num
        target_train_sample = self.target_data_split(target_train_data)
        # self.check2(id2label_dic,target_train_sample)
        target_test_sample = target_test_data
        # self.check3(id2label_dic,target_test_sample)
        data['target_train_data'] = target_train_sample
        data['target_test_data'] = target_test_sample
        # for line in target_train_sample:
        #     print(line)
        # exit()
        self.write(data)

def table_vec_and_dic():
    vecfpath = '/datastore/liu121/nosqldb2/skipgram_v200_w5'
    word_embed = gensim.models.KeyedVectors.load_word2vec_format(vecfpath, binary=False, datatype=np.float32)
    embed_mat = word_embed.syn0
    vocabulary = word_embed.index2word
    embed_mat = check_array(embed_mat, dtype='float32', order='C')
    dictionary = {}
    for i in range(len(vocabulary)):
        dictionary[vocabulary[i]] = i
    return embed_mat,dictionary

if __name__ == "__main__":
    data_configs =[
                   # {'max_len':822,#275
                   #  'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_unk.pkl',
                   #  'source_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_unk/data_train',
                   #  'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_unk/data_test_draw',
                   #  'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_unk/data_test_eval',
                   #  'target_train_jsonPath': '/datastore/liu121/nosqldb2/bbn_unk/draw_unk.json',
                   #  'groups_num':5,
                   #  'instances_num':[1,2,4,8,16]},

                    {'max_len': 100,  # bbn_bbn_kn
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_bbn_kn.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_train.txt',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_draw.txt',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_eval.txt',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/bbn_kn/draw_kn.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},
                    {'max_len': 100,  # conll_bbn_kn
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_bbn_kn.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/conll2003/conll_train',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_draw.txt',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_eval.txt',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/bbn_kn/draw_kn.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},


                    {'max_len': 200,  # 315 bbn_cadec
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_train.txt',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_draw',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_eval',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/cadec/json/draw.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},
                    {'max_len': 200,  # conll_cadec
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/conll2003/conll_train',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_draw',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_eval',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/cadec/json/draw.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},


                    {'max_len': 200,  # bbn_cadec_simple
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_cadec_simple.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_train.txt',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/cadec_simple/cadec_draw.txt',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/cadec_simple/cadec_eval.txt',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/cadec_simple/test_draw_cadec_cache_config.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},
                    {'max_len': 200,  # conll_cadec_simple
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_cadec_simple.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/conll2003/conll_train',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/cadec_simple/cadec_draw.txt',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/cadec_simple/cadec_eval.txt',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/cadec_simple/test_draw_cadec_cache_config.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},

                    {'max_len': 100,  #bbn_nvd
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_bbn_nvd.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_train.txt',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/nvd/nvd_test_draw',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/nvd/nvd_test_eval',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/nvd/draw.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},
                    {'max_len': 100,  #conll_nvd
                     'pkl_filePath': '/datastore/liu121/nosqldb2/emnlp_baseline/data/data_conll_nvd.pkl',
                     'source_Conll_filePath': '/datastore/liu121/nosqldb2/conll2003/conll_train',
                     'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/nvd/nvd_test_draw',
                     'target_test_Conll_filePath': '/datastore/liu121/nosqldb2/nvd/nvd_test_eval',
                     'target_train_jsonPath': '/datastore/liu121/nosqldb2/nvd/draw.json',
                     'groups_num': 5,
                     'instances_num': [1, 2, 4, 8, 16]},



                   ]
    table,dictionary = table_vec_and_dic()
    for data_config in data_configs:
        print('================================')
        print(data_config['pkl_filePath'])
        dg = DataGenerator(data_config,table,dictionary)
        dg.main()