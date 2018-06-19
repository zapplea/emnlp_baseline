import sys

sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader
from nerd.data.util.mc_generator import McGenerator
import pickle
import re
import os
import numpy as np
import json

class DataGenerator:
    def __init__(self, data_config):
        self.data_config = data_config
        f = open(self.data_config['table_filePath'], 'rb')
        self.dictionary = pickle.load(f)
        f.close()
        self.max = 30

    def headword(self, mention, tag):
        if tag[-1] == 'POS':
            return mention[-1]
        for i in range(1, len(tag) + 1):
            if tag[-i] == 'NN' or tag[-i] == 'NNP' or tag[-i] == 'NNPS' or tag[-i] == 'NNS' or \
                            tag[-i] == 'NX' or tag[-i] == 'POS' or tag[-i] == 'JJR':
                return mention[-i]
        for i in range(len(tag)):
            if tag[i] == 'NP':
                return mention[i]
        for i in range(1, len(tag) + 1):
            if tag[-i] == '$' or tag[-i] == 'ADJP' or tag[-i] == 'PRN':
                return mention[-i]
        for i in range(1, len(tag) + 1):
            if tag[-i] == 'CD':
                return mention[-i]
        for i in range(1, len(tag) + 1):
            if tag[-i] == 'JJ' or tag[-i] == 'JJS' or tag[-i] == 'RB' or tag[-i] == 'QP':
                return mention[-i]
        return mention[-1]

    def B(self, Bp_element,dic):
        """
        :param mc2tag_dic: 
        :param dic: it is dictionary that map type to id
        :return: 
        """
        Bp_words = {}
        for mention,tag,type in Bp_element:
            head = self.headword(mention, tag)
            tid = dic[type]
            if tid not in Bp_words:
                Bp_words[tid] = [head]
            else:
                Bp_words[tid].append(head)
        return Bp_words

    def type_dic(self,type_set):
        dic={}
        count =0
        for type in type_set:
            dic[type]=count
            count+=1
        return dic

    def map_word2id(self,dataset,dic):
        lc_id=[]
        mc_id=[]
        rc_id=[]
        dataset_id = []
        for ((lc,mc,rc),type) in dataset:
            for word in lc:
                if word in self.dictionary:
                    lc_id.append(self.dictionary[word])
                else:
                    lc_id.append(self.dictionary['#UNK#'])
            for word in mc:
                if word in self.dictionary:
                    mc_id.append(self.dictionary[word])
                else:
                    mc_id.append(self.dictionary['#UNK#'])
            try:
                if len(mc_id)>self.max:
                    raise ValueError()
            except Exception as err:
                print('+++++++++++++',len(mc_id),'+++++++++++++++')
                exit()
            if len(mc_id)<self.max:
                for i in range(self.max-len(mc_id)):
                    mc_id.append(self.dictionary['#PAD#'])
            for word in rc:
                if word in self.dictionary:
                    rc_id.append(self.dictionary[word])
                else:
                    rc_id.append(self.dictionary['#UNK#'])
            t_id = dic[type]
            dataset_id.append([[lc_id,mc_id,rc_id],[t_id]])
        return dataset_id


    def train_data_gen(self):
        train_data = BBNDataReader.readFile(filePath=self.data_config['trainConll_filePath'],are_tweets=self.data_config['are_tweets'])
        train_mc_gen = McGenerator(train_data)
        pos_candidates = train_mc_gen.get_all_positive_candidates()
        Bp_element = []
        type_set=[]
        dataset=[]
        for candidate in pos_candidates:
            lc = candidate.text[0]
            mc = candidate.text[1]
            rc = candidate.text[2]
            type=candidate.label[1][0]
            if type not in type_set:
                type_set.append(type)
            mc_tags = candidate.pos[1]
            Bp_element.append((mc,mc_tags,type))
            dataset.append([[lc,mc,rc],type])
        dic = self.type_dic(type_set)
        dataset_id = self.map_word2id(dataset,dic)
        Bp_words = self.B(Bp_element,dic)

        return dataset_id, Bp_words, dic


    def test_data_gen(self):
        train_data = BBNDataReader.readFile(filePath=self.data_config['testConll_filePath'],
                                            are_tweets=self.data_config['are_tweets'])
        train_mc_gen = McGenerator(train_data)
        pos_candidates = train_mc_gen.get_all_positive_candidates()
        Bp_element = []
        type_set = []
        dataset = []
        for candidate in pos_candidates:
            lc = candidate.text[0]
            mc = candidate.text[1]
            rc = candidate.text[2]
            type = candidate.label[1][0]
            if type not in type_set:
                type_set.append(type)
            mc_tags = candidate.pos[1]
            Bp_element.append((mc, mc_tags, type))
            dataset.append([[lc, mc, rc], type])
        dic = self.type_dic(type_set)
        dataset_id = self.map_word2id(dataset, dic)
        Bp_words = self.B(Bp_element, dic)
        return dataset_id, Bp_words,dic

    def write(self,dataset, Bp_words,):
        pass


if __name__ == "__main__":
    data_gen_configs = [
                        # {'trainConll_filePath': '/datastore/liu121/nosqldb2/bbn_unk/data_test_draw',
                        #  'train_jsonPath': '/datastore/liu121/nosqldb2/bbn_unk/draw_unk.json',
                        #  'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/train_unk.pkl',
                        #
                        #  'testConll_filePath': '/datastore/liu121/nosqldb2/bbn_unk/data_test_eval',
                        #  'test_jsonPath': '/datastore/liu121/nosqldb2/bbn_unk/eval_unk.json',
                        #  'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/test_unk.pkl',
                        #
                        #  'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                        #  't2id':'/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/t2id_bbn_unk.pkl',
                        #  'iteration': 5,
                        #  'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_unk/Bp_bbn_unk',
                        #  'are_tweets': False
                        #  },
                        {'trainConll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_draw',
                         'train_jsonPath':'',
                         'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/train_kn',

                         'testConll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_eval',
                         'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/test_kn.pkl',

                         'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                         't2id': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/t2id_bbn_kn.pkl',
                         'iteration': 5,
                         'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/Bp_bbn_kn',
                         'are_tweets': False
                         },
                        # {'trainConll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_draw',
                        #  'train_jsonPath': '/datastore/liu121/nosqldb2/cadec/json/draw.json',
                        #  'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/train',
                        #
                        #  'testConll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_eval',
                        #  'test_jsonPath': '/datastore/liu121/nosqldb2/cadec/json/eval.json',
                        #  'testData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/test.pkl',
                        #
                        #  'table_filePath': '/datastore/liu121/nosqldb2/ner_zero/data/table.pkl',
                        #  't2id': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/t2id_cadec.pkl',
                        #  'iteration': 5,
                        #  'Bp_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_cadec/Bp_cadec',
                        #  'are_tweets': False
                        #  },
                        ]
    for data_gen_config in data_gen_configs:
        dg = DataGenerator(data_gen_config)
        dg.train_data_gen()
        dg.test_data_gen()

