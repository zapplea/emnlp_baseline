import sys

sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader
import pickle
import re
import os
import numpy as np
import json


class Node:
    def __init__(self):
        self.text = '__empty__'
        self.type = '__empty__'
        self.parent = '__empty__'
        self.tag = '__empty__'
        self.left_child = '__empty__'
        self.right_child = '__empty__'
        self.pos = '__empty__'

    def __str__(self):
        return self.text


class DataGenerator:
    def __init__(self, data_config):
        self.data_config = data_config
        f = open(self.data_config['table_filePath'], 'rb')
        self.dictionary = pickle.load(f)
        f.close()
        self.max = 0

    def tokenConvert(self, token):
        # convert to lower case
        token = token.lower()
        # convert digits
        if (len(re.findall(r".*\d.*", token)) != 0):
            token = re.sub(r"\d", " ", token)
            tk = ""
            count = 0
            for char in token:
                if char == " ":
                    count += 1
                else:
                    if count == 0:
                        tk += char  # end of digits
                    else:
                        tk = tk + "NUM" + str(count) + char
                        count = 0
            if count > 0:
                tk += "NUM" + str(count)
        else:
            tk = token  # no digit in token
        if tk not in self.dictionary:
            tk = '#UNK#'
        return tk

    def generate_tree(self, sentence, labels, tag):
        tree = []
        for i in range(len(sentence)):
            word = sentence[i]
            word = self.tokenConvert(word)
            label = labels[i]
            node = Node()
            node.text = word
            node.type = label
            node.tag = tag[i]
            node.pos = i
            # print('node{}: {}'.format(str(i),node.text))
            if len(tree) == 0:
                tree.append(node)
            else:
                parent = tree[-1]
                node.parent = parent
                if parent.type == node.type:
                    parent.right_child = node
                else:
                    parent.left_child = node
                tree.append(node)
        return tree

    def visit_tree(self, tree):
        mct = []
        mention = []
        lcontxt = []
        rcontxt = []
        mention_tag = []
        mention_pos = []
        type_txt = ''
        for i in range(len(tree)):
            node = tree[i]
            parent = node.parent
            # the first node
            # if parent=='__empty__':
            if parent == '__empty__':
                if node.type != 'OTHER':
                    mention.append(node.text)
                    mention_tag.append(node.tag)
                    mention_pos.append(node.pos)
                    type_txt = node.type
                    type_txt = '/' + '/'.join(type_txt.split(':'))
                    lcontxt.append('#PAD#')
                    lcontxt.append('#PAD#')
            elif parent.right_child == '__empty__':
                if type_txt != '':
                    # fulfill the right context
                    # print('current{}: {}'.format(str(i),tree[i].text))
                    if i < len(tree):
                        rcontxt.append(tree[i].text)
                        # print('right{}: {}'.format(str(i),tree[i].text))
                        if i + 1 < len(tree):
                            rcontxt.append(tree[i + 1].text)
                            # print('right{}: {}'.format(str(i+1),tree[i+1].text))
                        else:
                            rcontxt.append('#PAD#')
                    else:
                        rcontxt.append('#PAD#')
                        rcontxt.append('#PAD#')
                    if len(mention) > self.max:
                        self.max = len(mention)
                    mct.append(
                        {'mention': ' '.join(mention), 'lcontxt': ' '.join(lcontxt), 'rcontxt': ' '.join(rcontxt),
                         'type': type_txt, 'mention_tag': mention_tag, 'mention_start': mention_pos[0],
                         'mention_end': mention_pos[-1]})
                mention = []
                mention_tag = []
                mention_pos = []
                lcontxt = []
                rcontxt = []
                type_txt = ''
                if node.type != 'OTHER':
                    mention.append(node.text)
                    mention_tag.append(node.tag)
                    mention_pos.append(node.pos)
                    type_txt = node.type
                    type_txt = '/' + '/'.join(type_txt.split(':'))
                    if i - 1 >= 0:
                        if i - 2 >= 0:
                            lcontxt.append(tree[i - 2].text)
                        else:
                            lcontxt.append('#PAD#')
                        lcontxt.append(tree[i - 1].text)
                    else:
                        lcontxt.append('#PAD#')
                        lcontxt.append('#PAD#')
            elif parent.left_child == '__empty__':
                if node.type != 'OTHER':
                    if type_txt == '':
                        mention.append(node.text)
                        mention_tag.append(node.tag)
                        mention_pos.append(node.pos)
                        type_txt = node.type
                        type_txt = '/' + '/'.join(type_txt.split(':'))
                        if i - 1 >= 0:
                            if i - 2 >= 0:
                                lcontxt.append(tree[i - 2].text)
                            else:
                                lcontxt.append('#PAD#')
                            lcontxt.append(tree[i - 1].text)
                        else:
                            lcontxt.append('#PAD#')
                            lcontxt.append('#PAD#')
                    else:
                        mention.append(node.text)
                        mention_tag.append(node.tag)
                        mention_pos.append(node.pos)
            if i == len(tree) - 1:
                if type_txt != '':
                    rcontxt.append('#PAD#')
                    rcontxt.append('#PAD#')
                    if len(mention) > self.max:
                        self.max = len(mention)
                    mct.append(
                        {'mention': ' '.join(mention), 'lcontxt': ' '.join(lcontxt), 'rcontxt': ' '.join(rcontxt),
                         'type': type_txt, 'mention_tag': mention_tag, 'mention_start': mention_pos[0],
                         'mention_end': mention_pos[-1]})
        return mct

    def mct_gen(self, data, conductor):
        """

        :param data: 
        :return: {(mention, lcontxt, rcontxt):[type,...],...}
        """
        # # {mention:[(lcontxt,rcontxt),...],...}
        # mc={}
        # # {mention:[type,...],...}
        # mt={}
        type_list = []
        mc2t_dic = {}
        mc2tag_dic = {}
        for key in conductor:
            indexes = conductor[key]
            for index in indexes:
                sentence_id = index[0]
                start = index[1]
                end = index[2] - 1
                sentence = data.text[sentence_id]
                labels = data.labels[sentence_id]
                tag = data.pos[sentence_id]
                tree = self.generate_tree(sentence, labels, tag)
                mct = self.visit_tree(tree)
                for element in mct:
                    if element['mention_start'] == start and element['mention_end'] == end:
                        type_txt = element['type']
                        mc = tuple([element['mention'], element['lcontxt'], element['rcontxt']])
                        if mc in mc2t_dic:
                            mc2t_dic[mc].add(type_txt)
                        else:
                            mc2t_dic[mc] = {type_txt}
                            mc2tag_dic[mc] = element['mention_tag']
                        if type_txt not in type_list:
                            type_list.append(type_txt)
        return mc2t_dic, type_list, mc2tag_dic

    def context_type2id(self, mc2t_dic, dic, filepath):
        """
        The context is mention + context
        :param mc2t_dic: {[lemma,lcontxt,rcontxt]:[type_txt],...} 
        :param mt: 
        :return: [[[lemma,lcontxt,rcontxt],[typeid,...]], ...]
        """
        mcid_tid = []
        for mc in mc2t_dic:
            type_set = mc2t_dic[mc]

            lemma = mc[0]
            lcontxt = mc[1]
            rcontxt = mc[2]
            words = lcontxt.split()
            lemma_ls = lemma.split()
            while len(lemma_ls) < 5:
                lemma_ls.extend(['#PAD#'])
            words.extend(lemma_ls)
            words.extend(rcontxt.split())
            mc_ids = []
            for word in words:
                mc_ids.append(self.dictionary[word])

            tids = []
            for type_txt in type_set:
                if type_txt in dic:
                    tid = dic[type_txt]
                    tids.append(tid)
            if len(tids) != 0:
                mcid_tid.append((mc_ids,tids))
        with open(filepath, 'wb') as f:
            pickle.dump(mcid_tid, f)

    def extract_topk(self, dic, topk):
        type_count = {}
        instance = {}
        for key in dic:
            flag = False
            type_list = dic[key]
            for type in type_list:
                if type not in type_count:
                    type_count[type] = 1
                else:
                    type_count[type] += 1
                if type_count[type] <= topk:
                    flag = True
            if flag:
                instance[key] = type_list
        return instance

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

    def B(self, new_dic, mc2tag_dic, dic):
        """
        :param mc2tag_dic: 
        :param dic: it is dictionary that map type to id
        :return: 
        """
        Bp_words = {}
        for key in new_dic:
            mention = key[0].split(' ')
            tag = mc2tag_dic[key]
            head = self.headword(mention, tag)
            type_list = new_dic[key]
            for type in type_list:
                tid = dic[type]
                if tid not in Bp_words:
                    Bp_words[tid] = [head]
                else:
                    Bp_words[tid].append(head)
        return Bp_words

    def data_gen(self):
        if False:
            fns = os.listdir('/datastore/liu121/nosqldb2/bbn_data_check')
            print(fns)
            outfilepath = '/datastore/liu121/nosqldb2/bbn_data/bbn.txt'
            with open(outfilepath, 'a+') as outf:
                for fn in fns:
                    if 'data_' not in fn:
                        continue
                    print(fn)
                    infilepath = '/datastore/liu121/nosqldb2/bbn_data/' + fn
                    with open(infilepath, 'r') as inf:
                        for line in inf:
                            outf.write(line)

        # train data
        train_data = BBNDataReader.readFile(filePath=self.data_config['trainConll_filePath'],
                                            are_tweets=self.data_config['are_tweets'])
        train_conductor = json.load(open(self.data_config['train_jsonPath']))
        # test data
        test_data = BBNDataReader.readFile(filePath=self.data_config['testConll_filePath'],
                                           are_tweets=self.data_config['are_tweets'])
        mc2t_dic, type_list =self.mct_gen_eval(test_data)

        type2id_dic = {}
        count = 0
        for type_txt in type_list:
            if type_txt not in type2id_dic:
                type2id_dic[type_txt] = count
                count+=1
        with open(self.data_config['t2id'],'wb') as f:
            pickle.dump(type2id_dic,f)

        self.context_type2id(mc2t_dic, type2id_dic, self.data_config['testData_filePath'])

        # train data
        # {(mention, lcontxt, rcontxt):[type,...],...}
        for i in range(self.data_config['iteration']):
            for j in [1, 2, 4, 8, 16]:
                affix = str(j) + '.' + str(i)
                t2indexes = train_conductor[str(j)][str(i)]
                mc2t_dic, type_list, mc2tag_dic = self.mct_gen(train_data, t2indexes)
                new_dic = mc2t_dic
                self.context_type2id(new_dic, type2id_dic, self.data_config['trainData_filePath'] + affix + '.pkl')
                Bp_words = self.B(new_dic, mc2tag_dic, type2id_dic)
                with open(self.data_config['Bp_filePath'] + affix + '.pkl', 'wb') as f:
                    pickle.dump(Bp_words, f)
                # test data

                # t2indexes = test_conductor[str(j)][str(i)]
                # mc2t_dic, type_list, mc2tag_dic = self.mct_gen(test_data, t2indexes)
                # new_dic = mc2t_dic
                # self.context2typeid(new_dic, type2id_dic,
                #                     self.data_config['testData_filePath'] + str(j) + '.' + str(i) + '.pkl')
        print('max: ', str(self.max))

    # TODO: generate eval data
    def generate_tree_eval(self,sentence, labels):
        tree = []
        for i in range(len(sentence)):
            word = sentence[i]
            word = self.tokenConvert(word)
            label = labels[i]
            node = Node()
            node.text = word
            node.type = label
            # print('node{}: {}'.format(str(i),node.text))
            if len(tree) == 0:
                tree.append(node)
            else:
                parent = tree[-1]
                node.parent = parent
                if parent.type == node.type:
                    parent.right_child = node
                else:
                    parent.left_child = node
                tree.append(node)
        return tree

    def visit_tree_eval(self,tree):
        mct = []
        mention = []
        lcontxt = []
        rcontxt = []
        type_txt = ''
        for i in range(len(tree)):
            node = tree[i]
            parent = node.parent
            # the first node
            # if parent=='__empty__':
            if parent == '__empty__':
                if node.type != 'OTHER':
                    mention.append(node.text)
                    type_txt = node.type
                    type_txt = '/' + '/'.join(type_txt.split(':'))
                    lcontxt.append('#PAD#')
                    lcontxt.append('#PAD#')
            elif parent.right_child == '__empty__':
                if type_txt != '':
                    # fulfill the right context
                    # print('current{}: {}'.format(str(i),tree[i].text))
                    if i < len(tree):
                        rcontxt.append(tree[i].text)
                        # print('right{}: {}'.format(str(i),tree[i].text))
                        if i + 1 < len(tree):
                            rcontxt.append(tree[i + 1].text)
                            # print('right{}: {}'.format(str(i+1),tree[i+1].text))
                        else:
                            rcontxt.append('#PAD#')
                    else:
                        rcontxt.append('#PAD#')
                        rcontxt.append('#PAD#')
                    mct.append(
                        {'mention': ' '.join(mention), 'lcontxt': ' '.join(lcontxt), 'rcontxt': ' '.join(rcontxt),
                         'type': type_txt})
                mention = []
                lcontxt = []
                rcontxt = []
                type_txt = ''
                if node.type != 'OTHER':
                    mention.append(node.text)
                    type_txt = node.type
                    type_txt = '/' + '/'.join(type_txt.split(':'))
                    if i - 1 >= 0:
                        if i - 2 >= 0:
                            lcontxt.append(tree[i - 2].text)
                        else:
                            lcontxt.append('#PAD#')
                        lcontxt.append(tree[i - 1].text)
                    else:
                        lcontxt.append('#PAD#')
                        lcontxt.append('#PAD#')
            elif parent.left_child == '__empty__':
                if node.type != 'OTHER':
                    if type_txt == '':
                        mention.append(node.text)
                        type_txt = node.type
                        type_txt = '/' + '/'.join(type_txt.split(':'))
                        if i - 1 >= 0:
                            if i - 2 >= 0:
                                lcontxt.append(tree[i - 2].text)
                            else:
                                lcontxt.append('#PAD#')
                            lcontxt.append(tree[i - 1].text)
                        else:
                            lcontxt.append('#PAD#')
                            lcontxt.append('#PAD#')
                    else:
                        mention.append(node.text)
            if i == len(tree) - 1:
                if type_txt != '':
                    rcontxt.append('#PAD#')
                    rcontxt.append('#PAD#')
                    mct.append(
                        {'mention': ' '.join(mention), 'lcontxt': ' '.join(lcontxt), 'rcontxt': ' '.join(rcontxt),
                         'type': type_txt})
        return mct

    def mct_gen_eval(self,data):
        """

        :param data: 
        :return: {(mention, lcontxt, rcontxt):[type,...],...}
        """
        # # {mention:[(lcontxt,rcontxt),...],...}
        # mc={}
        # # {mention:[type,...],...}
        # mt={}
        type_list = []
        data_len = len(data.text)
        mc2t_dic = {}
        for i in range(data_len):
            sentence = data.text[i]
            labels = data.labels[i]
            tree = self.generate_tree_eval(sentence, labels)
            mct = self.visit_tree_eval(tree)
            for element in mct:
                type_txt = element['type']
                mc = tuple([element['mention'], element['lcontxt'], element['rcontxt']])
                if mc in mc2t_dic:
                    mc2t_dic[mc].add(type_txt)
                else:
                    mc2t_dic[mc] = set(type_txt)
                if type_txt not in type_list:
                    type_list.append(type_txt)
                        #     if element['mention'] in mc:
                        #         mc[element['mention']].append([element['lcontxt'],element['rcontxt']])
                        #         mt[element['mention']].add(element['type'])
                        #     else:
                        #         mc[element['mention']]=[[element['lcontxt'],element['rcontxt']]]
                        #         mt[element['mention']]={element['type']}
        return mc2t_dic, type_list


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
                         'train_jsonPath': '/datastore/liu121/nosqldb2/bbn_kn/draw_kn.json',
                         'trainData_filePath': '/datastore/liu121/nosqldb2/ner_zero/data_bbn_kn/train_kn',

                         'testConll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_eval',
                         'test_jsonPath': '/datastore/liu121/nosqldb2/bbn_kn/eval_kn.json',
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
        dg.data_gen()

