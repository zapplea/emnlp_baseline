import sys
sys.path.append('/home/liu121/dlnlp')

import json
import re
import pickle
import random

from nerd.data.util.readers.BBNDataReader import BBNDataReader

class Node:
    def __init__(self):
        self.text='__empty__'
        self.type='__empty__'
        self.parent='__empty__'
        self.tag='__empty__'
        self.left_child='__empty__'
        self.right_child = '__empty__'
        self.pos = '__empty__'
    def __str__(self):
        return self.text

class DataSplit:
    def __init__(self,data_config):
        self.data_config = data_config
        f = open(self.data_config['table_filePath'], 'rb')
        self.dictionary = pickle.load(f)
        f.close()

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
                    if len(mention)<=5:
                        mct.append(
                            {'mention': ' '.join(mention), 'lcontxt': ' '.join(lcontxt), 'rcontxt': ' '.join(rcontxt),
                             'type': type_txt, 'mention_tag': mention_tag,'mention_start':mention_pos[0],'mention_end':mention_pos[-1]})
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
                    if len(mention)<=5:
                        mct.append(
                            {'mention': ' '.join(mention), 'lcontxt': ' '.join(lcontxt), 'rcontxt': ' '.join(rcontxt),
                             'type': type_txt, 'mention_tag': mention_tag,'mention_start':mention_pos[0],'mention_end':mention_pos[-1]})
        return mct

    def mct_gen(self, data):
        """

        :param data: 
        :return: {(mention, lcontxt, rcontxt):[type,...],...}
        """
        # # {mention:[(lcontxt,rcontxt),...],...}
        # mc={}
        # # {mention:[type,...],...}
        data_len = len(data.text)
        t2sentenceID={}
        sentence_id =-1
        for i in range(data_len):
            sentence_id+=1
            sentence = data.text[i]
            if len(sentence)>self.data_config['max_len']:
                continue

            labels = data.labels[i]
            tag = data.pos[i]
            tree = self.generate_tree(sentence, labels, tag)
            mct = self.visit_tree(tree)
            for element in mct:
                type_txt = element['type']
                mention_start = element['mention_start']
                mention_end = element['mention_end']+1
                if type_txt not in t2sentenceID:
                    t2sentenceID[type_txt]=[(sentence_id, mention_start, mention_end)]
                else:
                    t2sentenceID[type_txt].append((sentence_id, mention_start, mention_end))
        return t2sentenceID

    def split(self,t2sentenceID,is_train,**kwargs):
        # sample = {type:k[sentence id]}
        sample={}
        for key in t2sentenceID:
            ids = t2sentenceID[key]
            random.shuffle(ids)
            sample[key]=[]
            if is_train:
                for i in range(kwargs['k']):
                    if i>=len(ids):
                        index = random.randint(0,len(ids)-1)
                        sample[key].append(ids[index])
                    else:
                        sample[key].append(ids[i])
            else:
                for i in range(len(ids)):
                    sample[key].append(ids[i])
        return sample

if __name__ == "__main__":
    for flag in [True]:
        is_draw = flag
        if is_draw:
            data_config = {'Conll_filePath':'/datastore/liu121/nosqldb2/nvd/nvd_test_draw',
                           'table_filePath': '/datastore/liu121/nosqldb2/bbn_data/table.pkl',
                           'iteration':5,
                           'json_filePath':'/datastore/liu121/nosqldb2/nvd/draw.json',
                           'max_len':100}
        else:
            data_config = {'Conll_filePath': '/datastore/liu121/nosqldb2/nvd/nvd_test_eval',
                           'table_filePath': '/datastore/liu121/nosqldb2/bbn_data/table.pkl',
                           'iteration': 5,
                           'json_filePath': '/datastore/liu121/nosqldb2/nvd/eval.json',
                           'max_len':100}
        data = BBNDataReader.readFile(filePath=data_config['Conll_filePath'])
        ds = DataSplit(data_config)
        t2sentenceID = ds.mct_gen(data)
        samples={}

        for i in range(data_config['iteration']):
            for j in [1,2,4,8,16]:
                sample = ds.split(t2sentenceID,True,k=j)
                if str(j) not in samples:
                    samples[str(j)]={str(i):sample}
                else:
                    samples[str(j)][str(i)]=sample
                # print('{}, {}, {}'.format(str(j),str(i),str(len(sample))))
        json_filePath = data_config['json_filePath']
        with open(json_filePath,'w+') as f:
            json.dump(samples,f,indent=4,sort_keys=False)

