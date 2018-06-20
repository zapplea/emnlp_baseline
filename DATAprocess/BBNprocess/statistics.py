import sys
sys.path.append('/home/liu121/dlnlp')
from BBNDataReader import BBNDataReader
import numpy as np
import pickle
import operator
import re

class Node:
    def __init__(self):
        self.text='__empty__'
        self.type='__empty__'
        self.parent='__empty__'
        self.left_child='__empty__'
        self.right_child = '__empty__'
    def __str__(self):
        return self.text

global dictionary
f=open('/datastore/liu121/nosqldb2/bbn_data/table.pkl','rb')
dictionary = pickle.load(f)

def tokenConvert(token):
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
    if tk not in dictionary:
        tk='#UNK#'
    return tk

def generate_tree(sentence,labels):
    tree=[]
    for i in range(len(sentence)):
        word=sentence[i]
        word = tokenConvert(word)
        label=labels[i]
        node=Node()
        node.text=word
        node.type=label
        #print('node{}: {}'.format(str(i),node.text))
        if len(tree)==0:
            tree.append(node)
        else:
            parent=tree[-1]
            node.parent=parent
            if parent.type == node.type:
                parent.right_child = node
            else:
                parent.left_child = node
            tree.append(node)
    return tree


def visit_tree(tree):
    mct = []
    mention = []
    lcontxt = []
    rcontxt = []
    type_txt =''
    for i in range(len(tree)):
        node = tree[i]
        parent=node.parent
        # the first node
        # if parent=='__empty__':
        if parent =='__empty__':
            if node.type!='OTHER':
                mention.append(node.text)
                type_txt = node.type
                type_txt = '/' + '/'.join(type_txt.split(':'))
                lcontxt.append('#PAD#')
                lcontxt.append('#PAD#')
        elif parent.right_child == '__empty__':
            if type_txt != '':
                # fulfill the right context
                #print('current{}: {}'.format(str(i),tree[i].text))
                if i < len(tree):
                    rcontxt.append(tree[i].text)
                    #print('right{}: {}'.format(str(i),tree[i].text))
                    if i+1<len(tree):
                        rcontxt.append(tree[i+1].text)
                        #print('right{}: {}'.format(str(i+1),tree[i+1].text))
                    else:
                        rcontxt.append('#PAD#')
                else:
                    rcontxt.append('#PAD#')
                    rcontxt.append('#PAD#')
                mct.append({'mention':' '.join(mention),'lcontxt':' '.join(lcontxt),'rcontxt':' '.join(rcontxt),'type':type_txt})
            mention = []
            lcontxt = []
            rcontxt = []
            type_txt = ''
            if node.type != 'OTHER':
                mention.append(node.text)
                type_txt = node.type
                type_txt='/'+'/'.join(type_txt.split(':'))
                if i-1>=0:
                    if i-2>=0:
                        lcontxt.append(tree[i-2].text)
                    else:
                        lcontxt.append('#PAD#')
                    lcontxt.append(tree[i - 1].text)
                else:
                    lcontxt.append('#PAD#')
                    lcontxt.append('#PAD#')
        elif parent.left_child == '__empty__':
            if node.type!='OTHER':
                if type_txt=='':
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
        if i == len(tree)-1:
            if type_txt != '':
                rcontxt.append('#PAD#')
                rcontxt.append('#PAD#')
                mct.append({'mention':' '.join(mention),'lcontxt':' '.join(lcontxt),'rcontxt':' '.join(rcontxt),'type':type_txt})
    return mct

def mct_gen(data):
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
        tree = generate_tree(sentence, labels)
        mct = visit_tree(tree)
        for element in mct:
            type = element['type']
            ls = type.split('/')
            ls.remove('')
            temp_list = []
            type = ''
            for s in ls:
                type += '/' + s
                temp_list.append(type)
            mc = tuple([element['mention'], element['lcontxt'], element['rcontxt']])
            if mc in mc2t_dic:
                for type_txt in temp_list:
                    mc2t_dic[mc].add(type_txt)
            else:
                mc2t_dic[mc] = set(temp_list)
            for type_txt in temp_list:
                if type_txt not in type_list:
                    type_list.append(type_txt)
                    #     if element['mention'] in mc:
                    #         mc[element['mention']].append([element['lcontxt'],element['rcontxt']])
                    #         mt[element['mention']].add(element['type'])
                    #     else:
                    #         mc[element['mention']]=[[element['lcontxt'],element['rcontxt']]]
                    #         mt[element['mention']]={element['type']}
    return mc2t_dic, type_list

def longest_sentence_len(data):
    data_len = len(data.text)
    max = 0
    for i in range(data_len):
        text_len = len(data.text[i])
        if text_len>max:
            max=text_len
    print('longest length of sentence is {}'.format(str(max)))

    freq = {}
    for i in range(data_len):
        text_len = len(data.text[i])
        if text_len in freq:
            freq[text_len]+=1
        else:
            freq[text_len]=1
    sum = 0
    for key in freq:
        sum+=freq[key]
    for key in freq:
        freq[key] = freq[key]/sum
    result = sorted(freq.items(), key=operator.itemgetter(0))
    print('frequency of each sequence length')
    for i in range(len(result)):
        print(result[i])

def amount_of_sentences(data):
    data_len=len(data.text)
    print('number of sentences:',data_len)

if __name__ == "__main__":
    data = BBNDataReader.readFile(filePath='/datastore/liu121/nosqldb2/bbn_data/data_train')
    amount_of_sentences(data)
    # {(mention, lcontxt, rcontxt):[type,...],...}
    if False:
        longest_sentence_len(data)
    if False:
        mc2t_dic, type_list = mct_gen(data)
        new_dic={}
        length_dic={}
        for key in mc2t_dic:
            mention = key[0]
            ls = mention.split(' ')
            if len(ls) in length_dic:
                length_dic[len(ls)]+=1
            else:
                length_dic[len(ls)]=1
        result = sorted(length_dic.items(),key=operator.itemgetter(0))
        result = np.array(result)
        divisor = np.sum(result[:,1])
        for pair in result:
            print('length{}: {}'.format(pair[0],pair[1]/divisor))