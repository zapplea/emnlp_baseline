import sys
sys.path.append('/home/liu121/dlnlp')
from BBNDataReader import BBNDataReader
import pickle
import operator
import re
import os

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
    data_len=len(data.text)
    mc2t_dic = {}
    for i in range(data_len):
        sentence = data.text[i]
        labels = data.labels[i]
        tree = generate_tree(sentence,labels)
        mct = visit_tree(tree)
        for element in mct:
            type = element['type']
            ls = type.split('/')
            ls.remove('')
            temp_list=[]
            type=''
            for s in ls:
                type+='/'+s
                temp_list.append(type)
            mc = tuple([element['mention'],element['lcontxt'],element['rcontxt']])
            if mc in mc2t_dic:
                for type_txt in temp_list:
                    mc2t_dic[mc].add(type_txt)
            else:
                mc2t_dic[mc]=set(temp_list)
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

# =========================
# ===== training data =====
# =========================
def map_context2type(mc,mt):
    """
    The context is mention + context
    :param mc: 
    :param mt: 
    :return: [[[lemma,lcontxt,rcontxt],[typeTxts,...]], ...]
    """
    mct=[]
    for mention in mc:
        if len(mention.split(' '))>5:
            continue
        type_list= list(mt[mention])
        contxt_list = mc[mention]
        for context in contxt_list:
            mention_contxt=[mention]
            mention_contxt.extend(context)
            mct.append([mention_contxt,type_list])
    with open('/datastore/liu121/nosqldb2/bbn_data/contxt_typeTxt.pkl','wb') as f:
        pickle.dump(mct,f)

# =========================
# ====== train queue ======
# =========================
def generate_dic_tree(mt):
    """
    :param mt: {mention:[type, ...], ...}
    :return: {'typetxt':{'num':int,'typeTxt':{...},...},...} 
    """
    all_types=set()
    for mention in mt:
        if len(mention.split(' '))>5:
            continue
        type_list = mt[mention]
        for type in type_list:
            all_types.add(type)

    tree = {}
    branches = list(all_types)

    # f = open(config['typeInfo'], 'rb')
    # while True:
    #     try:
    #         dic = pickle.load(f)
    #         if dic['class'] == 'Notable Type':
    #             branches.append((dic['type'], dic['title']))
    #     except EOFError:
    #         break
    # f.close()
    for t in branches:
        tp = t
        type_path = tp.split('/')
        type_path.remove('')
        subtree = tree
        for node in type_path:
            if node in subtree:
                subtree = subtree[node]
                subtree['num'] += 1
            else:
                subtree[node] = {'num': 1}
                subtree = subtree[node]
    return tree


def dfs(tree):
    """
    visit the tree in DFS and return all types
    :return: 
    """
    new_tree={}
    for key in tree:
        subtree=tree[key]
        key='/'+key
        new_tree[key]=subtree
    stack = [new_tree]
    train_queue_with_num = []
    while len(stack) > 0:
        subtree = stack.pop(0)
        queue = []
        for key in subtree:
            if key == 'num':
                continue
            else:
                type_path=key.split('/')
                if len(type_path)==1:
                    key='/'+key
                new_tree = {}
                for sub_key in subtree[key]:
                    if sub_key == 'num':
                        continue
                    else:
                        new_tree[key + '/' + sub_key] = subtree[key][sub_key]
                stack.append(new_tree)
                queue.append((key, subtree[key]['num']))
        if len(queue) > 0:
            queue = sorted(queue, key=operator.itemgetter(1))
            train_queue_with_num.append(queue)
    return train_queue_with_num

# train_queue
def generate_train_queue(train_queue_with_num):
    """

    :param train_queue_with_num: 
    :param init: 
    :return: train_queue
    """
    train_queue = []
    code = {}
    code_count = 0
    for queue in train_queue_with_num:
        temp_queue = []
        for t in queue:
            tp = t[0]
            instance_num = t[1]
            code[tp] = code_count
            code_count += 1
            temp_queue.append(tp)
        train_queue.append(temp_queue)
    dic = {'train_queue': train_queue, 'code': code}
    with open('/datastore/liu121/nosqldb2/bbn_data/notable.pkl','wb') as f:
        pickle.dump(dic,f)

def context2typeid(mc2t_dic, type_list):
    """
    The context is mention + context
    :param mc: 
    :param mt: 
    :return: [[[lemma,lcontxt,rcontxt],[typeid,...]], ...]
    """
    # create type dictionary
    count=0
    # type 2 id
    dic={}
    mctid=[]
    for type_txt in type_list:
        if type_txt not in dic:
            dic[type_txt]=count
            count+=1
    for mc in mc2t_dic:
        type_set = mc2t_dic[mc]
        tids=[]
        for type_txt in type_set:
            tid=dic[type_txt]
            tids.append(tid)
        mctid.append([list(mc),tids])
    with open('/datastore/liu121/nosqldb2/bbn_data/contxt.pkl', 'wb') as f:
        pickle.dump(mctid, f)
        pickle.dump(dic,f)
    return dic


if __name__ == "__main__":
    if True:
        fns = os.listdir('/datastore/liu121/nosqldb2/bbn_data_check')
        print(fns)
        outfilepath='/datastore/liu121/nosqldb2/bbn_data/bbn.txt'
        with open(outfilepath,'w+') as outf:
            for fn in fns:
                if 'data_' not in fn:
                    continue
                print(fn)
                infilepath = '/datastore/liu121/nosqldb2/bbn_data/'+fn
                with open(infilepath,'r') as inf:
                    for line in inf:
                        outf.write(line)
    if False:
        data = BBNDataReader.readFile(filePath='/datastore/liu121/nosqldb2/bbn_data/bbn.txt')
        # {(mention, lcontxt, rcontxt):[type,...],...}
        mc2t_dic,type_list = mct_gen(data)
        new_dic={}
        max=0
        for key in mc2t_dic:
            mention = key[0]
            mention_ls = mention.split(' ')
            if len(mention_ls)<=5:
                new_dic[key]=mc2t_dic[key]
        dic = context2typeid(new_dic, type_list)
        print(len(dic))

        # # train queue
        # tree = generate_dic_tree(mt=dic['mention_type'])
        # train_queue_with_num=dfs(tree)
        # generate_train_queue(train_queue_with_num)