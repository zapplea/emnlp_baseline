import nltk
import math
import random

class DataGeneartor:
    def __init__(self,data_config):
        self.data_config = data_config
        self.data = self.loader()

    def loader(self):
        data = []
        with open(self.data_config['Conll_filePath'],'r') as f:
            instance = []
            label = []
            for line in f:
                if ' ' in line:
                    ls = line.split()
                    if len(ls)<2:
                        continue
                    instance.append(ls[0])
                    label_txt = ls[1]
                    if 'B-' in label_txt:
                        label_txt=label_txt.replace('B-','I-')
                    label.append(label_txt)
                elif line == '\n':
                    data.append((instance,label))
                    instance = []
                    label = []
                    continue
        return data

    def pos_tagger(self,tokens):
        words_tag_ls = nltk.pos_tag(tokens)
        tag_ls = []
        for words_tag in words_tag_ls:
            tag = words_tag[1]
            tag_ls.append(tag)
        return tag_ls

    def tag_chunk_generator(self):
        data = []
        for instance in self.data:
            tokens = instance[0]
            label = instance[1]
            postag_ls = self.pos_tagger(tokens)
            chunk_ls = ['CHUNKS']*len(postag_ls)
            data.append((tokens,postag_ls,chunk_ls,label))
        return data

    # def write(self,data):
    #     f = open(self.data_config['newConll_filePath'],'w')
    #     print('lenght of data: ',str(len(data)))
    #     for instance in data:
    #         length = len(instance[0])
    #         for i in range(length):
    #             token = instance[0][i]
    #             postag = instance[1][i]
    #             chunk = instance[2][i]
    #             label = instance[3][i]
    #             f.write('{}\t{}\t{}\t{}\n'.format(token,postag,chunk,label))
    #         f.write('\n')

    def write(self,conll_data,is_draw):
        # instance : {'tokens':tokens,'postags':postags,'chunktags':chunktags,'netags':netags}
        if is_draw:
            f = open(self.data_config['conll_draw_filePath'],'w+')
        else:
            f = open(self.data_config['conll_eval_filePath'],'w+')
        for instance in conll_data:
            length = len(instance[0])
            for i in range(length):
                token = instance[0][i]
                postag = instance[1][i]
                chunk = instance[2][i]
                label = instance[3][i]
                f.write('{}\t{}\t{}\t{}\n'.format(token, postag, chunk, label))
            f.write('\n')
            f.flush()
        f.close()

if __name__ == "__main__":
    data_config = {'Conll_filePath':'/datastore/liu121/nosqldb2/nvd/nvd_corpus.txt',
                   'conll_draw_filePath':'/datastore/liu121/nosqldb2/nvd/nvd_test_draw',
                   'conll_eval_filePath':'/datastore/liu121/nosqldb2/nvd/nvd_test_eval'}
    dg = DataGeneartor(data_config)
    conll_data = dg.tag_chunk_generator()
    random.shuffle(conll_data)
    half = int(math.ceil(len(conll_data) / 2))
    dg.write(conll_data[:half], True)
    dg.write(conll_data[half:], False)