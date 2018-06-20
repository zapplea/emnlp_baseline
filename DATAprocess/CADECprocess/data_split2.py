import sys

sys.path.append('/home/liu121/dlnlp')

from nerd.StanfordNLP.tokenizer_en import Tokenizer

import pathlib
import random
import nltk
import operator
import math


class Statistics:
    def __init__(self, data_config):
        self.data_config = data_config
        self.tokenizer = Tokenizer()

    # def sentence_char_pos2word_pos(self, sentence):
    #     """
    #     convert char position to word position.
    #     give each word a char range.
    #     :return: [{start:int, word:str}, ...]
    #     """
    #     tokens = self.tokenizer.tokenize(sentence)
    #     new_tokens = []
    #     for token in tokens:
    #         if token == "-LRB-":
    #             new_tokens.append('(')
    #         elif token == "-RRB-":
    #             new_tokens.append(')')
    #         elif token == "-LSB-":
    #             new_tokens.append('[')
    #         elif token == "-RSB-":
    #             new_tokens.append(']')
    #         else:
    #             new_tokens.append(token)
    #     tokens = new_tokens
    #     start = 0
    #     ls = []
    #     for token in tokens:
    #         ls.append({'start': start, 'word': token})
    #         start += len(token)
    #         sentence = sentence[len(token):]
    #         if sentence == "":
    #             break
    #
    #         while sentence[0] == " ":
    #             start += 1
    #             sentence = sentence[1:]
    #     return ls, tokens
    #
    # def label_char_pos2word_pos(self, sentence, start, end):
    #     """
    #
    #     :param sentence: [{start:int,word:str}, ...]
    #     :param index_ranges: [(start, end), ...]
    #     :return: [[word index1, word index2,...], ...]
    #     """
    #     words_index = []
    #     flag = False
    #     for i in range(len(sentence)):
    #         word_pos = sentence[i]
    #         char_start = word_pos['start']
    #         word = word_pos['word']
    #         char_end = char_start + len(word)
    #         if char_start == start:
    #             flag = True
    #         if flag:
    #             words_index.append(i)
    #         if char_end >= end:
    #             break
    #     if len(words_index) == 0:
    #         return 'empty'
    #     return words_index

    def load(self):
        """
        :return: 
        """
        p = pathlib.Path(self.data_config['original_folderPath'])
        file_names = list(p.glob('*'))
        data = []
        for name in file_names:
            n = str(name)
            label = []
            with open(n, 'r') as f:
                for line in f:
                    if '#' in line or ';' in line:
                        continue
                    line = line.replace('\n', '')
                    ls = line.split('\t')
                    space_pos = ls[1].find(' ')
                    type_txt = ls[1][:space_pos]
                    # '55 66'
                    index_range = ls[1][(space_pos + 1):]
                    start = int(index_range.split()[0])
                    end = int(index_range.split()[1])
                    # label.append({'type_txt': type_txt, 'start': start, 'end': end, 'entity': ls[-1]})
                    label.append((type_txt, start, end, ls[-1]))
            n = n.replace('/original/', '/text/')
            n = n.replace('.ann', '.txt')
            text = ''
            with open(n, 'r') as f:
                for line in f:
                    line = line.replace('\n', '')
                    if line == '':
                        text += ' '
                        continue
                    text = text + line + ' '
                text = text.strip()
            data.append({'text': text, 'label': label})
        return data, file_names

    def cut(self,data):
        new_data = []
        for i in range(len(data)):
            item = data[i]
            text=item['text']
            labels = item['label']
            labels = sorted(labels,key=operator.itemgetter(1))
            # split the text into pieces, based on labels
            type_list =[]
            text_list = []
            cur_pos = 0
            for label in labels:
                type_txt = label[0]
                start = label[1]
                entity = label[-1]
                end = start+len(entity)
                if cur_pos<start:
                    text_list.append(text[cur_pos:start])
                    type_list.append('O')
                candidate = text[start:end]
                if candidate!=entity:
                    print(candidate)
                    print(start,' ',end)
                    return i
                text_list.append(text[start:end])
                type_list.append(type_txt)

                cur_pos=end
            if cur_pos<len(text):
                text_list.append(text[cur_pos:])
                type_list.append('O')
            new_data.append({'text_list':text_list,'type_list':type_list})
        return new_data

    def pos_tagger(self,tokens):
        words_tag_ls = nltk.pos_tag(tokens)
        tag_ls = []
        for words_tag in words_tag_ls:
            tag = words_tag[1]
            tag_ls.append(tag)
        return tag_ls

    def split(self,data):
        """
        
        :param data: 
        :return: 
        """
        conll_data=[]
        for i in range(len(data)):
            text_list = data[i]['text_list']
            type_list = data[i]['type_list']
            tokens_list = self.tokenizer.tokenize_sents(text_list)
            tokens = []
            types=[]
            for i in range(len(tokens_list)):
                tokens.extend(tokens_list[i])
                types.extend([type_list[i]]*len(tokens_list[i]))
            pos=self.pos_tagger(tokens)
            chunks=['CHUNKS']*len(tokens)

            new_tokens = []
            for token in tokens:
                if token == "-LRB-":
                    new_tokens.append('(')
                elif token == "-RRB-":
                    new_tokens.append(')')
                elif token == "-LSB-":
                    new_tokens.append('[')
                elif token == "-RSB-":
                    new_tokens.append(']')
                else:
                    new_tokens.append(token)
            tokens = new_tokens

            conll_data.append({'tokens':tokens,'chunks':chunks,'pos':pos,'types':types,'type_list':types})
        return conll_data

    def write(self,conll_data,is_draw):
        # instance : {'tokens':tokens,'postags':postags,'chunktags':chunktags,'netags':netags}
        if is_draw:
            f = open(self.data_config['conll_draw_filePath'],'w+')
        else:
            f = open(self.data_config['conll_eval_filePath'],'w+')
        for instance in conll_data:
            tokens = instance['tokens']
            postags = instance['pos']
            chunks = instance['chunks']
            type_list = instance['type_list']
            for i in range(len(tokens)):
                f.write('{}\t{}\t{}\t{}\n'.format(tokens[i],postags[i],chunks[i],type_list[i]))
            f.write('\n')
            f.flush()
        f.close()

    def ratio(self,conll_data):
        total = {}
        draw = {}
        eval = {}
        half = int(math.ceil(len(conll_data) / 2))
        for i in range(len(conll_data)):
            instance = conll_data[i]
            type_list = instance['type_list']
            if i <half:
                for type_txt in type_list:
                    if type_txt in draw:
                        draw[type_txt]+=1
                    else:
                        draw[type_txt]=1
            else:
                for type_txt in type_list:
                    if type_txt in eval:
                        eval[type_txt]+=1
                    else:
                        eval[type_txt]=1
            for type_txt in type_list:
                if type_txt in total:
                    total[type_txt]+=1
                else:
                    total[type_txt]=1
        ratio=[]
        sum_total = 0
        sum_draw = 0
        sum_eval = 0
        for key in total:
            sum_total+=total[key]
            if key in draw:
                d = draw[key]/total[key]
                sum_draw+=draw[key]
            else:
                draw[key] = 0
                d = 0
            if key in eval:
                e = eval[key]/total[key]
                sum_eval += eval[key]
            else:
                eval[key] = 0
                e = 0
            ratio.append((key,d,e))

        self_ratio=[]
        for key in total:
            draw[key]=draw[key]/sum_draw
            total[key]=total[key]/sum_total
            eval[key]=eval[key]/sum_eval
            self_ratio.append((key,total[key],draw[key],eval[key]))

        return ratio,self_ratio

    def write_ratio(self,ratio):
        # ratio = sorted(ratio,key=operator.itemgetter(1,2),reverse=True)
        with open(self.data_config['conll_ratio_filePath'],'w+') as f:
            for instance in ratio:
                f.write('{}\t{:.2f}\t{:.2f}\n'.format(instance[0],instance[1],instance[2]))
                f.flush()

    def write_self_ratio(self,self_ratio):
        with open(self.data_config['conll_self_ratio_filePath'],'w+') as f:
            for instance in self_ratio:
                f.write('{}\t{:.2f}\t{:.2f}\t{:.2f}\n'.format(instance[0],instance[1],instance[2],instance[3]))
                f.flush()

if __name__ == "__main__":
    data_config = {'original_folderPath': '/datastore/liu121/nosqldb2/cadec/original',
                   'conll_draw_filePath':'/datastore/liu121/nosqldb2/cadec/Conll/cadec_draw',
                   'conll_eval_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/cadec_eval',
                   'conll_ratio_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/cadec_ratio',
                   'conll_self_ratio_filePath':'/datastore/liu121/nosqldb2/cadec/Conll/self_ratio'
                   }
    stat = Statistics(data_config)
    data, file_names = stat.load()
    data = stat.cut(data)
    if isinstance(data,int):
        print(file_names[data])
    else:
        print('successful')
    conll_data = stat.split(data)
    random.shuffle(conll_data)
    half = int(math.ceil(len(conll_data)/2))
    stat.write(conll_data[:half],True)
    stat.write(conll_data[half:],False)
    ratio,self_ratio = stat.ratio(conll_data)
    stat.write_ratio(ratio)
    stat.write_self_ratio(self_ratio)
    print('program finish')
