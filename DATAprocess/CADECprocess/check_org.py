import sys
sys.path.append('/home/liu121/dlnlp')

from nerd.StanfordNLP.tokenizer_en import Tokenizer

import pathlib
import json
import nltk

class Statistics:

    def __init__(self, data_config):
        self.data_config = data_config
        self.tokenizer = Tokenizer()

    def sentence_char_pos2word_pos(self,sentence):
        """
        convert char position to word position.
        give each word a char range.
        :return: [{start:int, word:str}, ...]
        """
        tokens = self.tokenizer.tokenize(sentence)
        new_tokens =[]
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
        start=0
        ls = []
        for token in tokens:
            ls.append({'start':start,'word':token})
            start+=len(token)
            sentence = sentence[len(token):]
            if sentence =="":
                break

            while sentence[0] == " ":
                start+=1
                sentence = sentence[1:]
        return ls,tokens

    def label_char_pos2word_pos(self,sentence,start,end):
        """
        
        :param sentence: [{start:int,word:str}, ...]
        :param index_ranges: [(start, end), ...]
        :return: [[word index1, word index2,...], ...]
        """
        words_index=[]
        flag = False
        for i in range(len(sentence)):
            word_pos = sentence[i]
            char_start = word_pos['start']
            word = word_pos['word']
            char_end = char_start+len(word)
            if char_start == start:
                flag = True
            if flag:
                words_index.append(i)
            if char_end >= end:

                break
        if len(words_index)==0:
            return 'empty'
        return words_index

    def load(self):
        """
        :return: 
        """
        p = pathlib.Path(self.data_config['original_folderPath'])
        file_names = list(p.glob('*'))
        data=[]
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
                    index_range = ls[1][(space_pos+1):]
                    start=int(index_range.split()[0])
                    end = int(index_range.split()[1])
                    label.append({'type_txt':type_txt,'start':start,'end':end,'entity':ls[-1]})
            n = n.replace('/original/', '/text/')
            n = n.replace('.ann', '.txt')
            text=''
            with open(n,'r') as f:
                for line in f:
                    line = line.replace('\n','')
                    if line == '':
                        text+=' '
                        continue
                    text=text+line+' '
                text = text.strip()
            data.append({'text':text,'label':label})
        return data,file_names

if __name__ == "__main__":
    data_config = {'original_folderPath':'/datastore/liu121/nosqldb2/cadec/original'}
    stat = Statistics(data_config)
    data,file_names = stat.load()
    wordindex_data = []
    # convert char index to word index
    count=0
    for i in range(len(data)):
        text = data[i]['text']
        # label = [{type:,start:,end:}]
        label = data[i]['label']
        # tokens = [{start:int, word:str}, ...]
        tokens_ls,tokens = stat.sentence_char_pos2word_pos(text)
        wordindex_label=[]
        for index in label:
            type_txt = index['type_txt']
            start = index['start']
            end = index['end']
            words_index = stat.label_char_pos2word_pos(tokens_ls,start,end)
            if isinstance(words_index,str):
                count+=1
                print(file_names[i])
                print(tokens_ls)
                print(type_txt,' ',start,' ',end,' ',index['entity'])
                print('=========================')
                continue
            wordindex_label.append({'type_txt':type_txt,'words_index':words_index})
        wordindex_data.append({'tokens':tokens,'wordindex_label':wordindex_label})
    print('count: ',count)
    # for i in range(len(wordindex_data)):
    #     label = wordindex_data[i]['wordindex_label']
    #     print(label)
    #     print('\n')