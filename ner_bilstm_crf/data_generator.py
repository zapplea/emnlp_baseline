import data
import numpy
import re
import pickle

class DataGen:
    def __init__(self,data_config):
        self.data_config = data_config

    def embedding_generator(self):
        pass

    def tokenConvert(self,inputstr): #string
            txt=inputstr.lower()
            if bool(re.search('.*\d.*',txt)):
                txt=re.sub('\d',' ',txt)
                chls=[]
                count=0
                for char in txt:
                    if char==' ':
                        count=count+1
                    else:
                        if count==0:
                            chls.append(char)
                        else:
                            chls.append('NUM'+str(count)+char)
                            count=0
                if count>0:
                    chls.append('NUM'+str(count))
                txt=''
                for s in chls:
                    txt=txt+s
            return txt

    def read_corpus(self,corpus_path):
        """
        read corpus and return the list of samples
        :param corpus_path:
        :return: data
        """
        data = []
        with open(corpus_path, encoding='utf-8') as fr:
            lines = fr.readlines()
        sent_, tag_ = [], []
        for line in lines:
            if line != '\n':
                [char, label] = line.strip().split()
                sent_.append(char)
                tag_.append(label)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

        return data

    def vocab_build(self,vocab_path, corpus_path, min_count):
        """

        :param vocab_path:
        :param corpus_path:
        :param min_count:
        :return:
        """
        data = self.read_corpus(corpus_path)
        word2id = {}
        for sent_, tag_ in data:
            for word in sent_:
                word = self.tokenConvert(word)
                # if word.isdigit():
                #     word = '<NUM>'
                if ('\u0041' <= word <= '\u005a') or ('\u0061' <= word <= '\u007a'):
                    word = '<ENG>'
                if word not in word2id:
                    word2id[word] = [len(word2id) + 1, 1]
                else:
                    word2id[word][1] += 1
        low_freq_words = []
        for word, [word_id, word_freq] in word2id.items():
            if word_freq < min_count and word != '<NUM>' and word != '<ENG>':
                low_freq_words.append(word)
        for word in low_freq_words:
            del word2id[word]

        new_id = 1
        for word in word2id.keys():
            word2id[word] = new_id
            new_id += 1
        word2id['<UNK>'] = new_id
        word2id['<PAD>'] = 0

        print(len(word2id))
        with open(vocab_path, 'wb') as fw:
            pickle.dump(word2id, fw)

    def main(self):
        pass

if __name__ == "__main__":
    data_config={}
    dg = DataGen(data_config)