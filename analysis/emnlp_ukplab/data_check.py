import sys
sys.path.append('/home/liu121/emnlp_baseline')
import argparse

class Check:
    def __init__(self, data_config):
        self.data_config = data_config

    def load(self):
        with open(self.data_config['train_conll_filePath'],'r') as f:
            lines = f.readlines()
            doc = []
            labels = []
            sentence=[]
            label=[]
            count=1
            for line in lines:
                count+=1
                result = line.split(' ')
                if result[0]=='\n':
                    doc.append(sentence)
                    labels.append(label)
                    sentence=[]
                    label=[]
                    continue
                sentence.append(result[1])
                label.append(result[2])
        return doc,labels

    def stat(self,doc,labels):
        freq={}
        for label in labels:
            for type in label:
                if type not in freq:
                    freq[type]=1
                else:
                    freq[type]+=1
        print('length of doc: ',len(doc))
        print('freq:')
        print(freq)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',type=str,default='train')
    parser.add_argument('--num', type=str, default='1.0')
    args=parser.parse_args()
    data_config = {'train_conll_filePath':'/datastore/liu121/nosqldb2/emnlp_ukplab/data/bbn_kn/bbn_kn__%s/%s.txt'%(args.num,args.name)}
    ch=Check(data_config)
    doc,labels=ch.load()
    ch.stat(doc,labels)