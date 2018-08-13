import sys
sys.path.append('/home/liu121/emnlp_baseline')

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
            for line in lines:
                result = line.split(' ')
                if result[0]=='\n':
                    doc.append(sentence)
                    labels.append(label)
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
        print('sentence numbers: ', str(len(doc)))
        print('freq:')
        print(freq)

if __name__=='__main__':
    data_config = {'train_conll_filePath':'/datastore/liu121/nosqldb2/emnlp_ukplab/data/bbn_kn/bbn_kn__1.0/train.txt'}
    ch=Check(data_config)
    doc,labels=ch.load()
