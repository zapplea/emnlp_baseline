# verify the ratio of instance involved by each type
import sys
sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader

class TrainVer:

    def __init__(self, config):
        self.config = config

    def conll_data_reader(self, filePath):
        data = BBNDataReader.readFile(filePath=filePath)
        return data

    def draw_labels(self,):
        labels = set()
        draw_data = self.conll_data_reader(self.config['draw_filePath'])
        for i in range(len(draw_data.labels)):
            instance_labels = draw_data.lables[i]
            for label in instance_labels:
                if label !='O' or label!='OTHER':
                    labels.add(label)
        print('draw labels: \n',labels)
        print('draw labels num: \n',len(labels))

    def verify(self):
        train_data = self.conll_data_reader(self.config['train_filePath'])
        labels={}
        for i in range(len(train_data.labels)):
            instance_labels = train_data.labels[i]
            for label in instance_labels:
                if label !='O' or label!='OTHER':
                    if label in labels:
                        labels[label]+=1
                    else:
                        labels[label]=1
        name = self.config['train_filePath'].split('/')[-1]
        print('number of instances in %s:' % name, len(train_data.labels))
        print('train labels num: \n',len(labels))
        print('ratio of labels: \n', labels)

if __name__=='__main__':
    configs=[{'train_filePath':'/datastore/liu121/nosqldb2/acl_hscrf/data/bbn_kn/bbn_kn__1.0/train.txt',
              'draw_filePath':'/datastore/liu121/nosqldb2/bbn_kn/data_test_draw.txt'},
             ]
    for config in configs:
        tv = TrainVer(config)
        tv.draw_labels()
        tv.verify()