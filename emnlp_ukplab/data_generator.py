import sys
sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader

from pathlib import Path
import json
import random

class DataGenerator:
    def __init__(self, data_config):
        self.data_config = data_config

    def conll_data_reader(self, filePath):
        data = BBNDataReader.readFile(filePath=filePath)
        return data

    def check_begin(self,label,index,labels):
        if index ==0:
            new_label='B-'+label
        else:
            if label == labels[index-1]:
                new_label='I-'+label
            else:
                new_label = 'B-'+label
        return new_label

    def target_data_gnerator(self):
        data = self.conll_data_reader(self.data_config['target_train_Conll_filePath'])
        data_len = len(data.text)
        # train_data = [[text,labels], ...]
        nn_data = []
        for i in range(data_len):
            text = data.text[i]
            labels = data.labels[i]
            if len(text) > self.data_config['max_len']:
                continue
            y_ = []
            for i in range(len(labels)):
                if 'I-' in labels[i]:
                    labels[i] = labels[i].replace('I-', '')

            for i in range(len(labels)):
                label = labels[i]
                if label == 'OTHER' or label == 'O':
                    label = 'O'
                    y_.append(label)
                else:
                    label = self.check_begin(label, i, labels)
                    y_.append(label)
            nn_data.append((text, y_))
        return nn_data

    # TODO: there are two kindes of target data: draw and eval. draw is used to train, and eval to predict.
    def target_data_split(self, data):
        """

        :param data: [[text_ids,labels_id],...]
        :return: 
        """
        sample = {}
        train_conductor = json.load(open(self.data_config['target_train_jsonPath']))
        for i in range(self.data_config['groups_num']):
            for j in self.data_config['instances_num']:
                affix = str(j) + '.' + str(i)
                sample[affix] = []
                t2indexes = train_conductor[str(j)][str(i)]
                for key in t2indexes:
                    for instance in t2indexes[key]:
                        id = instance[0]
                        if id >= len(data):
                            # print(affix,' ',str(id))
                            continue
                        instance = data[id]
                        sample[affix].append(instance)
        return sample

    def dev_generator(self,nn_data):
        random.shuffle(nn_data)
        random.shuffle(nn_data)

        return nn_data[-self.data_config['dev_nums']:]

    def write(self, sample,dev_data):
        name = self.data_config['conll_filePath'].split('/')[-2]
        for key in sample:
            data = sample[key]
            path = Path(self.data_config['conll_filePath']+name+key+'/')
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            with open(self.data_config['conll_filePath']+name+key+'/'+'train.txt', 'w') as f:
                for instance in data:
                    text = instance[0]
                    label = instance[1]
                    for i in range(len(text)):
                        f.write(str(i+1)+' '+text[i]+' '+label[i]+'\n')
                    f.write('\n')
                    f.flush()
            with open(self.data_config['conll_filePath'] + name + key + '/' + 'dev.txt', 'w') as f:
                for instance in dev_data:
                    text = instance[0]
                    label = instance[1]
                    # print('================')
                    # print('len_text: ', len(text))
                    # print(text)
                    # print('len_label: ',len(label))
                    # print(label)
                    for i in range(len(text)):
                        # print(i)
                        # print(text[i])
                        # print(label[i])
                        f.write(str(i+1)+' '+text[i]+' '+label[i]+'\n')
                    f.write('\n')
                    f.flush()

    def main(self):
        target_train_data = self.target_data_gnerator()
        test_sample = self.target_data_split(target_train_data)
        dev_data = self.dev_generator(target_train_data)

        self.write(test_sample,dev_data)

if __name__ == "__main__":
    data_configs = [

        {'max_len': 100,  # bbn_kn
         'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/bbn_kn/data_test_draw.txt',
         'target_train_jsonPath': '/datastore/liu121/nosqldb2/bbn_kn/draw_kn.json',
         'groups_num': 5,
         'instances_num': [1, 2, 4, 8, 16],
         'conll_filePath':'/datastore/liu121/nosqldb2/emnlp_ukplab/data/bbn_kn/'},

        {'max_len': 200,  # 315 cadec
         'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/cadec/Conll/data_test_draw',
         'target_train_jsonPath': '/datastore/liu121/nosqldb2/cadec/json/draw.json',
         'groups_num': 5,
         'instances_num': [1, 2, 4, 8, 16],
         'conll_filePath': '/datastore/liu121/nosqldb2/emnlp_ukplab/data/cadec/'},

        # {'max_len': 200,  # cadec_simple
        #  'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/cadec_simple/cadec_draw.txt',
        #  'target_train_jsonPath': '/datastore/liu121/nosqldb2/cadec_simple/test_draw_cadec_cache_config.json',
        #  'groups_num': 5,
        #  'instances_num': [1, 2, 4, 8, 16],
        #  'conll_filePath':'/datastore/liu121/nosqldb2/emnlp_ukplab/data/cadec_simple'},

        {'max_len': 100,  # nvd
         'target_train_Conll_filePath': '/datastore/liu121/nosqldb2/nvd/nvd_test_draw',
         'target_train_jsonPath': '/datastore/liu121/nosqldb2/nvd/draw.json',
         'groups_num': 5,
         'instances_num': [1, 2, 4, 8, 16],
         'conll_filePath': '/datastore/liu121/nosqldb2/emnlp_ukplab/data/nvd/'},
    ]

    for data_config in data_configs:
        data_config['dev_nums']=1000
        print(data_config['conll_filePath'],'\n')
        path = Path(data_config['conll_filePath'])
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        dg = DataGenerator(data_config)
        dg.main()