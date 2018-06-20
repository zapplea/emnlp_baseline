import sys
sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader

from pathlib import Path
import json


class DataGenerator:
    def __init__(self, data_config):
        self.data_config = data_config

    def conll_data_reader(self, filePath):
        data = BBNDataReader.readFile(filePath=filePath)
        return data

    def target_nn_data_generator(self, data):
        data_len = len(data.text)
        # train_data = [[text,labels], ...]
        nn_data = []
        for i in range(data_len):
            text = data.text[i]
            labels = data.labels[i]
            if len(text) > self.data_config['max_len']:
                continue
            y_ = []
            for label in labels:
                if label == 'OTHER':
                    label = 'O'
                y_.append(label)
            nn_data.append((text, y_))
        return nn_data

    def target_data_gnerator(self):
        target_draw_data = self.conll_data_reader(self.data_config['target_train_Conll_filePath'])
        target_train_data = self.target_nn_data_generator(target_draw_data)
        return target_train_data

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

    def write(self, sample):
        for key in sample:
            data = sample[key]
            with open(self.data_config['conll_filePath']+key, 'w') as f:
                for instance in data:
                    text = instance[0]
                    label = instance[1]
                    for i in range(len(text)):
                        f.write(text[i]+' '+label[i])
                    f.write('\n')
                    f.flush()

    def main(self):
        target_train_data = self.target_data_gnerator()
        sample = self.target_data_split(target_train_data)
        self.write(sample)

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
        print(data_config['conll_filePath'],'\n')
        path = Path(data_config['conll_filePath'])
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        dg = DataGenerator(data_config)
        dg.main()