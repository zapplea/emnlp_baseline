import sys
sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader

import operator

class Stat:
    def __init__(self,data_config):
        self.data_config = data_config

    def conll_data_reader(self):
        data = BBNDataReader.readFile(filePath=self.data_config['file_path'])
        return data

    def stat(self,stat_dic):
        data=self.conll_data_reader()
        for i in range(len(data.text)):
            text=data.text[i]
            if len(text) in stat_dic:
                stat_dic[len(text)]+=1
            else:
                stat_dic[len(text)]=1
        return stat_dic


def report(stat_dic,report_path):

    with open(report_path,'a+') as f:
        stat = sorted(stat_dic,key=operator.itemgetter(0))
        sum = 0
        for element in stat:
            f.write(str(element)+'\n')
            sum += element[1]
        f.write('sum: ' + str(sum)+'\n')
        f.write('\n\n')


if __name__ == "__main__":
    data_configs=[
                [{'file_path':'/datastore/liu121/nosqldb2/bbn_kn/data_train.txt'},
                 {'file_path':'/datastore/liu121/nosqldb2/bbn_kn/data_test_draw.txt'},
                 {'file_path':'/datastore/liu121/nosqldb2/bbn_kn/data_test_eval.txt'}],
                [{'file_path':'/datastore/liu121/nosqldb2/cadec/Conll/data_test_draw'},
                 {'file_path':'/datastore/liu121/nosqldb2/cadec/Conll/data_test_eval'}],
                [{'file_path':'/datastore/liu121/nosqldb2/nvd/nvd_test_draw'},
                 {'file_path':'/datastore/liu121/nosqldb2/nvd/nvd_test_eval'}],
                [{'file_path':'/datastore/liu121/nosqldb2/cadec_simple/cadec_draw.txt'},
                 {'file_path':'/datastore/liu121/nosqldb2/cadec_simple/cadec_eval.txt'}]
                  ]

    for configs in data_configs:
        stat_dic={}
        for config in configs:
            stat=Stat(config)
            stat_dic=stat.stat(stat_dic)
        report(stat_dic,'report.log')