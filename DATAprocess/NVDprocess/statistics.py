import sys
sys.path.append('/home/liu121/dlnlp')

from nerd.data.util.readers.BBNDataReader import BBNDataReader

class Statistics:
    def __init__(self):
        pass

    def loader(self,filePath):
        data = BBNDataReader.readFile(filePath=filePath)
        return data

    def count_type(self,data):
        text = data.text
        labels = data.labels
        freq = {}
        for label in labels:
            label_set = set(label)
            for l in label_set:
                if l not in freq:
                    freq[l] = 1
                else:
                    freq[l] += 1
        for key in freq:
            print('{} : {}'.format(key,freq[key]))
        print('============')


if __name__ == "__main__":
    data_config = ['/datastore/liu121/nosqldb2/nvd/nvd_test_draw','/datastore/liu121/nosqldb2/nvd/nvd_test_eval']

    stat = Statistics()
    for filePath in data_config:
        data = stat.loader(filePath)
        stat.count_type(data)