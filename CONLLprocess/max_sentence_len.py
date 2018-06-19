import sys
sys.path.append('/home/liu121/emnlp_baseline')
from nerd.data.util.readers.BBNDataReader import BBNDataReader


def conll_data_reader(filePath):
    data = BBNDataReader.readFile(filePath=filePath)
    return data

def max_length():
    paths=['/datastore/liu121/nosqldb2/conll2003/conll_train']
    max_len = 0
    for p in paths:
        data = conll_data_reader(p)
        for i in range(len(data.text)):
            text = data.text[i]
            if len(text)>max_len:
                max_len = len(text)
    print(max_len)

if __name__ == "__main__":
    max_length()