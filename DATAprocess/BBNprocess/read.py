import sys
sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader

data = BBNDataReader.readFile(filePath='/datastore/liu121/nosqldb2/bbn_data/data_test_draw')
print(data.text[2])
print(data.labels[2])
