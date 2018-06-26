# This script trains the BiLSTM-CNN-CRF architecture for NER in German using
# the GermEval 2014 dataset (https://sites.google.com/site/germeval2014ner/).
# The code use the embeddings by Reimers et al. (https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/)
from __future__ import print_function

import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle
import argparse

from keras import backend as K

# :: Change into the working dir of the script ::
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# :: Logging level ::
loggingLevel = logging.INFO
logger = logging.getLogger()
logger.setLevel(loggingLevel)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(loggingLevel)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


######################################################
#
# Data preprocessing
#
######################################################
parser = argparse.ArgumentParser()
parser.add_argument('--mod',type=str)
args = parser.parse_args()

seeds = {
    'bbn_kn':                                   #Name of the dataset
        {'columns': {1:'tokens', 2:'NER_bbn'},    #CoNLL format for the input data. Column 1 contains tokens, column 2 contains NER information using BIO encoding
         'label': 'NER_bbn',                      #Which column we like to predict
         'evaluate': True,                        #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None},                   #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    'cadec':
        {'columns': {1:'tokens', 2:'NER_cadec'},
         'label': 'NER_cadec',
         'evaluate': True,
         'commentSymbol': None},
    'nvd':
        {'columns': {1:'tokens', 2:'NER_nvd'},
         'label': 'NER_nvd',
         'evaluate': True,
         'commentSymbol': None}
}

seed=seeds[args.mod]

k_shot = ['1.0', '2.0', '4.0', '8.0', '16.0']
datasets={}
for k in k_shot:
    datasets[args.mod+'__'+k]=seed

# :: Path on your computer to the word embeddings. Embeddings by Reimers et al. will be downloaded automatically ::
embeddingsPath = '/datastore/liu121/nosqldb2/emnlp_ukplab/skipgram'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)
print('data prepare successful: %s' % pickleFile)
######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)
# print('mappings type: ',type(mappings))
# for key in mappings:
#     print(key)
#     print(mappings[key])
#     print('===============')

# print('embeddings type:',type(data))
# for key in data:
#     print(key)
#     for subkey in data[key]:
#         print('--',subkey)
        # for subsubkey in data[key][subkey]:
        #     print('----',subsubkey)

    # print('=========================')

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN', 'maxCharLength': 50}

print('#######################'+args.mod+' #######################')
model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.modelSavePath = "/datastore/liu121/nosqldb2/emnlp_ukplab/models/[ModelName]_[DevScore]_[TestScore]_[Epoch]_bbn.h5"
model.fit(epochs=100)