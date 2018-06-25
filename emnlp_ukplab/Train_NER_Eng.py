# This script trains the BiLSTM-CNN-CRF architecture for NER in German using
# the GermEval 2014 dataset (https://sites.google.com/site/germeval2014ner/).
# The code use the embeddings by Reimers et al. (https://www.ukp.tu-darmstadt.de/research/ukp-in-challenges/germeval-2014/)
from __future__ import print_function

import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

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
seeds = {
    'bbn_kn':                                   #Name of the dataset
        {'columns': {1:'tokens', 2:'NER_bbn'},    #CoNLL format for the input data. Column 1 contains tokens, column 2 contains NER information using BIO encoding
         'label': 'NER_BIO',                      #Which column we like to predict
         'evaluate': True,                        #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None},                   #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    'cadec':
        {'columns': {1:'tokens', 2:'NER_cadec'},
         'label': 'NER_BIO',
         'evaluate': True,
         'commentSymbol': None},
    'nvd':
        {'columns': {1:'tokens', 2:'NER_nvd'},
         'label': 'NER_BIO',
         'evaluate': True,
         'commentSymbol': None}
}

k_shot = ['1.0', '2.0', '4.0', '8.0', '16.0']
datasets={}
for key in seeds:
    for k in k_shot:
        datasets[key+'__'+k]=seeds[key]

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
exit()
# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'charEmbeddings': 'CNN', 'maxCharLength': 50}

print('BiLSTM')
model = BiLSTM(params)
print('setMappings')
model.setMappings(mappings, embeddings)
print('setDataset')
model.setDataset(datasets, data)
model.modelSavePath = "/datastore/liu121/nosqldb2/emnlp_ukplab/models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
print('fit')
model.fit(epochs=25)



