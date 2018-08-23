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
parser.add_argument('--k_shot',type=str)
args = parser.parse_args()
print(args.mod+args.k_shot)

seeds = {
    'bbn_kn':                                   #Name of the dataset
        {'columns': {1:'tokens', 2:'NER_BIO'},    #CoNLL format for the input data. Column 1 contains tokens, column 2 contains NER information using BIO encoding
         'label': 'NER_BIO',                      #Which column we like to predict
         'evaluate': True,                        #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None},                   #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    'cadec':
        {'columns': {1:'tokens', 2:'NER_BIO'},
         'label': 'NER_BIO',
         'evaluate': True,
         'commentSymbol': None},
    'nvd':
        {'columns': {1:'tokens', 2:'NER_BIO'},
         'label': 'NER_BIO',
         'evaluate': True,
         'commentSymbol': None}
}

seed=seeds[args.mod]

# k_shot = ['1.0']
datasets={}
# for k in k_shot:
#     datasets[args.mod+'__'+k]=seed
if args.k_shot!='16.0' and args.k_shot!='1.0' and args.k_shot!='2.0' and args.k_shot!='4.0' and args.k_shot!='8.0':
    print('k_shot doesn\'t exist')
    exit()

datasets[args.mod+'__'+args.k_shot]=seed

# datasets={}
# datasets[args.mod+'__'+args.shot]=seeds[args.mod]

# :: Path on your computer to the word embeddings. Embeddings by Reimers et al. will be downloaded automatically ::
embeddingsPath = '/datastore/liu121/nosqldb2/emnlp_ukplab/skipgram'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile_train, pickleFile_dev, pickleFile_test = perpareDataset(embeddingsPath, datasets,args.k_shot)
print('data prepare successful: %s, %s, and %s' % (pickleFile_train, pickleFile_dev, pickleFile_test))
######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data_train = loadDatasetPickle(pickleFile_train)
embeddings, mappings, data_dev = loadDatasetPickle(pickleFile_dev)
embeddings, mappings, data_test = loadDatasetPickle(pickleFile_test)

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
model.setDataset(datasets, data_train, data_dev, data_test)
model.modelSavePath = "/datastore/liu121/nosqldb2/emnlp_ukplab/models/[ModelName]_bbn.h5"
eval_result=model.fit(epochs=30)

def report(eval_result, filePath):
    with open(filePath, 'w+') as f:
        for key in eval_result:
            info = eval_result[key]
            f.write('===================='+key+'====================')
            f.write(info['epoch']+'\n')
            f.write(info["per_f1"] + "\n")
            f.write(info['per_pre'] + '\n')
            f.write(info['per_recall'] + '\n')
            f.write(info["micro_f1"] + '\n')
            f.write(info["micro_pre"] + '\n')
            f.write(info["micro_recall"] + '\n')
            f.write(info["macro_f1"] + '\n')
            f.write(info["macro_pre"] + '\n')
            f.write(info["macro_recall"] + '\n')

report(eval_result,'/datastore/liu121/nosqldb2/emnlp_ukplab/report/report_%s%s.txt'%(args.mod,str(args.k_shot)))

