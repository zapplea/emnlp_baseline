import sys

sys.path.append('/home/liu121/dlnlp')
from nerd.data.util.readers.BBNDataReader import BBNDataReader
from nerd.data.util.mc_generator import McGenerator

class LenMention:
    def __init__(self, config):
        self.config = config

    def stat(self):
        train_data = BBNDataReader.readFile(filePath=self.config['conll_filePath'])
        train_mc_gen = McGenerator(train_data)
        pos_candidates = train_mc_gen.get_all_positive_candidates()
        max_len=0
        freq={}
        for candidate in pos_candidates:
            mc = candidate.text[1]
            mc_len = len(mc)
            if mc_len > max_len:
                max_len=mc_len
            if mc_len in freq:
                freq[mc_len]+=1
            else:
                freq[mc_len]=1
        print('max length: \n',max_len)
        print('frequency: \n',freq)

if __name__=="__main__":
    configs = {'bbn_kn':{'conll_filePath':'/datastore/liu121/nosqldb2/bbn_kn/data_test_draw'},
               'cadec':{'conll_filePath':'/datastore/liu121/nosqldb2/cadec/Conll/data_test_draw'},
               'nvd':{'conll_filePath':'/datastore/liu121/nosqldb2/nvd/nvd_test_draw'}}
    for key in configs:
        print(key)
        lm = LenMention(configs[key])
        lm.stat()