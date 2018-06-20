# this file will delete "-DOCSTAR-" in the original data set

class Preprocess:
    def __init__(self,config):
        self.config = config

    def delete(self):
        conll_train=[]
        with open(self.config['org_filePath'],'r') as f:
            lastline=''
            for line in f:
                if "-DOCSTART-" in line:
                    continue
                else:
                    if lastline=='\n' and line=='\n':
                        continue
                    else:
                        conll_train.append(line)
                    lastline=line
        return conll_train

    def write(self,conll_train):
        with open(self.config['conll_outPath'],'w') as f:
            for line in conll_train:
                f.write(line)

if __name__=="__main__":
    configs=[#{'org_filePath':'/datastore/liu121/nosqldb2/conll2003/conll_train_original',
            #'conll_outPath':'/datastore/liu121/nosqldb2/conll2003/conll_train'},
             {'org_filePath':'/datastore/liu121/nosqldb2/conll2003/conll_testa_original',
            'conll_outPath':'/datastore/liu121/nosqldb2/conll2003/conll_testa'}]
    for config in configs:
        pp=Preprocess(config)
        conll_train = pp.delete()
        pp.write(conll_train)