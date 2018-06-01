import unittest
from classifier import Classifier
from datafeed import DataFeedTest

class ClassifierTest(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(ClassifierTest, self).__init__(*args, **kwargs)
        self.nn_config={'lstm_cell_size':300,
                        'batch_size':30,
                        'vocabulary_size':2981402,
                        'words_num':20,
                        'feature_dim':200,
                        'lr':0.03,
                        'reg_rate':0.003,
                        'source_NETypes_num':18,
                        'target_NETypes_num':19,
                        'epoch':100001,
                        'pad_index':1}
        self.dft=DataFeedTest()
        self.cl=Classifier(self.nn_config,self.dft)

    def test_classifier(self):
        self.cl.classifier()

if __name__ == "__main__":
    unittest.main()