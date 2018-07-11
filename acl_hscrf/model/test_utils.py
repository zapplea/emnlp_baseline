import utils
import codecs


class Test:
    def __init__(self):
        pass

    def test_iob2(self):
        print('\niob2:\n')
        tags = ['O', 'O', 'I-LOC', 'I-LOC', 'O', 'O', 'I-PER', 'I-PER', 'I-PER', 'O','I-PER']
        utils.iob2(tags)
        print(tags)

    def test_iob_iobes(self):
        print '\niob_iobes:\n'
        tags = ['O', 'O', 'I-LOC', 'I-LOC', 'O', 'O', 'I-PER', 'I-PER', 'I-PER', 'O', 'I-PER']
        result = utils.iob_iobes(tags)
        print result

    def loadcorpus(self):
        with codecs.open('/datastore/liu121/nosqldb2/acl_hscrf/data/bbn_kn/bbn_kn__1.0/train.txt', 'r', 'utf-8') as f:
            lines = f.readlines()
        return lines

    def test_get_crf_scrf_label(self):
        print '\nget_crf_scrf_label:\n'
        SCRF_l_map = {}
        SCRF_l_map['PER'] = 0
        SCRF_l_map['LOC'] = 1
        SCRF_l_map['ORG'] = 2
        SCRF_l_map['MISC'] = 3
        CRF_l_map = {}
        for pre in ['S-', 'B-', 'I-', 'E-']:
            for suf in SCRF_l_map.keys():
                CRF_l_map[pre + suf] = len(CRF_l_map)
        SCRF_l_map['<START>'] = 4
        SCRF_l_map['<STOP>'] = 5
        SCRF_l_map['O'] = 6
        CRF_l_map['<start>'] = len(CRF_l_map)
        CRF_l_map['<pad>'] = len(CRF_l_map)
        CRF_l_map['O'] = len(CRF_l_map)
        print 'CRF_l_map',CRF_l_map
        print 'SCRF_l_map',SCRF_l_map

    def test_read_corpus(self):
        print '\ntest_read_corpus:\n'
        lines = self.loadcorpus()
        features, labels = utils.read_corpus(lines)
        for i in range(3):
            print 'features_%s:\n'% str(i),features[i]
            print 'labels_%s:\n'% str(i),labels[i]

    def test_generate_corpus(self):
        print('\ngenerate_corpus:\n')
        lines = self.loadcorpus()
        features, labels, feature_map, label_map = utils.generate_corpus(lines, if_shrink_feature=False,
                                                                   thresholds=1)
        for i in range(1):
            print 'features_%s: '% str(i),features[i]
            print 'labels_%s:'% str(i),labels[i]
        print 'feature_map:\n',feature_map.keys()
        print 'label_map:\n',label_map


    def test_generate_corpus_char(self):
        print('\ngenerate_corpus_char:\n')
        lines = self.loadcorpus()
        train_features, train_labels, f_map, _, c_map = utils.generate_corpus_char(lines, if_shrink_c_feature=True, c_thresholds=5, if_shrink_w_feature=False)
        for i in range(1):
            print 'train_features_%s: '% str(i),train_features[i]
            print 'train_labels_%s:'% str(i),train_labels[i]

        print 'f_map:'

    def main(self):
        self.test_iob2()
        self.test_iob_iobes()
        self.test_get_crf_scrf_label()
        self.test_read_corpus()
        self.test_generate_corpus()

if __name__ =="__main__":
    # TODO: need to verify what the labels used to test, with B-, S-?
    test = Test()
    test.main()