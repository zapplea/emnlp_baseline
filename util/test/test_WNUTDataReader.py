import random
import unittest
from unittest import TestCase

from nerd.data.util import McGenerator
from nerd.data.util import WNUTDataReader


class TestWNUTDataReader(TestCase):

    def setUp(self):
        data = WNUTDataReader.readFile('/home/nikhil/workspace/data61/local/mwe_ner/ner/data/WNUT/train')
        self.random_seed = random.randint(0, 10**6)
        print("Using seed : %d" % (self.random_seed))
        self.mcGen = McGenerator(data, self.random_seed)

    def test_readFile(self):
        # TODO : Add read file checks
        return



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWNUTDataReader)
    unittest.TextTestRunner(verbosity=2).run(suite)