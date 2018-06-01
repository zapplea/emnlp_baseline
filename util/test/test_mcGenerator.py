import random
import unittest
from unittest import TestCase

from nerd.data.util.mc_generator import McGenerator
from nerd.data.util.readers.WNUTDataReader import WNUTDataReader

from nerd.config.constants import Constants
from nerd.data.util.containers.mention_candidate import MentionCandidate


class TestMcGenerator(TestCase):

    def setUp(self):
        data = WNUTDataReader.readFile('/home/nikhil/workspace/data61/local/mwe_ner/ner/data/WNUT/train')
        self.random_seed = random.randint(0, 10**6)
        print("Using seed : %d" % (self.random_seed))
        self.mcGen = McGenerator(data, self.random_seed)


    def test_getPositiveCandidate(self):
        context_window_size = 2
        for i in range(5000):
            candidate = self.mcGen.get_positive_candidate(context_window=context_window_size)


            # Ensure candidate is of the correct class
            self.assertIsInstance(candidate, MentionCandidate)


            # Ensure candidate length is greater than zero
            self.assertGreater(len(candidate.text[1]), 0)


            # Ensure candidate length is less than equal to max size specified
            self.assertEqual(len(candidate.text[0]), context_window_size)
            self.assertEqual(len(candidate.text[2]), context_window_size)


            # Ensure positive candidate label is uniform
            label = candidate.labels[1][0]
            for candidate_label in candidate.labels[1]:
                self.assertEqual(candidate_label, label)

                # Ensure positive label is actually positive
                self.assertNotIn(candidate_label, [Constants.PAD_LABEL, Constants.OTHER_LABEL])


            # Ensure left and right context do not contain candidate tokens
            for context_label in [candidate.labels[0][context_window_size-1], candidate.labels[2][0]]:
                self.assertNotIn(context_label, [label],
                                "sentence Index : %d, start_index : %d, end_index : %d ----> %s" %
                                (candidate.sentence_index, candidate.start_index, candidate.end_index, str(candidate.labels)))

    def test_getNegativeCandidate(self):

        size_max_candidate = 5
        context_window_size = 2

        for i in range(5000):
            candidate = self.mcGen.get_negative_candidate(max_candidate_size=size_max_candidate,
                                                          context_window=context_window_size)

            # Ensure candidate is of the correct class
            self.assertIsInstance(candidate, MentionCandidate)

            # Ensure candidate length is greater than zero
            self.assertGreater(len(candidate.text[1]), 0)


            # Ensure left and right context length is correct
            self.assertEqual(len(candidate.text[0]), context_window_size)
            self.assertEqual(len(candidate.text[2]), context_window_size)

            # Ensure candidate length is less than equal to max size specified
            self.assertLessEqual(len(candidate.text[1]), size_max_candidate)

            # Ensure candidate is actually negative
            for candidate_label in candidate.labels[1]:
                self.assertIn(candidate_label, [Constants.PAD_LABEL, Constants.OTHER_LABEL],
                              "sentence Index : %d, start_index : %d, end_index : %d ----> %s" %
                              (candidate.sentence_index, candidate.start_index, candidate.end_index,
                               str(candidate.labels))
                              )

            # Check case 700

            # TODO : 1) Check word embedding

    def test_getAllCandidatesFromSentence(self):
        ignore_labels = ['PAD', 'OTHER']

        #################################
        # Case 1: back to back entities #
        #################################
        labels = ['OTHER', 'OTHER', 'OTHER', 'geo-loc', 'other', 'other', 'other', 'OTHER', 'OTHER']
        candidates = self.mcGen.get_all_candidates_from_sentence(labels, ignore_labels)

        # Check 2 candidates found
        self.assertEqual(len(candidates), 2)

        # Ensure candidate one starts at index 3 and ends at 4
        start_index, end_index, label = candidates[0]
        self.assertEqual(start_index, 3)
        self.assertEqual(end_index, 4)


        # Ensure candidate two starts at index 4 and ends at 7
        start_index, end_index, label = candidates[1]
        self.assertEqual(start_index, 4)
        self.assertEqual(end_index, 7)

        # Case 2: entities at the end of a sentence
        #------------------------------------------
        labels = ['OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER', 'geo-loc','OTHER', 'other', 'other', 'other']
        candidates = self.mcGen.get_all_candidates_from_sentence(labels, ignore_labels)

        # Check 2 candidates found
        self.assertEqual(len(candidates), 2)

        # Ensure candidate one starts at index 6 and ends at 7
        start_index, end_index, label = candidates[0]
        self.assertEqual(start_index, 5)
        self.assertEqual(end_index, 6)

        # Ensure candidate two starts at index 8 and ends at 11
        start_index, end_index, label = candidates[1]
        self.assertEqual(start_index, 7)
        self.assertEqual(end_index, 10)

        #
        # Case 3: entities at the start of a sentence
        #---------------------------------------------
        labels = ['geo-loc','other', 'other', 'other','OTHER', 'OTHER', 'OTHER', 'OTHER', 'OTHER']
        candidates = self.mcGen.get_all_candidates_from_sentence(labels, ignore_labels)

        # Check 2 candidates found
        self.assertEqual(len(candidates), 2)

        # Ensure candidate one starts at index 0 and ends at 1
        start_index, end_index, label = candidates[0]
        self.assertEqual(start_index, 0)
        self.assertEqual(end_index, 1)

        # Ensure candidate two starts at index 1 and ends at 4
        start_index, end_index, label = candidates[1]
        self.assertEqual(start_index, 1)
        self.assertEqual(end_index, 4)

        #
        # Case 4: All entities
        #--------------------------------------------

        labels = ['geo-loc', 'other', 'other', 'other', 'person', 'person', 'organization', 'organization']
        candidates = self.mcGen.get_all_candidates_from_sentence(labels, ignore_labels)

        # Check 2 candidates found
        self.assertEqual(len(candidates), 4)

        # Ensure candidate one starts at index 0 and ends at 1
        start_index, end_index, label = candidates[0]
        self.assertEqual(start_index, 0)
        self.assertEqual(end_index, 1)

        # Ensure candidate two starts at index 1 and ends at 4
        start_index, end_index, label = candidates[3]
        self.assertEqual(start_index, 6)
        self.assertEqual(end_index, 8)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMcGenerator)
    unittest.TextTestRunner(verbosity=2).run(suite)