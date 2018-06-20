import logging
import random

from nerd.data.util.containers.text_data import TextData
from nerd.data.util.features.handcrafted_feature_generator import HandCraftFeatureGenerator

from nerd.config.constants import Constants
from nerd.config.hyper_parameters import HyperParameters
from nerd.data.util.containers.mention_candidate import MentionCandidate
from nerd.util.word_embedding_util import WordEmbeddingUtil


class McGenerator():


    def __init__(self, data, seed=None):

        assert data is not None and isinstance(data, TextData)

        self.data = data

        self.positive_random = random.Random()
        self.negative_random = random.Random()

        if seed is not None:
            self.positive_random.seed(seed)
            self.negative_random.seed(seed+1)


        # An dictionary of positive candidates in each sentence. Format as below:
        # {Sentence_Number: (Number_of_Candidates, list of (Start_index, End_index, Label))]
        self.positive_candidates_by_sentence = {}

        # A list of positive candidates in the dataset. Format as below:
        # (Sentence Number, start_index, end_index, label)
        self.positive_candidates = []

        self.max_mention_text_length = HyperParameters.MAX_MENTION_TEXT_LENGTH

        ignoreLabels = [data.id_to_label[0], data.id_to_label[1]]

        for i, text_labels in enumerate(data.labels):

            candidates = self.getAllCandidatesFromSentence(text_labels, ignoreLabels)

            # If candidates are retrieved
            if len(candidates) > 0:
                # Make note of them by the sentence
                self.positive_candidates_by_sentence[i] = (len(candidates), candidates)

                # and store each candidate separately in a list
                for candidate in candidates:

                    start_index, end_index, label = candidate
                    self.positive_candidates.append((i, start_index, end_index, label))

                    mention_text_length = end_index - start_index

                    if mention_text_length > HyperParameters.MAX_MENTION_TEXT_LENGTH:

                        logging.info("McGenerator Info : A MC was found of length %d, this is greater than the MAX_MENTION_TEXT_LENGTH of %d specified in the params file."
                                        % (end_index - start_index, HyperParameters.MAX_MENTION_TEXT_LENGTH))

                        if mention_text_length > self.max_mention_text_length:
                            self.max_mention_text_length = mention_text_length


        # A list to store the dict keys above for quick retrieval
        self.positive_candidate_keys = sorted(self.positive_candidates_by_sentence.keys())

        # Set the max mention text length for future reference
        HyperParameters.MAX_MENTION_TEXT_LENGTH = self.max_mention_text_length

        # Initialise feature generator
        self.feature_gen = HandCraftFeatureGenerator(self.data)

        print("McGenerator Info : Total of %d positive candidates in %d sentences."
              %(len(self.positive_candidates), len(data.text)))


    def getAllCandidatesFromSentence(self, labels, ignoreLabels):
        """
        Internal method to retrieve all positive candidates from the dataset
        
        
        :param labels: Dataset labels, array of sentence labels 
        :param ignoreLabels: Labels that shouldn't be considered positive. ie : OTHER and PAD
        :return: Returns tuple of format (start_index, end_index, label) 
        """

        start_index = -1
        candidate_label = None

        candidates = []

        for index, label in enumerate(labels + ['#EOS#']):  # Added an end of sentence label in case candidate is at the EOS
            if candidate_label is None:
                if label not in ignoreLabels:
                    candidate_label = label
                    start_index = index
            else:
                if not label == candidate_label:
                    end_index = index

                    candidates.append((start_index, end_index, candidate_label))

                    candidate_label = None

                    # To check for back to back candidates
                    if label not in ignoreLabels:
                        candidate_label = label
                        start_index = index


        return candidates

    def getCandidateFromSentence(self, sentence_index, start_index, end_index, context_window=2):
        """
        If the sentence number in the data set, start index, end_index and context_window size is known
        This method retrieves that particular candidate from the dataset.
        Note: To be used internally.
        
        :param sentence_index: Sentence number in the data set
        :param start_index: The starting position of the candidate in the sentence
        :param end_index: The position to the right of the end of the candidate 
        :param context_window: Length of context window, default 2
        :return: returns a mention candidate
        """

        ###################
        # Text and Labels #
        ###################

        # Fetch left context
        left_context_start = max(0, start_index - context_window)
        left_context_end   = start_index

        left_context = self.data[sentence_index].text[left_context_start:left_context_end]
        left_context_labels = self.data[sentence_index].labels[left_context_start:left_context_end]


        # Add padding if short
        if len(left_context) < context_window:
            left_context = [Constants.PAD_STRING]*(context_window - len(left_context)) + left_context
            left_context_labels = [Constants.PAD_LABEL]*(context_window - len(left_context_labels)) + left_context_labels



        # Fetch right context
        right_context_start = end_index
        right_context_end   = end_index + context_window

        right_context = self.data[sentence_index].text[right_context_start: right_context_end]
        right_context_labels = self.data[sentence_index].labels[right_context_start:right_context_end]

        # Add padding if short
        if len(right_context) < context_window:
            right_context += [Constants.PAD_STRING]*(context_window - len(right_context))
            right_context_labels += [Constants.PAD_LABEL]*(context_window - len(right_context_labels))

        candidate_text = self.data[sentence_index].text[start_index:end_index]
        candidate_labels = self.data[sentence_index].labels[start_index:end_index]

        ######################################
        # Add text_embeddings if initialized #
        ######################################
        left_context_vector = []
        candidate_vector = []
        right_context_vector = []

        if len(self.data[sentence_index].text_embeddings) > 0:

            candidate_vector = self.data[sentence_index].text_embeddings[start_index:end_index]
            left_context_vector = self.data[sentence_index].text_embeddings[left_context_start:left_context_end]
            right_context_vector = self.data[sentence_index].text_embeddings[right_context_start:right_context_end]

            if len(left_context_vector) < context_window:
                left_context_vector = [WordEmbeddingUtil.padding_vector] * (
                context_window - len(left_context_vector)) + left_context_vector

            if len(right_context_vector) < context_window:
                right_context_vector += [WordEmbeddingUtil.padding_vector] * (context_window - len(right_context_vector))

        #####################################
        # Add part of speech if initialized #
        #####################################
        left_context_pos = []
        candidate_pos = []
        right_context_pos = []

        if len(self.data[sentence_index].pos) > 0:
            candidate_pos = self.data[sentence_index].pos[start_index:end_index]
            left_context_pos = self.data[sentence_index].pos[left_context_start:left_context_end]
            right_context_pos = self.data[sentence_index].pos[right_context_start:right_context_end]

            if len(left_context_pos) < context_window:
                left_context_pos = [Constants.PAD_LABEL] * (context_window - len(left_context_pos)) + left_context_pos

            if len(right_context_pos) < context_window:
                right_context_pos += [Constants.PAD_LABEL] * (context_window - len(right_context_pos))

        ##################################
        # Generate sparse_feature_vector #
        ##################################
        sparse_feature_vector = self.feature_gen.get_sparse_feature_vector(mention_text=candidate_text,
                                                                           left_context=left_context,
                                                                           right_context=right_context,
                                                                           mention_pos=candidate_pos)

        mc = MentionCandidate(text = [left_context, candidate_text, right_context],
                              labels =[left_context_labels, candidate_labels, right_context_labels],
                              feature_vector = [left_context_vector, candidate_vector, right_context_vector],
                              sparse_feature_vector = sparse_feature_vector,
                              sentence_index = sentence_index,
                              start_index = start_index,
                              end_index = end_index,
                              context_window_size = context_window,
                              pos = [left_context_pos, candidate_pos, right_context_pos])

        return mc



    def getPositiveCandidate(self, context_window = 2):
        """
        Returns a positive mention candidate, randomly sampled from the data set.
        
        :param context_window: Determines the length of the context around the candidate
        :return: 
        """

        candidate = self.positive_random.choice(self.positive_candidates)
        sentence_index, start_index, end_index, label = candidate
        return self.getCandidateFromSentence(sentence_index, start_index, end_index, context_window)


    def getNegativeCandidate(self, context_window=2, max_candidate_size = 5):
        """
        Returns a negative mention candidate, randomly sampled from the data set.
        
        :param context_window: Determines the length of the context around the candidate
        :param max_candidate_size: 
        :return: 
        """
        sentence_index = 0
        possible_candidates = []

        # Loop and repick if candidate generation from that sentence isn't possible
        while len(possible_candidates) == 0:

            sentence_index = max(0, self.negative_random.randint(0, len(self.data.text)-1))
            sentence_length = len(self.data[sentence_index].text)
            candidate_size = self.negative_random.randint(1, max_candidate_size)

            possible_candidates = range(sentence_length)
            ban_list = []

            # If the sentence contains a positive candidate
            # Ensure that the positive candidate isn't chosen by adding indexes to a ban list
            if sentence_index in self.positive_candidates_by_sentence:
                num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]

                for start_index, end_index, label in candidates:
                    ban_list += list(range(start_index - context_window - candidate_size, end_index + context_window))

            possible_candidates = [x for x in possible_candidates if x not in ban_list]

        start_index = self.negative_random.choice(possible_candidates)
        end_index = start_index + candidate_size

        return self.getCandidateFromSentence(sentence_index, start_index, end_index, context_window)


