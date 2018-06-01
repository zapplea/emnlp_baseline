import logging
import random
import numpy as np
from scipy.sparse import coo_matrix

from nerd.data.util.containers.text_data import TextData
from nerd.data.util.features.handcrafted_feature_generator import HandCraftFeatureGenerator

from nerd.config.constants import Constants
from nerd.data.util.containers.mention_candidate import MentionCandidate
from nerd.util.math_util import MathUtil
from nerd.util.word_embedding_util import WordEmbeddingUtil


class McGenerator:

    def __init__(self, data, seed=None, init_feature_gen=None, include_handcraft_features=True, max_mc_size=5):

        assert data is not None and isinstance(data, TextData)

        self.data = data
        self.vector_dimension = None

        self.positive_random = random.Random()
        self.negative_random = random.Random()

        if seed is not None:
            self.positive_random.seed(seed)
            self.negative_random.seed(seed+1)

        # A dictionary of positive candidates in each sentence. Format as below:
        # {Sentence_Number: (Number_of_Candidates, list of (Start_index, End_index, Label))]
        self.positive_candidates_by_sentence = {}

        # A list of positive candidates in the dataset. Format as below:
        # (Sentence Number, start_index, end_index, label)
        self.positive_candidates = []

        # A dictionary of positive candidates by label
        # {label : [sentence_number, start_index, end_index]}
        self.positive_candidates_by_label = {}

        self.max_mention_text_length = max_mc_size

        ignore_labels = [data.id_to_label[0], data.id_to_label[1]]

        for i, text_labels in enumerate(data.labels):

            candidates = self._get_all_candidates_from_sentence(text_labels, ignore_labels)

            # If candidates are retrieved
            if len(candidates) > 0:
                # Make note of them by the sentence
                self.positive_candidates_by_sentence[i] = (len(candidates), candidates)

                # and store each candidate separately in a list
                for candidate in candidates:

                    start_index, end_index, label = candidate
                    self.positive_candidates.append((i, start_index, end_index, label))

                    mention_text_length = end_index - start_index

                    if mention_text_length > max_mc_size:

                        logging.info("McGenerator Info : A MC was found of length %d, this is greater than "
                                     "the MAX_MENTION_TEXT_LENGTH of %d specified in the params file."
                                     % (end_index - start_index, max_mc_size))

                        if mention_text_length > self.max_mention_text_length:
                            self.max_mention_text_length = mention_text_length

        # Get candidates by label
        for sentence_index, start_index, end_index, label in self.positive_candidates:
            if label not in self.positive_candidates_by_label:
                self.positive_candidates_by_label[label] = [(sentence_index, start_index, end_index)]
            else:
                self.positive_candidates_by_label[label].append((sentence_index, start_index, end_index))


        # A list to store the dict keys above for quick retrieval
        self.positive_candidate_keys = sorted(self.positive_candidates_by_sentence.keys())

        # Set the max mention text length for future reference
        max_mc_size = self.max_mention_text_length

        # Initialise feature generator
        self.feature_gen = HandCraftFeatureGenerator(self.data) if include_handcraft_features and init_feature_gen is None else init_feature_gen

        # Initialise dimension getters
        self.init_vector_dimensions()

        print("McGenerator Info : Total of %d positive candidates in %d sentences."
              % (len(self.positive_candidates), len(data.text)))

    def get_sentence_ids_for_label(self, label):
        """
        Return a list of sentence_ids, which the label occurs in.
        :param label: 
        :return: 
        """

        sentence_ids = []
        for sentence_index, start_index, end_index in self.positive_candidates_by_label[label]:
            sentence_ids.append(sentence_index)

        return sentence_ids

    def init_vector_dimensions(self):
        """Initialised dimension stats"""
        check_mc = self.get_positive_candidate()
        if len(check_mc.feature_vector[1]) > 0:
            self.word_embedding_dimension = len(check_mc.feature_vector[1][0])
            self.vector_dimension = len(check_mc.get_text_embedding())
        else:
            self.word_embedding_dimension = 0
            self.vector_dimension = 0

    def get_sparse_feature_dimension(self):
        """Returns dimension of the sparse feature vector"""
        return self.feature_gen.sparse_feature_size if self.feature_gen is not None else 0

    def get_word_embedding_dimension(self):
        """Returns the word embedding dimension"""
        if self.word_embedding_dimension is None:
            self.init_vector_dimensions()
        return self.word_embedding_dimension

    def get_dense_feature_dimension(self):
        """Returns dimension of the dense feature vector"""
        if self.vector_dimension is None:
            self.init_vector_dimensions()
        return self.vector_dimension

    def _get_all_candidates_from_sentence(self, labels, ignore_labels):
        """
        Internal method to retrieve all positive candidates from the dataset
        
        
        :param labels: Dataset labels, array of sentence labels 
        :param ignore_labels: Labels that shouldn't be considered positive. ie : OTHER and PAD
        :return: Returns tuple of format (start_index, end_index, label) 
        """

        start_index = -1
        candidate_label = None

        candidates = []

        # Added an end of sentence label in case candidate is at the EOS
        for index, label in enumerate(labels + ['#EOS#']):
            if candidate_label is None:
                if label not in ignore_labels:
                    candidate_label = label
                    start_index = index
            else:
                if not label == candidate_label:
                    end_index = index

                    if (end_index - start_index) <= self.max_mention_text_length:
                        candidates.append((start_index, end_index, candidate_label))

                    candidate_label = None

                    # To check for back to back candidates
                    if label not in ignore_labels:
                        candidate_label = label
                        start_index = index

        return candidates

    def get_candidate_from_sentence(self, sentence_index, start_index, end_index, context_window=2,
                                    add_sparse_features=True):
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
        left_context_end = start_index

        left_context = self.data[sentence_index].text[left_context_start:left_context_end]
        left_context_labels = self.data[sentence_index].labels[left_context_start:left_context_end]

        # Add padding if short
        if len(left_context) < context_window:
            left_context = [Constants.PAD_STRING]*(context_window - len(left_context)) + left_context
            left_context_labels = [Constants.PAD_LABEL]*(context_window-len(left_context_labels)) + left_context_labels

        # Fetch right context
        right_context_start = end_index
        right_context_end = end_index + context_window

        right_context = self.data[sentence_index].text[right_context_start: right_context_end]
        right_context_labels = self.data[sentence_index].labels[right_context_start:right_context_end]

        # Add padding if short
        if len(right_context) < context_window:
            right_context += [Constants.PAD_STRING]*(context_window - len(right_context))
            right_context_labels += [Constants.PAD_LABEL]*(context_window - len(right_context_labels))

        candidate_text = self.data[sentence_index].text[start_index:end_index]
        candidate_labels = self.data[sentence_index].labels[start_index:end_index]
        candidate_label_vector = self.data[sentence_index].label_vectors[start_index]

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
                left_context_vector = [WordEmbeddingUtil.padding_vector] * \
                                      (context_window - len(left_context_vector)) + left_context_vector

            if len(right_context_vector) < context_window:
                right_context_vector += [WordEmbeddingUtil.padding_vector] * (context_window-len(right_context_vector))

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
        sparse_vector_dimension = self.get_sparse_feature_dimension()
        if add_sparse_features and self.feature_gen is not None:
            sparse_feature_vector = self.feature_gen.get_sparse_feature_vector(mention_text=candidate_text,
                                                                               left_context=left_context,
                                                                               right_context=right_context,
                                                                               mention_pos=candidate_pos)
        else:

            sparse_feature_vector = coo_matrix((1, sparse_vector_dimension), dtype=np.float32)


        mc = MentionCandidate(text=[left_context, candidate_text, right_context],
                              labels=[left_context_labels, candidate_labels, right_context_labels],
                              label_vector= candidate_label_vector,
                              feature_vector=[left_context_vector, candidate_vector, right_context_vector],
                              sparse_feature_vector=sparse_feature_vector,
                              sparse_feature_dimension=sparse_vector_dimension,
                              sentence_index=sentence_index,
                              start_index=start_index,
                              end_index=end_index,
                              context_window_size=context_window,
                              pos=[left_context_pos, candidate_pos, right_context_pos])

        return mc

    def get_counts_by_positive_label(self):
        """
        Returns the counts of candidates by label
        :return: 
        """
        return {label : len(self.positive_candidates_by_label[label]) for label in self.positive_candidates_by_label.keys()}

    def get_all_positive_candidates_by_label(self, label, context_window=2):
        """
        Returns a list of positive candidates based on the label provided
        :return: positive_candidates_with_label : List <mention_candidate>
        """

        positive_candidates_with_label = []
        for sentence_index, start_index, end_index in self.positive_candidates_by_label[label]:
            mc = self.get_candidate_from_sentence(sentence_index, start_index, end_index, context_window)
            positive_candidates_with_label.append(mc)

        return positive_candidates_with_label

    def get_all_positive_candidates(self, context_window=2):
        """
        Returns a list of all the positive candidates from the data set
        :return: all_positive_candidates : List <mention_candidate>
        """

        all_positive_candidates = []
        for sentence_index, start_index, end_index, label in self.positive_candidates:
            mc = self.get_candidate_from_sentence(sentence_index, start_index, end_index, context_window)
            all_positive_candidates.append(mc)

        return all_positive_candidates

    def get_positive_candidate(self, context_window=2):
        """
        Returns a positive mention candidate, randomly sampled from the data set.
        
        :param context_window: Determines the length of the context around the candidate
        :return: 
        """

        candidate = self.positive_random.choice(self.positive_candidates)
        sentence_index, start_index, end_index, label = candidate
        return self.get_candidate_from_sentence(sentence_index, start_index, end_index, context_window)

    def get_negative_candidate(self, context_window=2, max_candidate_size=5):
        """
        Returns a negative mention candidate, randomly sampled from the data set.
        
        :param context_window: Determines the length of the context around the candidate
        :param max_candidate_size: Determines the length of the mention candidate 
        :return: 
        """
        sentence_index = 0
        candidate_size = 0
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

        return self.get_candidate_from_sentence(sentence_index, start_index, end_index, context_window)

    def get_negative_candidates_from_sentence(self, sentence_index, context_window=2, max_candidate_size=5, sample_window=None):
        """
        Returns all negative candidates from a sentence
        :param sentence_index: 
        :param context_window: 
        :param max_candidate_size: 
        :param sample_window: (start_index, end_index] Gets the negative candidates within this sample window
        :return: 
        """

        sentence_length = len(self.data[sentence_index].text)
        if sample_window is None:
            starting_positions = range(sentence_length)
        else:
            starting_positions = range(max(0, sample_window[0]), min(sentence_length, sample_window[1]))

        candidate_sizes = list(range(1, max_candidate_size + 1))
        # candidate_size = self.negative_random.randint(1, max_candidate_size)

        negative_candidates = []

        for candidate_size in candidate_sizes:
            possible_candidates = starting_positions
            ban_list = []

            # If the sentence contains a positive candidate
            # Ensure that the positive candidate isn't chosen by adding indexes to a ban list
            if sentence_index in self.positive_candidates_by_sentence:
                num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]

                for start_index, end_index, label in candidates:
                    ban_list += list(range(start_index - candidate_size, end_index))

            possible_candidates = [x for x in possible_candidates if x not in ban_list]

            for start_index in possible_candidates:
                end_index = start_index + candidate_size
                negative_candidates.append(self.get_candidate_from_sentence(sentence_index, start_index,
                                                                            end_index, context_window))

        return negative_candidates

    def get_positive_candidates_from_sentence(self, sentence_index, context_window=2):
        """
        Returns all positive candidates from a sentence
        :param sentence_index: 
        :param context_window:  
        :return: 
        """

        positive_candidates = []

        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]

            for start_index, end_index, label in candidates:
                positive_candidates.append(self.get_candidate_from_sentence(sentence_index=sentence_index,
                                                                            start_index=start_index,
                                                                            end_index=end_index,
                                                                            context_window=context_window))

        return positive_candidates



    def get_single_token_positive_candidates_from_sentence(self, sentence_index, context_window=2):
        """
        Get positive examples as single token candidates. Multi token candidates are split into multiple single token
        candidates.
        :param sentence_index: Sentence number to fetch from 
        :param context_window: context size around the single token
        :return: 
        """

        single_token_candidates = []

        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]

            for start_index, end_index, label in candidates:
                for i in range(start_index, end_index):
                    single_token_candidates.append(self.get_candidate_from_sentence(sentence_index=sentence_index,
                                                                                    start_index=i,
                                                                                    end_index=i+1,
                                                                                    context_window=context_window))

        return single_token_candidates

    def get_single_token_candidates_from_sentence(self, sentence_index, context_window=2):
        """
        Splits a sentence into multiple single tokens and returns them as list of mcs
        :param sentence_index: Sentence number to fetch from 
        :param context_window: context size around the single token
        :return: 
        """

        single_token_candidates = []
        sentence_length = len(self.data[sentence_index].text)

        for i in range(sentence_length):
            single_token_candidates.append(self.get_candidate_from_sentence(sentence_index=sentence_index,
                                                                            start_index=i,
                                                                            end_index=i+1,
                                                                            context_window=context_window))

        return single_token_candidates

    def get_variable_length_candidates_from_sentence(self, sentence_index, max_mc_size=5, context_window=2):
        """
        Splits a sentence into multiple candidate mcs upto a fixed max_mc_size.
        At each position i the candidate will be from (i-j)...i where j varies from [0,max_mc_size)
        :param sentence_index: 
        :param max_mc_size: 
        :param context_window: 
        :return: 
        """

        candidates = []
        sentence_length = len(self.data[sentence_index].text)

        for i in range(sentence_length):
            for j in range(max_mc_size):
                if i - j >= 0:
                    candidates.append(self.get_candidate_from_sentence(sentence_index=sentence_index,
                                                                       start_index=i-j,
                                                                       end_index=i + 1, context_window=context_window))
                else:
                    candidates.append(None)

        return candidates

    def prepare_candidates_for_testing(self, sentence_index, max_mc_size=5, context_window=2, dtype=np.float32):
        """
        Uses apis to prepare candidates and returns them as a list of vectors
        :param sentence_index: 
        :param max_mc_size: 
        :param context_window: 
        :return: 
        """

        candidates_for_neg_scores = self.get_single_token_candidates_from_sentence(sentence_index, context_window)

        candidates_for_pos_scores = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                                      max_mc_size=max_mc_size,
                                                                                      context_window=context_window)

        vectors_for_neg_scores = [mc.get_text_embedding() for mc in candidates_for_neg_scores]

        dummy_vector = np.zeros(len(vectors_for_neg_scores[0]), dtype=dtype)
        vectors_for_pos_scores = [mc.get_text_embedding() if mc is not None else dummy_vector
                                  for mc in candidates_for_pos_scores]

        return np.array(vectors_for_neg_scores, dtype=dtype), np.array(vectors_for_pos_scores, dtype=dtype)

    def get_label_for_candidate_window(self, sentence_index, window_start_position, window_end_position):
        """
        Checks if there is a true positive candidate that starts at the window_start_position and ends on or before the
        window end position in a window, if not returns the true negative candidate at window_start_position, 
         window_start_position + 1.
        :param sentence_index: 
        :param window_start_position: 
        :param window_end_position: 
        :return: 
        """
        label = 0
        true_start_index = window_start_position
        true_end_index = window_start_position + 1
        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]
            for start_index, end_index, _ in candidates:
                if start_index < window_end_position:
                    if start_index == window_start_position and end_index<=window_end_position:
                        label = 1
                        true_start_index = start_index
                        true_end_index = end_index
                else:
                    break

        return label, true_start_index, true_end_index

    def prepare_candidates_sparse_for_testing(self, sentence_index, positive_reference=None, negative_reference=None,
                                              ref_k=10, max_mc_size=5, context_window=2, dtype=np.float32):
        """
        Uses apis to prepare candidates and returns them as a list of vectors
        :param sentence_index: 
        :param max_mc_size: 
        :param context_window: 
        :return: 
        """

        candidates_for_neg_scores = self.get_single_token_candidates_from_sentence(sentence_index, context_window)

        candidates_for_pos_scores = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                                      max_mc_size=max_mc_size,
                                                                                      context_window=context_window)

        vectors_for_neg_scores = [mc.get_text_embedding() for mc in candidates_for_neg_scores]
        sparse_vectors_for_neg_scores = [mc.get_sparse_feature_vector() for mc in candidates_for_neg_scores]
        sparse_vectors_for_neg_scores = MathUtil.vstack_sparse_matrices(sparse_vectors_for_neg_scores)

        dummy_vector = np.zeros(len(vectors_for_neg_scores[0]), dtype=dtype)
        dummy_ref_vector = np.zeros(shape=(ref_k*2, len(vectors_for_neg_scores[0])), dtype=dtype)
        sparse_vector_dim = self.get_sparse_feature_dimension()
        dummy_sparse_vector = coo_matrix((1, sparse_vector_dim), dtype=dtype)

        vectors_for_pos_scores = [mc.get_text_embedding() if mc is not None else dummy_vector
                                  for mc in candidates_for_pos_scores]

        sparse_vectors_for_pos_scores = [mc.get_sparse_feature_vector() if mc is not None else dummy_sparse_vector
                                         for mc in candidates_for_pos_scores]

        sparse_vectors_for_pos_scores = MathUtil.vstack_sparse_matrices(sparse_vectors_for_pos_scores)

        '''Get reference vectors from memory'''
        ref_for_neg_scores = []
        ref_for_pos_scores = []
        if positive_reference is not None and negative_reference is not None:
            for mc in candidates_for_neg_scores:
                pos_ref = positive_reference.search_children(mc.get_text_embedding(), ref_k)
                neg_ref = negative_reference.search_children(mc.get_text_embedding(), ref_k)
                refs = np.concatenate((pos_ref, neg_ref), axis=0)
                ref_for_neg_scores.append(refs)

            for mc in candidates_for_pos_scores:
                if mc is not None and np.any(mc.get_text_embedding()):
                    pos_ref = positive_reference.search_children(mc.get_text_embedding(), ref_k)
                    neg_ref = negative_reference.search_children(mc.get_text_embedding(), ref_k)
                    refs = np.concatenate((pos_ref, neg_ref), axis=0)
                    ref_for_pos_scores.append(refs)
                else:
                    ref_for_pos_scores.append(dummy_ref_vector)

        return np.array(vectors_for_neg_scores, dtype=dtype), sparse_vectors_for_neg_scores,\
               np.array(vectors_for_pos_scores, dtype=dtype), sparse_vectors_for_pos_scores,\
               np.array(ref_for_neg_scores, dtype=dtype), np.array(ref_for_pos_scores, dtype=dtype)


    def prepare_candidates_for_training(self, sentence_index, start_index=None, end_index=None, max_mc_size=5,
                                        positive_reference=None, negative_reference=None, skip_mmat_ids=set(),
                                        context_window=2, ref_k=10, dtype=np.float32):

        sentence_length = len(self.data[sentence_index].text)
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = sentence_length

        x_p = []
        x_p_sparse = []
        x_p_ref = []

        y = []

        x_n = []
        x_n_sparse = []
        x_n_ref = []


        check_mc = self.get_positive_candidate()
        vector_dimension = len(check_mc.get_text_embedding())
        sparse_vector_dim = self.feature_gen.sparse_feature_size

        dummy_vector = np.zeros(vector_dimension, dtype=dtype)
        dummy_sparse_vector = coo_matrix((1, sparse_vector_dim), dtype=dtype)
        dummy_ref_vector = np.zeros((ref_k*2, vector_dimension), dtype=dtype)

        # Padding zero vectors for negative cases
        pad_vecs = dict()
        for d in range(max_mc_size+1):
            pad_vecs[d] = ([dummy_vector]*(max_mc_size-d), [dummy_sparse_vector]*(max_mc_size-d), [dummy_ref_vector]*(max_mc_size-d))

        # Precalculate all the single token mcs as they will all be re-used multiple times later for negative mcs
        negative_candidates = self.get_single_token_candidates_from_sentence(sentence_index, context_window=context_window)

        # Precalculate the corresponding reference vectors
        reference_vectors = []
        for negative_candidate in negative_candidates:
            negative_embedding = negative_candidate.get_text_embedding()
            pos_ref = positive_reference.search_children(negative_embedding, ref_k, skip_indexes=skip_mmat_ids)
            neg_ref = negative_reference.search_children(negative_embedding, ref_k, skip_indexes=skip_mmat_ids)
            reference_vector = np.concatenate((pos_ref, neg_ref), axis=0)
            reference_vectors.append(reference_vector)


        # Determine positive candidate positions
        positive_candidate_start = []
        positive_candidate_end = []

        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]
            for candidate_start_index, candidate_end_index, label in candidates:
                # save positive_candidate positions for future reference
                positive_candidate_start.append(candidate_start_index)
                positive_candidate_end.append(candidate_end_index)

        # Add sentence end index to support last index
        positive_candidate_start.append(end_index)
        positive_candidate_end.append(end_index)


        # Initialize next positive candidate
        next_positive_candidate_index = 0
        while start_index > positive_candidate_start[next_positive_candidate_index]:
            next_positive_candidate_index += 1

        # Main loop
        i = start_index
        while i < end_index:
            current_x_n = []
            current_x_n_sparse = []
            current_x_n_ref = []

            current_y = None

            current_x_p = None
            current_x_p_sparse = None
            current_x_p_ref = None

            label = 1 if (i == positive_candidate_start[next_positive_candidate_index]) else 0

            if label == 0:
                """ True Negative Candidate """
                for d in range(min((max_mc_size, i - start_index + 1))):
                    current_start_index = i - d
                    current_end_index = i + 1

                    # Compile x_n vectors
                    negative_embedding = negative_candidates[current_start_index].get_text_embedding()
                    negative_sparse = negative_candidates[current_start_index].get_sparse_feature_vector()
                    negative_ref = reference_vectors[current_start_index]

                    current_x_n.append(negative_embedding)
                    current_x_n_sparse.append(negative_sparse)
                    current_x_n_ref.append(negative_ref)

                    # Compile x_p vectors
                    if current_end_index - current_start_index == 1:
                        # if single token vector, positive mc is the same as negative
                        current_x_p = negative_embedding
                        current_x_p_sparse = negative_sparse
                        current_x_p_ref = negative_ref
                    else:
                        # Else construct multi token positive MC and compile x_hat vectors
                        positive_mc = self.get_candidate_from_sentence(sentence_index, current_start_index,
                                                                       current_end_index, context_window)

                        current_x_p = positive_mc.get_text_embedding()
                        current_x_p_sparse = positive_mc.get_sparse_feature_vector()

                        pos_ref = positive_reference.search_children(current_x_p, ref_k, skip_indexes=skip_mmat_ids)
                        neg_ref = negative_reference.search_children(current_x_p, ref_k, skip_indexes=skip_mmat_ids)

                        current_x_p_ref = np.concatenate((pos_ref, neg_ref), axis=0)

                    # Add the vectors to data list
                    x_n.extend(current_x_n)
                    x_n_sparse.extend(current_x_n_sparse)
                    x_n_ref.extend(current_x_n_ref)

                    # Extend x_n with padding dummy vectors
                    num_tokens = d + 1
                    x_n.extend(pad_vecs[num_tokens][0])
                    x_n_sparse.extend(pad_vecs[num_tokens][1])
                    x_n_ref.extend(pad_vecs[num_tokens][2])


                    y.append([label])

                    x_p.append(current_x_p)
                    x_p_sparse.append(current_x_p_sparse)
                    x_p_ref.append(current_x_p_ref)

                # Increment i
                i += 1

            else:
                """ True Positive candidate"""

                current_start_index = positive_candidate_start[next_positive_candidate_index]
                current_end_index = positive_candidate_end[next_positive_candidate_index]

                # Positive MC is of the right size
                if current_end_index - current_start_index <= max_mc_size:
                    # Increment positive candidate counter
                    next_positive_candidate_index += 1

                    positive_mc = self.get_candidate_from_sentence(sentence_index, current_start_index,
                                                                   current_end_index, context_window)

                    # Compile x_p vectors
                    current_x_p = positive_mc.get_text_embedding()
                    current_x_p_sparse = positive_mc.get_sparse_feature_vector()

                    pos_ref = positive_reference.search_children(current_x_p, ref_k, skip_indexes=skip_mmat_ids)
                    neg_ref = negative_reference.search_children(current_x_p, ref_k, skip_indexes=skip_mmat_ids)
                    current_x_p_ref = np.concatenate((pos_ref, neg_ref), axis=0)

                    # Compile x_n vectors
                    for d in range(current_start_index, current_end_index):
                        # sum up negative single token vectors between start and end of the positive mc
                        negative_embedding = negative_candidates[d].get_text_embedding()
                        negative_sparse = negative_candidates[d].get_sparse_feature_vector()
                        negative_ref = reference_vectors[d]

                        current_x_n.append(negative_embedding)
                        current_x_n_sparse.append(negative_sparse)
                        current_x_n_ref.append(negative_ref)

                    # Add the vectors to data list
                    x_p.append(current_x_p)
                    x_p_sparse.append(current_x_p_sparse)
                    x_p_ref.append(current_x_p_ref)

                    y.append([label])

                    x_n.extend(current_x_n)
                    x_n_sparse.extend(current_x_n_sparse)
                    x_n_ref.extend(current_x_n_ref)

                    # Extend x_n with padding dummy vectors
                    num_tokens = len(current_x_n)
                    x_n.extend(pad_vecs[num_tokens][0])
                    x_n_sparse.extend(pad_vecs[num_tokens][1])
                    x_n_ref.extend(pad_vecs[num_tokens][2])

                # Increment i
                i = current_end_index

        assert len(x_n) == len(x_p)*max_mc_size, 'Data compiled incorrectly'

        # Returning the generated data
        return x_p, x_p_sparse, x_p_ref, y, x_n, x_n_sparse, x_n_ref

    def prepare_candidates_for_joint_testing(self, sentence_index, negative_reference=None, k_shot=8, max_mc_size=5,
                                             time_steps=9, context_window=2, include_context=True, dtype=np.float32):
        """
        Uses apis to prepare candidates and returns them as a list of vectors
        :param sentence_index: 
        :param max_mc_size: 
        :param context_window: 
        :return: 
        """

        neg_candidates = self.get_single_token_candidates_from_sentence(sentence_index=sentence_index,
                                                                        context_window=context_window)

        pos_candidates = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                           max_mc_size=max_mc_size,
                                                                           context_window=context_window)

        # Initialise padding vectors
        vector_dimension = self.get_word_embedding_dimension()
        dummy_vector = np.zeros((time_steps, vector_dimension), dtype=dtype)
        neg_vec_size = (2*context_window + 1) if include_context else 1
        neg_padding_required = time_steps - neg_vec_size
        neg_cache_padding = np.zeros((1, k_shot, neg_padding_required, vector_dimension))

        pos_vectors = []
        pos_seq = []
        neg_vectors = []
        neg_seq = []
        cache_n = []
        cache_n_seq = []

        '''Get reference vectors from memory'''

        '''Construct positive candidates'''
        for mc in pos_candidates:
            # Append positive vector and mc length
            vec, vec_seq = mc.get_padded_text_embedding(include_context=include_context) if mc is not None else (dummy_vector, 0)
            pos_vectors.append(vec)
            pos_seq.append(vec_seq)

        '''Construct negative candidates'''
        for i, mc in enumerate(neg_candidates):
            # Append negative vector and mc length
            vec, vec_seq = mc.get_padded_text_embedding(include_context=include_context)
            neg_vectors.append(vec)
            neg_seq.append(vec_seq)

            # Construct mc specific cache for neg case
            negative_text_embedding = mc.get_text_embedding()
            neg_cache = negative_reference.search_children(negative_text_embedding, k_shot)
            neg_cache = np.array(neg_cache).reshape((1, k_shot, max_mc_size, vector_dimension))

            if not include_context:
                neg_cache = neg_cache[:, :, context_window:context_window + 1, :]
            neg_cache = np.concatenate((neg_cache, neg_cache_padding), axis=2)

            neg_cache_seq = np.ones((1, k_shot))*neg_vec_size

            cache_n.append(neg_cache)
            cache_n_seq.append(neg_cache_seq)

        return neg_vectors, neg_seq, pos_vectors, pos_seq, cache_n, cache_n_seq

    def extract_current_candidate(self, x, start_index, end_index, max_mc_size=5):
        """
        Given a list of candidates of dim : sentence_lengt*mc_size, this method candidates that was originally between 
        start_index and end_index
        """
        candidate_length = end_index - start_index
        candidate_group_position = (end_index - 1) * max_mc_size
        return x[candidate_group_position + candidate_length - 1]

    def extract_single_token_overlaps(self, x, start_index, end_index, max_mc_size=5):
        """
        Given a list of candidates of dim : sentence_lengt*mc_size, this method returns the single token candidates that
        overlap with the start_index and end_index        
        """
        results = []
        start, end, step = start_index*max_mc_size, end_index*max_mc_size, max_mc_size
        for i in range(start, end, step):
            results.append(x[i])
        return results

    def extract_overlaps(self, x, start_index, end_index, max_mc_size=5):
        """
        Given a list of candidates of dim : sentence_length*mc_size, this returns that candidates that overlap with the
        start index and end index provided.
        Note : the list of candidates are looking backwards (see get_variable_length_candidates_from_sentence(...))
        """
        results = []
        candidates_start = start_index * max_mc_size
        candidates_end = end_index * max_mc_size


        if start_index < max_mc_size:
            # append all candidates on the left side of end_index
            results.extend([mc for mc in x[candidates_start:candidates_end] if mc is not None])

            # append all on the right side of end index.
            for i in range(max_mc_size):
                num_right_candidates = max_mc_size - i - 1
                candidates_start = end_index * max_mc_size + i * max_mc_size + i + 1
                candidates_end = candidates_start + num_right_candidates
                results.extend([mc for mc in x[candidates_start:candidates_end] if mc is not None])
        else:
            # append all candidates on the left side of end_index
            results.extend(x[candidates_start:candidates_end])

            # append all on the right side of end index.
            for i in range(max_mc_size):
                num_right_candidates = max_mc_size - i - 1
                candidates_start = end_index * max_mc_size + i * max_mc_size + i + 1
                candidates_end = candidates_start + num_right_candidates
                results.extend(x[candidates_start:candidates_end])

        return results


    def prepare_candidates_for_joint_testing_with_context(self, sentence_index, negative_reference=None, k_shot=8,
                                                          max_mc_size=5, time_steps=9, context_window=2,
                                                          include_context=True, skip_mmat_ids=set(), dtype=np.float32):
        """
        Uses apis to prepare candidates and returns them as a list of vectors
        :param sentence_index: 
        :param max_mc_size: 
        :param context_window: 
        :return: 
        """

        neg_candidates = self.get_single_token_candidates_from_sentence(sentence_index=sentence_index,
                                                                        context_window=context_window)

        pos_candidates = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                           max_mc_size=max_mc_size,
                                                                           context_window=context_window)

        # Initialise padding vectors
        vector_dimension = self.get_word_embedding_dimension()
        dummy_vector = np.zeros((time_steps, vector_dimension), dtype=dtype)
        dummy_context = np.zeros((context_window*2, vector_dimension), dtype=dtype)
        neg_vec_size = (2*context_window + 1) if include_context else 1
        neg_padding_required = time_steps - neg_vec_size
        neg_cache_padding = np.zeros((1, k_shot, neg_padding_required, vector_dimension))

        pos_vectors = []
        pos_context = []
        pos_seq = []
        neg_vectors = []
        neg_context = []
        neg_seq = []
        cache_n = []
        cache_n_context = []
        cache_n_seq = []

        '''Get reference vectors from memory'''

        '''Construct positive candidates'''
        for mc in pos_candidates:
            # Append positive vector and mc length
            vec, vec_context, vec_seq = mc.get_padded_text_and_context_embedding() if mc is not None else (dummy_vector, dummy_context, 0)
            pos_vectors.append(vec)
            pos_context.append(vec_context)
            pos_seq.append(vec_seq)

        '''Construct negative candidates'''
        for i, mc in enumerate(neg_candidates):
            # Append negative vector and mc length
            vec, vec_context, vec_seq = mc.get_padded_text_and_context_embedding()
            neg_vectors.append(vec)
            neg_context.append(vec_context)
            neg_seq.append(vec_seq)

            # Construct mc specific cache for neg case
            negative_text_embedding = mc.get_text_embedding()
            neg_cache = negative_reference.search_children(negative_text_embedding, k_shot, skip_indexes=skip_mmat_ids)
            neg_cache = np.array(neg_cache).reshape((1, k_shot, max_mc_size, vector_dimension))

            neg_cache = neg_cache[:, :, context_window:context_window + 1, :]
            neg_cache = np.concatenate((neg_cache, neg_cache_padding), axis=2)
            neg_cache_context_left = neg_cache[:, :, 0:context_window, :]
            neg_cache_context_right = neg_cache[:, :, -context_window:, :]
            neg_cache_context = np.concatenate((neg_cache_context_left, neg_cache_context_right), axis=2)
            neg_cache_seq = np.ones((1, k_shot)) * neg_vec_size

            cache_n.append(neg_cache)
            cache_n_context.append(neg_cache_context)
            cache_n_seq.append(neg_cache_seq)


        return neg_vectors, neg_context, neg_seq, pos_vectors, pos_context, pos_seq, cache_n, cache_n_context, cache_n_seq

    def prepare_candidates_jigsaw_joint_training_with_context(self, sentence_index, start_index=None, end_index=None,
                                                              max_mc_size=5, negative_reference=None,
                                                              skip_mmat_ids=set(), label_dict=None,
                                                              context_window=2, k_shot=8, time_steps=9,
                                                              dtype=np.float32):

        """Extracts the mention candidates for training, randomly sample overlaps for false positive cases"""

        sentence_length = len(self.data[sentence_index].text)
        # Initialise inputs
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = sentence_length
        if label_dict is None:
            label_dict = dict()

        # Initialise outputs
        x_p = []
        x_p_context = []
        x_p_seq = []

        y_p = []
        y_n = []

        x_n = []
        x_n_context = []
        x_n_seq = []

        x_n_cache = []
        x_n_cache_context = []
        x_n_cache_seq = []

        # Get all possible candidates from sentence
        all_mcs = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                    max_mc_size=max_mc_size,
                                                                    context_window=context_window)

        # Initialise padding vectors
        pad_label = self.data.id_to_label[0]
        other_label = self.data.id_to_label[1]
        num_labels = len(label_dict[other_label]) - 1
        vector_dimension = self.get_word_embedding_dimension()
        dummy_vector = np.zeros((time_steps, vector_dimension), dtype=dtype)
        dummy_context = np.zeros((context_window*2, vector_dimension), dtype=dtype)
        dummy_label = np.zeros(num_labels)
        neg_vec_size =  1
        neg_padding_required = time_steps - neg_vec_size
        neg_cache_padding = np.zeros((1, k_shot, neg_padding_required, vector_dimension))

        # Determine positive candidate positions
        positive_candidate_start = []
        positive_candidate_end = []

        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]
            for candidate_start_index, candidate_end_index, label in candidates:
                # save positive_candidate positions for future reference
                positive_candidate_start.append(candidate_start_index)
                positive_candidate_end.append(candidate_end_index)

        # Add sentence end index to support last index
        positive_candidate_start.append(end_index)
        positive_candidate_end.append(end_index)

        # Initialize next positive candidate
        next_positive_candidate_index = 0
        while start_index > positive_candidate_start[next_positive_candidate_index]:
            next_positive_candidate_index += 1

        # Main loop
        i = start_index
        while i < end_index:

            if i == positive_candidate_start[next_positive_candidate_index]:
                """ Encountered true positive candidate """
                is_positive = True
                candidate_start_index = i
                candidate_end_index = positive_candidate_end[next_positive_candidate_index]
                # Increment counters
                i = positive_candidate_end[next_positive_candidate_index]
                next_positive_candidate_index +=1
                pos_sample_count = max_mc_size - 1

            else:
                """ Encountered true negative candidate """
                is_positive = False
                candidate_start_index = i
                candidate_end_index = i + 1
                pos_sample_count = max_mc_size
                # Increment counters
                i +=1

            """Sample the overlap counter examples"""
            neg_candidates = self.extract_single_token_overlaps(all_mcs, candidate_start_index, candidate_end_index,
                                                               max_mc_size)
            pos_candidates = self.extract_overlaps(all_mcs, candidate_start_index, candidate_end_index,
                                                   max_mc_size)

            # Randomly sample required amount
            possible_pos_samples = min(len(pos_candidates), pos_sample_count)
            neg_candidates = random.sample(neg_candidates, 1)
            pos_candidates = random.sample(pos_candidates, possible_pos_samples)

            """Get the current example"""
            # Current candidate
            current_mc = self.extract_current_candidate(all_mcs, candidate_start_index, candidate_end_index, max_mc_size)

            """Add all the examples to the result lists"""
            # True candidate
            if is_positive:
                y_p.append(label_dict[current_mc.get_label()][1:])
                vec, vec_context, vec_seq = current_mc.get_padded_text_and_context_embedding()
                x_p.append(vec)
                x_p_context.append(vec_context)
                x_p_seq.append(vec_seq)

            # False positive candidates
            for pos_candidate in pos_candidates:
                y_p.append(dummy_label)
                vec, vec_context, vec_seq = pos_candidate.get_padded_text_and_context_embedding()
                x_p.append(vec)
                x_p_context.append(vec_context)
                x_p_seq.append(vec_seq)

            # Remainder dummy labels if possible_pos_samples < pos_sample_count
            for j in range(possible_pos_samples, pos_sample_count):
                y_p.append(dummy_label)
                x_p.append(dummy_vector)
                x_p_context.append(dummy_context)
                x_p_seq.append(0)

            # False negative_candidate
            y_n_label = 0 if is_positive else 1
            y_n.append(y_n_label)
            vec, vec_context, vec_seq = neg_candidates[0].get_padded_text_and_context_embedding()
            x_n.append(vec)
            x_n_context.append(vec_context)
            x_n_seq.append(vec_seq)

            # Construct mc specific cache for neg case
            negative_text_embedding = neg_candidates[0].get_text_embedding()
            neg_cache = negative_reference.search_children(negative_text_embedding, k_shot)
            neg_cache = np.array(neg_cache).reshape((1, k_shot, max_mc_size, vector_dimension))

            neg_cache = neg_cache[:, :, context_window:context_window + 1, :]
            neg_cache = np.concatenate((neg_cache, neg_cache_padding), axis=2)
            neg_cache_context_left = neg_cache[:, :, 0:context_window, :]
            neg_cache_context_right = neg_cache[:, :, -context_window:, :]
            neg_cache_context = np.concatenate((neg_cache_context_left, neg_cache_context_right), axis=2)
            neg_cache_seq = np.ones((1, k_shot)) * neg_vec_size

            x_n_cache.append(neg_cache)
            x_n_cache_context.append(neg_cache_context)
            x_n_cache_seq.append(neg_cache_seq)

        num_candidates = len(y_n)
        # Reshape labels for concatenation
        y_n = np.array(y_n).reshape((num_candidates, 1))
        y_p = np.array(y_p).reshape(num_candidates, max_mc_size * num_labels)
        y = np.concatenate((y_n, y_p), axis=1).reshape((num_candidates, -1))


        return x_p, x_p_context, x_p_seq, y, x_n, x_n_context, x_n_seq, x_n_cache, x_n_cache_context, x_n_cache_seq


    def prepare_candidates_for_senner_jigsaw_joint_training(self, sentence_index, start_index=None, end_index=None,
                                                            max_mc_size=5, negative_reference=None,
                                                            skip_mmat_ids=set(), include_context=True,
                                                            label_dict=None, context_window=2, k_shot=8, time_steps=9,
                                                            dtype=np.float32):
        """Extracts the mention candidates for training, randomly sample overlaps for false positive cases"""

        sentence_length = len(self.data[sentence_index].text)
        # Initialise inputs
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = sentence_length
        if label_dict is None:
            label_dict = dict()

        # Initialise outputs
        x_p = []
        x_p_seq = []

        y_p = []
        y_n = []

        x_n = []
        x_n_seq = []

        x_n_cache = []
        x_n_cache_seq = []

        # Get all possible candidates from sentence
        all_mcs = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                    max_mc_size=max_mc_size,
                                                                    context_window=context_window)

        # Initialise padding vectors
        pad_label = self.data.id_to_label[0]
        other_label = self.data.id_to_label[1]
        num_labels = len(label_dict[other_label]) - 1
        vector_dimension = self.get_word_embedding_dimension()
        dummy_vector = np.zeros((time_steps, vector_dimension), dtype=dtype)
        dummy_label = np.zeros(num_labels)
        neg_vec_size =  (2 * context_window + 1) if include_context else 1
        neg_padding_required = time_steps - neg_vec_size
        neg_cache_padding = np.zeros((1, k_shot, neg_padding_required, vector_dimension))

        # Determine positive candidate positions
        positive_candidate_start = []
        positive_candidate_end = []

        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]
            for candidate_start_index, candidate_end_index, label in candidates:
                # save positive_candidate positions for future reference
                positive_candidate_start.append(candidate_start_index)
                positive_candidate_end.append(candidate_end_index)

        # Add sentence end index to support last index
        positive_candidate_start.append(end_index)
        positive_candidate_end.append(end_index)

        # Initialize next positive candidate
        next_positive_candidate_index = 0
        while start_index > positive_candidate_start[next_positive_candidate_index]:
            next_positive_candidate_index += 1

        # Main loop
        i = start_index
        while i < end_index:

            if i == positive_candidate_start[next_positive_candidate_index]:
                """ Encountered true positive candidate """
                is_positive = True
                candidate_start_index = i
                candidate_end_index = positive_candidate_end[next_positive_candidate_index]
                # Increment counters
                i = positive_candidate_end[next_positive_candidate_index]
                next_positive_candidate_index +=1
                pos_sample_count = max_mc_size - 1

            else:
                """ Encountered true negative candidate """
                is_positive = False
                candidate_start_index = i
                candidate_end_index = i + 1
                pos_sample_count = max_mc_size
                # Increment counters
                i +=1

            """Sample the overlap counter examples"""
            neg_candidates = self.extract_single_token_overlaps(all_mcs, candidate_start_index, candidate_end_index,
                                                               max_mc_size)
            pos_candidates = self.extract_overlaps(all_mcs, candidate_start_index, candidate_end_index,
                                                   max_mc_size)

            # Randomly sample required amount
            possible_pos_samples = min(len(pos_candidates), pos_sample_count)
            neg_candidates = random.sample(neg_candidates, 1)
            pos_candidates = random.sample(pos_candidates, possible_pos_samples)

            """Get the current example"""
            # Current candidate
            current_mc = self.extract_current_candidate(all_mcs, candidate_start_index, candidate_end_index, max_mc_size)

            """Add all the examples to the result lists"""
            # True candidate
            if is_positive:
                y_p.append(label_dict[current_mc.get_label()][1:])
                vec, vec_seq = current_mc.get_padded_text_embedding(include_context=include_context)
                x_p.append(vec)
                x_p_seq.append(vec_seq)

            # False positive candidates
            for pos_candidate in pos_candidates:
                y_p.append(dummy_label)
                vec, vec_seq = pos_candidate.get_padded_text_embedding(include_context=include_context)
                x_p.append(vec)
                x_p_seq.append(vec_seq)

            # Remainder dummy labels if possible_pos_samples < pos_sample_count
            for j in range(possible_pos_samples, pos_sample_count):
                y_p.append(dummy_label)
                x_p.append(dummy_vector)
                x_p_seq.append(0)

            # False negative_candidate
            y_n_label = 0 if is_positive else 1
            y_n.append(y_n_label)
            vec, vec_seq = neg_candidates[0].get_padded_text_embedding(include_context=include_context)
            x_n.append(vec)
            x_n_seq.append(vec_seq)

            # Construct mc specific cache for neg case
            negative_text_embedding = neg_candidates[0].get_text_embedding()
            neg_cache = negative_reference.search_children(negative_text_embedding, k_shot)
            neg_cache = np.array(neg_cache).reshape((1, k_shot, max_mc_size, vector_dimension))

            if not include_context:
                neg_cache = neg_cache[:, :, context_window:context_window + 1, :]
            neg_cache = np.concatenate((neg_cache, neg_cache_padding), axis=2)

            neg_cache_seq = np.ones((1, k_shot)) * neg_vec_size

            x_n_cache.append(neg_cache)
            x_n_cache_seq.append(neg_cache_seq)

        num_candidates = len(y_n)
        # Reshape labels for concatenation
        y_n = np.array(y_n).reshape((num_candidates, 1))
        y_p = np.array(y_p).reshape(num_candidates, max_mc_size * num_labels)
        y = np.concatenate((y_n, y_p), axis=1).reshape((num_candidates, -1))

        return x_p, x_p_seq, y, x_n, x_n_seq, x_n_cache, x_n_cache_seq


    def prepare_candidates_for_senner_greedy_joint_training(self, sentence_index, start_index=None, end_index=None,
                                                     max_mc_size=5, negative_reference=None,
                                                     skip_mmat_ids=set(), include_context=True,
                                                     label_dict=None, context_window=2, k_shot=8, time_steps=9,
                                                     dtype=np.float32):

        sentence_length = len(self.data[sentence_index].text)
        # Initialise inputs
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = sentence_length
        if label_dict is None:
            label_dict = dict()

        # Initialise outputs
        x_p = []
        x_p_seq = []

        y_p = []
        y_n = []

        x_n = []
        x_n_seq = []

        x_n_cache = []
        x_n_cache_seq = []

        neg_candidates = self.get_single_token_candidates_from_sentence(sentence_index=sentence_index,
                                                                        context_window=context_window)

        pos_candidates = self.get_variable_length_candidates_from_sentence(sentence_index=sentence_index,
                                                                           max_mc_size=max_mc_size,
                                                                           context_window=context_window)

        # Initialise padding vectors
        pad_label = self.data.id_to_label[0]
        other_label = self.data.id_to_label[1]
        non_entity_labels = [pad_label, other_label]
        num_labels = len(label_dict[other_label]) - 1
        vector_dimension = self.get_word_embedding_dimension()
        dummy_vector = np.zeros((time_steps, vector_dimension), dtype=dtype)
        dummy_label = np.zeros(num_labels)
        neg_vec_size =  (2 * context_window + 1) if include_context else 1
        neg_padding_required = time_steps - neg_vec_size
        neg_cache_padding = np.zeros((1, k_shot, neg_padding_required, vector_dimension))

        '''Construct positive candidates'''
        for mc in pos_candidates:
            # Append positive vector and mc length
            vec, vec_seq = mc.get_padded_text_embedding(include_context=include_context) if mc is not None else (dummy_vector, 0)
            x_p.append(vec)
            x_p_seq.append(vec_seq)

            is_true_positive = mc is not None and mc.get_label() not in non_entity_labels

            if is_true_positive:
                y_p.append(label_dict[mc.get_label()][1:])
            else:
                y_p.append(dummy_label)

        y_p = np.array(y_p).reshape(sentence_length, max_mc_size * num_labels)
        true_positive = np.sum(y_p, axis=1)

        '''Construct negative candidates'''
        for i, mc in enumerate(neg_candidates):
            # Append negative vector and mc length
            vec, vec_seq = mc.get_padded_text_embedding(include_context=include_context)
            x_n.append(vec)
            x_n_seq.append(vec_seq)

            # Construct mc specific cache for neg case
            negative_text_embedding = mc.get_text_embedding()
            neg_cache = negative_reference.search_children(negative_text_embedding, k_shot)
            neg_cache = np.array(neg_cache).reshape((1, k_shot, max_mc_size, vector_dimension))

            if not include_context:
                neg_cache = neg_cache[:, :, context_window:context_window + 1, :]
            neg_cache = np.concatenate((neg_cache, neg_cache_padding), axis=2)

            neg_cache_seq = np.ones((1, k_shot)) * neg_vec_size

            x_n_cache.append(neg_cache)
            x_n_cache_seq.append(neg_cache_seq)

            # y_n labels if no true positive at this index
            if true_positive[i] == 0:
                y_n.append(1)
            else:
                y_n.append(0)

        y_n = np.array(y_n).reshape((sentence_length, 1))
        y = np.concatenate((y_n, y_p), axis=1).reshape((sentence_length, -1))
        assert len(y[0]) == (num_labels*max_mc_size) + 1 , 'Expected len(y) = %d, got %s' % ((num_labels*max_mc_size) + 1,
                                                                                          y.shape)

        return x_p, x_p_seq, y, x_n, x_n_seq, x_n_cache, x_n_cache_seq

    def prepare_candidates_for_senner_joint_training(self, sentence_index, start_index=None, end_index=None,
                                                     max_mc_size=5, negative_reference=None,
                                                     skip_mmat_ids=set(), include_context=True,
                                                     label_dict=None, context_window=2, k_shot=8, time_steps=9,
                                                     dtype=np.float32):

        sentence_length = len(self.data[sentence_index].text)
        # Initialise inputs
        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = sentence_length
        if label_dict is None:
            label_dict = dict()

        # Initialise outputs
        x_p = []
        x_p_seq = []
        x_p_sparse = []

        y = []

        x_n = []
        x_n_seq = []
        x_n_sparse = []
        x_n_cache = []
        x_n_cache_seq = []

        x_n_train_mask = []

        vector_dimension = self.get_word_embedding_dimension()
        sparse_vector_dim = self.get_sparse_feature_dimension()

        dummy_vector = np.zeros((time_steps, vector_dimension), dtype=dtype)
        dummy_sparse_vector = coo_matrix((1, sparse_vector_dim), dtype=dtype)
        dummy_cache_vector = np.zeros((1, k_shot, time_steps, vector_dimension), dtype=dtype)
        dummy_cache_seq = np.ones((1, k_shot))

        # Padding zero vectors for negative cases
        neg_cache_length = 5 if include_context else 1
        pad_vecs = dict()
        neg_train_mask = dict()
        for d in range(max_mc_size+1):
            pad_length = max_mc_size - d
            pad_vecs[d] = ([dummy_vector]*pad_length, [dummy_sparse_vector]*pad_length,
                           [dummy_cache_vector]*pad_length, [neg_cache_length]*pad_length,
                           [dummy_cache_seq]*pad_length)
            neg_train_mask[d] = [np.concatenate(([1]*d, [0]*(max_mc_size - d)))]


        # Precalculate all the single token mcs as they will all be re-used multiple times later for negative mcs
        negative_candidates = self.get_single_token_candidates_from_sentence(sentence_index, context_window=context_window)

        # Precalculate the corresponding reference vectors
        reference_cache_seq = np.ones((1, k_shot)) * neg_cache_length
        reference_cache_n = []
        neg_padding_required = time_steps - max_mc_size if include_context else time_steps - 1
        neg_cache_padding = np.zeros((1, k_shot, neg_padding_required, vector_dimension))
        for negative_candidate in negative_candidates:
            negative_text_embedding = negative_candidate.get_text_embedding()

            # Get top k cache and pad results
            neg_cache = negative_reference.search_children(negative_text_embedding, k_shot)
            neg_cache = np.array(neg_cache).reshape((1, k_shot, max_mc_size, vector_dimension))
            if not include_context:
                neg_cache = neg_cache[:, context_window:context_window+1, :]
            neg_cache = np.concatenate((neg_cache, neg_cache_padding), axis=2)
            reference_cache_n.append(neg_cache)

        assert len(reference_cache_n) == sentence_length, 'Mismatch in number of negative references'


        # Determine positive candidate positions
        positive_candidate_start = []
        positive_candidate_end = []

        if sentence_index in self.positive_candidates_by_sentence:
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]
            for candidate_start_index, candidate_end_index, label in candidates:
                # save positive_candidate positions for future reference
                positive_candidate_start.append(candidate_start_index)
                positive_candidate_end.append(candidate_end_index)

        # Add sentence end index to support last index
        positive_candidate_start.append(end_index)
        positive_candidate_end.append(end_index)


        # Initialize next positive candidate to be after the start index
        next_positive_candidate_index = 0
        while start_index > positive_candidate_start[next_positive_candidate_index]:
            next_positive_candidate_index += 1

        # Main loop
        i = start_index
        while i < end_index:
            current_x_n = []
            current_x_n_seq = []
            current_x_n_sparse = []
            current_x_n_cache = []
            current_x_n_cache_seq = []

            current_y = None

            current_x_p = None
            current_x_p_seq = None
            current_x_p_sparse = None
            current_x_p_cache = None
            current_x_p_cache_seq = None

            # Check if iterator is on a positive candidate
            label = 1 if (i == positive_candidate_start[next_positive_candidate_index]) else 0

            if label == 0:
                """ True Negative Candidate """
                for d in range(min((max_mc_size, i - start_index + 1))):
                    current_start_index = i - d
                    current_end_index = i + 1

                    # Compile x_n vectors
                    negative_embedding, negative_seq = negative_candidates[current_start_index].get_padded_text_embedding(include_context=include_context)
                    negative_sparse = negative_candidates[current_start_index].get_sparse_feature_vector()
                    negative_cache = reference_cache_n[current_start_index]
                    negative_cache_seq = reference_cache_seq

                    current_x_n.append(negative_embedding)
                    current_x_n_seq.append(negative_seq)
                    current_x_n_sparse.append(negative_sparse)
                    current_x_n_cache.append(negative_cache)
                    current_x_n_cache_seq.append(negative_cache_seq)

                    # Compile x_p vectors
                    if current_end_index - current_start_index == 1:
                        # if single token vector, positive mc is the same as negative
                        current_x_p = negative_embedding
                        current_x_p_seq = negative_seq
                        current_x_p_sparse = negative_sparse
                    else:
                        # Else construct multi token positive MC and compile x_hat vectors
                        positive_mc = self.get_candidate_from_sentence(sentence_index, current_start_index,
                                                                       current_end_index, context_window)

                        current_x_p, current_x_p_seq = positive_mc.get_padded_text_embedding(include_context=include_context)
                        current_x_p_sparse = positive_mc.get_sparse_feature_vector()

                    """
                    For every stride of positive candidates we are adding the corresponding single token
                    negative candidates along with some padding.
                    """
                    # Add the vectors to data list
                    x_n.extend(current_x_n)
                    x_n_seq.extend(current_x_n_seq)
                    x_n_sparse.extend(current_x_n_sparse)
                    x_n_cache.extend(current_x_n_cache)
                    x_n_cache_seq.extend(current_x_n_cache_seq)


                    # Extend x_n with padding dummy vectors
                    num_tokens = d + 1
                    x_n.extend(pad_vecs[num_tokens][0])
                    x_n_sparse.extend(pad_vecs[num_tokens][1])
                    x_n_cache.extend(pad_vecs[num_tokens][2])
                    x_n_seq.extend(pad_vecs[num_tokens][3])
                    x_n_cache_seq.extend(pad_vecs[num_tokens][4])

                    # Add train mask to ignore dummy vectors during training
                    x_n_train_mask.append(neg_train_mask[num_tokens])


                    y.append(label_dict['OTHER'])

                    x_p.append(current_x_p)
                    x_p_seq.append(current_x_p_seq)
                    x_p_sparse.append(current_x_p_sparse)


                # Increment i
                i += 1

            else:
                """ True Positive candidate"""

                current_start_index = positive_candidate_start[next_positive_candidate_index]
                current_end_index = positive_candidate_end[next_positive_candidate_index]
                # Increment positive candidate counter
                next_positive_candidate_index += 1


                # Positive MC is of the right size
                if current_end_index - current_start_index <= max_mc_size:
                    positive_mc = self.get_candidate_from_sentence(sentence_index, current_start_index,
                                                                   current_end_index, context_window)

                    # Compile x_p vectors
                    current_x_p, current_x_p_seq = positive_mc.get_padded_text_embedding(include_context=include_context)
                    current_x_p_sparse = positive_mc.get_sparse_feature_vector()

                    # Compile x_n vectors
                    for d in range(current_start_index, current_end_index):
                        # sum up negative single token vectors between start and end of the positive mc
                        negative_embedding, negative_seq = negative_candidates[d].get_padded_text_embedding(include_context=include_context)
                        negative_sparse = negative_candidates[d].get_sparse_feature_vector()
                        negative_cache = reference_cache_n[d]
                        negative_cache_seq = reference_cache_seq

                        current_x_n.append(negative_embedding)
                        current_x_n_seq.append(negative_seq)
                        current_x_n_sparse.append(negative_sparse)
                        current_x_n_cache.append(negative_cache)
                        current_x_n_cache_seq.append(negative_cache_seq)

                    # Add the vectors to data list
                    x_p.append(current_x_p)
                    x_p_seq.append(current_x_p_seq)
                    x_p_sparse.append(current_x_p_sparse)

                    y.append(label_dict[positive_mc.get_label()])

                    x_n.extend(current_x_n)
                    x_n_seq.extend(current_x_n_seq)
                    x_n_sparse.extend(current_x_n_sparse)
                    x_n_cache.extend(current_x_n_cache)
                    x_n_cache_seq.extend(current_x_n_cache_seq)

                    # Extend x_n with padding dummy vectors
                    num_tokens = len(current_x_n)
                    x_n.extend(pad_vecs[num_tokens][0])
                    x_n_sparse.extend(pad_vecs[num_tokens][1])
                    x_n_cache.extend(pad_vecs[num_tokens][2])
                    x_n_seq.extend(pad_vecs[num_tokens][3])
                    x_n_cache_seq.extend(pad_vecs[num_tokens][4])

                    # Add train mask to ignore dummy vectors during training
                    x_n_train_mask.append(neg_train_mask[num_tokens])

                # Increment i
                i = current_end_index

        assert len(x_n) == len(x_p)*max_mc_size, 'Data compiled incorrectly'

        # Returning the generated data
        return x_p, x_p_seq, y, x_n, x_n_seq, x_n_cache, x_n_cache_seq, x_n_train_mask, x_p_sparse, x_n_sparse,

    def get_boundary_labels(self, sentence_index):
        """Gets the boundary labels for a sentence. 1 if it is a positive candidate, 0 otherwise"""
        sentence_length = len(self.data[sentence_index].text)
        boundary_labels = '0'*sentence_length
        if sentence_index in self.positive_candidates_by_sentence:
            boundary_labels = ''
            num_candidates, candidates = self.positive_candidates_by_sentence[sentence_index]
            last_candidate_end = 0
            for start_index, end_index, _ in candidates:
                boundary_labels += '0'*(start_index - last_candidate_end)
                boundary_labels += '1'*(end_index - start_index)
                last_candidate_end = end_index

            boundary_labels +='0'*(sentence_length-last_candidate_end)

        return list(boundary_labels)
