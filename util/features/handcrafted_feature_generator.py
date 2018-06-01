from __future__ import absolute_import, division, print_function

import logging
import sys
from datetime import datetime

from nerd.config.constants import Constants
from nerd.data.util.features.stanford_preprocess_twitter import stanford_tokenize

if sys.version_info > (3, 0):
    from IPython.utils.py3compat import xrange
import nltk
class HandCraftFeatureGenerator:

    def __init__(self, data, max_mc_size=5, context_window=2, are_tweets=False):
        t = datetime.now()
        self.max_mc_size = max_mc_size
        self.context_window = context_window
        self.feature_map = {} # type: dict
        self.prefix_set = set(data.prefix_list)
        self.suffix_set = set(data.suffix_list)
        self.are_tweets = are_tweets
        self.sparse_feature_size = 0
        self.tag_set = {'VERB', 'NOUN', 'PRON', 'ADJ', 'ADV', 'ADP', 'CONJ', 'DET', 'NUM', 'PRT', 'X', '.',
                        '0'}  # '0' is just padding
        self._initialise_sparse_feature_dict(data)
        t = datetime.now() - t
        logging.info("HandCraftFeatureGenerator initialised in %d seconds." % (t.total_seconds()))



    def pos_tag_features(self, pos_tags, mode):
        """
        Generate pos tag feataures for mention text or contexts, depending on mode
    
        The features are in the same order as the tag_set
    
        :param token_list:
        :return: list of feature indices that is 1
        """

        self._validate_inputs(pos_tags, mode)

        pos_tag_features = []

        for i in xrange(len(pos_tags)):
            # Stanford only uses PTB tags, but it's too complicated. We use universal tags
            universal_tag = nltk.mapping.map_tag("en-ptb", 'universal', pos_tags[i])

            feature_name = "pos_" + str(i + 1) + "_th_" + mode + "_token:" + universal_tag
            try:
                assert feature_name in self.feature_map
            except:
                print(feature_name)
                print(self.feature_map.keys())
                raise

            if self._feature_exists(feature_name):
                pos_tag_features.append(self._feature_index_lookup(feature_name))

        return pos_tag_features

    def _feature_exists(self, feature_name):
        """ Returns bool checking if feature exists"""
        if feature_name in self.feature_map:
            return True
        else:
            logging.debug("Could not find feature named '%s'" % feature_name)

        return False

    def _feature_index_lookup(self, feature_name):
        """
        Add the index of the feature if the feature is in the feature map
        :param feature_name: 
        :return: 
        """
        if feature_name in self.feature_map:
            # self.feature_indices.append(self.feature_map[feature_name])
            return self.feature_map[feature_name]
        else:
            logging.debug("Could not find feature named '%s'" % feature_name)



    def capitalisation_features(self, token_list):
        """
        # Return a binary feature vector of max mention length
        Feature A from
        http://www.clips.uantwerpen.be/conll2003/pdf/20407zha.pdf
         http://www.aclweb.org/anthology/P10-1040
        :param token_list:
        :return:
        """

        capitalisation_features = []
        # Individual token capitalisation
        for i in xrange(len(token_list)):
            if token_list[i] != Constants.PAD_STRING and any(l.isupper() for l in token_list[i]):
                feature_name = "capitalisation_" + str(i + 1) + "_th_mention_text_token"
                if self._feature_exists(feature_name):
                    capitalisation_features.append(self._feature_index_lookup(feature_name))

        # All cap feature
        if all((token[0].isupper() for token in token_list)):
            if self._feature_exists("all_tokens_capitalised"):
                capitalisation_features.append(self._feature_index_lookup("all_tokens_capitalised"))

        return capitalisation_features



    def numerical_features(self, token_list):
        """
        @TODO
        :param token_list: 
        :return: 
        """

        feature_vec = []
        for token in token_list:
            if token == Constants.PAD_STRING:
                feature_vec.append(0)
            else:
                if "." in token:
                    token = "".join(token.split(".")) # Remove the decimal point to see if the rest is a number
                feature_vec.append(int(token.isdigit()))
        pass


    def unigram_features(self, token_list, mode):

        self._validate_inputs(token_list, mode)

        unigram_features = []

        for i in xrange(len(token_list)):

            if self.are_tweets:

                processed_tokens = stanford_tokenize(token_list[i]).split(" ")
                for processed_token in processed_tokens:
                    feature_name = "unigram_" + str(i + 1) + "_th_" + mode + "_token:" + processed_token
                    if self._feature_exists(feature_name):
                        unigram_features.append(self._feature_index_lookup(feature_name))
            else:
                feature_name = "unigram_" + str(i + 1) + "_th_" + mode + "_token:" + token_list[i]
                if self._feature_exists(feature_name):
                    unigram_features.append(self._feature_index_lookup(feature_name))

        return unigram_features





    def _validate_inputs(self, token_list, mode):
        if mode == "mention_text":
            assert len(token_list) <= self.max_mc_size

        elif mode == "left_context" or mode == "right_context":
            assert len(token_list) <= self.context_window

        else:
            print("mode parameter has to be "
                  "'mention_text', 'left_context', or 'right_context")
            sys.exit(1)

    # Prefix and suffix features
    def prefix_features(self, token_list, mode):
        self._validate_inputs(token_list, mode)

        prefix_features = []
        for i in xrange(len(token_list)):
            prefix_3 = token_list[i][:3]
            prefix_4 = token_list[i][:4]

            if prefix_3 in self.prefix_set:
                feature_name = "prefix_" + str(i + 1) + "_th_" + mode + "_token:" + prefix_3
                if self._feature_exists(feature_name):
                    prefix_features.append(self._feature_index_lookup(feature_name))

            if prefix_4 in self.prefix_set:
                feature_name = "prefix_" + str(i + 1) + "_th_" + mode + "_token:" + prefix_4
                if self._feature_exists(feature_name):
                    prefix_features.append(self._feature_index_lookup(feature_name))

        return prefix_features



    def suffix_features(self, token_list, mode ):
        self._validate_inputs(token_list, mode)

        suffix_features = []
        for i in xrange(len(token_list)):
            suffix_3 = token_list[i][-3:]
            suffix_4 = token_list[i][-4:]

            if suffix_3 in token_list[i]:
                feature_name = "suffix_" + str(i + 1) + "_th_" + mode + "_token:" + suffix_3
                if self._feature_exists(feature_name):
                    suffix_features.append(self._feature_index_lookup(feature_name))

            if suffix_4 in token_list[i]:
                feature_name = "suffix_" + str(i + 1) + "_th_" + mode + "_token:" + suffix_4
                if self._feature_exists(feature_name):
                    suffix_features.append(self._feature_index_lookup(feature_name))

        return suffix_features

    def _initialise_sparse_feature_dict(self, data):
        """
        Initialise the sparse features dictionary mapping
        This method contains the cononical names for all sparse features
        :return: None
        """
        logging.info("%s Start generating sparse feature dictionary (mapping)" % (datetime.now().ctime()))

        # ===============================
        # initialise capitalisation feature
        # ===============================
        for i in xrange(self.max_mc_size):
            feature_name = "capitalisation_" + str(i + 1) + "_th_mention_text_token"
            self._allocate_feature(feature_name)

            self._allocate_feature("all_tokens_capitalised")
        # ===============================
        # initialise unigram features
        # ===============================
        # Unigram features for mention text
        for token in data.vocab_list:
            for i in xrange(self.max_mc_size):
                feature_name = "unigram_" + str(i + 1) + "_th_mention_text_token:" + token
                self._allocate_feature(feature_name)
        # Unigram features for left context
        for token in data.vocab_list:
            for i in xrange(self.context_window):
                feature_name = "unigram_" + str(i + 1) + "_th_left_context_token:" + token
                self._allocate_feature(feature_name)
        # Unigram features for right context
        for token in data.vocab_list:
            for i in xrange(self.context_window):
                feature_name = "unigram_" + str(i + 1) + "_th_right_context_token:" + token
                self._allocate_feature(feature_name)
        # ===============================
        # Prefix feature for ...
        # ===============================
        # mc text
        for token in data.prefix_list:
            for i in xrange(self.max_mc_size):
                feature_name = "prefix_" + str(i + 1) + "_th_mention_text_token:" + token
                self._allocate_feature(feature_name)
        # left context
        for token in data.prefix_list:
            for i in xrange(self.context_window):
                feature_name = "prefix_" + str(i + 1) + "_th_left_context_token:" + token
                self._allocate_feature(feature_name)
        # right context
        for token in data.prefix_list:
            for i in xrange(self.context_window):
                feature_name = "prefix_" + str(i + 1) + "_th_right_context_token:" + token
                self._allocate_feature(feature_name)
        # ===============================
        # Suffix feature for ...
        # ===============================
        # mc text
        for token in data.suffix_list:
            for i in xrange(self.max_mc_size):
                feature_name = "suffix_" + str(i + 1) + "_th_mention_text_token:" + token
                self._allocate_feature(feature_name)
        # left context
        for token in data.suffix_list:
            for i in xrange(self.context_window):
                feature_name = "suffix_" + str(i + 1) + "_th_left_context_token:" + token
                self._allocate_feature(feature_name)
        # right context
        for token in data.suffix_list:
            for i in xrange(self.context_window):
                feature_name = "suffix_" + str(i + 1) + "_th_right_context_token:" + token
                self._allocate_feature(feature_name)

        # ===============================
        # POS feature for ...
        # ===============================
        # mc text
        for pos_name in self.tag_set:
            for i in xrange(self.max_mc_size):
                feature_name = "pos_" + str(i + 1) + "_th_mention_text_token:" + pos_name
                self._allocate_feature(feature_name)
        # left context
        """
        for pos_name in HyperParameters.TAG_SET:
            for i in xrange(self.context_window):
                feature_name = "pos_" + str(i + 1) + "_th_left_context_token:" + pos_name
                MentionCandidate._allocate_feature(feature_name)
        # right context
        for pos_name in HyperParameters.TAG_SET:
            for i in xrange(self.context_window):
                feature_name = "pos_" + str(i + 1) + "_th_right_context_token:" + pos_name
                MentionCandidate._allocate_feature(feature_name)
        """
        assert isinstance(self.sparse_feature_size, int)
        assert self.sparse_feature_size == len(self.feature_map)

        # GlobalVars.sparse_feature_dim = self.sparse_feature_size
        # logging.info(
        #     "%s: Total number of sparse features: %d" % (datetime.now().ctime(), GlobalVars.sparse_feature_dim))

        # MentionCandidate.print_feature_dict()

        return

    def print_feature_dict(self):
        for feature in self.feature_map.keys():
            logging.info(feature)
        logging.info("=" * 40)
        logging.info("%d features in total" % (len(self.feature_map)))

    def _allocate_feature(self, feature_name):
        """
        Given a feature name, automatically allocate a feature to the next available integer in the feature index dictionary
        :param feature_name: str
        :return: None
        """
        if feature_name not in self.feature_map:
            self.feature_map[feature_name] = self.sparse_feature_size
            self.sparse_feature_size += 1

        # Previously used for padding features, not required anymore
        # assert self.sparse_feature_size != 1
        return


    def get_sparse_feature_vector(self, mention_text, left_context, right_context, mention_pos):

        feature_indices = set()

        feature_indices.update(self.capitalisation_features(mention_text))

        # hand_craft_features.extend(capitalisation_features(self.left_context))
        # hand_craft_features.extend(capitalisation_features(self.right_context))
        # B Unigram: for mention text, left context, right context
        mention_unigram_features = self.unigram_features(mention_text, mode="mention_text")
        lc_unigram_features = self.unigram_features(left_context, mode="left_context")
        rc_unigram_features = self.unigram_features(right_context, mode="right_context")

        # E Digit - not sure if that improves performance
        # hand_craft_features_indices.extend(numerical_features(self.mention_text))
        # hand_craft_features.extend(numerical_features(self.left_context))
        # hand_craft_features.extend(numerical_features(self.right_context))
        # F Token prefix (length three and four), and token suffix (length from one to four)

        # Prefix and suffix features
        prefix_features = self.prefix_features(mention_text, mode="mention_text")
        suffix_features = self.suffix_features(mention_text, mode="mention_text")

        # G POS: for mention text, left context, right context
        pos_features = self.pos_tag_features(mention_pos, mode="mention_text")
        # self.feature_gen.pos_tag_features(self.left_context_pos, mode="left_context")
        # self.feature_gen.pos_tag_features(self.right_context, mode="left_context")


        # Add features to set
        feature_indices.update(mention_unigram_features)
        feature_indices.update(lc_unigram_features)
        feature_indices.update(rc_unigram_features)
        feature_indices.update(prefix_features)
        feature_indices.update(suffix_features)
        feature_indices.update(pos_features)


        # Sort the feature indices in numerical order
        feature_list = list(feature_indices)
        feature_list.sort()


        return feature_list