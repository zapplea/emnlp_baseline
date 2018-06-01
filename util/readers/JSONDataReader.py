import json
import logging

from nerd.data.util.containers.text_data import TextData
from nerd.util.word_embedding_util import WordEmbeddingUtil


class JSONDataReader:
    '''
    Reads a JSON dataset file and returns a text data object with attributes populated
    See TextData class desc for more info on the attributes
    '''
    @classmethod
    def readFile(cls, filePath, modelPath = None, model_is_binary = False, are_tweets = False, initData = None):

        logging.info("Loading OntoNotes(JSON)-like file from %s" % (filePath))


        data = TextData(initData)

        with open(filePath) as file:

            # Raw text information
            data.text = []
            data.labels = []

            # Sequence length of the sentence
            data.seq_length = []

            # Numeric conversion of raw text
            data.text_embeddings = []
            data.label_vectors = []

            # Feature related data
            vocab_set = set()
            prefix_set = set()
            suffix_set = set()



            # STRING CONSTANTS
            # JSON attributes
            TOKENS = 'tokens'
            MENTIONS = 'mentions'
            START = 'start'
            END = 'end'
            LABELS = 'labels'

            # token labels
            OTHER_LABEL = 'OTHER'

            for line in file.readlines():

                file_data = json.loads(line)

                sens = file_data[TOKENS]  # Sentence
                sens_label = [OTHER_LABEL] * len(sens)  # Labels of that sentence

                for i in range(len(file_data[MENTIONS])):
                    mention_data = file_data[MENTIONS][i]
                    start_index = mention_data[START]
                    end_index = mention_data[END]

                    if len(mention_data[LABELS]) > 0 and not mention_data[LABELS][0] == '/other':
                        label = mention_data[LABELS][len(mention_data[LABELS])-1]
                    else:
                        continue

                    label = [label] * (end_index - start_index)
                    sens_label[start_index:end_index] = label

                sens_label_ids = [data.getLabelId(label) for label in sens_label]

                # Load data
                data.text.append(sens)
                data.seq_length.append(len(sens))
                data.labels.append(sens_label)
                data.label_ids.append(sens_label_ids)

                # Add feature related data
                for text_token in sens:
                    # Add word to vocab
                    vocab_set.add(text_token)
                    # Add prefix and suffix data
                    if len(text_token) > 2:
                        # Pick the first 3 letters and last 3 letters of the word
                        prefix_set.add(text_token[:3])
                        suffix_set.add(text_token[-3:])
                    if len(text_token) > 3:
                        # Pick the first 4 letters and last 4 letters of the word
                        prefix_set.add(text_token[:4])
                        suffix_set.add(text_token[-4:])


        # Store max length for future use in pad method
        data.max_length = data.get_max_length()

        # Convert set to lists for indexing
        data.vocab_list.extend(list(vocab_set))
        data.prefix_list.extend(list(prefix_set))
        data.suffix_list.extend(list(suffix_set))


        # Convert labels to one hot vectors
        vec_len = len(data.label_to_id)
        for ids in data.label_ids:
            label_vecs = []
            for id in ids:
                label_vector = [0] * vec_len
                label_vector[id] = 1
                label_vecs.append(label_vector)

            data.label_vectors.append(label_vecs)

        # Convert data into vectors of word embeddings
        if modelPath is not None:
            data.text_embeddings = WordEmbeddingUtil.query_embeddings_from_corpus(text=data.text,
                                                                                  modelPath=modelPath,
                                                                                  modelIsBinary=model_is_binary,
                                                                                  are_tweets=are_tweets)
        else:
            data.text_embeddings = [[] for x in data.text]

        # # Initialise data structure for generating char-based features
        # text_char_ids = CharEmbeddingUtil.get_char_ids(text, HyperParameters.MAX_CHAR_LENGTH)
        # _max_char_length = CharEmbeddingUtil.max_char_length(text_char_ids)
        #
        # # text_embeddings = [3,3]*len(text) #FOR TESTING PLEASE REMOVE
        data.pos = WordEmbeddingUtil.getPOSTags(data.text)

        return data