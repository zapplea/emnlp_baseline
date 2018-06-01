import logging

from nerd.config.constants import Constants
from nerd.data.util.containers.text_data import TextData
from nerd.util.word_embedding_util import WordEmbeddingUtil


class WNUTDataReader:
    '''
        Reads a WNUT-like file and returns a dataset object
        with the class attributes populated
        See Class desc for description of attributes
    '''

    @classmethod
    def readFile(cls, filePath, modelPath = None, model_is_binary = False, are_tweets = True, initData = None ):

        logging.info("Loading WNUT-like file from %s" % (filePath))


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

            vocab_set = set()
            prefix_set = set()
            suffix_set = set()

            # For internal aggregation
            sens = []
            text_labels = []
            numeric_labels = []


            for line in file.readlines():
                words = line.split()

                # If a blank line encountered its the end of the sentence.
                if len(words) == 0:
                    if len(sens) > 0:
                        data.text.append(sens)
                        data.seq_length.append(len(sens))

                        sens = []

                        data.labels.append(text_labels)
                        data.label_ids.append(numeric_labels)

                        text_labels = []
                        numeric_labels = []

                        # No other info in MYSTERIOUS_DATASET dataset
                        # other_data.append(other_datum)
                        # other_datum = []

                else:
                    # Data structure in the MYSTERIOUS_DATASET is as follows
                    # First word is the text
                    text_token = words[0]
                    sens.append(text_token)


                    # Second word is the label
                    label = words[1]

                    if label.upper() == 'O':  # in WNUT 2015, O represents Other
                        label = Constants.OTHER_LABEL
                    elif label.startswith("B-") or label.startswith("I-"):
                        label = label[2:]


                    text_labels.append(label)
                    numeric_labels.append(data.getLabelId(label))


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
                label_vector = [0]*vec_len
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
        data.pos = WordEmbeddingUtil.getPOSTags(data.text)

        return data
