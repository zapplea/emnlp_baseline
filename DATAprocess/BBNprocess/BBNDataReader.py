import logging

from nerd.config.constants import Constants
from nerd.data.util.containers.text_data import TextData
from nerd.util.word_embedding_util import WordEmbeddingUtil


class BBNDataReader:
    '''
    Reads a BBN dataset file and returns a text data object with attributes populated
    See TextData class desc for more info on the attributes
    '''
    @staticmethod
    def readFile(filePath, modelPath = None, model_is_binary = False, are_tweets = False, initData = None):

        logging.info("Loading BBN data file from %s" % (filePath))


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
            sens_pos = []
            text_labels = []
            numeric_labels = []

            for line in file.readlines():
                words = line.split()

                if len(words) == 0:
                    if len(sens) > 0:

                        data.text.append(sens)
                        data.seq_length.append(len(sens))
                        sens = []

                        data.pos.append(sens_pos)
                        sens_pos = []

                        data.labels.append(text_labels)
                        data.label_ids.append(numeric_labels)
                        text_labels = []
                        numeric_labels = []

                else:
                    # First word is the text
                    text_token = words[0]
                    sens.append(words[0])

                    # Second word is the pos
                    sens_pos.append(words[1])

                    if len(words) >= 3:
                        #TODO: confirm this check is required
                        label = words[len(words ) -1]
                        if label == 'O':
                            label = Constants.OTHER_LABEL

                    else:
                        label = Constants.OTHER_LABEL

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



        # Store max length for future use in pad metho
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
        # # Add POS labels to the text
        data.pos = WordEmbeddingUtil.getPOSTags(data.text)
        #
        # # check dim
        # assert len(text_embeddings) == len(text)
        # assert len(text) == len(text_char_ids)



        # return DataSetUtil(text, text_embeddings, text_char_ids, labels, label_vectors, pos,
        #                    other_data, [], _max_length, _max_char_length, seq_length)

        return data

if __name__=="__main__":
    print('run')