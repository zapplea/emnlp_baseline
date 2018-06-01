import logging

from nerd.data.util.containers.text_data import TextData


class ConLLDataReader:



    @classmethod
    def readFile(cls, file_path, modelPath = None, model_is_binary = False, are_tweets = False, initData = None):

        logging.info("Loading CoNLL format file from %s" % file_path)
        data = TextData(initData)
        with open(file_path) as file:

            # Raw text information
            data.text = []
            data.labels= []

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

            # Ignore tags, previously used for Yago dataset, unintialised now
            IGNORE_TAGS = set()

            # labels_datum = []
            # labels = []
            #
            # sens = []
            #
            # other_data = []
            # other_datum = []

            for line in file.readlines():
                words = line.split()

                if len(words) == 0 or words[0] in IGNORE_TAGS:
                    if len(sens) > 0:
                        data.text.append(sens)
                        seq_length.append(len(sens))
                        sens = []

                        labels.append(labels_datum)
                        labels_datum = []

                        other_data.append(other_datum)
                        other_datum = []

                else:
                    sens.append(words[0])
                    if len(words) > 3 and words[1].isupper() and words[len(words) - 1].isupper():
                        # Creating label for first and last token after the word in YAGO-Set
                        label = words[1] + '_' + words[len(words) - 1]
                        labels_datum.append(DataSetUtil.getLabelId(label))

                        other_tags = words[len(words) - 2]
                        other_datum.append(other_tags)

                    else:
                        labels_datum.append(DataSetUtil.getLabelId('OTHER'))

        # Store max length for future use in pad method
        _max_length = DataSetUtil.get_max_length(data.text)

        # Convert labels to one hot vectors
        for l in labels:
            label_vector = []
            for i in xrange(len(l)):
                label_vector.append(DataSetUtil.getOneHotVector(key=l[i], size=len(DataSetUtil.label_to_id)))
            label_vectors.append(label_vector)

        # Convert data into vectors of word embeddings
        data.text_embeddings = WordEmbeddingUtil.query_embeddings_from_corpus(data.text)

        # Initialise data structure for generating char-based features
        data.text_char_ids = CharEmbeddingUtil.get_char_ids(data.text, HyperParameters.MAX_CHAR_LENGTH)
        _max_char_length = CharEmbeddingUtil.max_char_length(data.text_char_ids)

        # Add POS labels to the data.text
        pos = WordEmbeddingUtil.getPOSTags(text)

        # check dim
        assert len(text_embeddings) == len(text)
        assert len(text) == len(text_char_ids)

        return DataSetUtil(text, text_embeddings, text_char_ids, labels, label_vectors, pos,
                           other_data, [], _max_length, _max_char_length, seq_length)