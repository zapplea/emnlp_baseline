import json
import math
import os

import numpy as np
from typing import Tuple, Set, Dict

from nerd.config.constants import Constants
from nerd.util.dot_dict import dotdict
from operator import itemgetter

from nerd.util.file_util import FileUtil


class TextData(object):
    """
        TextData : Class to ease data processing
        Important members of this class are as follows
            text            : gets the raw text information from the dataset (ie. words) Type: list of list [sentence] where sentence = [token]
            text_embeddings : converts text to embeddings using WordEmbeddingUtil Type: List of list of embeddings ; [sentence] where sentence = [word_embedding]
            label_text      : actual labels from the file
            label           : gets the label information from the dataset, has internal dictionary
            labelVectors    : creates one hot encoding of labels 
            other_data      : used to store any other info from the dataset (eg: YAGO ID)
            max_length      : Initialized to the length of the largest sentence. Can be changed to
                              modify the padding applied.
    """

    def __init__(self, object=None):
        self.label_to_id = {Constants.PAD_LABEL: 0, Constants.OTHER_LABEL: 1}
        self.id_to_label = {0: Constants.PAD_LABEL, 1: Constants.OTHER_LABEL}
        self.max_seq_length = 0

        # The text data with paddings
        self.text = []
        self.labels = []

        # To store the sequence length of each sentence
        self.seq_length = []

        # Converted numeric forms
        self.text_embeddings = []
        self.label_ids = []
        self.label_vectors = []

        # Sparse Feature related data
        self.pos = []
        self.prefix_list = []
        self.suffix_list = []
        self.vocab_list = [Constants.PAD_STRING, Constants.UNK_STRING]

        # Initialize values based on other data if passed
        if object is not None:
            self.label_to_id = object.label_to_id
            self.id_to_label = object.id_to_label
            self.max_seq_length = object.max_seq_length

            # Get the sparse feature related data
            self.prefix_list.extend(object.prefix_list)
            self.suffix_list.extend(object.suffix_list)

            if len(object.vocab_list) > 2:  # Start from index 2 so that we skip pad and unk
                self.vocab_list.extend(object.vocab_list[2:])

    def remove_sentences(self, sentence_ids):
        """
        Remove a sentence in text data based on sentence_id.
        :return: 
        """
        sentence_ids = sorted(sentence_ids, reverse=True)
        print(sentence_ids)
        for sentence_id in sentence_ids:
            self.text.pop(sentence_id)
            self.labels.pop(sentence_id)
            self.pos.pop(sentence_id)

            self.seq_length.pop(sentence_id)
            self.text_embeddings.pop(sentence_id)
            self.label_ids.pop(sentence_id)
            self.label_vectors.pop(sentence_id)

        return

    def create_new_set_from_sentences(self, sentence_set):
        """
        <WIP> only copies text data and labels for now.
        :param sentence_set:
        :return:
        """


        new_label_to_id = {Constants.PAD_LABEL: 0, Constants.OTHER_LABEL: 1}
        new_id_to_label = {0: Constants.PAD_LABEL, 1: Constants.OTHER_LABEL}
        new_max_seq_length = 0

        # The text data with paddings
        new_text = []
        new_labels = []
        new_pos = []

        # To store the sequence length of each sentence
        new_seq_length = []

        new_obj = TextData()

        new_obj.label_to_id = self.label_to_id
        new_obj.id_to_label = self.id_to_label

        for id in sentence_set:
            new_text.append(self.text[id])
            new_labels.append(self.labels[id])
            new_pos.append(self.pos[id])
            new_seq_length.append(self.seq_length[id])

        # Set these new data in the new obj
        new_obj.text = new_text
        new_obj.labels = new_labels
        new_obj.pos = new_pos

        return new_obj




    def getLabelId(self, label):
        """
        Keeps track of labels encountered and returns a label id
        based on the label provided
        :param label: String that need to be checked against the dictionary
        :return: label_id (int)
        """
        label_id = len(self.id_to_label)
        if label not in self.label_to_id.keys():
            self.label_to_id[label] = label_id
            self.id_to_label[label_id] = label
        else:
            label_id = self.label_to_id[label]

        return label_id

    def get_max_length(self, threshold=np.inf):
        """
        Return the max length of a sentence in the data set.
        :return: <int> max length of text sentence
        """
        return max(len(sentence) if  len(sentence) < threshold else 0 for sentence in self.text)

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return json.dumps([self[i] for i in range(len(self.text))])

    def __getitem__(self, item):
        _text = self.text[item]
        _labels = self.labels[item]
        _label_vectors = self.label_vectors[item]

        _text_embeddings = []
        if len(self.text_embeddings) >= item:
            _text_embeddings = self.text_embeddings[item]

        _pos = []
        if len(self.pos) >= item:
            _pos = self.pos[item]

        return dotdict({'text': _text,
                        'labels': _labels,
                        'text_embeddings': _text_embeddings,
                        'label_vectors': _label_vectors,
                        'pos': _pos})

    def _separate_field(self, field: list, data_1_sent_indices: list) -> Tuple[list, list]:
        data_2_sent_indices = list(set(range(len(field))) - set(data_1_sent_indices))
        field_1 = itemgetter(*data_1_sent_indices)(field)
        field_2 = itemgetter(*data_2_sent_indices)(field)
        return field_1, field_2

    def split_dataset(self, data_1_sent_indices: list) -> Tuple[object, object]:
        """
        Select the sentences in the sent_indices into a separate TextData object and the rest into another TextData object
        :param data_1_sent_indices:
        :return:
        """
        data_2_sent_indices = list(set(range(len(self.text))) - set(data_1_sent_indices))

        data_1 = TextData()
        data_2 = TextData()

        data_1.label_to_id, data_2.label_to_id = self.label_to_id, self.label_to_id
        data_1.id_to_label, data_2.id_to_label = self.id_to_label, self.id_to_label
        # The text data with paddings
        data_1.text, data_2.text = self._separate_field(field=self.text, data_1_sent_indices=data_1_sent_indices)
        data_1.labels, data_2.labels = self._separate_field(field=self.labels, data_1_sent_indices=data_1_sent_indices)

        data_1.seq_length, data_2.seq_length = self.seq_length, self.seq_length
        # Converted numeric forms
        data_1.text_embeddings, data_2.text_embeddings = self._separate_field(field=self.text_embeddings,
                                                                              data_1_sent_indices=data_1_sent_indices)
        data_1.label_ids, data_2.label_ids = self._separate_field(field=self.label_ids,
                                                                  data_1_sent_indices=data_1_sent_indices)
        data_1.label_vectors, data_2.label_vectors = self._separate_field(field=self.label_vectors,
                                                                          data_1_sent_indices=data_1_sent_indices)

        data_1.pos, data_2.pos = self._separate_field(field=self.pos, data_1_sent_indices=data_1_sent_indices)

        return data_1, data_2

    def remove_labels(self, labels_to_keep: Set[int], replace_label_id: int = 1):
        for i, sentence in enumerate(self.label_ids):
            for j, label_id in enumerate(sentence):
                if label_id not in labels_to_keep and label_id != 0:
                    self.label_ids[i][j] = replace_label_id  # marked as not an entity
                    self.labels[i][j] = self.id_to_label[replace_label_id]
                    # One hot representation of type other
                    vec_len = len(self.id_to_label)
                    label_vector = [0] * vec_len
                    label_vector[replace_label_id] = 1
                    self.label_vectors[i][j] = label_vector


    def label_info(self) -> Dict[int, int]:
        label_id_dict = dict()
        for sent_idx, label_ids in enumerate(self.label_ids):
            for label_id in label_ids:
                if label_id not in label_id_dict:
                    label_id_dict[label_id] = {sent_idx}
                else:
                    label_id_dict[label_id] |= {sent_idx}
        label_id_count = dict()
        for label_id, sent_set in label_id_dict.items():
            label_id_count[label_id] = len(label_id_dict[label_id])

        return label_id_count

    def export_to_conll_like_format(self, export_file_path: str, two_column_format=False):
        """
        Conll like format means:
        text    pos
        Pierre	NNP	CHUNKS	I-PERSON

        :return:
        """
        with open(export_file_path, "w") as f:
            for sent_idx, sent in enumerate(self.text):
                for word_idx, word in enumerate(sent):
                    pos = self.pos[sent_idx][word_idx]
                    label = self.labels[sent_idx][word_idx]
                    if label not in self.label_to_id:
                        raise ValueError("Unexpected key {} in label_to_id dict {}".format(label, self.label_to_id))
                    if self.label_to_id[label] == 1:
                        label = "O"
                    if two_column_format:
                        f.write("{}\t{}\n".format(word, label))
                    else:
                        f.write("{}\t{}\t{}\t{}\n".format(word, pos, "CHUNKS", label))

                f.write("\n")  # End of sentence


    def get_num_labels(self):
        """ Returns number of labels defined by length of id_to_label keys"""
        return len(self.id_to_label.keys())

    def get_vector_dimension(self):
        """
        Returns last dimension of text embeddings
        :return:
        """

        if len(self.text_embeddings) > 0 and len(self.text_embeddings[0]) > 0:
            return len(self.text_embeddings[0][0])
        else:
            return 0

    def write_out_batches_with_padding(self, batch_size, padded_length, output_path, num_sentences=None):
        """
        Preprares batches for sequence tagging with padding at the end.
        format :

                batch_data = {
                    'input_data' : input_data,
                    'seq_len' : seq_length,
                    'labels' : labels,
                    'text' : input_data_text,
                    'labels_text' : labels_text,
                    'id_to_label' : self.id_to_label
                }

        :param padded_length:
        :return:
        """

        assert len(self.text_embeddings) > 0 , "Text embeddings are not initialized."

        FileUtil.validate_folder(output_path)

        # Number of batches
        num_batches = int(math.floor(len(self.text_embeddings) / batch_size)) + 1

        # Padding constants
        PAD_TEXT = Constants.PAD_STRING
        PAD_LABEL = Constants.PAD_LABEL
        PAD_VECTOR = np.zeros(len(self.text_embeddings[0][0]))


        batch_no = 1

        for i in range(num_batches):
            batch_start = i*batch_size
            batch_end = min(len(self.text), (i + 1) * batch_size)

            # Batch_data
            input_data = []
            seq_length = []
            labels = []

            input_data_text = []
            labels_text = []

            for j in range(batch_start, batch_end):

                if len(self.text[j]) > padded_length:
                    continue
                if num_sentences is not None and j > num_sentences:
                    break

                assert len(self.text_embeddings[j]) > 0 , 'Sentence %d incorrect Text data: %s' % (j, str(self.text[j]))
                # Size of padding to add at the end of data row
                padding_size = padded_length - len(self.text[j])

                # Prepare data
                padded_input_data = self.text_embeddings[j] + [PAD_VECTOR] * padding_size
                padded_seq_length = self.seq_length[j]
                padded_labels = self.label_ids[j ] + [self.label_to_id[PAD_LABEL]] * padding_size

                padded_input_text = self.text[j] + [PAD_TEXT] * padding_size
                padded_labels_text = self.labels[j] + [PAD_LABEL] * padding_size

                # Add to batch
                input_data.append(padded_input_data)
                seq_length.append(padded_seq_length)
                labels.append(padded_labels)

                input_data_text.append(padded_input_text)
                labels_text.append(padded_labels_text)


            if len(input_data) > 0:
                # Construct and dump batch
                batch_file_name = 'batch_%06d.dat' % batch_no
                batch_data = {
                    'input_data' : input_data,
                    'seq_len' : seq_length,
                    'labels' : labels,
                    'text' : input_data_text,
                    'labels_text' : labels_text,
                    'id_to_label' : self.id_to_label
                }

                FileUtil.dump_object(batch_data, os.path.join(output_path, batch_file_name))

                # Increment count
                batch_no += 1




    # TODO: Add code to merge data_set
