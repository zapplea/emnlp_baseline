import numpy as np

from nerd.util.math_util import MathUtil


class MentionCandidate(object):

    def __init__(self, text = None, labels = None, label_vector = None,
                 sentence_index = None, start_index = None, end_index = None,
                 context_window_size = None,  feature_vector = None, sparse_feature_vector = None,
                 sparse_feature_dimension = None, pos = None):
        self.text = text
        self.labels = labels
        self.label_vector = label_vector
        self.feature_vector = feature_vector
        self.sparse_feature_vector = sparse_feature_vector
        self.sparse_feature_dimension = sparse_feature_dimension
        self.sentence_index = sentence_index
        self.start_index = start_index
        self.end_index = end_index
        self.context_window_size = context_window_size
        self.pos = pos

    def get_label(self):
        """
        Returns the string label of the mention candidate
        :return: 
        """
        return self.labels[1][0]

    def get_padded_text_and_context_embedding(self, max_mc_size=5, dtype=np.float32):
        """
        Returns the mc embeddings with a padding to max features size. 
        for example if the the max_mc_size=5 and context_window_size=2, this will return
        a matrix of shape (max_mc_size + 2*context_window_size, vector_dimension)
        Padding is all zero vectors. Returns the true mc length
        :param max_mc_size: 
        :param dtype: 
        :return: 
        """

        feature_vector_dimension = len(self.feature_vector[0][0])
        mc_size = len(self.feature_vector[1])

        true_seq_length = mc_size
        max_seq_length = max_mc_size + (2*self.context_window_size)

        # Construct padding vector
        padding = []
        if true_seq_length < max_seq_length:
            num_padding_vecs = max_seq_length - true_seq_length
            padding = [[0]*feature_vector_dimension for i in range(num_padding_vecs)]

        # Construct_padded text_embeddings
        text_embeddings = []
        text_embeddings.extend(self.feature_vector[1])

        # Add paddings
        text_embeddings.extend(padding)

        # Construct context embeddings
        context_embeddings = []
        context_embeddings.extend(self.feature_vector[0])
        context_embeddings.extend(self.feature_vector[2])

        return text_embeddings, context_embeddings, true_seq_length

    def get_padded_text_embedding(self, max_mc_size=5, include_context=True, dtype=np.float32):
        """
        Returns the mc embeddings with a padding to max features size. 
        for example if the the max_mc_size=5 and context_window_size=2, this will return
        a matrix of shape (max_mc_size + 2*context_window_size, vector_dimension)
        Padding is all zero vectors. Returns the true mc length
        :param max_mc_size: 
        :param dtype: 
        :return: 
        """

        feature_vector_dimension = len(self.feature_vector[0][0])
        context_size = len(self.feature_vector[0])
        mc_size = len(self.feature_vector[1])

        true_seq_length = mc_size + (2*context_size) if include_context else mc_size
        max_seq_length = max_mc_size + (2*self.context_window_size)

        # Construct padding vector
        padding = []
        if true_seq_length < max_seq_length:
            num_padding_vecs = max_seq_length - true_seq_length
            padding = [[0]*feature_vector_dimension for i in range(num_padding_vecs)]

        # Construct_padded text_embeddings
        text_embeddings = []
        if include_context:
            # Add lcontext, mc_embedding, rcontext
            for i in range(len(self.feature_vector)):
                text_embeddings.extend(self.feature_vector[i])
        else:
            # Add mc_embeddings
            text_embeddings.extend(self.feature_vector[1])

        # Add paddings
        text_embeddings.extend(padding)

        return text_embeddings, true_seq_length

    def get_text_embedding(self, dtype=np.float32):
        """
        Concatenates feature vector and returns one long vector
        :return: List of numeric data
        """
        text_embeddings = []
        for emb in self.feature_vector[0]:
            text_embeddings.extend(emb)

        text_embeddings.extend(list(np.mean(self.feature_vector[1], axis=0, dtype=dtype)))

        for emb in self.feature_vector[2]:
            text_embeddings.extend(emb)

        return text_embeddings

    def get_mention_embedding_summed(self, dtype=np.float32):
        """
        Returns the sum of the mention embeddings (without context)
        :param dtype: 
        :return: 
        """

        return np.sum(self.feature_vector[1], axis=0)

    def get_text_embedding_with_mention_summed(self, dtype=np.float32):
        """
        Returns sum of the mention embedding with context concatenated on the left and right
        :param dtype: 
        :return: 
        """
        text_embeddings = []
        for emb in self.feature_vector[0]:
            text_embeddings.extend(emb)

        text_embeddings.extend(list(np.sum(self.feature_vector[1], axis=0, dtype=dtype)))

        for emb in self.feature_vector[2]:
            text_embeddings.extend(emb)

        return text_embeddings

    def get_sparse_feature_vector(self):
        """
        Returns the sparse feature vector
        :return: 
        """
        one_indices = [self.sparse_feature_vector]
        vector_dimension = self.sparse_feature_dimension
        sparse_vector = MathUtil.construct_sparse_matrix_of_ones(one_indices, shape=(1, vector_dimension))

        return sparse_vector

    def get_mc_repre(self):
        """
        Concatenates feature vector with sparse feature vector to return the full
        MC Representations, lovingly called mc_repre
        :return: 
        """

        mc_repre = self.get_text_embedding()
        mc_repre.extend(self.sparse_feature_vector)

        return mc_repre

    def __str__(self):
        return  "Text    : "+str(self.text) + "\n" + \
                "Labels  : "+str(self.labels)+ "\n"+\
                "POS     : "+str(self.pos)+ "\n"+\
                "WordVec : "+str(self.feature_vector)+ "\n"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    # return comparison
    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self == other
        return NotImplemented


