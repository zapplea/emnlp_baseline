import tensorflow as tf

class CasingEmbeddingData:
    def __init__(self):
        """
        how to use the program?
        eg.
        sentence = ['I', 'went', 'to', 'ANU', 'today', '#PAD#', '#PAD#', '#PAD#', '#PAD#', '#PAD#']
        ced = CasingEmbeddingData()
        casing_sentence = ced.casingSentence(sentence)
        casing_idList = ced.casingSentence2idList(casing_sentence)
        
        # the casing_idList is the final result
        :param shape: 
        """
        self.casing_vocab=self.getCasingVocab()

    def casingSentence2idList(self,casing_sentence):
        """
        convert casing sentence to list of casing ids
        :param casing_sentence: 
        :return: 
        """
        casing_idlist = []
        for tokenId in range(len(casing_sentence)):
            id = self.casing_vocab[casing_sentence[tokenId]]
            casing_idlist.append(id)
        return casing_idlist

    def getCasingVocab(self):
        """
        will return table of casing word, like ['PADDING':0,'other':1, ...]
        :return: 
        """
        entries = ['PADDING', 'other', 'numeric', 'mainly_numeric', 'allLower', 'allUpper', 'initialUpper',
                   'contains_digit']
        return {entries[idx]: idx for idx in range(len(entries))}

    def casingSentence(self,sentence):
        """
        will convert a sentence to a casing word sentence.
        :param sentence: [str, ...] eg.['I', 'went', 'to', 'ANU', 'today', '#PAD#', '#PAD#', '#PAD#', '#PAD#', '#PAD#']. 
        the sentence should be padded to the same length and the padding word should be represented by '#PAD#'.
        be careful, the digits in the sentence should not be converted to form like NUM4.
        And the word should not be converted to lower or upper.
        :return: eg. ['allUpper', 'allLower', ...]
        """
        casing_sentence = []
        for tokenIdx in range(len(sentence)):
            token = sentence[tokenIdx]
            casing_sentence.append(self.getCasing(token))
        return casing_sentence

    def getCasing(self,word):
        """Returns the casing for a word"""
        if word == '#PAD#':
            casing ='PADDING'
        else:
            casing = 'other'

            numDigits = 0
            for char in word:
                if char.isdigit():
                    numDigits += 1

            digitFraction = numDigits / float(len(word))

            if word.isdigit():  # Is a digit
                casing = 'numeric'
            elif digitFraction > 0.5:
                casing = 'mainly_numeric'
            elif word.islower():  # All lower case
                casing = 'allLower'
            elif word.isupper():  # All upper case
                casing = 'allUpper'
            elif word[0].isupper():  # is a title, initial char upper, then all lower
                casing = 'initialUpper'
            elif numDigits > 0:
                casing = 'contains_digit'

        return casing

class CassingEmbedding:
    def __init__(self,casing_vocab,casingVecLen,maxSentenceLen):
        """
        how to use?
        eg.
        with graph.as_default():
            cemb = CasingEmbedding(casing_vocab,casingVecLen,maxSentenceLen)
            casingInput = cemb.input()
            cemb.lookup_table()
        # then in the training process
        casingX = graph.get_collection('casingInput')[0]
        sess.run(train_step, feed_dict={casingX:casingSentences, ...})
        
        :param casing_vocab: 
        :param casingVecLen: length of casing vector 
        :param maxSentenceLen: maximum length of sentences
        """
        self.casingVocab = casing_vocab
        self.casingVecLen=casingVecLen
        self.maxSentenceLen=maxSentenceLen

    def parameter_initializer(self,dtype='float32'):
        stdv=1/tf.sqrt(tf.constant(self.casingVecLen,dtype=dtype))
        init = tf.random_normal((len(self.casingVocab),self.casingVecLen),stddev=stdv,dtype=dtype)
        return init

    def casing_embedding(self):
        """
        return casing embedding table.
        :return: 
        """
        embedding = tf.get_variable(name='casingEmb', initializer=self.parameter_initializer(), dtype='float32')
        return embedding

    def input(self):
        """
        the placeholder to input the casing_sentences
        :return: 
        """
        casingX = tf.placeholder(name='caisngInput',shape=(None,self.maxSentenceLen),dtype='int32')
        tf.add_to_collection('casingInput',casingX)
        return casingX

    def lookup_table(self,):
        table = self.casing_embedding()
        casingX = self.input()
        return tf.nn.embedding_lookup(table,casingX)