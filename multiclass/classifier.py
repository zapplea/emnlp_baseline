import tensorflow as tf
import sklearn
import numpy as np
from datetime import datetime

class Classifier:
    def __init__(self,nn_config,datafeed):
        self.nn_config = nn_config
        self.df = datafeed

    def X_input(self, graph):
        """

        :param graph: 
        :return: shape = (batch size, words num, feature dim)
        """
        X = tf.placeholder(name='X', shape=(None, self.nn_config['words_num']), dtype='int32')
        return X

    def sequence_length(self, X, graph):
        """

        :param X: (batch size, max words num)
        :param graph: 
        :return: 
        """
        ones = tf.ones_like(X, dtype='int32') * self.nn_config['pad_index']
        condition = tf.equal(ones, X)
        seq_len = tf.reduce_sum(tf.where(condition, tf.zeros_like(X, dtype='int32'), tf.ones_like(X, dtype='int32')),
                                axis=1, name='seq_len')
        return seq_len

    def Y_input(self, graph):
        """

        :param graph: 
        :return: shape = (batch size, words num)
        """
        Y_ = tf.placeholder(name='Y_', shape=(None, self.nn_config['words_num']), dtype='int32')
        return Y_

    def lookup_mask(self, X, graph):
        ones = tf.ones_like(X, dtype='int32') * self.nn_config['pad_index']
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        multiple = [1, 1, self.nn_config['feature_dim']]
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiple)
        return mask

    def lookup_table(self, X, mask, graph):
        table = tf.placeholder(name='table', shape=(self.nn_config['vocabulary_size'], self.nn_config['feature_dim']),
                               dtype='float32')
        table = tf.Variable(table, dtype='float32')
        X = tf.nn.embedding_lookup(table, X)
        X = tf.multiply(X, mask)
        return X

    def bilstm(self, X, seq_len, graph):
        """

        :param X: shape = (batch size, words num, feature dim)
        :param mask: shape = (batch size, words num, feature dim)
        :return: 
        """
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        # outputs.shape = [(batch size, max time step, lstm cell size),(batch size, max time step, lstm cell size)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=X,
                                                     sequence_length=seq_len, dtype='float32')
        # outputs.shape = (batch size, max time step, 2*lstm cell size)
        outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
        graph.add_to_collection('bilstm_outputs', outputs)
        return outputs

    def softmax_log_mask(self, X, graph):
        """

        :param X: (batch size, max words num)
        :param graph: 
        :return: (batch size, max words num)
        """
        condition = tf.equal(X, tf.ones_like(X, dtype='int32') * self.nn_config['pad_index'])
        mask = tf.where(condition, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        return mask

    def Y_2one_hot(self, Y_, graph):
        """

        :param Y_: (batch size, max words num)
        :param mask: (batch size, max words num, labels number)
        :return: (batch size, words num, labes num)
        """
        # add mask to mask tag of #PAD# to 0 vec, otherwise #PAD# will update parameter
        Y_one_hot = tf.one_hot(Y_, depth=self.nn_config['target_NETypes_num'], axis=2, dtype='float32')
        return Y_one_hot

    def multiclass_score(self, X, graph):
        """

        :param X: (batch size, max words num, 2*lstm cell size)
        :param graph: 
        :return: (batch size, max words num, target NETypes num)
        """
        W_s = tf.get_variable(name='W_s', initializer=tf.random_uniform(
            shape=(2 * self.nn_config['lstm_cell_size'], self.nn_config['source_NETypes_num']), dtype='float32'))
        graph.add_to_collection('reg_multiclass', tf.contrib.layers.l2_regularizer(self.nn_config['reg_linear_rate'])(W_s))
        W_t = tf.get_variable(name='W_t',
                              initializer=tf.random_uniform(
                                  shape=(self.nn_config['source_NETypes_num'], self.nn_config['target_NETypes_num']),
                                  dtype='float32'))
        graph.add_to_collection('reg_multiclass', tf.contrib.layers.l2_regularizer(self.nn_config['reg_linear_rate'])(W_t))
        # X.shape = (batch size*words num, 2*lstm cell size)
        X = tf.reshape(X, shape=(-1, 2 * self.nn_config['lstm_cell_size']))
        # score.shape = (batch size, words num, target NETypes num)
        # score = tf.reshape(tf.matmul(tf.matmul(X, W_s), W_t),
        #                    shape=(-1, self.nn_config['words_num'], self.nn_config['target_NETypes_num']))
        # score.shape = (batch size*words num, target NETypes num)
        score = tf.matmul(tf.matmul(X, W_s), W_t)
        # score = tf.matmul(tf.matmul(X, W_s), W_t)
        graph.add_to_collection('multiclass_score', score)
        return score

    def loss_multiclass(self, score, Y_, mask, graph):
        """

        :param score: (batch size* max words num, target NETypes num)
        :param Y_: (batch size, max words num, target NETypes num)
        :param graph: 
        :return: 
        """
        regularizer = graph.get_collection('reg_multiclass')
        regularizer.extend(graph.get_collection('bilstm_reg'))
        loss = tf.reduce_mean(tf.add(tf.reduce_sum(
            tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(Y_), logits=score, dim=-1), mask),
            axis=1), tf.reduce_sum(regularizer)),
                              name='loss_multiclass')
        return loss

    def test_loss_multiclass(self, score, Y_, mask, graph):

        loss = tf.reduce_mean(
            tf.reduce_sum(tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_, logits=score), mask),
                          axis=1), name='test_loss_multiclass')
        return loss

    def opt_multiclass(self, loss, graph):
        train_op = tf.train.GradientDescentOptimizer(self.nn_config['lr'], name='train_op_crf_source').minimize(loss)
        graph.add_to_collection('train_op_multiclass', train_op)
        return train_op

    def tag_sequence_mask(self, X, graph):
        """

        :param X: (batch size, max words num)
        :param graph: 
        :return: (batch size, max words num)
        """
        condition = tf.equal(X, tf.ones_like(X, dtype='int32') * self.nn_config['pad_index'])
        mask = tf.where(condition, tf.zeros_like(X, dtype='int32'), tf.ones_like(X, dtype='int32'))
        return mask

    def pred_multiclass(self, score, mask, graph):
        """

        :param score: (batch size, max words num, target NETypes num)
        :param mask: (batch size, max words num)
        :return: 
        """
        Y = tf.multiply(tf.argmax(score, axis=2, output_type='int32'), mask)
        graph.add_to_collection('pred_multiclass', Y)
        return Y

    def f1(self, Y_, Y, labels_num):
        """

        :param Y_: (batch size, max words num)
        :param Y: (batch size, max words num
        :return: 
        """
        Y = np.reshape(Y, newshape=(-1,)).astype('float32')
        Y_ = np.reshape(Y_, newshape=(-1,)).astype('float32')
        ls_y = []
        ls_y_ = []
        for i in range(Y.shape[0]):
            if Y_[i] != 0:
                ls_y.append(Y[i])
                ls_y_.append(Y_[i])
        Y = np.array(ls_y, dtype='float32')
        Y_ = np.array(ls_y_, dtype='float32')

        f1_macro = sklearn.metrics.f1_score(Y_, Y, labels=list(range(1, labels_num)), average='macro')
        f1_micro = sklearn.metrics.f1_score(Y_, Y, labels=list(range(1, labels_num)), average='micro')
        return f1_macro, f1_micro

    def optimize(self, loss, graph):
        train_op = tf.train.GradientDescentOptimizer(self.nn_config['lr']).minimize(loss)
        return train_op

    def classifier(self):
        graph = tf.Graph()
        with graph.as_default():
            X_id = self.X_input(graph)
            seq_len = self.sequence_length(X_id, graph)
            tag_seq_mask = self.tag_sequence_mask(X_id, graph)
            Y_ = self.Y_input(graph)
            mask = self.lookup_mask(X_id, graph)
            # X.shape = (batch size, words num, feature dim)
            X = self.lookup_table(X_id, mask, graph)
            with tf.variable_scope('bilstm') as vs:
                # X.shape = (batch size, max time step, 2*lstm cell size)
                X = self.bilstm(X, seq_len, graph)
                graph.add_to_collection('bilstm_reg',
                                        tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                            graph.get_tensor_by_name(
                                                'bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
                graph.add_to_collection('bilstm_reg',
                                        tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                            graph.get_tensor_by_name(
                                                'bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))

            soft_log_mask = tf.reshape(self.softmax_log_mask(X_id, graph),shape=(-1,self.nn_config['target_NETypes_num']))
            Y_one_hot = self.Y_2one_hot(Y_, graph)
            score = self.multiclass_score(X, graph)
            pred = self.pred_multiclass(tf.reshape(score,shape=(-1,self.nn_config['words_num'],self.nn_config['target_NETypes_num'])), tag_seq_mask, graph)
            loss = self.loss_multiclass(score, tf.reshape(Y_one_hot, shape=(-1, self.nn_config['target_NETypes_num'])), soft_log_mask, graph)
            test_loss = self.test_loss_multiclass(score, tf.reshape(Y_one_hot, shape=(-1, self.nn_config['target_NETypes_num'])), soft_log_mask, graph)
            train_op = self.optimize(loss, graph)
            graph.add_to_collection('train_op_multiclass', train_op)
            saver = tf.train.Saver()
        return graph, saver

    def train(self):
        print('create graph')
        graph, saver = self.classifier()
        print('complete graph')
        with graph.device('/:gpu0'):
            with graph.as_default():
                X = graph.get_tensor_by_name('X:0')
                Y_ = graph.get_tensor_by_name('Y_:0')
                table = graph.get_tensor_by_name('table:0')

                # relationship between source and target
                train_op_multiclass = graph.get_collection('train_op_multiclass')[0]
                # accuracy_multiclass = graph.get_collection('accuracy_multiclass')[0]
                # loss_multiclass = graph.get_tensor_by_name('loss_multiclass:0')
                test_loss_multiclass = graph.get_tensor_by_name('test_loss_multiclass:0')
                pred_multiclass = graph.get_collection('pred_multiclass')[0]
                train_loss_multiclass = graph.get_tensor_by_name('loss_multiclass:0')
                init = tf.global_variables_initializer()

            report = open(self.nn_config['report'], 'a+')
            table_data = self.df.table_generator()
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init, feed_dict={table: table_data})
                print('epoch:\n')
                print(str(self.nn_config['epoch']) + '\n')
                report.write('multiclass\n')
                report.flush()
                start = datetime.now()
                for i in range(self.nn_config['epoch']):
                    X_data, Y_data = self.df.target_data_generator('train', batch_num=i,
                                                                   batch_size=self.nn_config['batch_size'])
                    sess.run(train_op_multiclass, feed_dict={X: X_data, Y_: Y_data})
                    # train_loss = sess.run(test_loss_multiclass, feed_dict={X: X_data, Y_: Y_data})
                    if i % self.nn_config['mod'] == 0 and i != 0:
                        X_data, Y_data = self.df.target_data_generator('test')
                        pred, test_loss,train_loss = sess.run([pred_multiclass, test_loss_multiclass,train_loss_multiclass],
                                              feed_dict={X: X_data, Y_: Y_data})
                        f1_macro, f1_micro = self.f1(Y_data, pred, self.nn_config['target_NETypes_num'])
                        end = datetime.now()
                        time_cost = end - start
                        report.write(
                            'epoch:{}, time_cost:{}, test_loss:{},train_loss:{} macro_f1:{}, micro_f1:{}\n'.format(str(i), str(time_cost),
                                                                                                 str(test_loss),
                                                                                                 str(train_loss),
                                                                                                 str(f1_macro),
                                                                                                 str(f1_micro)))
                        report.flush()
                        start = end