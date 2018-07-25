import tensorflow as tf
from datetime import datetime
import numpy as np
import copy

from util.eval.evaluate_overlap import evaluate

class Classifier:
    def __init__(self, nn_config, datafeed,data_config,metrics):
        self.nn_config = nn_config
        self.df = datafeed
        self.data_config = data_config
        self.mt = metrics

    def X_input(self,graph):
        """
        
        :param graph: 
        :return: shape = (batch size, words num, feature dim)
        """
        X = tf.placeholder(name='X',shape=(None,self.nn_config['words_num']),dtype='int32')
        return X

    def sequence_length(self,X,graph):
        """
        
        :param X: (batch size, max words num)
        :param graph: 
        :return: 
        """
        ones = tf.ones_like(X,dtype='int32')*self.nn_config['pad_index']
        condition = tf.equal(ones,X)
        seq_len = tf.reduce_sum(tf.where(condition,tf.zeros_like(X,dtype='int32'),tf.ones_like(X,dtype='int32')),axis=1,name='seq_len')
        return seq_len


    def Y_input(self,graph):
        """
        
        :param graph: 
        :return: shape = (batch size, words num)
        """
        Y_ = tf.placeholder(name='Y_',shape=(None,self.nn_config['words_num']),dtype='int32')
        return Y_

    def lookup_mask(self,X,graph):
        ones = tf.ones_like(X, dtype='int32')*self.nn_config['pad_index']
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        multiple = [1, 1, self.nn_config['feature_dim']]
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiple)
        return mask

    def lookup_table(self,X,mask,graph):
        table = tf.placeholder(name='table',shape=(self.nn_config['vocabulary_size'],self.nn_config['feature_dim']),dtype='float32')
        table = tf.Variable(table,dtype='float32')
        X = tf.nn.embedding_lookup(table,X)
        X = tf.multiply(X,mask)
        return X


    def bilstm(self,X,seq_len,graph):
        """
        
        :param X: shape = (batch size, words num, feature dim)
        :param mask: shape = (batch size, words num, feature dim)
        :return: 
        """
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(self.nn_config['lstm_cell_size'])
        # outputs.shape = [(batch size, max time step, lstm cell size),(batch size, max time step, lstm cell size)]
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,cell_bw=bw_cell,inputs=X,sequence_length=seq_len,dtype='float32')

        graph.add_to_collection('bilstm_reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name('bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
        graph.add_to_collection('bilstm_reg',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
                                    graph.get_tensor_by_name('bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))

        # outputs.shape = (batch size, max time step, 2*lstm cell size)
        outputs = tf.concat(outputs,axis=2,name='bilstm_outputs')
        graph.add_to_collection('bilstm_outputs',outputs)
        return outputs

    # def bilstm(self, X, seq_len, graph):
    #     """
    #
    #     :param X: shape = (batch size, words num, feature dim)
    #     :param mask: shape = (batch size, words num, feature dim)
    #     :return:
    #     """
    #     cell = tf.contrib.rnn.LSTMCell(self.nn_config['lstm_cell_size'])
    #     cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.nn_config['dropout'])
    #     cell = tf.contrib.rnn.MultiRNNCell([cell] * self.nn_config['bilstm_num_layers'])
    #     # outputs.shape = [(batch size, max time step, lstm cell size),(batch size, max time step, lstm cell size)]
    #     outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=X,
    #                                                  sequence_length=seq_len, dtype='float32')
    #
    #     for v in tf.all_variables():
    #         if v.name.startswith('bilstm'):
    #             print(v.name)
    #     exit()
    #     # graph.add_to_collection('bilstm_reg',
    #     #                         tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
    #     #                             graph.get_tensor_by_name('bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
    #     # graph.add_to_collection('bilstm_reg',
    #     #                         tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
    #     #                             graph.get_tensor_by_name('bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))
    #
    #     graph.add_to_collection('bilstm_reg',
    #                             tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(
    #                                 graph.get_tensor_by_name('bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))
    #
    #     # outputs.shape = (batch size, max time step, 2*lstm cell size)
    #     outputs = tf.concat(outputs, axis=2, name='bilstm_outputs')
    #     graph.add_to_collection('bilstm_outputs', outputs)
    #     return outputs

    # ###################
    #     source crf
    # ###################
    def crf_source(self,X,Y_,seq_len,graph):
        """
        
        :param X: (batch size, max time step, 2*lstm cell size)
        :param Y_: (batch size, words num)
        :param seq_len: 
        :param graph: 
        :return: (batch size, words num), (batch size, words num)
        """
        # p(x,y)
        W_s = tf.get_variable(name='W_s',initializer=tf.zeros(shape=(2*self.nn_config['lstm_cell_size'],self.nn_config['source_NETypes_num']),dtype='float32'))
        graph.add_to_collection('reg_crf_source', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_s))
        W_trans = tf.get_variable(name='W_trans_crf_source',initializer=tf.zeros(shape=(self.nn_config['source_NETypes_num'],self.nn_config['source_NETypes_num']),dtype='float32'))
        graph.add_to_collection('reg_crf_source', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_trans))
        # X.shape = (batch size*words num, 2*lstm cell size)
        X = tf.reshape(X,shape=(-1,2*self.nn_config['lstm_cell_size']))
        # score.shape = (batch size, words num, 2*lstm cell size)
        score = tf.reshape(tf.matmul(X,W_s),shape=(-1,self.nn_config['words_num'],self.nn_config['source_NETypes_num']))
        # log_likelihood.shape=(batch_size,)
        log_likelihood, transition_params= tf.contrib.crf.crf_log_likelihood(score,Y_,seq_len,W_trans)
        viterbi_seq, _ = tf.contrib.crf.crf_decode(score,transition_params,seq_len)
        graph.add_to_collection('pred_crf_source',viterbi_seq)
        return log_likelihood, viterbi_seq

    def loss_crf_source(self,log_likelihood,graph):
        """
        
        :param log_likelihood: shape=(batch size,) 
        :param graph: 
        :return: 
        """
        regularizer = graph.get_collection('reg_crf_source')
        regularizer.extend(graph.get_collection('bilstm_reg'))
        loss = tf.reduce_mean(tf.add(-log_likelihood,tf.reduce_sum(regularizer)),name='loss_crf_source')
        return loss

    def test_loss_crf_source(self,log_likelihood,graph):
        loss = tf.reduce_mean(-log_likelihood,name='test_loss_crf_source')
        return loss


    def opt_crf_source(self,loss,graph):
        train_op = tf.train.GradientDescentOptimizer(self.nn_config['lr']).minimize(loss)
        graph.add_to_collection('train_op_crf_source',train_op)
        return train_op

    # ######################################################
    #     relationship between source data and target data
    # ######################################################
    # def softmax_log_mask(self,X,graph):
    #     """
    #
    #     :param X: (batch size, max words num)
    #     :param graph:
    #     :return: (batch size, max words num)
    #     """
    #     condition = tf.equal(X, tf.ones_like(X, dtype='int32') * self.nn_config['pad_index'])
    #     mask = tf.where(condition, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
    #     mask = tf.tile(tf.expand_dims(mask,axis=2),multiples=[1,1,self.nn_config['target_NETypes_num']])
    #     return mask

    # def Y_2one_hot(self,Y_,mask,graph):
    #     """
    #
    #     :param Y_: (batch size, max words num)
    #     :param mask: (batch size, max words num, labels number)
    #     :return: (batch size, words num, labes num)
    #     """
    #     # add mask to mask tag of #PAD# to 0 vec, otherwise #PAD# will update parameter
    #     Y_one_hot = tf.multiply(tf.one_hot(Y_,depth=self.nn_config['target_NETypes_num'],axis=2,dtype='float32'),mask)
    #     return Y_one_hot

    def softmax_log_mask(self,X,graph):
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

    def multiclass_score(self,X,graph):
        """
        
        :param X: (batch size, max words num, 2*lstm cell size)
        :param graph: 
        :return: (batch size, max words num, target NETypes num)
        """
        W_s = graph.get_tensor_by_name('W_s:0')
        graph.add_to_collection('reg_multiclass',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_s))
        W_t = tf.get_variable(name='W_t',
                              initializer=tf.random_uniform(
                                  shape=(self.nn_config['source_NETypes_num'], self.nn_config['target_NETypes_num']),
                                  dtype='float32'))
        graph.add_to_collection('reg_multiclass',
                                tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_t))
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

    def loss_multiclass(self,score,Y_,mask,graph):
        """
        
        :param score: (batch size, max words num, target NETypes num)
        :param Y_: (batch size, max words num, target NETypes num)
        :param graph: 
        :return: 
        """
        regularizer = graph.get_collection('reg_multiclass')
        regularizer.extend(graph.get_collection('bilstm_reg'))
        term2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(Y_), logits=score, dim=-1)

        term1 = tf.multiply(term2, mask)
        reg = tf.reduce_sum(regularizer, keepdims=True)
        loss = tf.reduce_mean(tf.add(term1, reg),
                              name='loss_multiclass')
        return loss

    def test_loss_multiclass(self,score,Y_,mask,graph):
        loss = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_, logits=score), mask),
                              name='test_loss_multiclass')
        return loss

    def opt_multiclass(self,loss,graph):
        train_op = tf.train.GradientDescentOptimizer(self.nn_config['lr'], name='train_op_crf_source').minimize(loss)
        graph.add_to_collection('train_op_multiclass',train_op)
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

    def pred_multiclass(self,score,mask,graph):
        """
        
        :param score: (batch size, max words num, target NETypes num)
        :param mask: (batch size, max words num)
        :return: 
        """
        Y = tf.multiply(tf.argmax(score,axis=2,output_type='int32'),mask)
        graph.add_to_collection('pred_multiclass',Y)
        return Y

    # #####################
    #      target crf
    # #####################
    def crf_target(self, X, Y_, seq_len, graph):
        """

        :param X: (batch size, max time step, 2*lstm cell size)
        :param Y_: (batch size, max words num)
        :param seq_len: 
        :param graph: 
        :return: (batch size, max words num), (batch size, max words num)
        """
        # p(x,y)
        # W_s.shape = (2*lstm cell size, target NETypes num)
        # W_t = tf.get_variable(name='stage3_W_t',
        #                       initializer=tf.zeros(
        #                           shape=(2 * self.nn_config['lstm_cell_size'], self.nn_config['target_NETypes_num']),
        #                           dtype='float32'))
        W_t = tf.get_variable(tf.zeros(
                                  shape=(2 * self.nn_config['lstm_cell_size'], self.nn_config['target_NETypes_num']),
                                  dtype='float32'),dtype='float32',name='stage3_W_t')
        graph.add_to_collection('stage3_W_t',W_t)
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_t))
        W_trans = tf.get_variable(name='W_trans_crf_target',
                                  initializer=tf.zeros(shape=(self.nn_config['target_NETypes_num'],
                                                              self.nn_config['target_NETypes_num']),
                                                       dtype='float32'))
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_trans))
        # X.shape = (batch size*words num, 2*lstm cell size)
        X = tf.reshape(X, shape=(-1, 2 * self.nn_config['lstm_cell_size']))
        # score.shape = (batch size, words num, 2*lstm cell size)
        score = tf.reshape(tf.matmul(X, W_t),
                           shape=(-1, self.nn_config['words_num'], self.nn_config['target_NETypes_num']))
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(score, Y_, seq_len, W_trans)
        viterbi_seq, _ = tf.contrib.crf.crf_decode(score, transition_params, seq_len)
        graph.add_to_collection('pred_crf_target', viterbi_seq)
        return log_likelihood, viterbi_seq

    def loss_crf_target(self, log_likelihood, graph):
        regularizer = graph.get_collection('reg_crf_target')
        regularizer.extend(graph.get_collection('bilstm_reg'))
        # loss = tf.add(tf.reduce_mean(-log_likelihood),
        #               tf.truediv(tf.reduce_sum(regularizer),
        #                          tf.constant(self.nn_config['batch_size'],dtype='float32')),
        #               name='loss_crf_target')
        loss = tf.reduce_mean(tf.add(-log_likelihood, tf.reduce_sum(regularizer)), name='loss_crf_target')
        return loss

    def test_loss_crf_target(self,log_likelihood,graph):
        loss = tf.reduce_mean(-log_likelihood,name='test_loss_crf_target')
        return loss

    def opt_crf_target(self, loss, graph):
        train_op = tf.train.GradientDescentOptimizer(self.nn_config['lr']).minimize(loss)
        graph.add_to_collection('train_op_crf_target', train_op)
        return train_op

    def optimize(self, loss, graph):
        train_op = tf.train.GradientDescentOptimizer(self.nn_config['lr']).minimize(loss)
        return train_op

    def accuracy(self,Y_,Y,tag_seq_mask,seq_len,graph):
        """
        :param Y_:  (batch size, max words num) 
        :param Y: 
        :param tag_seq_mask: (batch size, max words num); used to mask #PAD#
        :param seq_len: (batch size,)
        :param graph: 
        :return: 
        """
        condition = tf.equal(Y_,Y)
        correct_labels_num=tf.reduce_sum(tf.multiply(tf.where(condition,tf.ones_like(Y,dtype='float32'),
                                                     tf.zeros_like(Y,dtype='float32')),tf.cast(tag_seq_mask,dtype='float32')))
        total_labels_num = tf.reduce_sum(tf.cast(seq_len,dtype='float32'))
        accuracy = tf.truediv(correct_labels_num,total_labels_num)
        return accuracy

    # def f1(self,Y_,Y,labels_num):
    #     """
    #
    #     :param Y_: (batch size, max words num)
    #     :param Y: (batch size, max words num
    #     :return:
    #     """
    #     Y = np.reshape(Y,newshape=(-1,)).astype('float32')
    #     Y_ = np.reshape(Y_,newshape=(-1,)).astype('float32')
    #     ls_y=[]
    #     ls_y_=[]
    #     for i in range(Y.shape[0]):
    #         if Y_[i] !=0:
    #             ls_y.append(Y[i])
    #             ls_y_.append(Y_[i])
    #     Y = np.array(ls_y,dtype='float32')
    #     Y_ = np.array(ls_y_,dtype='float32')
    #
    #     f1_macro = sklearn.metrics.f1_score(Y_, Y, labels=list(range(1,labels_num)),average='macro')
    #     f1_micro = sklearn.metrics.f1_score(Y_, Y, labels=list(range(1,labels_num)),average='micro')
    #     return f1_macro,f1_micro

    def classifier(self):
        graph = tf.Graph()
        if self.nn_config['stage1'] == 'True':
            with graph.as_default():
                # X.shape = (batch size, words num)
                X_id = self.X_input(graph)
                seq_len = self.sequence_length(X_id,graph)
                tag_seq_mask = self.tag_sequence_mask(X_id, graph)
                Y_ = self.Y_input(graph)
                mask = self.lookup_mask(X_id,graph)
                # X.shape = (batch size, words num, feature dim)
                X = self.lookup_table(X_id,mask,graph)
                with tf.variable_scope('bilstm') as vs:
                    # X.shape = (batch size, max time step, 2*lstm cell size)
                    X = self.bilstm(X,seq_len,graph)

                # crf source
                log_likelihood,viterbi_seq=self.crf_source(X,Y_,seq_len,graph)
                loss = self.loss_crf_source(log_likelihood,graph)
                test_loss = self.test_loss_crf_source(log_likelihood,graph)
                train_op = self.optimize(loss,graph)
                graph.add_to_collection('train_op_crf_source', train_op)

                # multiclass
                # train relationship between source data and target data
                # Y_one_hot.shape = (batch size, words num, target NETypes num)
                soft_log_mask = tf.reshape(self.softmax_log_mask(X_id, graph), shape=(-1,))
                Y_one_hot = self.Y_2one_hot(Y_, graph)
                score = self.multiclass_score(X, graph)
                pred = self.pred_multiclass(tf.reshape(score,
                                                       shape=(-1, self.nn_config['words_num'],
                                                              self.nn_config['target_NETypes_num'])),
                                            tag_seq_mask,
                                            graph)
                loss = self.loss_multiclass(score,
                                            tf.reshape(Y_one_hot, shape=(-1, self.nn_config['target_NETypes_num'])),
                                            soft_log_mask, graph)
                test_loss = self.test_loss_multiclass(score, tf.reshape(Y_one_hot,
                                                                        shape=(-1, self.nn_config['target_NETypes_num'])),
                                                      soft_log_mask,
                                                      graph)
                train_op = self.optimize(loss, graph)
                graph.add_to_collection('train_op_multiclass', train_op)

                # crf target
                log_likelihood, viterbi_seq = self.crf_target(X, Y_, seq_len, graph)
                loss = self.loss_crf_target(log_likelihood, graph)
                test_loss = self.test_loss_crf_target(log_likelihood, graph)
                train_op = self.optimize(loss, graph)
                graph.add_to_collection('train_op_crf_target',train_op)
                saver = tf.train.Saver()
        else:
            with graph.as_default():
                saver = tf.train.import_meta_graph(self.nn_config['model'])
        return graph,saver

    def reporter(self,report,best_score):
        report.write('=========================\n')
        report.write('epoch: ' + str(best_score['epoch']))
        report.write('loss: %s' % str(best_score['loss']))
        report.write(best_score["per_f1"] + '\n')
        report.write(best_score["per_pre"] + '\n')
        report.write(best_score["per_recall"] + '\n')
        report.write(best_score["micro_f1"] + '\n')
        report.write(best_score["micro_pre"] + '\n')
        report.write(best_score["micro_recall"] + '\n')
        report.write(best_score["macro_f1"] + '\n')
        report.write(best_score["macro_pre"] + '\n')
        report.write(best_score["macro_recall"] + '\n')
        report.write('=========================\n')
        report.flush()

    def stdout(self,best_score):
        print('=========================\n')
        print('epoch: ' + str(best_score['epoch']))
        print('loss: %s' % str(best_score['loss']))
        print(best_score["per_f1"] + '\n')
        print(best_score["per_pre"] + '\n')
        print(best_score["per_recall"] + '\n')
        print(best_score["micro_f1"] + '\n')
        print(best_score["micro_pre"] + '\n')
        print(best_score["micro_recall"] + '\n')
        print(best_score["macro_f1"] + '\n')
        print(best_score["macro_pre"] + '\n')
        print(best_score["macro_recall"] + '\n')
        print('=========================\n')

    def train(self):
        graph,saver = self.classifier()
        with graph.device('/:gpu0'):
            with graph.as_default():
                X = graph.get_tensor_by_name('X:0')
                Y_ = graph.get_tensor_by_name('Y_:0')
                table = graph.get_tensor_by_name('table:0')

                # crf source
                train_op_crf_source = graph.get_collection('train_op_crf_source')[0]
                # accuracy_crf_source = graph.get_collection('accuracy_crf_source')[0]
                #loss_crf_source = graph.get_tensor_by_name('loss_crf_source:0')
                test_loss_crf_source = graph.get_tensor_by_name('test_loss_crf_source:0')
                pred_crf_source = graph.get_collection('pred_crf_source')[0]

                # relationship between source and target
                train_op_multiclass = graph.get_collection('train_op_multiclass')[0]
                # accuracy_multiclass = graph.get_collection('accuracy_multiclass')[0]
                #loss_multiclass = graph.get_tensor_by_name('loss_multiclass:0')
                test_loss_multiclass = graph.get_tensor_by_name('test_loss_multiclass:0')
                pred_multiclass = graph.get_collection('pred_multiclass')[0]
                train_loss_multiclass=graph.get_tensor_by_name('loss_multiclass:0')

                # crf target
                train_op_crf_target = graph.get_collection('train_op_crf_target')[0]
                # accuracy_crf_target = graph.get_collection('accuracy_crf_target')[0]
                #loss_crf_target = graph.get_tensor_by_name('loss_crf_target:0')
                test_loss_crf_target = graph.get_tensor_by_name('test_loss_crf_target:0')
                pred_crf_target = graph.get_collection('pred_crf_target')[0]
                train_loss_crf_target = graph.get_tensor_by_name('loss_crf_target:0')

                W_s = graph.get_tensor_by_name('W_s:0')
                W_t = graph.get_tensor_by_name('W_t:0')
                stage3_W_t = graph.get_collection('stage3_W_t')[0]
                print(stage3_W_t)
                print('====================')
                init = tf.global_variables_initializer()

            report = open(self.nn_config['report'], 'a+')
            table_data = self.df.table_generator()
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                report.write('session\n')
                if self.nn_config['stage1'] == 'True':
                    print('start training stage1')
                    sess.run(init, feed_dict={table: table_data})
                    report.write('=================crf_source=================\n')
                    best_score={}
                    early_stop_count=0
                    for i in range(self.nn_config['epoch_stage1']):
                        dataset = self.df.source_data_generator('train')
                        for X_data,Y_data in dataset:
                            sess.run(train_op_crf_source,feed_dict={X:X_data,Y_:Y_data})

                        dataset = self.df.source_data_generator('test')
                        for X_data,Y_data in dataset:
                            pred,loss = sess.run([pred_crf_source,test_loss_crf_source],feed_dict={X:X_data,Y_:Y_data})
                            # f1_macro,f1_micro = self.f1(Y_data,pred,self.nn_config['source_NETypes_num'])
                            source_id2label_dic=self.df.source_id2label_generator()
                            true_labels = Y_data
                            pred_labels = pred
                            I = self.mt.word_id2txt(X_data, true_labels, pred_labels, source_id2label_dic)
                            self.mt.conll_eval_file(I)
                            eval_result = evaluate(self.data_config['conlleval_filePath'])
                            eval_result['epoch']=i
                            eval_result['loss']=loss
                            if len(best_score)==0:
                                best_score=copy.deepcopy(eval_result)
                                # self.stdout(best_score)
                            else:
                                if best_score['micro_f1']<eval_result['micro_f1']:
                                    saver.save(sess, self.nn_config['model'])
                                    best_score=copy.deepcopy(eval_result)
                                else:
                                    early_stop_count+=1
                        if early_stop_count>=self.nn_config['early_stop']:
                            self.reporter(report,best_score)
                            break
                    if early_stop_count < self.nn_config['early_stop']:
                        self.reporter(report, best_score)
                else:
                    print(stage3_W_t)
                    saver.restore(sess,self.nn_config['model_sess'])
                    report.write('=================multiclass=================\n')
                    report.flush()
                    print('start training stage2')
                    best_score = {}
                    early_stop_count = 0
                    # for i in range(self.nn_config['epoch_stage2']):
                    #     # dataset= self.df.target_data_generator('train')
                    #     # print('train stage2')
                    #     # for X_data,Y_data in dataset:
                    #     #     sess.run(train_op_multiclass,feed_dict={X:X_data,Y_:Y_data})
                    #     #train_loss = sess.run(test_loss_multiclass, feed_dict={X: X_data, Y_: Y_data})
                    #     dataset = self.df.target_data_generator('test')
                    #     print('test stage2')
                    #     for X_data,Y_data in dataset:
                    #         pred,loss= sess.run([pred_multiclass,test_loss_multiclass],feed_dict={X:X_data,Y_:Y_data})
                    #         target_id2label_dic = self.df.target_id2label_generator()
                    #         true_labels = Y_data
                    #         pred_labels = pred
                    #         I = self.mt.word_id2txt(X_data, true_labels, pred_labels, target_id2label_dic)
                    #         self.mt.conll_eval_file(I)
                    #         eval_result = evaluate(self.data_config['conlleval_filePath'])
                    #         eval_result['epoch'] = i
                    #         eval_result['loss'] = loss
                    #         if len(best_score)==0:
                    #             best_score=copy.deepcopy(eval_result)
                    #         else:
                    #             if best_score['micro_f1']<eval_result['micro_f1']:
                    #                 best_score=copy.deepcopy(eval_result)
                    #             else:
                    #                 early_stop_count+=1
                    #     if early_stop_count>=self.nn_config['early_stop']:
                    #         self.reporter(report, best_score)
                    #         break
                    # if early_stop_count < self.nn_config['early_stop']:
                    #     self.reporter(report, best_score)

                    report.write('\n')
                    report.write('=================crf_target=================\n')
                    W_s_data = sess.run(W_s)
                    W_t_data = sess.run(W_t)
                    print(stage3_W_t)
                    stage3_W_t.load(np.ones((300, 42), dtype='float32'))
                    print(sess.run(stage3_W_t))
                    stage3_W_t.load(np.matmul(W_s_data,W_t_data))
                    exit()
                    print('start training stage3')
                    best_score = {}
                    early_stop_count = 0
                    for i in range(self.nn_config['epoch_stage3']):
                        dataset = self.df.target_data_generator('train')
                        for X_data,Y_data in dataset:
                            sess.run(train_op_crf_target, feed_dict={X: X_data, Y_: Y_data})
                        #train_loss = sess.run(test_loss_crf_target, feed_dict={X: X_data, Y_: Y_data})
                        dataset = self.df.target_data_generator('test')
                        for X_data,Y_data in dataset:
                            pred,loss= sess.run([pred_crf_target,test_loss_crf_target],feed_dict={X:X_data,Y_:Y_data})
                            target_id2label_dic = self.df.target_id2label_generator()
                            true_labels = Y_data
                            pred_labels = pred
                            I = self.mt.word_id2txt(X_data, true_labels, pred_labels, target_id2label_dic)
                            self.mt.conll_eval_file(I)
                            eval_result = evaluate(self.data_config['conlleval_filePath'])
                            eval_result['epoch'] = i
                            eval_result['loss'] = loss
                            if len(best_score)==0:
                                best_score=copy.deepcopy(eval_result)
                            else:
                                if best_score['micro_f1']<eval_result['micro_f1']:
                                    best_score=copy.deepcopy(eval_result)
                                else:
                                    early_stop_count+=1
                        if early_stop_count>=self.nn_config['early_stop']:
                            self.reporter(report, best_score)
                            break
                    if early_stop_count < self.nn_config['early_stop']:
                        self.reporter(report, best_score)