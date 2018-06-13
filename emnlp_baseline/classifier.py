import tensorflow as tf
from datetime import datetime
import sklearn
import numpy as np
import math

class Classifier:
    def __init__(self, nn_config, datafeed):
        self.nn_config = nn_config
        self.df = datafeed

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
        # outputs.shape = (batch size, max time step, 2*lstm cell size)
        outputs = tf.concat(outputs,axis=2,name='bilstm_outputs')
        graph.add_to_collection('bilstm_outputs',outputs)
        return outputs

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
        W_s = tf.get_variable(name='W_s',initializer=tf.random_normal(shape=(2*self.nn_config['lstm_cell_size'],self.nn_config['source_NETypes_num']),dtype='float32'))
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
        # W_s.shape = (2*lstm cell size, source NETypes num)
        W_s = graph.get_tensor_by_name('W_s:0')
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_s))
        # W_t.shape = (source_NETypes_num, target_NETypes_num)
        W_t = graph.get_tensor_by_name('W_t:0')
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_t))
        W_trans = tf.get_variable(name='W_trans_crf_target',
                                  initializer=tf.zeros(shape=(self.nn_config['target_NETypes_num'],
                                                              self.nn_config['target_NETypes_num']),
                                                       dtype='float32'))
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_trans))
        # X.shape = (batch size*words num, 2*lstm cell size)
        X = tf.reshape(X, shape=(-1, 2 * self.nn_config['lstm_cell_size']))
        # score.shape = (batch size, words num, 2*lstm cell size)
        score = tf.reshape(tf.matmul(tf.matmul(X,W_s), W_t),
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

    def f1(self,Y_,Y,labels_num):
        """
        
        :param Y_: (batch size, max words num)
        :param Y: (batch size, max words num
        :return: 
        """
        Y = np.reshape(Y,newshape=(-1,)).astype('float32')
        Y_ = np.reshape(Y_,newshape=(-1,)).astype('float32')
        ls_y=[]
        ls_y_=[]
        for i in range(Y.shape[0]):
            if Y_[i] !=0:
                ls_y.append(Y[i])
                ls_y_.append(Y_[i])
        Y = np.array(ls_y,dtype='float32')
        Y_ = np.array(ls_y_,dtype='float32')

        f1_macro = sklearn.metrics.f1_score(Y_, Y, labels=list(range(1,labels_num)),average='macro')
        f1_micro = sklearn.metrics.f1_score(Y_, Y, labels=list(range(1,labels_num)),average='micro')
        return f1_macro,f1_micro

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
                    graph.add_to_collection('bilstm_reg',
                                            tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(graph.get_tensor_by_name('bilstm/bidirectional_rnn/fw/basic_lstm_cell/kernel:0')))
                    graph.add_to_collection('bilstm_reg',
                                            tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(graph.get_tensor_by_name('bilstm/bidirectional_rnn/bw/basic_lstm_cell/kernel:0')))

                # crf source
                log_likelihood,viterbi_seq=self.crf_source(X,Y_,seq_len,graph)
                loss = self.loss_crf_source(log_likelihood,graph)
                test_loss = self.test_loss_crf_source(log_likelihood,graph)
                train_op = self.optimize(loss,graph)
                graph.add_to_collection('train_op_crf_source', train_op)
                accuracy_crf_source = self.accuracy(Y_,viterbi_seq,tag_seq_mask,seq_len,graph)
                graph.add_to_collection('accuracy_crf_source',accuracy_crf_source)

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
                accuracy_crf_target = self.accuracy(Y_,viterbi_seq,tag_seq_mask,seq_len,graph)
                graph.add_to_collection('accuracy_crf_target',accuracy_crf_target)
                saver = tf.train.Saver()
        else:
            with graph.as_default():
                saver = tf.train.import_meta_graph(self.nn_config['model'])
        return graph,saver

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

                init = tf.global_variables_initializer()
            report = open(self.nn_config['report'], 'a+')
            report.write(self.nn_config['stage1']+'\n')
            table_data = self.df.table_generator()
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                print('start training')
                report.write('session\n')
                if self.nn_config['stage1'] == 'True':
                    sess.run(init, feed_dict={table: table_data})
                    report.write('=================crf_source=================\n')
                    start = datetime.now()
                    for i in range(self.nn_config['epoch_stage1']):
                        dataset = self.df.source_data_generator('train')
                        for X_data,Y_data in dataset:
                            sess.run(train_op_crf_source,feed_dict={X:X_data,Y_:Y_data})

                        dataset = self.df.source_data_generator('test')
                        for X_data,Y_data in dataset:
                            pred,loss = sess.run([pred_crf_source,test_loss_crf_source],feed_dict={X:X_data,Y_:Y_data})
                            f1_macro,f1_micro = self.f1(Y_data,pred,self.nn_config['source_NETypes_num'])
                            end = datetime.now()
                            time_cost = end-start
                            report.write('epoch:{}, time_cost:{}, loss:{}, macro_f1:{}, micro_f1:{}\n'.format(str(i),str(time_cost),str(loss),str(f1_macro),str(f1_micro)))
                            report.flush()
                            start = end
                    saver.save(sess,self.nn_config['model'])
                else:
                    # sess.run(init, feed_dict={table: table_data})
                    W_s = graph.get_tensor_by_name('W_s:0')
                    saver.restore(sess,self.nn_config['model_sess'])
                    W_s_norm = tf.norm(W_s)
                    print('W_s_norm: ', str(sess.run(W_s_norm)))
                    report.write('=================multiclass=================\n')
                    report.flush()
                    start = datetime.now()
                    for i in range(self.nn_config['epoch_stage2']):
                        dataset= self.df.target_data_generator('train')
                        for X_data,Y_data in dataset:
                            sess.run(train_op_multiclass,feed_dict={X:X_data,Y_:Y_data})
                        #train_loss = sess.run(test_loss_multiclass, feed_dict={X: X_data, Y_: Y_data})
                        dataset = self.df.target_data_generator('test')
                        for X_data,Y_data in dataset:
                            pred,loss= sess.run([pred_multiclass,test_loss_multiclass],feed_dict={X:X_data,Y_:Y_data})
                            f1_macro,f1_micro = self.f1(Y_data,pred,self.nn_config['target_NETypes_num'])
                            end = datetime.now()
                            time_cost = end - start
                            report.write('epoch:{}, time_cost:{}, loss:{}, macro_f1:{}, micro_f1:{}\n'.format(str(i), str(time_cost), str(loss),str(f1_macro),str(f1_micro)))
                            report.flush()
                            start = end
                        # if i%self.nn_config['mod'] == 0 and i!=0:
                        #     X_data,Y_data = self.df.target_data_generator('test')
                        #     length = X_data.shape[0]
                        #     slides = []
                        #     avg = 300
                        #     for j in range(1, avg + 1):
                        #         slides.append(j / avg)
                        #     slice_pre = 0
                        #     pred_labels = []
                        #     losses = []
                        #     for slide in slides:
                        #         slice_cur = int(math.floor(slide * length))
                        #         pred,loss=sess.run([pred_multiclass,test_loss_multiclass],
                        #                             feed_dict={X: X_data[slice_pre:slice_cur],
                        #                                 Y_: Y_data[slice_pre:slice_cur]})
                        #         pred_labels.append(pred)
                        #         losses.append(loss)
                        #         slice_pre = slice_cur
                        #     pred_labels = np.concatenate(pred_labels, axis=0)
                        #     f1_macro, f1_micro = self.f1(Y_data,pred_labels,self.nn_config['target_NETypes_num'])
                        #     end = datetime.now()
                        #     time_cost = end - start
                        #     report.write('stage2:\nepoch:{}, time_cost:{}, test_loss:{}, train_loss:{} macro_f1:{}, micro_f1:{}\n'.format(str(i), str(time_cost), str(np.mean(losses)),str(train_loss),str(f1_macro),str(f1_micro)))
                        #     report.write('norm:'+str(np.sum(sess.run(tf.get_collection('reg_crf_source')))+np.sum(sess.run(tf.get_collection('reg_multiclass')))+np.sum(sess.run(tf.get_collection('reg_crf_target')))) + '\n')
                        #     report.flush()
                        #     start = end
                    report.write('\n')
                    report.write('=================crf_target=================\n')
                    start = datetime.now()
                    for i in range(self.nn_config['epoch_stage3']):
                        dataset = self.df.target_data_generator('train')
                        for X_data,Y_data in dataset:
                            sess.run(train_op_crf_target, feed_dict={X: X_data, Y_: Y_data})
                        #train_loss = sess.run(test_loss_crf_target, feed_dict={X: X_data, Y_: Y_data})
                        dataset = self.df.target_data_generator('test')
                        for X_data,Y_data in dataset:
                            pred,loss = sess.run([pred_crf_target,test_loss_crf_target],feed_dict={X:X_data,Y_:Y_data})
                            f1_macro, f1_micro = self.f1(Y_data,pred,self.nn_config['target_NETypes_num'])
                            end = datetime.now()
                            time_cost = end - start
                            report.write('epoch:{}, time_cost:{}, loss:{}, macro_f1:{}, micro_f1:{}\n'.format(str(i), str(time_cost), str(loss),str(f1_macro),str(f1_micro)))
                            report.flush()
                            start = end
                        # if i%self.nn_config['mod'] == 0 and i!=0:
                        #     X_data,Y_data = self.df.target_data_generator('test')
                        #     length = X_data.shape[0]
                        #     slides = []
                        #     avg = 300
                        #     for j in range(1, avg + 1):
                        #         slides.append(j / avg)
                        #     slice_pre = 0
                        #     pred_labels = []
                        #     losses = []
                        #     for slide in slides:
                        #         slice_cur = int(math.floor(slide * length))
                        #         pred,loss=sess.run([pred_crf_target, test_loss_crf_target],
                        #                             feed_dict={X: X_data[slice_pre:slice_cur],
                        #                                 Y_: Y_data[
                        #                                     slice_pre:slice_cur]})
                        #         pred_labels.append(pred)
                        #         losses.append(loss)
                        #         slice_pre = slice_cur
                        #     pred_labels = np.concatenate(pred_labels, axis=0)
                        #     f1_macro, f1_micro = self.f1(Y_data,pred_labels,self.nn_config['target_NETypes_num'])
                        #     end = datetime.now()
                        #     time_cost = end - start
                        #     print(
                        #         'stage3:\nepoch:{}, time_cost:{}, test_loss:{}, train_loss:{} macro_f1:{}, micro_f1:{}\n'.format(
                        #             str(i), str(time_cost), str(np.mean(losses)), str(train_loss), str(f1_macro),
                        #             str(f1_micro)))
                        #     print('norm:' + str(np.sum(sess.run(tf.get_collection('reg_crf_source'))) + np.sum(
                        #         sess.run(tf.get_collection('reg_multiclass'))) + np.sum(
                        #         sess.run(tf.get_collection('reg_crf_target')))) + '\n')
                        #     report.flush()
                        #     start = end

                    # final test
                    dataset = self.df.target_data_generator('test')
                    for X_data,Y_data in dataset:
                        length = X_data.shape[0]
                        slides =[]
                        avg=300
                        for j in range(1,avg+1):
                            slides.append(j/avg)
                        slice_pre=0
                        pred_labels=[]
                        for slide in slides:
                            slice_cur = int(math.floor(slide*length))
                            pred_labels.append(sess.run(pred_crf_target,feed_dict={X:X_data[slice_pre:slice_cur],Y_:Y_data[slice_pre:slice_cur]}))
                            slice_pre=slice_cur
                        pred_labels = np.concatenate(pred_labels,axis=0)

                        true_labels = Y_data
                        report.close()
                        print('finsh')
                        return true_labels,pred_labels,X_data