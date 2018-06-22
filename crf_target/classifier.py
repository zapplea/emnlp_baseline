import tensorflow as tf
from datetime import datetime
import sklearn
import numpy as np
import math
from util.eval.evaluate_overlap import evaluate

class Classifier:
    def __init__(self, nn_config, datafeed,data_config, metrics):
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
        # outputs.shape = (batch size, max time step, 2*lstm cell size)
        outputs = tf.concat(outputs,axis=2,name='bilstm_outputs')
        graph.add_to_collection('bilstm_outputs',outputs)
        return outputs

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
        W_t = tf.get_variable(name='W_t',
                              initializer=tf.zeros(
                                  shape=(2 * self.nn_config['lstm_cell_size'], self.nn_config['source_NETypes_num']),
                                  dtype='float32'))
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_t))

        W_trans = tf.get_variable(name='W_trans_crf_target',
                                  initializer=tf.zeros(shape=(self.nn_config['source_NETypes_num'],
                                                              self.nn_config['source_NETypes_num']),
                                                       dtype='float32'))
        graph.add_to_collection('reg_crf_target', tf.contrib.layers.l2_regularizer(self.nn_config['reg_rate'])(W_trans))
        # X.shape = (batch size*words num, 2*lstm cell size)
        X = tf.reshape(X, shape=(-1, 2 * self.nn_config['lstm_cell_size']))
        # score.shape = (batch size, words num, 2*lstm cell size)
        # score = tf.reshape(tf.matmul(tf.matmul(X,W_s), W_t),
        #                    shape=(-1, self.nn_config['words_num'], self.nn_config['target_NETypes_num']))
        score = tf.reshape(tf.matmul(X, W_t),
                           shape=(-1, self.nn_config['words_num'], self.nn_config['source_NETypes_num']))
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
        with graph.as_default():
            # X.shape = (batch size, words num)
            X_id = self.X_input(graph)
            seq_len = self.sequence_length(X_id,graph)
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

            # crf target
            log_likelihood, viterbi_seq = self.crf_target(X, Y_, seq_len, graph)
            loss = self.loss_crf_target(log_likelihood, graph)
            test_loss = self.test_loss_crf_target(log_likelihood, graph)
            train_op = self.optimize(loss, graph)
            graph.add_to_collection('train_op_crf_target',train_op)
            saver = tf.train.Saver()
        return graph,saver

    def train(self):
        graph,saver = self.classifier()
        with graph.device('/:gpu0'):
            with graph.as_default():
                X = graph.get_tensor_by_name('X:0')
                Y_ = graph.get_tensor_by_name('Y_:0')
                table = graph.get_tensor_by_name('table:0')

                # crf target
                train_op_crf_target = graph.get_collection('train_op_crf_target')[0]
                # accuracy_crf_target = graph.get_collection('accuracy_crf_target')[0]
                #loss_crf_target = graph.get_tensor_by_name('loss_crf_target:0')
                test_loss_crf_target = graph.get_tensor_by_name('test_loss_crf_target:0')
                pred_crf_target = graph.get_collection('pred_crf_target')[0]
                train_loss_crf_target = graph.get_tensor_by_name('loss_crf_target:0')

                # W_s = tf.norm(graph.get_tensor_by_name('W_s:0'))
                W_t = tf.norm(graph.get_tensor_by_name('W_t:0'))

                init = tf.global_variables_initializer()
            report = open(self.nn_config['report'], 'a+')
            table_data = self.df.table_generator()
            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                sess.run(init, feed_dict={table: table_data})
                report.write('session\n')
                report.write('\n')
                report.write('=================crf_target=================\n')
                start = datetime.now()
                print('start training stage3')
                for i in range(self.nn_config['epoch_stage3']):
                    dataset = self.df.source_data_generator('train')
                    train_loss_list=[]
                    for X_data,Y_data in dataset:
                        sess.run(train_op_crf_target, feed_dict={X: X_data, Y_: Y_data})
                        train_loss = sess.run(test_loss_crf_target,feed_dict={X: X_data, Y_: Y_data})
                        train_loss_list.append(train_loss)
                    train_loss=np.mean(train_loss_list)
                    #train_loss = sess.run(test_loss_crf_target, feed_dict={X: X_data, Y_: Y_data})
                    dataset = self.df.source_data_generator('test')
                    for X_data,Y_data in dataset:
                        # pred,test_loss,train_loss, W_s_data, W_t_data = sess.run([pred_crf_target,test_loss_crf_target,train_loss_crf_target, W_s, W_t],feed_dict={X:X_data,Y_:Y_data})
                        pred, test_loss, W_t_data = sess.run(
                            [pred_crf_target, test_loss_crf_target, W_t],
                            feed_dict={X: X_data, Y_: Y_data})
                        f1_macro, f1_micro = self.f1(Y_data,pred,self.nn_config['source_NETypes_num'])
                        end = datetime.now()
                        time_cost = end - start
                        # report.write('epoch:{}, time_cost:{}, test_loss:{}, train_loss:{}, macro_f1:{}, micro_f1:{}, W_t:{}\n'.
                        #              format(str(i), str(time_cost), str(test_loss),str(train_loss),str(f1_macro),str(f1_micro),str(W_t_data)))

                        true_labels=Y_data
                        pred_labels = pred
                        id2label_dic = self.df.source_id2label_generator()
                        I = self.mt.word_id2txt(X_data, true_labels, pred_labels, id2label_dic)
                        self.mt.conll_eval_file(I)
                        eval_result=evaluate(self.data_config['conlleval_filePath'])
                        report.write('=========================\n')
                        report.write('loss: %s'% str(test_loss))
                        report.write(eval_result["per_f1"]+'\n')
                        report.write(eval_result["per_pre"]+'\n')
                        report.write(eval_result["per_recall"] + '\n')
                        report.write(eval_result["micro_f1"] + '\n')
                        report.write(eval_result["micro_pre"] + '\n')
                        report.write(eval_result["micro_recall"] + '\n')
                        report.write(eval_result["macro_f1"] + '\n')
                        report.write(eval_result["macro_pre"] + '\n')
                        report.write(eval_result["macro_recall"] + '\n')
                        report.write('=========================\n')
                        report.flush()
                        start = end

                saver.save(sess, self.nn_config['model'])
                # # final test
                # dataset = self.df.source_data_generator('test')
                # for X_data, Y_data in dataset:
                #     length = X_data.shape[0]
                #     slides = []
                #     avg = 300
                #     for j in range(1, avg + 1):
                #         slides.append(j / avg)
                #     slice_pre = 0
                #     pred_labels = []
                #     for slide in slides:
                #         slice_cur = int(math.floor(slide * length))
                #         pred_labels.append(sess.run(pred_crf_target, feed_dict={X: X_data[slice_pre:slice_cur],
                #                                                                 Y_: Y_data[slice_pre:slice_cur]}))
                #         slice_pre = slice_cur
                #     pred_labels = np.concatenate(pred_labels, axis=0)
                #
                #     true_labels = Y_data
                #     report.close()
                #     print('finsh')
                #     return true_labels, pred_labels, X_data