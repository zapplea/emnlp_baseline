import tensorflow as tf
import numpy as np
from datetime import datetime
import sklearn

class NerZero:
    def __init__(self,nn_config,**kwargs):
        self.nn_config = nn_config
        if len(kwargs)>0:
            self.df = kwargs['df']

    def features_input(self,graph):
        X = tf.placeholder(shape=(None,self.nn_config['words_num']),dtype='int32')
        graph.add_to_collection('features_embeddings',X)
        return X

    def lookup_table(self,ids, mask, graph):
        table = tf.placeholder(shape=(2981402, 200), dtype='float32')
        graph.add_to_collection('table', table)
        table = tf.Variable(table, name='table',dtype='float32',trainable=False)
        embeddings = tf.nn.embedding_lookup(table, ids, partition_strategy='mod', name='embeddings')
        embeddings = tf.multiply(embeddings, mask)
        lcontxt = tf.reshape(embeddings[:, :2], (-1, 400))
        mention = tf.reduce_mean(embeddings[:, 2:-2], axis=1)
        rcontxt = tf.reshape(embeddings[:, -2:], (-1, 400))
        new_embeddings = tf.concat([lcontxt, mention, rcontxt], axis=1)
        return new_embeddings

    def mask_matrix(self,X):
        ones = tf.ones_like(X, dtype='int32')
        is_one = tf.equal(X, ones)
        mask = tf.where(is_one, tf.zeros_like(X, dtype='float32'), tf.ones_like(X, dtype='float32'))
        multiple = [1, 1, 200]
        mask = tf.tile(tf.expand_dims(mask, axis=2), multiple)
        return mask


    def labels_input(self,graph):
        """
        True labels use one hot ot represent and false labels use minus-one hot to represent
        :param graph: 
        :return: shape = (batch size,)
        """
        Y_ = tf.placeholder(shape=(None,),dtype='int32')
        graph.add_to_collection('labels',Y_)
        return Y_

    def A(self,graph):
        A = tf.get_variable(name='A',
                            initializer=tf.random_uniform(shape=(self.nn_config['feature_vec_dim'],self.nn_config['label_embed_dim']),
                                                                   dtype='float32'))
        graph.add_to_collection('reg',tf.contrib.layers.l2_regularizer(self.nn_config['lambda'])(A))

        graph.add_to_collection('A',A)
        return A

    # need a way to choose head words
    def B(self,graph):
        B_place = tf.placeholder(shape=(self.nn_config['label_embed_dim'],self.nn_config['labels_num']),dtype='float32')
        graph.add_to_collection('B',B_place)
        B = tf.Variable(B_place,dtype='float32',trainable=self.nn_config['is_B_trainable'])
        graph.add_to_collection('reg', tf.contrib.layers.l2_regularizer(self.nn_config['lambda'])(tf.subtract(B,B_place)))
        # F_norm = self.nn_config['lambda'] * tf.reduce_sum(tf.square(tf.subtract(B,B_place)))
        # graph.add_to_collection('F_norm',F_norm)
        return B

    def score(self,X,A,B):
        """
        
        :param X: shape = (batch size, feature vector dim)
        :param Y_: shape = (batch size, labels number)
        :param A: shape = (feature vector dim, labels embedding dim)
        :param B: shape = (labels embedding dim, labels number)
        :return: (batch size, number of labels)
        """
        #batch_M*M_d = batch_d
        phi = tf.matmul(X,A)
        # phi*B = batch_d*d*N
        score = tf.matmul(phi,B)
        return score

    def Lrank(self,Y_,f_Y,graph):
        """
        calculate L(rank(x,y)) for all labels. Outside this function, false labels will be masked.
        :param Y_: shape = (batch size,labels number)
        :param f_Y: (batch size,number of labels)
        :return: shape = (batch size, number of labels)
        """
        # true label is True in condition
        condition = tf.equal(Y_,tf.constant(1,dtype='float32'))
        f_Y_false = tf.where(condition,tf.ones_like(f_Y,dtype='float32')*tf.constant(-np.inf,dtype='float32'),f_Y)
        # I(1+f(x,y',A,B)>f(x,y,A,B))
        T = tf.tile(tf.expand_dims(f_Y,axis=2),multiples=[1,1,self.nn_config['labels_num']])
        F = tf.add(tf.constant(1,dtype='float32'),tf.tile(tf.expand_dims(f_Y_false,axis=1),multiples=[1,self.nn_config['labels_num'],1]))
        condition = tf.greater(F,T)
        # Rank.shape = (batch size, number of labels)
        Rank = tf.reduce_sum(tf.where(condition,tf.ones_like(condition,dtype='float32'),tf.zeros_like(condition,dtype='float32')),axis=2)
        Rank = tf.cast(Rank,dtype='int32')

        table = tf.placeholder(shape=(self.nn_config['labels_num']+1,),dtype='float32')
        graph.add_to_collection('ratio_table',table)
        table = tf.Variable(table,dtype='float32',trainable=False)
        Lrank = tf.nn.embedding_lookup(table,Rank)
        graph.add_to_collection('Lrank_of_one_mention',Lrank)
        return Lrank

    # def loss(self,Y_,f_Y,graph):
    #     """
    #     need to add Frobenius norm
    #     :param Y_: (batch size , number of labels)
    #     :param f_Y: (batch size , number of labels)
    #     :return: shape = (batch size, number of labels)
    #     """
    #     condition = tf.equal(Y_, tf.constant(1, dtype='float32'))
    #     # mask = tf.where(condition,tf.zeros_like(Y_,dtype='float32'),tf.ones_like(Y_,dtype='float32'))
    #     # mask = tf.tile(tf.expand_dims(mask,axis=1),multiples=[1,self.nn_config['labels_num'],1])
    #
    #     f_Y_true = tf.tile(tf.expand_dims(f_Y,axis=2),multiples=[1,1,self.nn_config['labels_num']])
    #
    #     # masked_f_Y.shape = (batch size, labels number); for true label, the value is inf
    #     masked_f_Y=tf.where(condition,tf.ones_like(f_Y,dtype='float32')*tf.constant(-np.inf,dtype='float32'),f_Y)
    #     f_Y_false = tf.tile(tf.expand_dims(masked_f_Y,axis=1),multiples=[1,self.nn_config['labels_num'],1])
    #     # f_loss.shape = (batch size, lables number, labels number)
    #     f_loss = tf.add(tf.subtract(tf.constant(1, dtype='float32'), f_Y_true), f_Y_false)
    #     f_zero = tf.zeros_like(f_loss,dtype='float32')
    #     # shape = (batch size, lables number, labels number,1)
    #     loss = tf.reduce_sum(tf.reduce_max(tf.concat([tf.expand_dims(f_loss,axis=3),tf.expand_dims(f_zero,axis=3)],axis=3),axis=3),axis=2)
    #
    #     # max = tf.concat([f_loss,])
    #     #
    #     # loss = tf.reduce_sum(tf.multiply(max,mask),axis=2)
    #     return loss

    def loss(self,Y_,f_Y,graph):
        """
        need to add Frobenius norm
        :param Y_: (batch size , number of labels)
        :param f_Y: (batch size , number of labels)
        :return: shape = (batch size, number of labels)
        """
        condition = tf.equal(Y_, tf.constant(1, dtype='float32'))
        mask = tf.where(condition,tf.zeros_like(Y_,dtype='float32'),tf.ones_like(Y_,dtype='float32'))
        mask = tf.tile(tf.expand_dims(mask,axis=1),multiples=[1,self.nn_config['labels_num'],1])

        f_Y_true = tf.tile(tf.expand_dims(f_Y,axis=2),multiples=[1,1,self.nn_config['labels_num']])
        f_Y_false = tf.tile(tf.expand_dims(f_Y,axis=1),multiples=[1,self.nn_config['labels_num'],1])

        # f_loss.shape = (batch size, lables number, labels number)
        f_loss = tf.multiply(tf.add(tf.subtract(tf.constant(1, dtype='float32'), f_Y_true), f_Y_false),mask)
        f_zero = tf.zeros_like(f_loss,dtype='float32')
        # shape = (batch size, lables number, labels number,1)
        loss = tf.reduce_sum(tf.reduce_max(tf.concat([tf.expand_dims(f_loss,axis=3),tf.expand_dims(f_zero,axis=3)],axis=3),axis=3),axis=2)

        # max = tf.concat([f_loss,])
        #
        # loss = tf.reduce_sum(tf.multiply(max,mask),axis=2)
        return loss

    def optimizer(self,loss,graph):
        train_step = tf.train.AdamOptimizer(self.nn_config['lr']).minimize(loss)
        graph.add_to_collection('train_step',train_step)

    def prediction(self,score,graph):
        """
        :param score: shape = (batch size, number of labels)
        :param graph: 
        :return: (number of batch size,)
        """
        # phi.shape = (batch size, labels number)

        # condition = tf.greater(score,tf.constant(self.nn_config['pred_threshold'],dtype='float32'))
        # pred = tf.where(condition,tf.ones_like(score,dtype='float32'),tf.zeros_like(score,dtype='float32'))
        # pred.shape = (batch size,)
        pred = tf.argmax(score,output_type='int32',axis=1)
        graph.add_to_collection('pred',pred)
        return pred

    def f1(self,Y_,pred):
        """
        
        :param pred: shape=(batch size,); its value is type id
        :param Y_: 
        :param graph: 
        :return: 
        """
        f1_macro = sklearn.metrics.f1_score(Y_, pred, labels=list(range(self.nn_config['labels_num'])), average='macro')
        f1_micro = sklearn.metrics.f1_score(Y_, pred, labels=list(range(self.nn_config['labels_num'])), average='micro')
        f1_per_class = sklearn.metrics.f1_score(Y_, pred, labels=list(range(self.nn_config['labels_num'])), average=None)
        return f1_macro, f1_micro, f1_per_class


    def accuracy(self,Y,Y_,graph):
        """
        
        :param Y: use (1,1,0,0,....)
        :param Y_: (number of batch size, number of labels)
        :param graph: 
        :return: 
        """
        b = tf.equal(Y, Y_)
        b = tf.where(b, tf.zeros_like(b, dtype='float32'), tf.ones_like(b, dtype='float32'))
        b = tf.reduce_sum(b, axis=1)
        b = tf.equal(b, tf.zeros_like(b, dtype='float32'))
        precision = tf.reduce_mean(tf.where(b, tf.ones_like(b, dtype='float32'), tf.zeros_like(b, dtype='float32')),
                                   name='precision')
        graph.add_to_collection('precision', precision)
        return precision


    def classifier(self):
        graph=tf.Graph()
        with graph.as_default():
            X = self.features_input(graph)
            mask = self.mask_matrix(X)
            X = self.lookup_table(X,mask,graph)
            # Y_.shape = (batch size, ); its value is type id of the name entity
            Y_id = self.labels_input(graph)
            Y_ = tf.one_hot(Y_id,depth=self.nn_config['labels_num'],axis=1,dtype='float32')
            A = self.A(graph)
            B = self.B(graph)
            # f_Y.shape =(batches num, number of labels)
            f_Y = self.score(X,A,B)
            # Lrank.shape=(batch size, number of labels)
            Lrank = self.Lrank(Y_,f_Y,graph)
            # loss.shape=(batch size, number of labels)
            Loss = self.loss(Y_,f_Y,graph)

            # condition = tf.equal(Y_, tf.constant(1, dtype='float32'))
            # # mask.shape=(batch size, number of labels)
            # # mask can directly be Y_ itself.
            # Mask = tf.where(condition, tf.ones_like(condition, dtype='float32'),
            #                 tf.zeros_like(condition, dtype='float32'))
            loss = tf.reduce_sum(Lrank*Loss*Y_,axis=1)
            graph.add_to_collection('test_loss', tf.reduce_mean(loss))
            batch_loss = tf.reduce_mean(loss)+tf.truediv(tf.reduce_sum(graph.get_collection('reg')[0]),
                                                                                        tf.constant(self.nn_config['batch_size'],dtype='float32'))
            self.optimizer(batch_loss,graph)
            pred = self.prediction(f_Y,graph)
            # precision = self.accuracy(Y=pred,Y_=Y_,graph=graph)
            saver=tf.train.Saver()
        return graph,saver

    def train(self):
        graph,saver = self.classifier()
        report=open(self.nn_config['report_path'],'a+')
        with graph.device(self.nn_config['gpu']):
            with graph.as_default():
                X = graph.get_collection('features_embeddings')[0]
                B = graph.get_collection('B')[0]
                Y = graph.get_collection('labels')[0]
                train_step = graph.get_collection('train_step')[0]
                # accuracy = graph.get_collection('precision')[0]
                loss = graph.get_collection('test_loss')[0]
                prediction = graph.get_collection('pred')[0]
                table = graph.get_collection('table')[0]
                ratio_table = graph.get_collection('ratio_table')[0]
                init = tf.global_variables_initializer()
            with tf.Session(graph=graph,config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                if self.nn_config['stage1'] == "True":
                    B_data = self.df.B_generator()
                    table_data = self.df.table_generator()
                    ratio_table_data = self.df.ratio_table_generator()
                    sess.run(init,feed_dict={B:B_data,table:table_data,ratio_table:ratio_table_data})
                    start = datetime.now()
                    for i in range(self.nn_config['epoch']):
                        X_data, Y_data = self.df.data_generator('train',batch_num=i,batch_size=self.nn_config['batch_size'])
                        sess.run(train_step,feed_dict={X:X_data,Y:Y_data,B:B_data})
                        if i%self.nn_config['mod'] ==0 and i !=0:
                            X_data, Y_data = self.df.data_generator('test')
                            pred_value,loss_value = sess.run([prediction,loss],feed_dict={X:X_data,Y:Y_data})
                            f1_macro,f1_micro= self.f1(Y_data,pred_value)
                            end = datetime.now()
                            time_cost = end - start
                            start = end
                            report.write('time_cost:{},loss:{}, f1_macro:{}, f1_micro:{}\n'.format(str(time_cost),str(loss_value),str(f1_macro), str(f1_micro)))
                            report.write('==================================\n')
                            report.flush()
                    saver.save(sess, self.nn_config['model'])
                if self.nn_config['stage1'] == "False":
                    saver.restore(sess, self.nn_config['model_sess'])
                    X_data,Y_data = self.df.data_generator('test')
                    pred_labels = sess.run(prediction,feed_dict={X:X_data,Y:Y_data})
                    true_labels = Y_data
                    report.close()
                    print('finish train')
                    return true_labels,pred_labels, X_data
