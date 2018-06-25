import numpy as np
import sklearn
import pickle

class Metrics:
    def __init__(self,conll_filePath,dictionary):
        self.conll_filePath = conll_filePath
        self.dictionary = dictionary
        self.id2word = {v: k for k, v in dictionary.items()}

    def word_id2txt(self,X_data, true_labels, pred_labels, id2label):
        I = []
        for i in range(len(X_data)):
            instance = X_data[i]
            txt = []
            true_length = 0
            for id in instance:
                if id != 1:
                    word = self.id2word[id]
                    txt.append(word)
                    true_length += 1

            tlabel = true_labels[i]
            tlabel_txt = []
            cur_length = 0
            for id in tlabel:
                if id not in id2label:
                    print('true_id: ',str(tlabel))
                    print('label: ',id2label)
                type_txt = id2label[id]
                type_txt = type_txt.replace('I-', '')
                type_txt = type_txt.replace('B-','')
                tlabel_txt.append(type_txt)
                cur_length += 1
                if cur_length >= true_length:
                    break

            plabel = pred_labels[i]
            plabel_txt = []
            cur_length = 0
            for id in plabel:
                if id not in id2label:
                    print('pred_id: ',str(plabel))
                    print('label: ',id2label)
                type_txt = id2label[id]
                type_txt = type_txt.replace('I-', '')
                plabel_txt.append(type_txt)
                cur_length += 1
                if cur_length >= true_length:
                    break

            I.append((plabel_txt, tlabel_txt, txt))
        return I

    def conll_eval_file(self, I):
        with open(self.data_config['conlleval_filePath'], 'w+') as f:
            for t in I:
                pred_labels_txt = t[0]
                true_labels_txt = t[1]
                txt = t[2]
                for i in range(len(txt)):
                    f.write(txt[i] + '\t')
                    f.write(true_labels_txt[i] + '\t')
                    f.write(pred_labels_txt[i] + '\n')
                    f.flush()
                f.write('\n')
                f.flush()

    @staticmethod
    def measure(true_labels, pred_labels, id2label_dic):
        true_labels = np.reshape(true_labels, newshape=(-1,)).astype('float32')
        pred_labels = np.reshape(pred_labels, newshape=(-1,)).astype('float32')

        # delete other
        ls_pred = []
        ls_true = []
        for i in range(pred_labels.shape[0]):
            if true_labels[i] != 0 and true_labels != 1:
                ls_pred.append(pred_labels[i])
                ls_true.append(true_labels[i])
        pred_labels = np.array(ls_pred, dtype='float32')
        true_labels = np.array(ls_true, dtype='float32')
        id2label_dic.pop(0)
        all_labels_list = list(range(1, len(id2label_dic)))
        # all_labels_list = list(range(len(id2label_dic)))

        f1 = sklearn.metrics.f1_score(true_labels, pred_labels, labels=all_labels_list, average=None)
        f1_dic = {}
        for i in range(len(f1)):
            label = id2label_dic[i]  # when O is deleted, it should be id2label[i+1]
            f1_dic[label] = f1[i]
        precision = sklearn.metrics.precision_score(true_labels, pred_labels, labels=all_labels_list,
                                                    average=None)
        pre_dic = {}
        for i in range(len(precision)):
            label = id2label_dic[i]  # when O is deleted, it should be id2label[i+1]
            pre_dic[label] = precision[i]

        recall = sklearn.metrics.recall_score(true_labels, pred_labels, labels=all_labels_list, average=None)
        recall_dic = {}
        for i in range(len(recall)):
            label = id2label_dic[i]  # when O is deleted, it should be id2label[i+1]
            recall_dic[label] = recall[i]

        f1_micro = sklearn.metrics.f1_score(true_labels, pred_labels, labels=all_labels_list, average='micro')
        precision_micro = sklearn.metrics.precision_score(true_labels, pred_labels, labels=all_labels_list,
                                                          average='micro')
        recall_micro = sklearn.metrics.recall_score(true_labels, pred_labels, labels=all_labels_list,
                                                    average='micro')
        f1_macro = sklearn.metrics.f1_score(true_labels, pred_labels, labels=all_labels_list, average='macro')
        precision_macro = sklearn.metrics.precision_score(true_labels, pred_labels, labels=all_labels_list,
                                                          average='macro')
        recall_macro = sklearn.metrics.recall_score(true_labels, pred_labels, labels=all_labels_list,
                                                    average='macro')
        confusion_matrix = sklearn.metrics.confusion_matrix(true_labels, pred_labels, labels=all_labels_list)

        # dictionary of measure
        metrics_dic = {}
        metrics_dic['f1_macro'] = f1_macro
        metrics_dic['f1_micro'] = f1_micro
        metrics_dic['precision_macro'] = precision_macro
        metrics_dic['precision_micro'] = precision_micro
        metrics_dic['recall_macro'] = recall_macro
        metrics_dic['recall_micro'] = recall_micro
        metrics_dic['confusion_matrix'] = confusion_matrix
        metrics_dic['f1<per-type>'] = f1_dic
        metrics_dic['recall<per-type>'] = recall_dic
        metrics_dic['precision<per-type>'] = pre_dic
        return metrics_dic