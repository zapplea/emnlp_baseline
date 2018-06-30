import argparse
import os
import sys
sys.path.append('/home/liu121/dlnlp')
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

par_dir = os.pardir
for i in range(3):
    par_dir = os.path.join(os.pardir, par_dir)

curr_path = os.path.abspath(__file__)
root_path = os.path.abspath(os.path.join(curr_path, par_dir))
sys.path.append(str(root_path))

from nerd.util.file_util import FileUtil


def get_precision(_tp, _fp):
    return _tp / (_tp + _fp) if _tp > 0 else 0

def get_recall(_tp, _fn):
    return _tp / (_tp + _fn) if _tp > 0 else 0

def get_f1(_p, _r):
    return 2*_p*_r / (_p + _r) if _p > 0 or _r > 0 else 0

def increment_dict(dictionary, key):
    if key not in dictionary:
        dictionary[key] = 1
    else:
        dictionary[key] = dictionary[key] + 1


def evaluate(_file_path, _delimeter='\t', other_label='O'):
    """
    Soft evaluation of a conlleval file by considering chunks of labels,
    If there is overlap between ground truth and predicted chunk it is considered a true positive
    else false negative for ground truth and false_positive for each negative chunk mispredictced in the ground truth
    expects file in the following format:
    <TOKEN><DELIMETER><GROUND_TRUTH_LABEL><DELIMETER><PREDICTED_LABEL>\n
    <TOKEN><DELIMETER><GROUND_TRUTH_LABEL><DELIMETER><PREDICTED_LABEL>\n
    <TOKEN><DELIMETER><GROUND_TRUTH_LABEL><DELIMETER><PREDICTED_LABEL>\n
    ...

    :param _file_path: Path to the conll_eval format file
    :param _delimeter: Tag that separates each token in a line
    :return:
    """

    # Read the file

    true_positive = dict()
    false_positive = dict()
    false_negative = dict()

    print('Evaluating file @ %s' % _file_path)
    if FileUtil.is_file(_file_path):
        with open(_file_path, 'r') as file:
            prev_pred_label = ''
            chunk_mispredicted_labels = []
            chunk_predicted_correctly = True
            chunk_label = ''
            other_labels = [label.strip() for label in other_label.split(',')]
            other_labels.append(chunk_label)

            all_labels_set = set()
            lines = file.readlines()
            lines.append('%s%s' % (_delimeter, _delimeter))
            for line in lines:

                tokens = [token.strip() for token in line.split(_delimeter)]

                if len(tokens) == 3:
                    word = tokens[0]
                    true_label = tokens[1].replace('B-', '').replace('I-', '')
                    pred_label = tokens[2].replace('B-', '').replace('I-', '')


                    # If previous chunk ended
                    if chunk_label != true_label:

                        # Increment scores if not an other chunk
                        if chunk_label not in other_labels:
                            if not chunk_predicted_correctly:
                                # If the chunk was not predicted correctly add as false negative
                                increment_dict(false_negative, chunk_label)

                                # All the mislabelled labels add as false positive
                                for mispredicted_label in chunk_mispredicted_labels:
                                    if mispredicted_label not in other_labels:
                                        increment_dict(false_positive, mispredicted_label)
                            else:
                                increment_dict(true_positive, chunk_label)

                        # Reset trackers
                        chunk_mispredicted_labels = []
                        chunk_predicted_correctly = False
                        chunk_label = true_label


                    # Add true and pred labels to all labels if not other label
                    if true_label in other_labels:

                        # We only need to increment the false_positive once for a chunk
                        if true_label != pred_label and pred_label != prev_pred_label:
                            increment_dict(false_positive, pred_label)

                    else:
                        # Maintain a list of all positive labels
                        all_labels_set.add(true_label)

                        # Check if chunk was predicted correctly
                        if true_label == pred_label:
                            chunk_predicted_correctly = True
                        else:
                            # Maintain a list of mispredicted labels
                            # Add to the list if the previous label was not the same, hence its a different subchunk
                            if len(chunk_mispredicted_labels) == 0 or not chunk_mispredicted_labels[-1] != pred_label:
                                chunk_mispredicted_labels.append(pred_label)

                    # Make note of the currently predicted label
                    prev_pred_label = pred_label
            print('tp: ',true_positive)
            print('fp: ',false_positive)
            print('fn: ',false_negative)
            exit()
            # Evaluation Metrics
            precision_per_class = dict()
            recall_per_class = dict()
            f1_per_class = dict()

            total_tp = 0
            total_fp = 0
            total_fn = 0

            for label in all_labels_set:
                if label in true_positive:
                    tp = true_positive[label]
                    fp = false_positive[label] if label in false_positive else 0
                    fn = false_negative[label] if label in false_negative else 0

                    # Increment overall counters
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

                    # Calculate per class metrics
                    precision =  get_precision(tp, fp)
                    recall = get_recall(tp, fn)
                    f1 = get_f1(precision, recall)

                    # Store per class metrics
                    precision_per_class[label] = precision
                    recall_per_class[label] = recall
                    f1_per_class[label] = f1


                else:
                    precision_per_class[label] = 0
                    recall_per_class[label] = 0
                    f1_per_class[label] = 0



            precision_micro = get_precision(total_tp, total_fp)
            recall_micro = get_recall(total_tp, total_fn)
            f1_micro = get_f1(precision_micro, recall_micro)

            f1_macro = np.mean([value for key, value in f1_per_class.items()])
            precision_macro = np.mean([value for key, value in precision_per_class.items()])
            recall_macro = np.mean([value for key, value in recall_per_class.items()])

            result = {
                "per_f1" : 'F1        (per class): %s' % f1_per_class,
                "per_pre" : 'Precision (per class): %s' % precision_per_class,
                "per_recall" : 'Recall    (per class): %s' % recall_per_class,
                "micro_f1" : 'F1            (micro): %s' % f1_micro,
                "micro_pre" : 'Precision     (micro): %s' % precision_micro,
                "micro_recall" : 'Recall        (micro): %s' % recall_micro,
                "macro_f1" : 'F1            (macro): %s' % f1_macro,
                "macro_pre" : 'Precision     (macro): %s' % precision_macro,
                "macro_recall" : 'Recall        (macro): %s' % recall_macro,
            }

    else:
        result = 'Could not find file @ %s' % _file_path
        print(result)
        exit()
    return result


# parser = argparse.ArgumentParser('Script to perform overlap evaluation on results file or folder')
# parser.add_argument('-f', '--results_file', default='/home/nikhil/testing_folder',
#                     help='path to the results file or folder, '
#                          '(Note: if folder ensure all files in folder are for evaluation.')
#
# parser.add_argument('-d', '--delimeter', default='\t', help='split delimeter between labels eg: \\t')
# parser.add_argument('-o', '--other_label', default='O', help='Labels to ignore as list eg : "O, PERSON, PERCENTAGE"')
# args = parser.parse_args()
#
#
#
# if __name__ == '__main__':
#
#     delimeter = args.delimeter
#     file_path = args.results_file
#     other_label = args.other_label
#
#     if FileUtil.is_file(file_path):
#         print('Using file %s' % file_path)
#         evaluate(file_path, delimeter)
#
#     elif FileUtil.is_folder(file_path):
#         print('Using folder %s' % file_path)
#         for file_name in FileUtil.get_files_in_folder(file_path):
#             evaluate(file_name, delimeter)
#
#     else:
#         print('Error: Not a file or folder ( %s )' % file_path)
if __name__ == "__main__":
    path='/datastore/liu121/nosqldb2/acl_hscrf/conll/conll_eval.txt'
    eval_result = evaluate(path)
    print('========\n')
    print(eval_result["per_f1"] + '\n')
    print(eval_result["per_pre"] + '\n')
    print(eval_result["per_recall"] + '\n')
    print(eval_result["micro_f1"] + '\n')
    print(eval_result["micro_pre"] + '\n')
    print(eval_result["micro_recall"] + '\n')
    print(eval_result["macro_f1"] + '\n')
    print(eval_result["macro_pre"] + '\n')
    print(eval_result["macro_recall"] + '\n')
    print('========\n')