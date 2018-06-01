import argparse
import os
import sys

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

# par_dir = os.pardir
# for i in range(3):
#     par_dir = os.path.join(os.pardir, par_dir)
#
# curr_path = os.path.abspath(__file__)
# root_path = os.path.abspath(os.path.join(curr_path, par_dir))
# sys.path.append(str(root_path))
sys.path.append('/home/liu121/dlnlp')
from nerd.util.file_util import FileUtil

parser = argparse.ArgumentParser('Script to perform overlap evaluation on results file')
parser.add_argument('-f', '--results_file',
                    default='/home/nikhil/workspace/data/senner_joint/BBN/16_shot/results/run_170318_1124/conlleval_epoch_1_config_4',
                    help='path to the results file, results should be in conlleval format.')
parser.add_argument('-d', '--delimeter', default='\t', help='split delimeter')
parser.add_argument('-o', '--other_label', default='OTHER', help='Tag used to mark other')
args = parser.parse_args()

if __name__ == '__main__':

    delimeter = args.delimeter
    file_path = args.results_file
    other_label = args.other_label

    # Read the file
    if FileUtil.is_file(file_path):
        with open(file_path, 'r') as file:
            prev_true_label = ''
            prev_pred_label = ''
            prev_overlapped = False

            true_chunks = []
            pred_chunks = []

            all_labels_set = set()

            for line in file:
                if len(line.strip()) > 0:
                    tokens = [token.strip() for token in line.split(delimeter)]
                    word = tokens[0]
                    true_label = tokens[1].replace('B-', '').replace('I-', '')
                    pred_label = tokens[2].replace('B-', '').replace('I-', '')

                    # Add true and pred labels to all labels if not other label
                    if true_label != other_label:
                        all_labels_set.add(true_label)
                    if pred_label != other_label:
                        all_labels_set.add(pred_label)

                    # Process chunk
                    if true_label == prev_true_label:
                        # In a chunk
                        if pred_label == true_label:
                            prev_overlapped = True
                    else:
                        # End of chunk
                        if prev_true_label != '':
                            # If not initial label and prev chunk overlapped
                            if prev_overlapped:
                                true_chunks.append(prev_true_label)
                                pred_chunks.append(prev_true_label)
                            else:
                                true_chunks.append(prev_true_label)
                                pred_chunks.append(prev_pred_label)

                        # Initialise labels for next chunk
                        if true_label == other_label:
                            prev_true_label = ''
                            prev_overlapped = False
                            prev_pred_label = ''
                        else:
                            prev_true_label = true_label
                            prev_overlapped = (pred_label == true_label)
                            prev_pred_label = pred_label

            all_labels_list = list(all_labels_set)
            all_labels_list.sort()

            f1 = f1_score(true_chunks, pred_chunks, labels=all_labels_list, average=None)
            precision = precision_score(true_chunks, pred_chunks, labels=all_labels_list, average=None)
            recall = recall_score(true_chunks, pred_chunks, labels=all_labels_list, average=None)

            f1_micro = f1_score(true_chunks, pred_chunks, labels=all_labels_list, average='micro')
            precision_micro = precision_score(true_chunks, pred_chunks, labels=all_labels_list, average='micro')
            recall_micro = recall_score(true_chunks, pred_chunks, labels=all_labels_list, average='micro')

            f1_macro = f1_score(true_chunks, pred_chunks, labels=all_labels_list, average='macro')
            precision_macro = precision_score(true_chunks, pred_chunks, labels=all_labels_list, average='macro')
            recall_macro = recall_score(true_chunks, pred_chunks, labels=all_labels_list, average='macro')
            confusion_matrix = confusion_matrix(true_chunks, pred_chunks, labels=all_labels_list)

            print('##############################')
            print('Confusion Matrix     : \n%s' % confusion_matrix)
            print('Labels               : %s' % all_labels_list)
            print('F1        (per class): %s' % f1)
            print('Precision (per class): %s' % precision)
            print('Recall    (per class): %s' % recall)
            print('F1            (micro): %s' % f1_micro)
            print('Precision     (micro): %s' % precision_micro)
            print('Recall        (micro): %s' % recall_micro)
            print('F1            (macro): %s' % f1_macro)
            print('Precision     (macro): %s' % precision_macro)
            print('Recall        (macro): %s' % recall_macro)
            print('##############################')
    else:
        print('Could not find file @ %s' % file_path)