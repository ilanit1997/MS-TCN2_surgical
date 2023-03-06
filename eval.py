#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import numpy as np
import argparse
from batch_gen import convert_file_to_list
# from clearml import Task, Logger


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)

    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)



def eval(dataset, folds, test_files, weight_type, final_eval=0):
    print('#####################')
    print('Starting evaluation')
    evaluation_metrics = dict()
    evaluation_metrics['acc'] = list()
    evaluation_metrics['edit'] = list()
    overlap = [.1, .25, .5]
    for s in overlap:
        evaluation_metrics[f"F1@{s}"] = list()

    ClearMLlogger = Logger.current_logger()

    for fold in folds:
        print(f'Fold {fold}')
        ground_truth_path = '/datashare/APAS/transcriptions_gestures/'
        recog_path = f"./results/test/{dataset}fold{fold}/" + weight_type
        list_of_videos = test_files[int(fold)]

        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0

        for vid in list_of_videos:
            gt_file = ground_truth_path + vid[:-4] + ".txt"

            # gt_content = read_file(gt_file).split('\n')[0:-1]
            gt_content = convert_file_to_list(gt_file)
            recog_file = recog_path + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

            # print(len(gt_content))
            # print(gt_content[:5])
            # print(len(recog_content))
            # print(recog_content[:5])
            for i in range(min(len(gt_content), len(recog_content))):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
        acc = (100 * float(correct) / total)
        edit = ((1.0 * edit) / len(list_of_videos))

        print("Acc: %.4f" % (acc))
        evaluation_metrics['acc'].append(acc)
        ClearMLlogger.report_scalar(title="AccuracyPerFold", iteration=0, series=f"fold{fold}", value=acc)

        print('Edit: %.4f' % (edit))
        evaluation_metrics['edit'].append(edit)
        ClearMLlogger.report_scalar(title="EditPerFold", iteration=0, series=f"fold{fold}", value=edit)

        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s] + fp[s])
            recall = tp[s] / float(tp[s] + fn[s])

            f1 = 2.0 * (precision * recall) / (precision + recall)

            f1 = np.nan_to_num(f1) * 100

            print('F1@%0.2f: %.4f' % (overlap[s], f1))
            evaluation_metrics[f'F1@{overlap[s]}'].append(f1)
            ClearMLlogger.report_scalar(title=f"F1@{overlap[s]}PerFold", iteration=0, series=f"fold{fold}", value=f1)

        print()

    avg_acc_folds = sum(evaluation_metrics['acc']) / len(evaluation_metrics['acc'])
    ClearMLlogger.report_scalar(title="AverageFolds", series="Accuracy", iteration=0, value=avg_acc_folds)
    print("Average acccuracy on folds: %.4f" % (avg_acc_folds))

    avg_edit_folds = sum(evaluation_metrics['edit']) / len(evaluation_metrics['edit'])
    ClearMLlogger.report_scalar(title="AverageFolds", series="Edit", iteration=0, value=avg_edit_folds)
    print('Average edit distance on folds: %.4f' % avg_edit_folds)

    avg_f1_list = list()
    for s in range(len(overlap)):
        avg_f1_folds = sum(evaluation_metrics[f"F1@{overlap[s]}"]) / len(evaluation_metrics[f"F1@{overlap[s]}"])
        avg_f1_list.append(avg_f1_folds)
        ClearMLlogger.report_scalar(title="AverageFolds", series=f"F1@{overlap[s]}",iteration=0,  value=avg_f1_folds)
        print('Average F1@%0.2f on folds: %.4f' % (overlap[s], avg_f1_folds))

    if final_eval:
        return {"Avg. Accuracy": avg_acc_folds, "Avg. Edit Distance": avg_edit_folds}.update({"Avg. f1 " + str(ol): avg_f1 for ol, avg_f1 in zip(overlap, avg_f1_list)})




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="valid")
    parser.add_argument('--fold', default='1')
    parser.add_argument('--weight_type', type=str)

    args = parser.parse_args()
    dataset = args.dataset
    folds = args.fold.split(",")
    weight_type = args.weight_type

    evaluation_metrics = dict()
    evaluation_metrics['acc'] = list()
    evaluation_metrics['edit'] = list()
    overlap = [.1, .25, .5]
    for s in overlap:
        evaluation_metrics[f"F1@{s}"] = list()

    ClearMLlogger = Logger.current_logger()

    for fold in folds:
        print(f'Fold {fold}')
        ground_truth_path = '/datashare/APAS/transcriptions_gestures/'
        recog_path = f"./results/test/{dataset}{fold}/" + weight_type
        file_list = f"/datashare/APAS/folds/{dataset} {fold}.txt"

        list_of_videos = read_file(file_list).split('\n')[:-1]

        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0

        for vid in list_of_videos:
            gt_file = ground_truth_path + vid[:-4] + ".txt"

            # gt_content = read_file(gt_file).split('\n')[0:-1]
            gt_content = convert_file_to_list(gt_file)
            recog_file = recog_path + vid.split('.')[0]
            recog_content = read_file(recog_file).split('\n')[1].split()

            # print(len(gt_content))
            # print(gt_content[:5])
            # print(len(recog_content))
            # print(recog_content[:5])
            for i in range(min(len(gt_content), len(recog_content))):
                total += 1
                if gt_content[i] == recog_content[i]:
                    correct += 1

            edit += edit_score(recog_content, gt_content)

            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
                tp[s] += tp1
                fp[s] += fp1
                fn[s] += fn1
        acc = (100 * float(correct) / total)
        edit = ((1.0 * edit) / len(list_of_videos))

        print("Acc: %.4f" % (acc))
        evaluation_metrics['acc'].append(acc)
        ClearMLlogger.report_scalar(title="Accuracy", series=f"fold{fold}", iteration=0, value=acc)

        print('Edit: %.4f' % (edit))
        evaluation_metrics['edit'].append(edit)
        ClearMLlogger.report_scalar(title="Edit", series=f"fold{fold}", iteration=0,value=edit)

        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])

            f1 = 2.0 * (precision*recall) / (precision+recall)

            f1 = np.nan_to_num(f1)*100

            print('F1@%0.2f: %.4f' % (overlap[s], f1))
            evaluation_metrics[f'F1@{overlap[s]}'].append(f1)
            ClearMLlogger.report_scalar(title=f"F1@{overlap[s]}", series=f"fold{fold}", iteration=0,value=f1)

        print()

    avg_acc_folds = sum(evaluation_metrics['acc'])/len(evaluation_metrics['acc'])
    ClearMLlogger.report_scalar(title=f"AvgAccuracy", series=f"all_folds",iteration=0, value=avg_acc_folds)
    print("Average acccuracy on folds: %.4f" % (avg_acc_folds))

    avg_edit_folds = sum(evaluation_metrics['edit'])/len(evaluation_metrics['edit'])
    ClearMLlogger.report_scalar(title=f"AvgEdit", series=f"all_folds",iteration=0, value=avg_edit_folds)
    print('Average edit distance on folds: %.4f' % avg_edit_folds)

    for s in range(len(overlap)):
        avg_f1_folds = sum(evaluation_metrics[f"F1@{overlap[s]}"])/len(evaluation_metrics[f"F1@{overlap[s]}"])
        ClearMLlogger.report_scalar(title=f"AvgF1@{overlap[s]}", series=f"all_folds",iteration=0, value=avg_f1_folds)
        print('Average F1@%0.2f on folds: %.4f' % (overlap[s], avg_f1_folds))

# if __name__ == '__main__':
#     main()
