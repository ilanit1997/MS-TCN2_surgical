#!/usr/bin/python2.7

import torch
from model import Trainer
import pandas as pd
import numpy as np
import os
import argparse
import random
import cv2
import glob
from batch_gen import convert_file_to_list
from eval import read_file

import seaborn as sns
import matplotlib.pyplot as plt


# task = Task.init(project_name='ProjectCV', task_name='PredictEvalVisualize', reuse_last_task_id=False)
#
# task.set_user_properties(
#   {"name": "backbone", "description": "network type", "value": "mstcn++"}
# )

def weight_parameters(w_type):
    w = None
    experimental = 1
    learn_from_domain = False
    if w_type == "framewise":  # give more weight to lower resolution dilations
        w = torch.tensor(np.array(range(num_layers_R, 0, -1)) / sum(range(num_layers_R, 0, -1)), dtype=torch.float32,
                         device=device)
    elif w_type == "smooth":  # give more weight to higher resolution dilations
        w = torch.tensor(np.array(range(1, num_layers_R + 1)) / sum(range(1, num_layers_R + 1)), dtype=torch.float32,
                         device=device)
    elif w_type == "none":
        w = None
        experimental = 0
    elif w_type == "learned":
        w = num_layers_R  # here you can control the initialization of the learned parameters if you modify the Weighting class properly
    elif w_type == "uniform":  # weight equally all dilations
        w = torch.tensor(np.ones(num_layers_R) / num_layers_R, dtype=torch.float32, device=device)
    elif w_type == "learned_smooth":  # weight equally all dilations
        w = torch.tensor(np.array(range(1, num_layers_R + 1)) / sum(range(1, num_layers_R + 1)), dtype=torch.float32,
                         device=device)
        learn_from_domain = True
    elif w_type == "learned_framewise":  # weight equally all dilations
        w = torch.tensor(np.array(range(num_layers_R, 0, -1)) / sum(range(num_layers_R, 0, -1)), dtype=torch.float32,
                         device=device)
        learn_from_domain = True
    elif w_type == "learned_uniform":  # weight equally all dilations
        w = torch.tensor(np.ones(num_layers_R) / num_layers_R, dtype=torch.float32, device=device)
        learn_from_domain = True
    elif w_type == "learned_framewise_exp":  # weight equally all dilations
        arr = np.array(range(num_layers_R, 0, -1))
        arr_exps = np.exp(arr)
        arr_exps = arr_exps / sum(arr_exps)
        w = torch.tensor(arr_exps, dtype=torch.float32, device=device)
        learn_from_domain = True

    elif weight_type == "learned_smooth_exp":  # weight equally all dilations
        arr = np.array(range(1, num_layers_R + 1))
        arr_exps = np.exp(arr)
        arr_exps = arr_exps / sum(arr_exps)
        w = torch.tensor(arr_exps, dtype=torch.float32, device=device)
        learn_from_domain = True
    return w, experimental, learn_from_domain


def prepare_segments(per_frame_labels, mapping_dict=None):
    # print(per_frame_labels)
    label_list = list()
    segment_start_list = list()
    segment_end_list = list()

    # initialize lists
    if mapping_dict is not None:
        label_list.append(mapping_dict[per_frame_labels[0]])
    else:
        label_list.append(per_frame_labels[0])
    previous_label = per_frame_labels[0]
    segment_start_list.append(0)

    for frame_index, label in enumerate(per_frame_labels[1:]):
        if label == previous_label:
            continue
        else:
            if mapping_dict is not None:
                label_list.append(mapping_dict[label])
            else:
                label_list.append(label)
            previous_label = label
            segment_end_list.append(frame_index+1)
            segment_start_list.append(frame_index+1)
    segment_end_list.append(frame_index + 2)
    # print(label_list)
    # print(segment_start_list)
    # print(segment_end_list)
    return label_list, segment_start_list, segment_end_list


def plot_segments(predictions, ground_truth, mapping_dict, current_time=0, graph_path=None):
    colors = ['red', 'yellow', 'blue', 'green', 'black', 'pink']
    fig, ax = plt.subplots()
    fig.set_figheight(1)
    # plot ground truth
    gt_labels, gt_seg_starts, gt_seg_ends = prepare_segments(ground_truth, mapping_dict)
    gt_df = pd.DataFrame({"Label": gt_labels, "Start": gt_seg_starts, "End": gt_seg_ends})

    for label_index in range(len(gt_labels)):
        ax.plot([gt_df['Start'][label_index], gt_df['End'][label_index]], [0, 0], color=colors[gt_df['Label'][label_index]], linewidth = 8)

    offset = -1
    diff = -1
    if isinstance(predictions[0], list):
        for i, predictions_type in enumerate(predictions):
            pred_labels, pred_seg_starts, pred_seg_ends = prepare_segments(predictions_type, mapping_dict)
            pred_df = pd.DataFrame({"Label": pred_labels, "Start": pred_seg_starts, "End": pred_seg_ends})
            for label_index in range(len(pred_labels)):
                ax.plot([pred_df['Start'][label_index], pred_df['End'][label_index]], [offset + diff * i, offset + diff * i], color=colors[pred_df['Label'][label_index]], linewidth =8)
            ax.set_ylim(offset + len(predictions) * diff, 1)
    else:
        pred_labels, pred_seg_starts, pred_seg_ends = prepare_segments(predictions, mapping_dict)
        pred_df = pd.DataFrame({"Label": pred_labels, "Start": pred_seg_starts, "End": pred_seg_ends})
        for label_index in range(len(pred_labels)):
            ax.plot([pred_df['Start'][label_index], pred_df['End'][label_index]], [offset, offset], color=colors[pred_df['Label'][label_index]], linewidth = 8)
        ax.set_ylim(-2, 1)

    ax.plot(current_time, 1, 'bo')

    # ax.set_yticklabels(["", 'pred1', "gt", "time"])
    ax.set_xticklabels([])
    ax.set_xticks([])
    # ax.legend(["G0", "G1", "G2", "G3", "G4", "G5"])

    # plt.show()
    plt.savefig(graph_path)
    plt.close()


def create_video_graph(video, gt_path, prediction_path, graph_path, weight_types, timestep=0, mapping_dict=None):
    # get gt and predictions
    gt_content = convert_file_to_list(gt_path + f"/{video}.txt")
    predictions = list()
    for weight_type in weight_types:
        predictions.append(read_file(prediction_path + f"/final_predict_{weight_type}_{video}.txt").split('\n')[1].split())

    plot_segments(predictions, gt_content, mapping_dict=mapping_dict, current_time=timestep, graph_path=graph_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', default="test")
# parser.add_argument('--fold', default='1')
#
# parser.add_argument('--features_dim', default='1280', type=int)
# parser.add_argument('--bz', default='1', type=int)
# parser.add_argument('--lr', default='0.0005', type=float)
#
#
# parser.add_argument('--num_f_maps', default='64', type=int)
#
# # Need input
# parser.add_argument('--num_epochs', type=int)
# parser.add_argument('--num_layers_PG', type=int)
# parser.add_argument('--num_layers_R', type=int)
# parser.add_argument('--num_R', type=int)
# parser.add_argument('--weight_types', type=str)
# parser.add_argument('--experimental', type=int)
# parser.add_argument('--predict', type=int)
# parser.add_argument('--eval', type=int)
# parser.add_argument('--visualize', type=int)
# parser.add_argument('--videos', type=str)
# parser.add_argument('--create_videos', type=int)
#
# args = parser.parse_args()
#
# dataset = args.dataset
# folds = args.fold.split(",")
# # print(folds)
# num_epochs = args.num_epochs
# features_dim = args.features_dim
# bz = args.bz
# lr = args.lr
#
# num_layers_PG = args.num_layers_PG
# num_layers_R = args.num_layers_R
# num_R = args.num_R
# num_f_maps = args.num_f_maps
# weight_types = args.weight_types.split(", ")
# videos = args.videos.split(", ")
# experimental = args.experimental
# learn_from_domain = False
# perform_predict = args.predict
# perform_eval = args.eval
# perform_visualize = args.visualize
# create_videos = args.create_videos

#TODO: REMOVIE when running on remote
videos = "P040_tissue2".split(", ")
num_epochs = 15
num_layers_PG = 11
num_layers_R = 10
num_R = 3
num_f_maps = 64
lr = '0.0005'
bz = '1'
features_dim = 1280
weight_types = "none".split(', ')
predict = 1
visualize = 1
create_videos = 1
perform_predict = 0

sample_rate = 1

# Directories
mapping_file = "/datashare/APAS/mapping_gestures.txt"
gt_path = '/datashare/APAS/transcriptions_gestures/'

# mapping_file = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/mapping_gestures.txt'
# video_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/videos'
# gt_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/ground_truth'
# features_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/features'
# model_dir = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/models'
# results_dir = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/predictions'
# output_video_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/output_video'
# graph_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/temp/output.jpg'

mapping_file = 'on_computer/mapping_gestures.txt'
video_path = 'on_computer/videos'
gt_path = 'on_computer/ground_truth'
features_path = 'on_computer/features'
model_dir = 'on_computer/models'
results_dir = 'on_computer/predictions'
output_video_path = 'on_computer/output_video'
graph_path = 'on_computer/temp/output.jpg'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
num_classes = len(actions_dict)

for weight_type in weight_types:
    if perform_predict:
        w, experimental, learn_from_domain = weight_parameters(weight_type)
        trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, "create_videos", "",
                          refinement_weighting=w, experimental=experimental, fold=7,
                          learn_from_domain=learn_from_domain)
        predict_videos = ["/" + vid + ".npy" for vid in videos]
        trainer.final_predict(model_dir, results_dir, features_path, predict_videos, num_epochs, actions_dict, device,
                        sample_rate, weighting_method=weight_type)

if create_videos:
    if not os.path.exists(output_video_path):
        os.makedirs(output_video_path)
    for video in videos:
        img_array = []
        for timestep, filename in enumerate(glob.glob(f'{video_path}/{video}_side/*.jpg')):
            # create graph for time step
            create_video_graph(video, gt_path, results_dir, graph_path, weight_types, timestep=timestep,
                               mapping_dict=actions_dict)
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)

            graph = cv2.imread(graph_path)
            graph = cv2.resize(graph, (width, graph.shape[0]))

            legend = cv2.resize(cv2.imread('on_computer/temp/legend.jpg'), (width/5, graph.shape[0]))

            graph = cv2.resize(cv2.hconcat([graph, legend]), (width, graph.shape[0]))

            im_with_plt = cv2.resize(cv2.vconcat([img, graph]), (width, height))
            img_array.append(im_with_plt)
            if timestep == 100:
                break
        print(f'{output_video_path}/{video}.avi')
        out = cv2.VideoWriter(f'{output_video_path}/{video}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
