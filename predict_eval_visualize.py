#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import numpy as np
import os
import argparse
import random
from clearml import Task
from eval import eval
from os.path import isfile, join
from os import listdir
import seaborn as sns
import matplotlib.pyplot as plt


task = Task.init(project_name='ProjectCV', task_name='PredictEvalVisualize', reuse_last_task_id=False)

task.set_user_properties(
  {"name": "backbone", "description": "network type", "value": "mstcn++"}
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="test")
parser.add_argument('--fold', default='1')

parser.add_argument('--features_dim', default='1280', type=int)
parser.add_argument('--bz', default='1', type=int)
parser.add_argument('--lr', default='0.0005', type=float)


parser.add_argument('--num_f_maps', default='64', type=int)

# Need input
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--num_layers_PG', type=int)
parser.add_argument('--num_layers_R', type=int)
parser.add_argument('--num_R', type=int)
parser.add_argument('--weight_types', type=str)
parser.add_argument('--experimental', type=int)
parser.add_argument('--predict', type=int)
parser.add_argument('--eval', type=int)
parser.add_argument('--visualize', type=int)

args = parser.parse_args()

dataset = args.dataset
folds = args.fold.split(",")
# print(folds)
num_epochs = args.num_epochs
features_dim = args.features_dim
bz = args.bz
lr = args.lr

num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_R = args.num_R
num_f_maps = args.num_f_maps
weight_types = args.weight_types.split(", ")
experimental = args.experimental
learn_from_domain = False
perform_predict = args.predict
perform_eval = args.eval
perform_visualize = args.visualize

sample_rate = 1

mapping_file = "/datashare/APAS/mapping_gestures.txt"
gt_path = '/datashare/APAS/transcriptions_gestures/'
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])
num_classes = len(actions_dict)



fold_files = [(f"/datashare/APAS/folds/valid {i}.txt",
               f"/datashare/APAS/folds/test {i}.txt",
               f"/datashare/APAS/features/fold{i}/") for i in folds]

def fold_split(features_path, val_path, test_path, foldi):
    with open(val_path, 'r') as f:
        val_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
    with open(test_path, 'r') as f:
        test_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]

    other_test_files = []
    for i in range(5):
        if i!= foldi:
            curr_path = f"/datashare/APAS/folds/test {i}.txt"
            with open(curr_path, 'r') as f:
                curr_files = [vid.split('.')[0] + '.npy' for vid in f.readlines()]
                other_test_files.extend(curr_files)
    # train_files = [f for f in listdir(features_path) if isfile(join(features_path, f))]
    train_files = list(set(other_test_files) - set(test_files + val_files))
    print('train files:')
    print(train_files)
    print('val files:')
    print(val_files)
    print('test files:')
    print(test_files)
    return train_files, val_files, test_files

kpi_list = list()
for weight_type in weight_types:
    test_files = []
    if perform_predict:
        i = 0
        for val_path_fold, test_path_fold, features_path_fold in fold_files:
            print(f"{dataset}  : {i}")

            vid_list_file, vid_list_file_val, vid_list_file_test = fold_split(features_path_fold, val_path_fold,
                                                                              test_path_fold, foldi=i)
            fold = features_path_fold.split("/")[-2]
            i+=1
            model_dir = f"./models/test/{dataset}{fold}"
            results_dir = f"./results/final_test/{dataset}{fold}" # note that changed from test to final test, in order to not overwrite anything

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)


            trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, f"fold{fold}", f"fold{fold}",
                              refinement_weighting=w, experimental=experimental, fold=fold, learn_from_domain=learn_from_domain)

            batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold,
                                             sample_rate)
            batch_gen_train.read_data(vid_list_file)
            batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold, sample_rate)
            batch_gen_val.read_data(vid_list_file_val)


            # trainer.train(model_dir, batch_gen_train, batch_gen_val,  num_epochs=num_epochs, batch_size=bz, learning_rate=lr,
            #               device=device, weighting_method=weight_type)


            trainer.predict(model_dir, results_dir, features_path_fold, vid_list_file_test, num_epochs, actions_dict, device,
                            sample_rate, weighting_method=weight_type, final_predict_mode=True, final_predict_epoch=15)
            test_files.append(vid_list_file_test)
    else:
        for i, (val_path_fold, test_path_fold, features_path_fold) in enumerate(fold_files):
            vid_list_file, vid_list_file_val, vid_list_file_test = fold_split(features_path_fold, val_path_fold,
                                                                          test_path_fold, foldi=i)
            test_files.append(vid_list_file_test)

    if perform_eval:
        kpi_dict = eval(dataset, folds, test_files, weight_type, final_eval=True)
        kpi_list.append(kpi_dict)

if perform_visualize:
    visualization_dict = {"weight_type": [], "kpi": [], "result": []}
    for i, weight in enumerate(weight_types):
        for key in kpi_list[0].keys():
            visualization_dict["weight_type"].append(weight)
            visualization_dict["kpi"].append(key)
            visualization_dict["result"].append(kpi_list[i][key])
    sns.lineplot(data=visualization_dict, x="weight_type", y="result", hue="weight_type")
    plt.show()
