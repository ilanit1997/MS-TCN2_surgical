#!/usr/bin/python2.7

import torch
import numpy as np
import random
import pandas as pd

class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        self.list_of_examples = vid_list_file
        random.shuffle(self.list_of_examples)


    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            # print(vid)
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            # file_ptr = open(self.gt_path + vid[:-4] + ".txt", 'r')
            # content = file_ptr.read().split('\n')[:-1]
            content = convert_file_to_list(self.gt_path + vid[:-4] + ".txt", self.actions_dict)
            classes = np.zeros(min(np.shape(features)[1], len(content)))

            for i in range(len(classes)):
                classes[i] = content[i]

            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))

        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)

        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask

def convert_file_to_list(path, mapping_dict=None):
    """
    Explodes concise tool usage file to a list of ground truths for each file (arm) passed to it
    :param df:
    :return:
    """
    df = pd.read_csv(path, header=None, sep=' ', names=['start', 'end', 'label'])

    if mapping_dict == None:
        ground_truth = list()
        for index, row in df.iterrows():
            ground_truth.extend([str(row[2])] * (row[1] - row[0]))

    else:
        ground_truth = np.zeros(df.iloc[-1, 1])  # last row end time, maximum time

        for index, row in df.iterrows():
            ground_truth[row[0]:row[1]+1] = mapping_dict[row[2]]
    return ground_truth
