#!/usr/bin/python2.7

import torch
from model import Trainer
import numpy as np
import os
import argparse
import random
import cv2

import seaborn as sns
import matplotlib.pyplot as plt


# task = Task.init(project_name='ProjectCV', task_name='PredictEvalVisualize', reuse_last_task_id=False)
#
# task.set_user_properties(
#   {"name": "backbone", "description": "network type", "value": "mstcn++"}
# )

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
parser.add_argument('--videos', type=str)
parser.add_argument('--create_videos', type=int)

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
videos = args.videos.split(", ")
experimental = args.experimental
learn_from_domain = False
perform_predict = args.predict
perform_eval = args.eval
perform_visualize = args.visualize
create_videos = args.create_videos

sample_rate = 1

# Directories
mapping_file = "/datashare/APAS/mapping_gestures.txt"
gt_path = '/datashare/APAS/transcriptions_gestures/'

video_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/videos'
gt_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/ground_truth'
features_path = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/features'
model_dir = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/models'
results_dir = 'C:/Users/dovid/PycharmProjects/MS-TCN2_surgical/on_computer/predictions'
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
        # TODO: ask Ilanit about this
        # model_dir = f"./models/test/valid0"
        # results_dir = f"./results/final_test/{dataset}{fold}" # note that changed from test to final test, in order to not overwrite anything




        trainer = Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes, "create_videos", "")

        # batch_gen_train = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
        # batch_gen_train.read_data(vid_list_file)
        # batch_gen_val = BatchGenerator(num_classes, actions_dict, gt_path, features_path_fold, sample_rate)
        # batch_gen_val.read_data(vid_list_file_val)


        # trainer.train(model_dir, batch_gen_train, batch_gen_val,  num_epochs=num_epochs, batch_size=bz, learning_rate=lr,
        #               device=device, weighting_method=weight_type)

        trainer.predict(model_dir, results_dir, features_path, videos, num_epochs, actions_dict, device,
                        sample_rate, weighting_method=weight_type, final_predict_mode=True, final_predict_epoch=15)

if perform_visualize:
    for video in videos:
        # file_name = 'P025_tissue2'
        # file_name = 'P024_balloon1'
        # create video reader object
        cap = cv2.VideoCapture(f'{video_path}/{video}.wmv')
        # create video writer object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(f'{output_videos}/{file_name}.avi', fourcc, 10.0, (640, 640), isColor=True)

        ground_truth_df_left = pd.read_csv(f'tool_usage/tools_left/{file_name}.txt', header=None, sep=' ',
                                           names=['start', 'end', 'label'])
        ground_truth_df_right = pd.read_csv(f'tool_usage/tools_right/{file_name}.txt', header=None, sep=' ',
                                            names=['start', 'end', 'label'])

        # create smoothing objects for experiments
        smoother25_nosmoothing = Smoothing(window_size=15, confidence_weighting_method='no_smooth',
                                           bbox_weighting_method=None)
        smoother25_log = Smoothing(window_size=25, confidence_weighting_method="log", bbox_weighting_method='linear')
        smoother25_linear = Smoothing(window_size=25, confidence_weighting_method="linear", bbox_weighting_method=None)
        smoother25_superlinear = Smoothing(window_size=25, confidence_weighting_method="super-linear",
                                           bbox_weighting_method=None)

        smoothing_experiments = [smoother25_nosmoothing, smoother25_log, smoother25_linear, smoother25_superlinear]
        # tools_path = 'C:\\Users\\dovid\\PycharmProjects\\CV_hw1_old\\HW1_dataset\\HW1_dataset\\tool_usage'
        evaluator = MetricEvaluator(ground_truth_df_left, ground_truth_df_right)

        experiments = [(se, MetricEvaluator(ground_truth_df_left, ground_truth_df_right)) for se in
                       smoothing_experiments]

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        ## load repo from original git yolov5
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

        model.conf = 0.6  ## allow preds over this threshold
        model.max_det = 2  ## predict max 2 classes

        #

        i = 0
        # Read until video is completed
        size = (640, 640)
        while (cap.isOpened()):
            print(f"frame {i}")
            i += 1
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                # prepare image for infer
                frame = cv2.resize(frame, size)
                frame_to_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ## infer
                output = model(frame_to_rgb)
                output.render()
                output_df = output.pandas().xyxy[0]

                # # uncomment if show video
                outputs_smooth = smoother25_log.smooth(curr_output=output_df)
                boxes1 = [output["bbox"] for output in outputs_smooth]
                labels = [output["prediction"] for output in outputs_smooth]

                # # save predictions in evaluator
                # [evaluator.convert_yolo_output_to_tool(label) for label in labels]
                # evaluator.calculate_all_metrics()

                # experiments
                for se, ee in experiments:
                    smoothed = se.smooth(curr_output=output_df)
                    smoothed_labels = [output["prediction"] for output in smoothed]
                    [ee.convert_yolo_output_to_tool(label) for label in smoothed_labels]
                    finished = ee.calculate_all_metrics()
                    if finished:
                        break
                    ee.history_to_pickle(experiments_dir + '/' + file_name + se.smoother_params)
                if finished:
                    break

                frame = bbv.draw_multiple_rectangles(frame, boxes1, bbox_color=(255, 0, 0))
                frame = bbv.add_multiple_labels(frame, labels, boxes1, text_bg_color=(255, 0, 0))

                ## Left
                real_label_left = extract_label(ground_truth_df_left, i)
                draw_text(frame, text=real_label_left, font_scale=1, pos=(500, 20), text_color_bg=(255, 0, 0),
                          draw='left')
                ## Right
                real_label_right = extract_label(ground_truth_df_right, i)
                draw_text(frame, text=real_label_right, font_scale=1, pos=(10, 20), text_color_bg=(255, 0, 0),
                          draw='right')

                # record output in video
                out.write(frame)
                # Display the resulting frame
                # cv2.imshow('Frame', frame)
                # last_frame = frame.copy()

                # evaluator.history_to_pickle("metric_evaluation_test")

                # Press Q on keyboard to  exit
            # if cv2.waitKey(33) & 0xFF == ord('q'):
            #     break

            # Break the loop
            else:
                break
        # When everything done, release the video capture object
        cap.release()
        out.release()

    # Closes all the frames
    cv2.destroyAllWindows()
