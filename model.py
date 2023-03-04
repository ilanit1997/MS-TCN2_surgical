#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from clearml import Task, Logger


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,
                 refinement_weighting=None, experimental=1, learn_from_domain=False):
        super(MS_TCN2, self).__init__()

        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        if experimental:
            self.refinement_weighting = refinement_weighting
            self.Rs = nn.ModuleList([copy.deepcopy(
                TradeoffRefinement(num_layers_R, num_f_maps, num_classes, num_classes,
                                   weighting=self.refinement_weighting, learn_from_domain=learn_from_domain))
                for s in
                range(num_R)])
        else:
            self.Rs = nn.ModuleList(
                [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class TradeoffRefinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, weighting=None, learn_from_domain=False):
        super(TradeoffRefinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.weighter = Weighting(weighting, learn_from_domain)

    def forward(self, x):
        out = self.conv_1x1(x)
        dilation_output_list = list()
        for layer in self.layers:
            out = layer(out)
            dilation_output_list.append(out)

        # concat list into a vector
        dilation_outputs = torch.stack(dilation_output_list)

        averaged_outputs = self.weighter(dilation_outputs)
        out = self.conv_out(averaged_outputs)
        return out


class Weighting(nn.Module):
    def __init__(self, initial_weights=None, learn_from_domain=False):
        """
        :param initial_weights: currently the size of the refinement layer num_layers_R
        """
        super(Weighting, self).__init__()
        if isinstance(initial_weights, int):
            ## prepare weights for learned paramer
            self.w = torch.nn.Parameter(torch.tensor(np.ones(initial_weights) / initial_weights, dtype=torch.float32))
            self.w.requires_grad = True
        elif learn_from_domain:
            # just want to intialize and still have it be learned
            self.w = torch.nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
            self.w.requires_grad = True
        else:
            self.w = initial_weights
            self.w.requires_grad = False

    def forward(self, x):
        print('Parameters are:')
        print(self.w)
        return torch.einsum("ijkl, i->jkl", x, self.w)




class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return out + x


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split,
                 refinement_weighting=None, experimental=1, fold=0, learn_from_domain=False):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes,
                             refinement_weighting=refinement_weighting, experimental=experimental,
                             learn_from_domain=learn_from_domain)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        self.fold = fold

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def _train(self, batch_gen, epoch, best_acc, name='train'):
        batch_i_size = 5
        if name == 'valid':
            self.model.eval()
        else:
            self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        batch_i = 0
        batch_loss = 0
        while batch_gen.has_next():
            batch_i += 1
            batch_input, batch_target, mask = batch_gen.next_batch(self.batch_size)
            batch_input, batch_target, mask = batch_input.to(self.device), batch_target.to(self.device), mask.to(
                self.device)
            if name == 'train':
                self.optimizer.zero_grad()
            predictions = self.model(batch_input)

            loss = 0
            for p in predictions:
                loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                loss += 0.15 * torch.mean(torch.clamp(
                    self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                    max=16) * mask[:, :, 1:])

            batch_loss += loss / batch_i_size
            epoch_loss = loss.item()
            if name == 'train' and batch_i % batch_i_size == 0:
                batch_loss.backward()
                self.optimizer.step()
                batch_loss = 0

            _, predicted = torch.max(predictions[-1].data, 1)
            correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total += torch.sum(mask[:, 0, :]).item()

        loss = epoch_loss / len(batch_gen.list_of_examples)
        accuracy = float(correct) / total

        if name == 'valid' and accuracy > best_acc:
            torch.save(self.model.state_dict(),
                       self.save_dir + "/epoch-" + str(epoch + 1) + "_" + self.weighting_method + ".model")
            torch.save(self.optimizer.state_dict(),
                       self.save_dir + "/epoch-" + str(epoch + 1) + "_" + self.weighting_method + ".opt")
            best_acc = accuracy
            self.best_acc_epoch = epoch + 1

        logger.info("%s: [epoch %d ]: epoch loss = %f,   acc = %f" % (name, epoch + 1, loss, accuracy))

        ClearMLlogger = Logger.current_logger()
        ClearMLlogger.report_scalar(title=f"{name}_acc", series=f"accuracy_{self.fold}", iteration=(epoch + 1),
                                    value=accuracy)
        ClearMLlogger.report_scalar(title=f"{name}_loss", series=f"loss_{self.fold}", iteration=(epoch + 1),
                                    value=loss)
        batch_gen.reset()
        return best_acc

    def train(self, save_dir, batch_gen_train, batch_gen_val, num_epochs, batch_size, learning_rate, device,
              weighting_method=''):
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.weighting_method = weighting_method
        self.device = device
        self.model.to(device)
        print('starting training and validation')
        best_acc = 0
        self.best_acc_epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            best_acc = self._train(batch_gen_train, epoch, best_acc, name='train')
            best_acc = self._train(batch_gen_val, epoch, best_acc, name='valid')
            print(f'Best accuracy on validation: {best_acc} from epoch {self.best_acc_epoch}')

    def predict(self, model_dir, results_dir, features_path, vid_list_files, epoch, actions_dict, device, sample_rate,
                weighting_method='', final_predict_mode=True, final_predict_epoch=15):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(
                torch.load(model_dir + "/epoch-" + str(self.best_acc_epoch) + "_" + weighting_method + ".model"))
            print('#####################')
            print("Predicting")
            list_of_vids = vid_list_files
            for vid in list_of_vids:
                # print vid
                features = np.load(features_path + vid)
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                                                                    list(actions_dict.values()).index(
                                                                        predicted[i].item())]] * sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + weighting_method + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()