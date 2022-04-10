#!/usr/bin/python
# -*- coding: utf-8 -*-
import IPython
import numpy as np
import os
from tensorboard_logger import configure, log_value
import torch
import torch.nn as nn



import pandas as pd
import json
from pdb import set_trace
__author__ = "Chen Shao"
__email__ = "chen.shao@student.kit.edu"


def error_ratio(pred, target):
    if type(pred) is not np.ndarray:
        pred = np.array(pred)
    if type(target) is not np.ndarray:
        target = np.array(target)       
        
    return np.mean(np.divide(np.abs(pred - target), np.abs(target)))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        try: 
            self.val = val 
            self.sum += self.val * n
        except:
            self.val = val if val.size() == self.sum.size() else nn.ConstantPad1d((self.sum.size(0)-val.size(0), 0), 0)(val.transpose(0, 1)).transpose(0,1)
            self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_value(self):
        return self.avg




class Logger(object):
    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            # if the directory does not exist we create the directory
            os.makedirs(log_dir)
        else:                      
            # clean previous logged data under the same directory name
            self._remove(log_dir)

        # configure the project
        configure(log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains


class RunningAverage:
    def __init__(self):
        self.avg = 0
        self.count = 0

    def append(self, value):
        self.avg = (value + self.count * self.avg) / (self.count + 1)
        self.count += 1

    def get_value(self):
        return self.avg


class RunningAverageDict:
    def __init__(self):
        self._dict = None      

    def update(self, new_dict, n=1):
        if self._dict is None:
            self._dict = dict()
            for key, value in new_dict.items():
                self._dict[key] = AverageMeter()

        for key, value in new_dict.items():
            self._dict[key].update(value, n=1)

    def get_value(self):
        return {key: value.get_value() for key, value in self._dict.items()}


def compute_errors(gt, pred):
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).cpu().data.numpy().mean()
    a2 = (((thresh < 1.25).cpu().data.numpy()) ** 2).mean()
    a3 = (((thresh < 1.25).cpu().data.numpy()) ** 3).mean()

    abs_rel = torch.mean(torch.abs((gt - pred) / gt))
    sq_rel = torch.mean(torch.abs(((gt - pred) ** 2) / gt))
    mae_rel = torch.max(torch.abs((gt - pred) / gt))

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(torch.abs(gt)) - torch.log(torch.abs(pred))) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    mae = torch.abs(gt-pred).mean()
    # TODO r2 as metrics to monitor the training process 
    log_10 = (torch.abs(torch.log10(torch.abs(gt)) - torch.log10(torch.abs(pred)))).mean()

    error_ratio = torch.mean(torch.abs(pred - gt) / torch.abs(gt))
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                sq_rel=sq_rel, error_ratio=error_ratio,mae_rel=mae_rel, mae=mae)

def compute_errors_numpy(gt, pred):
    thresh = np.array([(gt / pred), (pred / gt)]).T.max(axis=1)
    a1 = (thresh < 1.25).mean()
    a2 = ((thresh < 1.25) ** 2).mean()
    a3 = (((thresh < 1.25)) ** 3).mean()

    abs_rel = np.mean(np.abs((gt - pred) / gt))
    sq_rel = np.mean(np.abs(((gt - pred) ** 2) / gt))
    mae_rel = np.max(np.abs((gt - pred) / gt))

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(np.abs(gt)) - np.log(np.abs(pred))) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    mae = np.abs(gt-pred).mean()
    # TODO r2 as metrics to monitor the training process
    log_10 = (np.abs(np.log10(np.abs(gt)) - np.log10(np.abs(pred)))).mean()

    error_ratio = np.mean(np.abs(pred - gt) / np.abs(gt))
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                sq_rel=sq_rel, error_ratio=error_ratio,mae_rel=mae_rel, mae=mae)


def csv_result_file(id, target_array, pred_array, resultFile):
    content = {}
    content.update({"id_array": id.squeeze(), "target": target_array.squeeze(), "pred": pred_array.squeeze()})
    df = pd.DataFrame(data=content)
    df = df.set_index('id_array')
    df = df.sort_values("id_array")
    df.to_csv(resultFile, sep='\t', encoding='utf-8')
    return df


# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:
    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]
"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import numpy as np

# Our Modules
import datasets
from datasets import qm_utils, xyz_utils
from models.MPNN import MPNN
# from LogMetric import AverageMeter, Logger, RunningAverage, RunningAverageDict, compute_errors
from pathlib import Path, PurePath, PurePosixPath
import os
import uuid
import wandb
from datetime import datetime as dt
import sys
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import seaborn as sns

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"

# os.environ['WANDB_MODE'] = 'dryrun'

global args, best_er1
PROJECT = "MPNN-Displace-Reaction-Training-Tuning"
logging = True


def main_worker(gpu, ngpus_per_node, args):
    # state gloabl variable for wandb and train
    global PROJECT
    args.gpu = gpu
    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath

    if ngpus_per_node == 1:
        args.gpu = 0

    args.last_epoch = -1
    args.epoch = 0
    args.rank = 0

    ############################# Dataloader & Preprocessing ########################
    # main worker
    print('Prepare files')
    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    idx = np.random.permutation(len(files))
    idx = idx.tolist()

    index = int(len(files) * (args.split))
    valid_ids = [files[i] for i in idx[index + 1:]]
    del index

    data_valid = datasets.Qm9(root, args, valid_ids, edge_transform=utils.qm9_edges, e_representation='raw_distance')

    # Define model and optimizer
    print('Define model')

    # Select one graph
    g_tuple, l = data_valid[0]
    g, h_t, e = g_tuple

    print('\tStatistics')
    stat_dict = datasets.utils.get_graph_stats(data_valid, ['target_mean', 'target_std'])

    data_valid.set_target_transform(lambda x: datasets.utils.normalize_data(x, stat_dict['target_mean'],
                                                                            stat_dict['target_std']))
    # Data Loader
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True, persistent_workers=False)

    ################################ Model  ##########################################
    print('\tCreate model')
    in_n = [len(h_t[0]), len(list(e.values())[0])]
    hidden_state_size = 73
    message_size = 73
    n_layers = 3
    l_target = 1
    type = 'regression'
    model = MPNN(in_n, hidden_state_size, message_size, n_layers, l_target, type=type)
    del in_n, hidden_state_size, message_size, n_layers, l_target, type

    args.multigpu = False
    print('Check cuda')
    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    # train for one epoch
    infer(model, valid_loader, args, device=args.gpu, root=args.root)


def infer(model, valid_loader, args, device=None, root="."):
    global PROJECT, best_er1, step
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ##################################### Loss:MSE ############################################
    print('Optimizer')
    criterion_mse = nn.MSELoss()
    ##################################### Evaluation ##########################################
    evaluation = lambda output, target: torch.mean(torch.abs(output - target) / torch.abs(target))
    ###########################################################################################

    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_er1 = checkpoint['best_er1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded best model '{}' (epoch {})".format(best_model_file, checkpoint['epoch']))
            del checkpoint
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    model.eval()
    metrics, val_mse, val_err, output, target = validate_eval(args, model, valid_loader, criterion_mse, evaluation)
    evaldir = Path(args.resume).joinpath("plots")
    if Path(evaldir).exists() is False:
        evaldir.mkdir()
    scatter_histogram_bestfit(args, (output, target), evaldir.joinpath("scatter.png"))

    heatmap(args, (output, target), evaldir)
    csvdir = Path(args.resume).joinpath("eval.csv")
    write_csv(args, metrics, val_mse, val_err, csvdir)
    print("Validated: {}".format(metrics))


def validate_eval(args, model, val_loader, criterion, evaluation, device='cpu'):
    with torch.no_grad():
        metrics = RunningAverageDict()

        val_mse = AverageMeter()
        val_error_ratio = AverageMeter()
        output_array, target_array = np.array([]), np.array([])

        for i, (g, h, e, target) in tqdm(enumerate(val_loader), desc=f"Loop: Validation\n", total=len(val_loader)):

            # Prepare input data
            if args.cuda:
                g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
            g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)
            target_array = np.concatenate((target_array, target.squeeze().cpu().data.numpy()), axis=0)

            # Compute output
            output = model(g, h, e)

            output_array = np.concatenate((output_array, output.squeeze().cpu().data.numpy()), axis=0)
            # Logs
            val_mse.update(criterion(output, target).data.item(), g.size(0))
            val_error_ratio.update(evaluation(output, target).data.item(), g.size(0))

            metrics.update(compute_errors(output, target), g.size(0))

        print(' * Average Error Ratio {err.avg:.3f}; Average Loss {loss.avg:.3f}'
              .format(err=val_error_ratio, loss=val_mse))

    return metrics.get_value(), val_mse, val_error_ratio, output_array, target_array


def scatter_histogram_bestfit(data, dir):
    (output, target) = data

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    # start with a square Figure
    fig = plt.figure(figsize=(9, 9))

    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    # use the previously defined function
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)

    # the scatter plot:
    ax.scatter(target, output)
    ax.set_xlabel("target(eV)", fontsize=20)
    ax.set_ylabel("prediction(eV)", fontsize=20)

    # now determine nice limits by hand:
    binwidth = (target.max() - target.min()) / 60
    binsx = np.arange(target.min(), target.max() + binwidth, binwidth)
    ax_histx.hist(target, bins=binsx)
    binsy = np.arange(output.min(), output.max() + binwidth, binwidth)
    ax_histy.hist(output, bins=binsy, orientation='horizontal')

    xmin, xmax = np.amin(target), np.amax(target)
    ax.plot(np.arange(xmin, xmax, np.abs(xmin - xmax) / target.shape[0]),
            np.arange(xmin, xmax, np.abs(xmin - xmax) / target.shape[0]), color='green')
    ax.set_title("Prediction of MPNN ", fontsize=12)
    plt.savefig(dir)
    

def heatmap(data, dir):
    (output, target) = data
    output, target = output.squeeze(), target.squeeze()
    heatmap, xedges, yedges = np.histogram2d(target, output, bins=100)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    fig, ax = plt.subplots(figsize=(9, 9))

    ax.imshow(heatmap.T, extent=extent, origin='lower')
    ax.set_xlabel("energy difference (eV)", fontsize=20)
    ax.set_ylabel("prediction (eV)", fontsize=20)
    ax.set_title("2D-histogram distribution estimation", fontsize=20)
    fig.savefig(Path(dir).joinpath("heatmap.png"))

    fig, ax = plt.subplots(figsize=(9, 9))
    ax = sns.heatmap(heatmap.T, vmin=heatmap.min(), vmax=heatmap.max(), cbar=False, xticklabels=False, yticklabels=False)
    ax.set_xlabel("predicted free energy (eV)", fontsize=20)
    ax.set_ylabel("target free energy(eV)", fontsize=20)
    ax.set_title("2D-histogram distribution estimation", fontsize=20)
    fig.savefig(Path(dir).joinpath("seabornheatmap.png"))

    plt.clf()
    plt.plot(np.sort(np.abs(target - output)), "ro")
    plt.savefig(Path(dir).joinpath("error_sort.png"))
    return 0


def histogramm(pred, target, png_dir):
    N_hist = 60
    error = (target - pred)
    n, bins, patches = plt.hist(error, N_hist, density=True, facecolor='b', alpha=0.75)


    plt.xlabel('Error')
    plt.ylabel('Probability Density')
    plt.title('Histogram of Error Distribution')
    # plt.text('DMPNN')
    plt.xlim(-0.5, 0.5)
    plt.grid(True)
    plt.savefig(png_dir)


def write_csv(metrics, val_mse, val_err, dir):
    csv_file = open(dir, "w+")
    metrics.update({"ID": Path(dir).parent, "val_mse": val_mse, "val_error_ratio": val_err})

    writer = csv.writer(csv_file)
    for key, value in metrics.items():
        writer.writerow([key, value])

    csv_file.close()


def save_args(args, argsdir):
    try:
        with open(argsdir, "w") as jsonFile:
            json.dump(vars(args), jsonFile, indent=2, sort_keys=True)
    except:
        args_dict = vars(args).copy()
        for keys, values in args_dict.items():
            try:
                if not isinstance(values, (str, int, float, bool, list)):
                    args_dict[keys] = str(values)
            except:
                if values is None:
                    args_dict[keys] = "None"
        with open(argsdir, "w") as jsonFile:
            json.dump(args_dict, jsonFile, indent=2, sort_keys=True)
        del args_dict

    return


if __name__ == '__main__':

    # Argument parser
    parser = argparse.ArgumentParser(description='Neural message passing')

    parser.add_argument('--dataset', default='Displace-Reaction-03302021', help='dataset ID')
    parser.add_argument('--datasetPath', default='data/data/data_xyz-00227f42-3874-4de0-9abe-eb0f953a0a80',
                        help='dataset path')
    parser.add_argument('--logPath', default='./log/qm9/mpnn/', help='log path')
    parser.add_argument('--plotLr', default=False, help='allow plotting the data')
    parser.add_argument('--plotPath', default='./plot/qm9/mpnn/', help='plot path')
    parser.add_argument('--resume', default='./checkpoint/checkpoint_cluster/qm9/mpnn/',
                        help='path to latest checkpoint')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    # Optimization Options
    parser.add_argument('--bs', '--batch-size', type=int, default=40, metavar='N',
                        help='Input batch size for training (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Enables CUDA training')
    parser.add_argument('--epochs', type=int, default=360, metavar='N',
                        help='Number of epochs to train (default: 360)')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # train
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--split", type=float, default=0.8, help="fraction of training set")

    best_er1 = 0

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        print(arg_filename_with_prefix)
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # standardize variable
    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'valid'

    # Folder to save
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    # Configurate gpu
    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if ngpus_per_node == 1:
        args.gpu = 0

    # main_worker(args.gpu, ngpus_per_node, args)

