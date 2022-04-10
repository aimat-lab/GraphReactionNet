# !/usr/bin/python
# -*- coding: utf-8 -*-
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from datasets.dataloader import SMILES_Loader, DepthDataLoader
from datasets.qm9 import XYZ
from datasets.qm_utils import get_graph_stats, qm9_edges, qm9_nodes
from datasets.xyz_utils import replace_nodes, replace_edges, get_graph_stats, collate_g, statistic_normalize
from pathlib import Path, PurePath, PosixPath
import torch
import timeit
import argparse
import sys, os
import json
import numpy as np 
__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"

"""
11.04.2021 this file provides a template using our dataloader, copy this file to main can use.
"""


# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x


def is_rank_zero(args):
    return args.rank == 0


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


# def main():
# Argument parser
parser = argparse.ArgumentParser(description='Eenergy Prediction on SMILES', fromfile_prefix_chars='@',
                                 conflict_handler='resolve')

parser.convert_arg_line_to_args = convert_arg_line_to_args
parser.add_argument("--epochs", default=75, type=int, help='number of total epochs to run')
# for dataset
parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
parser.add_argument('--datasetCSVPath', default='data/CSV/clean_good_luck.csv', help='dataset path')

# parser.add_argument('--datasetXYZPath', default="../data/pickle/combine_train0415_test1015-15b14b0c-a18d-48bc-ae0c-b8d46257ed3b",
#                     help='dataset path')
# parser.add_argument("--batch_size", default=4, type=int, help="batch size")
parser.add_argument("--transform", default="transforms.Compose([ToTensor(), Normalize(mean=0, std=1)])", type=str)
parser.add_argument("--root", default=".", type=str,
                    help="Root folder to save data in")
parser.add_argument("--input", default="resultant", type=str,
                    help="embedding of graph: reactant -> add supernode between reactant, product -> main product")
parser.add_argument("--epochs", default=25, type=int, help='number of total epochs to run')
parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")

parser.add_argument('--dataset', default='local', help='local')
# parser.add_argument('--logPath', default='./log/local/mpnn/', help='log path')
# parser.add_argument('--plotLr', default=False, help='allow plotting the data')
# parser.add_argument('--plotPath', default='./plot/local/mpnn/', help='plot path')
# parser.add_argument('--resume', default='./checkpoint/qm9/mpnn/',
#                     help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
# parser.add_argument('--epochs', type=int, default=360, metavar='N',
#                     help='Number of epochs to train (default: 360)')
# parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
#                     help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
# parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
#                     help='Learning rate decay factor [.01, 1] (default: 0.6)')
# parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
#                     help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
# parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
#                     help='SGD momentum (default: 0.9)')
# i/o
# parser.add_argument('--log-interval', type=int, default=20, metavar='N',
#                     help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

global args, best_er1
best_er1 = 0
if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

args.num_threads = args.workers
args.mode = 'train'

if args.root != "." and not os.path.isdir(args.root):
    os.makedirs(args.root)

try:
    node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
    nodes = node_str.split(',')

    args.world_size = len(nodes)
    args.rank = int(os.environ['SLURM_PROCID'])

except KeyError as e:
    # We are NOT using SLURM
    args.world_size = 1
    args.rank = 0
    nodes = ["127.0.0.1"]

ngpus_per_node = torch.cuda.device_count()
args.num_workers = args.workers
args.ngpus_per_node = ngpus_per_node

if ngpus_per_node == 1:
    args.gpu = 0
# transform csv to xyz
data_loader = SMILES_Loader(args, 'online_eval', vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                            target_transform=None, e_representation='raw_distance')
target_mean = data_loader.target_mean
target_std = data_loader.target_std

# xyzpath = data_loader.csv2xyz(args.datasetCSVPath, Path(args.datasetCSVPath).parent)
# print(f"{data_loader.len} reactions are saved into {xyzpath}")
# # transform xyz to h5 plese give
input_embedding = "supernode"
input = "reactant-graph"
args.datasetXYZPath = "data/CSV/clean_good_luck-411155d1-e3cf-48e2-8f90-8738c63a9d63" #xyzpath
xyzpath =  "data/CSV/clean_good_luck-411155d1-e3cf-48e2-8f90-8738c63a9d63"#xyzpath
data_loader = XYZ(args, vertex_transform=replace_nodes, edge_transform=replace_edges,
                  target_transform='normalize', e_representation='raw_distance', input=input,
                  input_embedding=input_embedding)

args.cuda = not args.no_cuda and torch.cuda.is_available()

train_loader = torch.utils.data.DataLoader(data_loader,
                                           batch_size=args.batch_size, shuffle=True,
                                           collate_fn=collate_g,
                                           num_workers=args.prefetch, pin_memory=True)

#
# v = statistic_normalize(train_loader, target_mean, target_std)
# print(v)
# v = {'nodes': ([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1., 53.,
#     1.,  1.,  1.,  1.,  0.,  0.,  3.,  5.]), 'edges': ([10.163048, 1., 1., 1., 1., 1.,
#    1., 1., 1.]), 'target_mean': -0.8217515381403844, 'target_std': 0.24135232818666932}
v = {'target_mean': -0.8217515381403844, 'target_std': 0.24135232818666932}
# print(get_graph_stats(data_loader, 'degrees'))

# reload with normalize constants
data_loader = XYZ(args, vertex_transform=replace_nodes, edge_transform=replace_edges,
                target_transform='normalize', e_representation='raw_distance', input=input,
                input_embedding=input_embedding, normalize_dict=v)
start = timeit.default_timer()
data_loader.xyz2h5(Path(xyzpath), \
                   Path(f"{xyzpath}_{input}{input_embedding}_withSMILES_onlyTrNorm1.pickle"))
print(f"{timeit.default_timer() - start} s.")
print("End of rgb_exr_to_hdf5.py")
print(f"Pickle file is saved into {xyzpath}_{input}{input_embedding}_withSMILES.pickle")

# if __name__ == '__main__':
#     main()


