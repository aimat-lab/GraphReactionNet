#!/usr/bin/python
# -*- coding: utf-8 -*-
""""
TODO:
generate all form of graph embedding, and train.
"""
from datasets.utils import get_graph_stats, qm9_edges, qm9_nodes, target_encoder
from pathlib import Path, PurePath, PosixPath
import torch
import timeit
import argparse
from datasets import utils, XYZ
from datasets.dataloader import SMILES_Loader, DepthDataLoader
import sys, os
import numpy as np
import pandas as pd
from datasets.get_koordination import get_coords_from_smiles

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"

"""
11.04.2021 this file provides a template using our dataloader
"""

# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

def is_rank_zero(args):
    return args.rank == 0

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Eenergy Prediction on SMILES', fromfile_prefix_chars='@', conflict_handler='resolve')

    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument("--epochs", default=75, type=int, help='number of total epochs to run')
    # for dataset
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    # parser.add_argument('--datasetCSVPath', default='../data/CSV/new_training_set_opt04022021.csv', help='dataset path')

    parser.add_argument('--datasetXYZPath', default='../data/data/new_training_set_opt04022021-327e8893-346f-493e-93f2-dd0aeb16f16b', help='dataset path')
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--transform", default="transforms.Compose([ToTensor(), Normalize(mean=0, std=1)])", type=str)
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--input", default="resultant", type=str,
                        help="embedding of graph: reactant -> add supernode between reactant, product -> main product")
    parser.add_argument("--epochs", default=25, type=int, help='number of total epochs to run')
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")

    # parser.add_argument('--dataset', default='local', help='local')
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
    parser.add_argument('--epochs', type=int, default=360, metavar='N',
                        help='Number of epochs to train (default: 360)')
    parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 1e-2]), default=1e-3, metavar='LR',
                        help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
    parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                        help='Learning rate decay factor [.01, 1] (default: 0.6)')
    parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                        help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    # i/o
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='How many batches to wait before logging training status')
    # Accelerating
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

    # global args, best_er1
    # best_er1 = 0
    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # data = pd.read_csv(args.datasetCSVPath)
    # print(data.head)
    # Acid_Halide = data["Acid_Halide"].to_list()
    # Alcohol = data["Alcohol"].to_list()
    # Ester = data["Ester"].to_list()
    # Target = data["Energy_difference"].to_list()
    # HX_energy = data["HX_energy"].to_list()
    # values_HX, counts_HX = np.unique(HX_energy, return_counts=True)
    #
    # # mean std for XYZ class
    # mean = np.array(Target).mean()
    # std = np.array(Target).std()
    # data["Acid"] = ""
    #
    # df1 = data[data["HX_energy"] == values_HX[0]].assign(Acid="Cl")
    # df2 = data[data["HX_energy"] == values_HX[1]].assign(Acid="Br")
    # df3 = data[data["HX_energy"] == values_HX[2]].assign(Acid="I")
    # data_new = pd.concat([df1, df2, df3])
    # data_new = data_new[['Acid_Halide', 'Alcohol', 'Ester', 'Acid', 'Halide_energy', 'Alcohol_energy',
    #                      'Ester_energy', 'Energy_difference', 'HX_energy']]
    #
    # # In the dataset, you can find the "HX_energy". If it is around -137, that means the HX is HCl. If it is around -124, that is HBr. And -117 means HI.
    # Acid = data_new["Acid"].to_list()
    # values_acid, counts_acid = np.unique(Acid, return_counts=True)
    # print(f"There are {len(values_acid)} esters.\n Distribution are {counts_acid}")
    # values_ester, counts_ester = np.unique(Ester, return_counts=True)
    # print(f"There are {len(values_ester)} esters.\n Distribution are {counts_ester}")
    # values, counts = np.unique(Acid_Halide, return_counts=True)
    # print(f"There are {len(values)} acid halide.\n Distribution are {counts}")
    # alcohol_values, alcohol_counts = np.unique(Alcohol, return_counts=True)
    # print(f"There are {len(alcohol_values)} alcohol.\n Distribution are {alcohol_counts}.")
    #
    # mols = data_new[data_new.keys()[:4]].to_numpy().reshape(-1, 1)
    # print(f"It contains {mols.shape[0]} unique molecules.")
    # elements = []
    # for i in range(mols.shape[0]):
    #     try:
    #         coords, elems = get_coords_from_smiles(mols[i][0], "suffix", "rdkit")
    #         elements += list(set(elems))
    #         elements = list(set(elements))
    #         print(f"{len(elements)}")
    #     except:
    #        pass
    # print(f"There are {len(elements)} elements in the dataset, \n these are {elements}")
    # new_csv = Path(args.datasetCSVPath).parent.joinpath(f"new_{Path(args.datasetCSVPath).parts[-1]}")
    # data_new.to_csv(new_csv)
    print(f"add acid in original csv dataset.")

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if ngpus_per_node == 1:
        args.gpu = 0
    # transform csv to xyz
    # args.datasetCSVPath = "../data/data/new_training_set_opt04022021.csv"
    # data_loader = SMILES_Loader(args, 'train', vertex_transform=qm9_nodes, edge_transform=qm9_edges,
    #             target_transform=target_encoder, e_representation='raw_distance')
    # xyzfolder = data_loader.csv2xyz(args.datasetCSVPath, Path(args.datasetCSVPath).parent)
    # args.datasetXYZPath = xyzfolder
    # # transform xyz to h5 plese give
    # config: please see datasets.qm9
    input_embedding = "virtual-edge"
    input = "product-graph"

    # xyzfolder = "../data/data/new_test-d29ac32c-2e19-4660-a3a4-2d27cfe23955"

    data_loader = XYZ(args, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                 target_transform=None, e_representation='raw_distance', input=input,
                      input_embedding=input_embedding) # , mean=mean, std=std

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    start = timeit.default_timer()
    xyzfolder = Path(args.datasetXYZPath)
    data_loader.xyz2h5(xyzfolder, xyzfolder.parent.joinpath(f"{xyzfolder.parts[-1]}_{input}_{input_embedding}.pickle"))

    print(f"{timeit.default_timer() -start} s.")

    print("End of rgb_exr_to_hdf5.py")

if __name__ == '__main__':
    main()


