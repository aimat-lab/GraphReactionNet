#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
qm9.py:

Usage:

"""
# TODO normalize the input vector 
# TODO try another normalization method like norm 2 

# Networkx should be imported before torch
import networkx as nx
from typing import List, Type, Union, Tuple 
from numpy.lib.shape_base import split
import torch.utils.data as data
import numpy as np


from .qm_utils import qm9_edges, qm9_nodes
from .xyz_utils import  get_graph_stats, replace_edges, replace_nodes, input_embedding
import os, sys
from pdb import set_trace
from joblib import Parallel, delayed
import multiprocessing
from visualization.Plotter import Plotter
import os
from torch.utils.data import Dataset, DataLoader
from pathlib import Path, PosixPath, PurePath
from datetime import datetime as dt
import pickle
import pandas as pd 
from IPython import embed 

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from .graph_reader import xyz_graph_reader, xyz_qm9_reader

__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"



# TODO adapt to GGNN
class Qm9(data.Dataset):

    # Constructor
    def __init__(self, root, args, ids, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                 target_transform=None, e_representation='raw_distance'):
        self.root = root
        # self.gpu = args.gpu
        self.ids = ids
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation

    def __getitem__(self, index):
        g, target = xyz_qm9_reader(os.path.join(self.root, self.ids[index]))
        # g = input_embedding(g)
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g, self.e_representation)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return (g, h, e), target

    def __len__(self):
        return len(self.ids)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform



class XYZ(data.Dataset):

    # Constructor
    def __init__(self, args, vertex_transform=replace_nodes, edge_transform=replace_edges,
                 target_transform=None, e_representation='raw_distance', input="reactant-graph", input_embedding="supernode",
                 normalize_dict=None):
        self.root = args.datasetXYZPath
        self.gpu = args.gpu
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation
        self.target_normalize = True
        self.input = input
        self.input_embedding = input_embedding
        self.whole_files = [f for f in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, f))]

        self.idx = np.arange(len(self.whole_files)).tolist()
        # statistical analysis for further processing, initialized
        if normalize_dict:
            self.target_mean = normalize_dict['target_mean']
            self.target_std = normalize_dict['target_std']
            try:
                self.nodes_norm = normalize_dict['nodes']
                self.edges_norm = normalize_dict['edges']
            except:
                self.nodes_norm = None
                self.edges_norm = None 
                print("Without normalization for nodes and edges...")
        # if (self.mean is None) or (self.std is None):
        #     print('\tStatistics')

        #     num_cores = multiprocessing.cpu_count()
        #     inputs = [int(i*len(self.idx)/num_cores) for i in range(num_cores)] + [len(self.idx)]
        #     # for debug please run this:
        #     res = self.get_statistic(inputs[10], inputs[11])
        #     for i in range(num_cores):
        #         res = self.get_statistic(inputs[i], inputs[i + 1])

        #     res = Parallel(n_jobs=num_cores)(delayed(self.get_statistic)(inputs[i], inputs[i+1]) for i in range(num_cores))
        #     param = np.array([file_res['params'] for core_res in res for file_res in core_res])
        #     self.target_mean = np.mean(param.astype(np.float), axis=0)
        #     self.target_std = np.std(param.astype(np.float), axis=0)

    


    def __getitem__(self, index):        
        ind, g, target, smiles = xyz_graph_reader(os.path.join(self.root, self.whole_files[index]))
        if self.input == "reactant-graph":
            g = input_embedding(g[:2], self.input_embedding)
        elif self.input == "product":
            g = g[-1]
        elif self.input == "product-graph":
            g = input_embedding(g[-2:], self.input_embedding)

        if self.vertex_transform is not None:
            h = self.vertex_transform(g)
        
        if self.edge_transform is not None:
            g, e = self.edge_transform(g, self.e_representation)

        return ind, (g, h, e), target, smiles
        

    def __len__(self):
        return len(self.idx)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform

    def target_transform(self, target):
        return (target-self.target_mean)/self.target_std

    def get_statistic(self, idx1, idx2):
        vals = []
        v = {}
        for i in range(idx1, idx2):
            _, _, target, _ = xyz_graph_reader(os.path.join(self.root, self.whole_files[i]))
            v['params'] = target
            
            vals.append(v)
        return vals

    def get_values(self, obj, start, end, prop):
        vals = []
        for i in range(start, end):
            v = {}
            if 'degrees' in prop:
                v['degrees'] = set(sum(obj[i][0][0].sum(axis=0, dtype='int').tolist(), []))
            if 'edge_labels' in prop:
                v['edge_labels'] = set(sum(list(obj[i][0][2].values()), []))
            if 'target_mean' in prop or 'target_std' in prop:
                v['params'] = obj[i][1]
            vals.append(v)
        return vals


    def get_graph_stats(self, graph_obj_handle, prop='degrees'):
        # if prop == 'degrees':
        num_cores = multiprocessing.cpu_count()
        inputs = [int(i*len(graph_obj_handle)/num_cores) for i in range(num_cores)] + [len(graph_obj_handle)]
        res = Parallel(n_jobs=num_cores)(delayed(self.get_values)(graph_obj_handle, inputs[i], inputs[i+1], prop) for i in range(num_cores))

        stat_dict = {}

        if 'degrees' in prop:
            stat_dict['degrees'] = list(set([d for core_res in res for file_res in core_res for d in file_res['degrees']]))
        if 'edge_labels' in prop:
            stat_dict['edge_labels'] = list(set([d for core_res in res for file_res in core_res for d in file_res['edge_labels']]))
        if 'target_mean' in prop or 'target_std' in prop:
            param = np.array([file_res['params'] for core_res in res for file_res in core_res])
        if 'target_mean' in prop:
            stat_dict['target_mean'] = np.mean(param, axis=0)
        if 'target_std' in prop:
            stat_dict['target_std'] = np.std(param, axis=0)

        return stat_dict

    def xyz2h5(self, xyzpath: Path, outFile: Path, **kwargs) -> Path:
        """Transformer from XYZ to pickle format.

            This class is derived from the core
            :class: Smiles2XYZ

            See Also:
                See the parent class :class:~plenpy.datasets.qm9 SMILES
                for the basic Attributes.

            Config_Params:
                input: determines if reactants or products as dataset: "reactant", "product"
                input_embedding: determines if only one molecules used to generate dataset, or
                    two graphs,
                    "supernode" : combines two graph with supernode index by LEN-1.
                    "virtual-edge": add virtual edges between all nodes.
                    "sparse-graph": disjoint unite two graphs.
            """
        xyz_paths = sorted(list(xyzpath.iterdir()))
        dataset = {}
        for c, react in enumerate(xyz_paths):
            if c % 10 == 0:
                print("counter:", c)
            batch = {}
            # write data to a hdf5 file (8 byte = 64 bit float)
            ind, g, target, mols = xyz_graph_reader(react)
            if self.input == "reactant-graph":
                g = input_embedding(g[:2], self.input_embedding)
            elif self.input == "product":
                g = g[-1]
            elif self.input == "product-graph":
                g = input_embedding(g[-2:], self.input_embedding)

            if self.vertex_transform is not None:
                h = self.vertex_transform(g)
            
            if self.edge_transform is not None:
                g, e = self.edge_transform(g, self.e_representation)

            if self.target_transform is not None:
                target = (target-self.target_mean) / self.target_std

            if self.nodes_norm is not None:
                self.nodes_norm = np.array(self.nodes_norm).astype(np.int32)
                h = np.array(h)
                h = np.nan_to_num(h / self.nodes_norm[None, ...])# TODO 
            
            if self.edges_norm is not None:
                self.edges_norm = np.array(self.edges_norm)
                for k, val in e.items():
                    e[k] = e[k] / self.edges_norm[None, ...]


            batch.update({"g": g})
            batch.update({"h": h})
            batch.update({"e": e})
            batch.update({"target": target})
            batch.update({"smiles": mols})
            dataset.update({f"{ind}": batch})
        dataset.update({"target_mean": self.target_mean, "target_std": self.target_std})

        with open(outFile, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"XYZ2Pickel.pickle is saved into {outFile}")
        return outFile

    def xyz2h5_delOutlier(self, xyzpath: Path, outFile: Path, outlier:List, **kwargs) -> Path:
        """Transformer from XYZ to pickle format.

            This class is derived from the core
            :class: Smiles2XYZ

            See Also:
                See the parent class :class:~plenpy.datasets.qm9 SMILES
                for the basic Attributes.

            Config_Params:
                input: determines if reactants or products as dataset: "reactant", "product"
                input_embedding: determines if only one molecules used to generate dataset, or
                    two graphs,
                    "supernode" : combines two graph with supernode index by LEN-1.
                    "virtual-edge": add virtual edges between all nodes.
                    "sparse-graph": disjoint unite two graphs.
            """
        xyz_paths = sorted(list(xyzpath.iterdir()))
        dataset = {}
        for c, react in enumerate(xyz_paths):
            if c % 10 == 0:
                print("counter:", c)
            batch = {}
            # write data to a hdf5 file (8 byte = 64 bit float)
            ind, g, target, mols = xyz_graph_reader(react)
            if self.input == "reactant-graph":
                g = input_embedding(g[:2], self.input_embedding)
            elif self.input == "product":
                g = g[-1]
            elif self.input == "product-graph":
                g = input_embedding(g[-2:], self.input_embedding)

            if self.vertex_transform is not None:
                h = self.vertex_transform(g)
            
            if self.edge_transform is not None:
                g, e = self.edge_transform(g, self.e_representation)

            if self.target_transform is not None:
                target = (target-self.target_mean) / self.target_std

            batch.update({"g": g})
            batch.update({"h": h})
            batch.update({"e": e})
            batch.update({"target": target})
            batch.update({"smiles": mols})
            dataset.update({f"{ind}": batch})
        dataset.update({"target_mean": self.target_mean, "target_std": self.target_std})

        with open(outFile, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"XYZ2Pickel.pickle is saved into {outFile}")
        return outFile


class PICKLE(Dataset):
    def __init__(self, args, mode, pickle_file, transform=None, **kwargs):

        with open(pickle_file, 'rb') as handle:
            self.pickle = pickle.load(handle)
        # self.split = args.split
        self.pickle_file = pickle_file
        self.idx = None
        self.split_mode = args.split_mode
        self.mode = mode

        try:
            self.idx = kwargs["index_list"]
        except:
            print(f"No given index list, generate new using mode {self.split_mode}")

            # for train and test use random permutated index
            if self.split_mode == "overlapping":
                self.train_id, self.valid_id, self.test_id = self.split_dataset_with_overlapping()
            elif self.split_mode == "one_molecule_overlapping":
                self.train_id, self.valid_id, self.test_id = self.split_dataset_partly_overlapping(config="alcohol")
            elif self.split_mode == "no_overlapping":
                self.train_id, self.valid_id, self.test_id = self.split_dataset_no_overlapping()


            if self.mode == 'train':
                self.idx = self.train_id
            elif self.mode == 'eval':
                self.idx = self.valid_id
            elif self.mode == 'test':
                self.idx = self.test_id

        self.transform = transform


    def __len__(self):
        return len(self.idx)

    def train_id(self):
        return self.train_id
    
    def valid_id(self):
        return self.valid_id

    def test_id(self):
        return self.test_id


    def __getitem__(self, idx):
        """
        Returns the item at given index as a dictionary with 'rgb' and 'depth' 
        containing ALL the n images of the sequence with the given index. 
        shape: (n, width, height, channels)
        """

        idx = self.idx[idx]
        try:
            g, h, e, target, smiles = self.pickle.get(str(idx))['g'], self.pickle.get(str(idx))['h'], self.pickle.get(str(idx))['e'], self.pickle.get(str(idx))['target'], self.pickle.get(str(idx))['smiles']
            return idx, (g, h, e), target, smiles
        except:
            try:
                g, h, e, target = self.pickle.get(str(idx))['g'], self.pickle.get(str(idx))['h'], self.pickle.get(str(idx))['e'], self.pickle.get(str(idx))['target']
                return idx, (g, h, e), target, None
            except:
                embed()

    def split_dataset_with_overlapping(self) -> Tuple[List[int], List[int], List[int]]:
        np.random.seed(dt.now().microsecond)
        index_keys_fromdataset = list(self.pickle.keys())[:-2]
        idx = np.random.permutation(index_keys_fromdataset)
        idx = idx.tolist()
        index = ((len(self.pickle.keys()) - 2) * np.cumsum(np.array([0.8, 0.1, 0.1]))).astype("int32")
        test_ids = idx[index[1]+1:]
        valid_ids = idx[index[0]:index[1]+1]
        train_ids = idx[0:index[0]]
        return train_ids, valid_ids, test_ids

    def split_dataset_partly_overlapping(self, **kwargs) -> Tuple[List[int], List[int], List[int]]:
        # TODO split dataset into partly overlapping 
        config_params = kwargs["config"]

        df = self.pickle2df(self.pickle)
        set_alcohol, set_acid_halide = np.random.permutation(list(set(df["alcohol"]))), np.random.permutation(list(set(df["acid_halide"])))
        len_train, len_valid, len_test = ((len(self.pickle.keys()) - 2) * np.array([0.8, 0.1, 0.1])).astype("int32")
        train_ids, valid_ids, test_ids = [], [], []
        if config_params == "alcohol":
            for alcohols in set_alcohol:
                if len(train_ids) < len_train:
                    train_ids += df[df["alcohol"] == alcohols].index.to_list()
                # elif len(valid_ids) < len_valid:
                #     valid_ids += df[df["alcohol"] == alcohols].index.to_list()
                else:
                    np.random.seed(dt.now().microsecond)
                    rest =  np.random.permutation(list(set(df.index.tolist()) - set(train_ids))).tolist()
                    valid_ids, test_ids = rest[:len_valid], rest[len_valid:]
                    test_ids = list(set(df.index.tolist()) - set(train_ids) - set(valid_ids))

        if config_params == "acid_halide":
            for acid_halide in set_acid_halide:
                if len(train_ids) < len_train:
                    train_ids += df[df["acid_halide"] == acid_halide].index.to_list()
                # elif len(valid_ids) < len_valid:
                #     valid_ids += df[df["acid_halide"] == acid_halide].index.to_list()
                else:
                    np.random.seed(dt.now().microsecond)
                    rest =  np.random.permutation(list(set(df.index.tolist()) - set(train_ids))).tolist()
                    valid_ids, test_ids = rest[:len_valid], rest[len_valid:]
                    test_ids = list(set(df.index.tolist()) - set(train_ids) - set(valid_ids))
                    test_ids = list(set(df.index.tolist()) - set(train_ids) - set(valid_ids))
        print(f"\n{len(train_ids)} train samples, {len(valid_ids)} valid samples, {len(test_ids)} test samples.")
        print(f"Split of the dataset with config {config_params} finished...")
        return train_ids, valid_ids, test_ids

    def split_dataset_no_overlapping(self) -> Tuple[List[int], List[int], List[int]]:
        # TODO split dataset into without overlapping 

        df = self.pickle2df(self.pickle)
        set_alcohol, set_acid_halide = np.random.permutation(list(set(df["alcohol"]))), np.random.permutation(list(set(df["acid_halide"])))
        len_train, len_valid, len_test = ((len(self.pickle.keys()) - 2) * np.array([0.8, 0.1, 0.1])).astype("int32")
        train_ids, valid_ids, test_ids = [], [], []
        for alcohols, acid_halide in zip(set_alcohol, set_acid_halide):
            if len(train_ids) < len_train:
                train_ids += df[df["alcohol"] == alcohols].index.to_list()
            elif len(train_ids) < len_train:
                train_ids += df[df["acid_halide"] == acid_halide].index.to_list()
            else:
                np.random.seed(dt.now().microsecond)
                rest =  np.random.permutation(list(set(df.index.tolist()) - set(train_ids))).tolist()
                valid_ids, test_ids = rest[:len_valid], rest[len_valid:]

        print(f"\n{len(train_ids)} train samples, {len(valid_ids)} valid samples, {len(test_ids)} test samples.")
        return train_ids, valid_ids, test_ids

    def pickle2df(self, pickle_content):
        content = {}
        id = list(pickle_content.keys())[:-2]
        acid_halide, alcohol, ester, acid, target = [], [], [], [], []
        for ids in id:
            acid_halide.append(pickle_content[str(ids)]['smiles'][0])
            alcohol.append(pickle_content[str(ids)]['smiles'][1])
            ester.append(pickle_content[str(ids)]['smiles'][2])
            acid.append(pickle_content[str(ids)]['smiles'][3])
            target.append(pickle_content[str(ids)]['target'])
        content.update({"id_array": id, "acid_halide": acid_halide, "alcohol": alcohol,
                        "ester": ester, "acid": acid})
        df = pd.DataFrame(data=content)
        df = df.set_index('id_array')
        return df


class DatasetfromPickle(Dataset):
    def __init__(self, args, mode, pickle_file, transform=None):

        with open(pickle_file, 'rb') as handle:
            self.pickle = pickle.load(handle)
        self.split = args.split
        self.pickle_file = pickle_file
        self.idx = None
        self.shuffle = True # if random shuffle train and validate
        self.split = 0.6
        # for train and test use random permutated index
        if mode == 'train':
            # to identify the outlier
            if self.shuffle is True:
                np.random.seed(dt.now().microsecond)
                # get index from pickle dict
                id_list = list(self.pickle.keys())[:-2]
                idx = np.random.permutation(id_list)
                self.idx = idx.tolist()


        elif mode == "eval" or mode == "test":
            # get index from pickle dict
            idx = list(self.pickle.keys())[:-2]
            index = int((len(self.pickle.keys()) - 2) * args.split)
            valid_ids = idx[:index]
            test_ids = idx[index:]
            if mode == "eval":
                self.idx = valid_ids
            elif mode == "test":
                self.idx = test_ids

        self.transform = transform


    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        """
        Returns the item at given index as a dictionary with 'rgb' and 'depth'
        containing ALL the n images of the sequence with the given index.
        shape: (n, width, height, channels)
        """

        idx = self.idx[idx]
        g, h, e, target, smiles = self.pickle.get(str(idx))['g'], self.pickle.get(str(idx))['h'], self.pickle.get(str(idx))['e'], self.pickle.get(str(idx))['target'], self.pickle.get(str(idx))['smiles']
        return idx, (g, h, e), target, smiles


# !/usr/bin/python
# -*- coding: utf-8 -*-

from .qm_utils import get_graph_stats, qm9_edges, qm9_nodes
from .xyz_utils import  target_encoder, replace_edges, replace_nodes
from pathlib import Path, PurePath, PosixPath
import torch
import timeit
import argparse

from .dataloader import SMILES_Loader, DepthDataLoader
import sys, os

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


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Eenergy Prediction on SMILES', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')

    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument("--epochs", default=75, type=int, help='number of total epochs to run')
    # for dataset
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument('--datasetCSVPath', default='data/data/new_training_set_opt.csv', help='dataset path')

    parser.add_argument('--datasetXYZPath', default='data/data/data-b1c238c6-7693-4fdf-8082-f237ff1319cb',
                        help='dataset path')
    parser.add_argument('--split_mode', '--split-mode', default='data/data/data-b1c238c6-7693-4fdf-8082-f237ff1319cb',
        help='dataset path')
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--transform", default="transforms.Compose([ToTensor(), Normalize(mean=0, std=1)])", type=str)
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--input", default="resultant", type=str,
                        help="embedding of graph: reactant -> add supernode between reactant, product -> main product")
    parser.add_argument("--epochs", default=25, type=int, help='number of total epochs to run')
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")

    parser.add_argument('--dataset', default='local', help='local')
    parser.add_argument('--logPath', default='./log/local/mpnn/', help='log path')
    parser.add_argument('--plotLr', default=False, help='allow plotting the data')
    parser.add_argument('--plotPath', default='./plot/local/mpnn/', help='plot path')
    parser.add_argument('--resume', default='./checkpoint/qm9/mpnn/',
                        help='path to latest checkpoint')
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
    data_loader = SMILES_Loader(args, 'train', vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                                target_transform=target_encoder, e_representation='raw_distance')
    data_loader.csv2xyz(args.datasetCSVPath, Path(args.datasetCSVPath).parent)

    # transform xyz to h5 plese give
    input_embedding = "virtual-edge"
    input = "reactant-graph"
    data_loader = XYZ(args, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                      target_transform=None, e_representation='raw_distance', input=input,
                      input_embedding=input_embedding)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    start = timeit.default_timer()
    data_loader.xyz2h5(Path("data/data/data-b1c238c6-7693-4fdf-8082-f237ff1319cb"), \
                       Path(f"data-b1c238c6-7693-4fdf-8082-f237ff1319cb_{input}{input_embedding}.pickle"))

    start = timeit.default_timer()
    print(f"{timeit.default_timer() - start} s.")

    print("End of rgb_exr_to_hdf5.py")


# if __name__ == '__main__':
#     main()


