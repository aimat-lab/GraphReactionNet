#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    utils.py: Functions to process dataset graphs.

    Usage:

"""

from __future__ import print_function
from torch.utils.data import DataLoader

import rdkit
import torch
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import numpy as np
import shutil
import os
import json
from pdb import set_trace
__author__ = "chen shao"
__email__ = "chen.shao@student.kit.edu"


def replace_nodes(graphs, hydrogen=False):
    h = []
    for n, d in graphs.nodes(data=True):
        h_t = []
        # Add Supernode Type 
        # h_t += [int(d["NodeType"] == "Supernode")]
        h_t += [i for i, x in enumerate(['None', "Supernode"]) if d["NodeType"] == x]
        h_t += [int(d['AtomSymbol'] == x) for x in ['H', 'C', 'N', 'O', 'S', 'F', 'P', 'B', 'Br', 'Cl', 'I']]
        # Atomic number
        h_t.append(d['NumAtomic'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['IsAromatic']))
        # FormalCharge
        h_t.append(d["FormalCharge"])
        # NumExplicit
        h_t.append(d["NumExplicit"])
        # NumImplicit
        h_t.append(d["NumImplicit"])
        # ChiralTag
        h_t+=[i for i, x in enumerate([None, rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW]) if d["ChiralTag"] == x]
        # Hybradization
        h_t+=[i for i, x in enumerate([None, rdkit.Chem.rdchem.HybridizationType.OTHER, rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED, rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3, rdkit.Chem.rdchem.HybridizationType.SP3D, rdkit.Chem.rdchem.HybridizationType.SP3D2]) if d['Hybridization'] == x]
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['TotalNum'])
        h.append(h_t)
    return h
# for keys in graphs.nodes(data=True)[0].keys():  
#     for n, d in graphs.nodes(data=True): 
#         print(d[keys])

def replace_edges(g, e_representation='raw_distance'):
    remove_edges = []
    e={}    
    for n1, n2, d in g.edges(data=True):
        e_t = []
        # Raw distance function
        if e_representation == 'chem_graph':
            if d['BondType'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t += [i+1 for i, x in enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'distance_bin':
            if d['BondType'] is None:
                step = (6-2)/8.0
                start = 2
                b = 9
                for i in range(0, 9):
                    if d['distance'] < (start+i*step):
                        b = i
                        break
                e_t.append(b+5)
            else:
                e_t += [i+1 for i, x in enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['BondType']]
        elif e_representation == 'raw_distance':
            if d['BondType'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t.append(d['distance'])
                e_t += [int(d['BondType'] == x) for x in ["virtual-edge", rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                                        rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
                e_t += [int(d['StereoType'] == x) for x in [rdkit.Chem.rdchem.BondStereo.STEREONONE, rdkit.Chem.rdchem.BondStereo.STEREOE,
                                                        rdkit.Chem.rdchem.BondStereo.STEREOZ]]

        else:
            print('Incorrect Edge representation transform')
            quit()
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)
    #return nx.to_numpy_matrix(g), e
    return nx.to_numpy_matrix(g), e


def target_encoder(target):
    return np.array([target[-1]])
    

def normalize_data(data, mean, std):
    return (data-mean)/std


def get_values(obj, start, end, prop):
    vals = []
    for i in range(start, end):
        v = {}
        if 'degrees' in prop:
            # physical interpretation: how to do graph normalization 
            v['degrees'] = set(sum(obj[i][0][0].sum(axis=0, dtype='int').tolist(), []))
        if 'edge_labels' in prop:
            v['edge_labels'] = set(sum(list(obj[i][0][2].values()), []))
        if 'target_mean' in prop or 'target_std' in prop:
            v['params'] = obj[i][1]
        vals.append(v)
    return vals   



def get_graph_stats(graph_obj_handle, prop='degrees'):
    # if prop == 'degrees':
    num_cores = multiprocessing.cpu_count()
    inputs = [int(i*len(graph_obj_handle)/num_cores) for i in range(num_cores)] + [len(graph_obj_handle)]
    res = Parallel(n_jobs=num_cores)(delayed(get_values)(graph_obj_handle, inputs[i], inputs[i+1], prop) for i in range(num_cores))

    stat_dict = {}

    if 'degrees' in prop:
        stat_dict['degrees'] = list(set([d for core_res in res for file_res in core_res for d in file_res['degrees']]))
    if 'edge_labels' in prop:
        stat_dict['edge_labels'] = list(set([d for core_res in res for file_res in core_res for d in file_res['edge_labels']]))
    if 'target_mean' in prop or 'target_std' in prop:
        param = np.array([file_res['params'] for core_res in res for file_res in core_res])
    if 'target_mean' in prop:
        stat_dict['target_mean'] = np.mean(param.astype(np.float), axis=0)
    if 'target_std' in prop:
        stat_dict['target_std'] = np.std(param.astype(np.float), axis=0)
    return stat_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.type_as(target)
    target = target.type_as(pred)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def collate_g(batch):
    batch_sizes = np.max(
        np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
                                len(list(input_b[2].values())[0])]
                                if input_b[2] else [len(input_b[1]), len(input_b[1][0]), 0, 0]
                                for (id, input_b, target_b, mols) in batch]), axis=0)


    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    mols = np.empty([len(batch), 4], dtype="S10")

    id = np.zeros((len(batch), len(np.asarray([batch[0][0]]))))
    target = np.zeros((len(batch), len(np.asarray([batch[0][2]]))))

    for i in range(len(batch)):

        num_nodes = len(batch[i][1][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][1][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][1][1] # last four atoms has different dimensions for node feature why

        # Edges
        for edge in batch[i][1][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][1][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][1][2][edge]

        # id
        id[i, :] = batch[i][0]
        # Target
        target[i, :] = batch[i][2]
        # print(np.asarray(batch[i][3]).shape)
        mols[i, :] = batch[i][3]

    id = torch.IntTensor(id)
    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    # mols = torch.StringType(mols)
    target = torch.FloatTensor(target)

    return id, g, h, e, target, mols

def collate_batch_tuple(batch):

    (input_b1, input_b2, input_b3), target_b = batch

    # batch_sizes = np.max(
    #     np.array([[len(input_b[1]), len(input_b[1][0]), len(input_b[2]),
    #                             len(list(input_b[2].values())[0])]
    #                             if input_b[2] else [len(input_b[1]), len(input_b[1][0]), 0, 0]
    #                             for (input_b, target_b) in batch]), axis=0)
    input_b, target_b = batch[:3], batch[-1]
    batch_sizes = np.max(
        np.array([[len(batch[0][1]), len(batch[0][1][0]), len(batch[0][2]),
                                len(list(batch[0][2].values())[0])]
                                if batch[0][2] else [len(batch[0][1]), len(batch[0][1][0]), 0, 0]]), axis=0)



    g = np.zeros((len(batch), batch_sizes[0], batch_sizes[0]))
    h = np.zeros((len(batch), batch_sizes[0], batch_sizes[1]))
    e = np.zeros((len(batch), batch_sizes[0], batch_sizes[0], batch_sizes[3]))

    target = np.zeros((len(batch), len(np.asarray([batch[0][1]]))))

    for i in range(len(batch)):

        num_nodes = len(batch[i][0][1])

        # Adjacency matrix
        g[i, 0:num_nodes, 0:num_nodes] = batch[i][0][0]

        # Node features
        h[i, 0:num_nodes, :] = batch[i][0][1] # last four atoms has different dimensions for node feature why

        # Edges
        for edge in batch[i][0][2].keys():
            e[i, edge[0], edge[1], :] = batch[i][0][2][edge]
            e[i, edge[1], edge[0], :] = batch[i][0][2][edge]

        # Target
        target[i, :] = batch[i][1]

    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)

    return g, h, e, target


def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


def save2dict(graph):
    dict_graph = {}
    for i, d in graph.nodes(data=True):
        dict_graph.update({str(i):d})
    with open("plots/current_graph.json", "w") as write_file:
        json.dump(dict_graph, write_file, indent=4)


def input_embedding(g, config: str):
    g = nx.disjoint_union(g[0], g[1])
    if config == "sparse-graph":
        return g
    elif config == "supernode":
        supernode_index = len(g.nodes())
        g.add_node(supernode_index,
            NodeType="Supernode",
            AtomSymbol = "None",
            NumAtomic = 0,
            acceptor= 0,
            donor= 0,
            IsAromatic= 0,
            FormalCharge = 0,
            NumExplicit = 0,
            NumImplicit = 0,
            ChiralTag = None,
            Hybridization = None,
            TotalNum= 0,
            coord=np.zeros((1, 3)))

        for n, dn in g.nodes(data=True):
            for m, dm in g.nodes(data=True):
                if dn["NodeType"] == 'Supernode' or dm["NodeType"] == 'Supernode':
                    g.add_edge(n, m, BondType="virtual-edge", StereoType=None,
                        distance=np.linalg.norm(g.nodes[n]['coord']-g.nodes[m]['coord']))
        # TODO remove self-circle here
        g.remove_edge(*(supernode_index, supernode_index))
        return g
    elif config == "virtual-edge":
        existed_edge_list = list(dict(g.edges()).keys())
        to_add_edge = []
        for i in range(g.number_of_nodes()):
            for j in range(g.number_of_nodes()):
                if (i, j) not in existed_edge_list:
                    to_add_edge.append((i, j))
        for (i, j) in to_add_edge:
            g.add_edge(i, j, BondType="virtual-edge", StereoType=None,
                       distance=np.linalg.norm(g.nodes[i]['coord'] - g.nodes[j]['coord']))
    return g

# TODO combine graphs without global node <--> compare with reactant with global node
def combine_graphs(g):
    """This function combine two reactant graphs without global node:
    print("The nodes of G are relabeled 0 to len(G)-1, and the nodes of H are relabeled len(G) to len(G)+len(H)-1. \
    Graph, edge, and node attributes are propagated from G and H to the union graph.  \
    If a graph attribute is present in both G and H the value from H is used.")

    nodes_dict = {}
    for n, dn in g[0].nodes(data=True):
        nodes_dict.update({n: dn})
    nodes_dict2 = {}
    for n, dn in g[1].nodes(data=True):
        nodes_dict2.update({n: dn})
    # print nodes
    for n, dn in g[0].nodes(data=True):
        print(n, dn["AtomSymbol"])
    for n, dn in g[1].nodes(data=True):
        print(n, dn["AtomSymbol"])

    combine_graph_dict = {}
    g_combine = nx.disjoint_union(g[0], g[1])
    for n, dn in g_combine.nodes(data=True):
        combine_graph_dict.update({n: dn})
    index = 0
    for ind, val in combine_graph_dict.items():
        # print(ind, combine_graph_dict[ind])
        try:
            print(combine_graph_dict[ind] == nodes_dict[ind])
            index = ind
        except:
            print(combine_graph_dict[ind] == nodes_dict2[ind-index-1])
    print("checking is finished, edge features remains the same.)

    Visualization:
    delete_edge = []
    for (i, j, dict_edge) in g_combine.edges(data=True):
        if dict_edge["distance"] == 0:
            delete_edge.append((i, j))
    for (i, j) in delete_edge:
        g_combine.remove_edge((i, j))
    Plotter.plot_graph(nx.to_numpy_matrix(g_combine))
    """

    g = nx.disjoint_union(g[0], g[1])
    return g


def qm9_nodes(g, hydrogen=False):
    h = []
    for n, d in g.nodes(data=True):
        h_t = []
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
        # Atomic number
        h_t.append(d['a_num'])
        # Partial Charge
        h_t.append(d['pc'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # Hybradization
        h_t += [int(d['hybridization'] == x) for x in [rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]]
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['num_h'])
        h.append(h_t)
    return h


def qm9_edges(g, e_representation='raw_distance'):
    remove_edges = []
    e = {}
    for n1, n2, d in g.edges(data=True):
        e_t = []
        # Raw distance function
        if e_representation == 'chem_graph':
            if d['b_type'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t += [i + 1 for i, x in
                        enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'distance_bin':
            if d['b_type'] is None:
                step = (6 - 2) / 8.0
                start = 2
                b = 9
                for i in range(0, 9):
                    if d['distance'] < (start + i * step):
                        b = i
                        break
                e_t.append(b + 5)
            else:
                e_t += [i + 1 for i, x in
                        enumerate([rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                                   rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC])
                        if x == d['b_type']]
        elif e_representation == 'raw_distance':
            if d['b_type'] is None:
                remove_edges += [(n1, n2)]
            else:
                e_t.append(d['distance'])
                e_t += [int(d['b_type'] == x) for x in
                        [rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE,
                         rdkit.Chem.rdchem.BondType.TRIPLE, rdkit.Chem.rdchem.BondType.AROMATIC]]
        else:
            print('Incorrect Edge representation transform')
            quit()
        if e_t:
            e[(n1, n2)] = e_t
    for edg in remove_edges:
        g.remove_edge(*edg)
    return nx.to_numpy_matrix(g), e

def statistic_normalize(loader: DataLoader, target_mean: float, target_std:float)-> dict:
    v = {}
    for i, (ind, g, h, e, target, smiles) in enumerate(loader):
        if i  == 0:
            v['nodes'] = h.data.numpy().squeeze().max(axis=0)
            v['edges'] = e.data.numpy().squeeze().reshape(-1, 9).max(axis=0)
        else:
            v['nodes'] = np.concatenate((v['nodes'][None, ...], h.data.numpy().squeeze().max(axis=0)[None, ...]), axis=0).max(axis=0) 
            v['edges'] = np.concatenate((v['edges'][None, ...], e.data.numpy().squeeze().reshape(-1, 9).max(axis=0)[None, ...]), axis=0).max(axis=0) 
            pass
        # print("index", ind)
        # print("g", g.shape)
        # print("h", len(h))
        # print("target", target)
        # print("smiles", smiles)
        print(v['nodes'])
        print(v['edges'])

    v['target_mean'] = target_mean
    v['target_std'] = target_std
    return v