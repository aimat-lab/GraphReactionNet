"""This files contains dataloader for smiles data in excel"""
from .logg import get_logger
import random
import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from typing import List, Tuple
from pathlib import Path, PurePath, PurePosixPath
from datetime import datetime
import warnings
import os
import io
import networkx as nx
import matplotlib.pyplot as plt
import IPython
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import MolFromSmarts as smt2mol
from rdkit.Chem.AllChem import ReactionFromSmarts as smt2rxn
from .graph_reader import atom_feature, bond_feature, init_graph_local
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from .qm_utils import get_graph_stats, qm9_edges, qm9_nodes
from .xyz_utils import target_encoder
from .xtb_utils import run_xtb
from .get_koordination import get_coords_from_smiles
from rdkit import Chem
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple, Type, Any
from csv import reader
import uuid
from pathlib import Path, PurePath, PurePosixPath
import h5py 
import glob 

logger = get_logger()

def preprocessing_transforms(mode):
    pass 

def _recursive_list(subpath: str):
    return sorted(os.walk(str(subpath), followlinks=False),
                  key=lambda x: x[0])


def _random_choose(dicts: dict):
    # update random seed using datetime
    np.random.seed(datetime.now().microsecond)
    params = dicts.copy()
    # enable the methods in augmentation when it is enabled
    keys = list(params.keys())
    for (k, v) in params.items():
        if v is False:
            keys.remove(k)
    # how many augmentation methods will be used? all - len
    len = random.randint(0, keys.__len__())
    keys_use = random.sample(keys, len)
    for k in list(params.keys()):
        if k not in keys_use:
            params.pop(k)
    return params


def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s

class DepthDataLoader(object):
    def __init__(self, args, mode, format):
        if mode == 'train':
            if format == "csv":
                self.training_samples = SMILES_Loader(args, mode, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                    target_transform=None, e_representation='raw_distance')
            elif format == "xyz":
                self.training_samples = XYZ_Loader(args, mode, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                    target_transform=None, e_representation='raw_distance')
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.training_samples)
            else:
                self.train_sampler = None

            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=(self.train_sampler is None),
                                   num_workers=1,  # 1 for debug, args.num_threads for train
                                   pin_memory=True,
                                   sampler=self.train_sampler)

        elif mode == 'online_eval':
            if format == "csv":
                self.online_eval_samples = SMILES_Loader(args, mode, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                    target_transform=None, e_representation='raw_distance')
            elif format == "xyz":
                self.online_eval_samples = XYZ_Loader(args, mode, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                    target_transform=None, e_representation='raw_distance')

            if args.distributed:  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(self.online_eval_samples, 1,
                                   shuffle=False,
                                   num_workers=1,  
                                   pin_memory=False,
                                   sampler=self.eval_sampler)

        elif mode == 'test':
            if format == "csv":
                self.testing_samples = SMILES_Loader(args, mode, transform=preprocessing_transforms(mode))
            elif format == "xyz":
                self.testing_samples = XYZ_Loader(args, mode, transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)

        else:
            print('mode should be one of \'train, test, online_eval\'. Got {}'.format(mode))



class SMILES_Loader(Dataset):
    def __init__(self, args, mode, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                 target_transform=target_encoder, e_representation='raw_distance'):

        self.mode = mode
        self.directory = args.datasetCSVPath
        self.gpu = args.gpu
        self.shuffle = False
        self.batch_index = 0
        self.split = 1
        self.seed = None
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation

        # encode important dataset information
        self.data = []
        self.label = []
        with open(self.directory, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for i, row in enumerate(csv_reader):
                # row variable is a list that represents a row in csv
                if i == 0:
                    # extract keys 
                    self.graph_keys = row[:3]
                    self.label_keys = row[3:]
                else:
                    self.data.append(row[:3])
                    self.label.append(row[3:])
        self.len = len(self.data)            
        print(f"Reading {self.len} SMILES from file {self.directory} ...")

        self.target = pd.read_csv(self.directory)['Energy_difference'].to_numpy()
        self.target_mean = self.target.mean()
        self.target_std = self.target.std()
        # for visualize augmented data
        self.save_to_dir = True


    def __len__(self):
        if self.mode == "train":
            return len(self.training_index)
        elif self.mode == "online_eval":
            return len(self.validation_index)

    def __getitem__(self, idx):
        print(f"{idx} read out...")
        g, l = self.__GraphEncoder__(self.data[idx], self.label[idx])
        
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g, self.e_representation)

        if self.target_transform is not None:
            l = self.target_transform(l)
        return (np.asarray(g), h, e), l


    def __GraphEncoder__(self, smiles, label):
        # target graph
        g = init_graph_local(smiles, label, self.graph_keys, self.label_keys)
        for mols in smiles:
            m = Chem.MolFromSmiles(mols)
            m = Chem.AddHs(m) # delete H atoms

            fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            feats = factory.GetFeaturesForMol(m)

            # Create nodes
            for i in range(0, m.GetNumAtoms()):
                atom_i = m.GetAtomWithIdx(i)

                # one-hot rep for some features
                g.add_node(i, 
                NodeType="None",
                AtomSymbol = atom_i.GetSymbol(), 
                NumAtomic = atom_i.GetAtomicNum(), 
                acceptor=0, 
                donor=0,
                IsAromatic=atom_i.GetIsAromatic(), 
                FormalCharge = atom_i.GetFormalCharge(),
                NumExplicit = atom_i.GetNumExplicitHs(),
                NumImplicit = atom_i.GetNumImplicitHs(),
                ChiralTag = atom_i.GetChiralTag(),
                Hybridization = atom_i.GetHybridization(),
                TotalNum=atom_i.GetTotalNumHs()) # what does it mean??

            for i in range(0, len(feats)):
                if feats[i].GetFamily() == 'Donor':
                    node_list = feats[i].GetAtomIds()
                    for i in node_list:
                        g._node[i]['donor'] = 1
                elif feats[i].GetFamily() == 'Acceptor':
                    node_list = feats[i].GetAtomIds()
                    for i in node_list:
                        g._node[i]['acceptor'] = 1
            
            # Read Edges
            for i in range(0, m.GetNumAtoms()):
                for j in range(0, m.GetNumAtoms()):
                    e_ij = m.GetBondBetweenAtoms(i, j)
                    if e_ij is not None:
                        g.add_edge(i, j, BondType=e_ij.GetBondType(), StereoType=e_ij.GetStereo())
                    else:
                        # Unbonded
                        g.add_edge(i, j, BondType=None, StereoType=None)
        return g , label 

    # XYZ file reader for QM9 dataset
    def xyz_graph_reader(self, graph_file):

        with open(graph_file,'r') as f:
            # Number of atoms
            na = int(f.readline())

            # Graph properties
            properties = f.readline()
            g, l = init_graph_local(properties)
            
            atom_properties = []
            # Atoms properties
            for i in range(na):
                a_properties = f.readline()
                a_properties = a_properties.replace('.*^', 'e')
                a_properties = a_properties.replace('*^', 'e')
                a_properties = a_properties.split()
                atom_properties.append(a_properties)

            # Frequencies
            f.readline()

            # SMILES
            smiles = f.readline()
            smiles = smiles.split()
            smiles = smiles[0]
            
            m = Chem.MolFromSmiles(smiles)
            m = Chem.AddHs(m)

            fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            feats = factory.GetFeaturesForMol(m)

            # Create nodes
            for i in range(0, m.GetNumAtoms()):
                atom_i = m.GetAtomWithIdx(i)

                g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                        aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                        num_h=atom_i.GetTotalNumHs(), coord=np.array(atom_properties[i][1:4]).astype(np.float),
                        pc=float(atom_properties[i][4]))

            for i in range(0, len(feats)):
                if feats[i].GetFamily() == 'Donor':
                    node_list = feats[i].GetAtomIds()
                    for i in node_list:
                        g.nodes[i]['donor'] = 1
                elif feats[i].GetFamily() == 'Acceptor':
                    node_list = feats[i].GetAtomIds()
                    for i in node_list:
                        g.nodes[i]['acceptor'] = 1

            # Read Edges
            for i in range(0, m.GetNumAtoms()):
                for j in range(0, m.GetNumAtoms()):
                    e_ij = m.GetBondBetweenAtoms(i, j)
                    if e_ij is not None:
                        g.add_edge(i, j, BondType=e_ij.GetBondType(), StereoType=e_ij.GetStereo(),
                                distance=np.linalg.norm(g.nodes[i]['coord']-g.nodes[j]['coord']))
                    else:
                        # Unbonded
                        g.add_edge(i, j, BondType=None, StereoType=e_ij.GetStereo(),
                                distance=np.linalg.norm(g.nodes[i]['coord'] - g.nodes[j]['coord']))
        return g , l

    @classmethod
    def csv2xyz(cls, csvpath: str, xyzpath: Path, **kwargs) -> Path:
        xyzfolder = xyzpath.joinpath(f"{Path(Path(csvpath).parts[-1]).stem}-{uuid.uuid4()}")
        if os.path.exists( xyzfolder ) is False:
            os.makedirs(xyzfolder)

        # generate random id for new generation 
        with open(csvpath, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for i, row in enumerate(csv_reader):
                # new .xyz for i-th reaction
                if i == 0:
                    continue
                elif i % 10 == 0:
                    print("counter: ", i)
                # else:
                file_path = Path(xyzfolder).joinpath("reaction-{:04d}.xyz".format(i))
                content = ""
                content += "{:06d}\n{:s}\t{:s}\t{:s}\t{:s}\n".format(i, row[1], row[2], row[3], row[4])
                for mols in row[1:5]:
                    try:
                        coor, elements = get_coords_from_smiles(mols, "suffix", "rdkit")
                    except:
                        # print(f"{i} reactions has no 3d structure,  try again...")
                        coor, elements = get_coords_from_smiles(mols, "suffix", "any")
                        # continue
                    content += "{:d}\n".format(len(coor))
                    for koords, atoms in zip(coor, elements):
                        content +="{:s}\t{:06f}\t{:06f}\t{:06f}\n".format(atoms, koords[0], koords[1], koords[2])
                    content += "\n"
                content+="{:s}\n".format(row[-2])
                with open(file_path, 'w+') as xyz_file:
                    xyz_file.write(content)
                print(f"{file_path} saved.")
        del mols, coor, elements, koords, atoms, content, file_path, i, row
        return xyzfolder

    def __target_mean__(self):
        return self.target_mean

    def __target_std__(self):
        return self.tatget_std



class XYZ_Loader(Dataset):
    def __init__(self, args, mode, vertex_transform=qm9_nodes, edge_transform=qm9_edges,
                 target_transform=None, e_representation='raw_distance'):

        self.mode = mode
        self.directory = args.datasetPath
        self.gpu = args.gpu
        self.shuffle = True
        self.batch_index = 0
        self.split = 1
        self.seed = None
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation

        # encode important dataset information
        self.datalist = [Path(self.directory).joinpath(xyz) for xyz in os.listdir(self.directory)]

        self.len = len(self.datalist)            
        logger.info(f"Reading {self.len} reactions from file {self.directory} ...")

        # for visualize augmented data
        self.save_to_dir = False

        if self.split is not None:
            self.validation_index = np.arange(0, int(self.len*self.split)) # keep invariant in train
            self.training_index = np.arange(int(self.len*self.split)+1, self.len)
            logger.info(f"{len(self.validation_index)} molecules are used as validation set...")
            logger.info(f"{len(self.training_index)} molecules are used as training set...")

    def __len__(self):
        if self.mode == "train":
            return len(self.training_index)
        elif self.mode == "online_eval":
            return len(self.validation_index)

    def __getitem__(self, idx):
        if self.mode == "train":
            idx += len(self.validation_index)
        print(f"{idx} read out...")
        g, l = self.xyz_graph_reader(self.datalist[idx])
        g = self.add_supernode(g)
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g, self.e_representation)

        if self.target_transform is not None:
            l = self.target_transform(l)
        return (np.asarray(g), h, e), l

    # XYZ file reader for QM9 dataset
    def xyz_graph_reader(self, graph_file):

        with open(graph_file,'r') as f:
            # Number of atoms
            na = int(f.readline())

            # SMILES
            mols = f.readline().split()
            
            # Atoms properties
            atom_properties1 = []
                        
            n1 = int(f.readline())
            for i in range(n1):
                a_properties = f.readline()
                a_properties = a_properties.split()
                atom_properties1.append(a_properties)

            f.readline()
            atom_properties2 = []
            n2  = int(f.readline())
            for i in range(n2):
                a_properties = f.readline()
                a_properties = a_properties.split()
                atom_properties2.append(a_properties)

            f.readline()
            atom_properties3 = []
            n3  = int(f.readline())
            for i in range(n3):
                a_properties = f.readline()
                a_properties = a_properties.split()
                atom_properties3.append(a_properties)

            # Target: energy difference
            f.readline()
            target = f.readline().split()[0]

            g_list = []
            properties = [atom_properties1, atom_properties2, atom_properties3]
            
            for smiles, atom_properties in zip(mols, properties):
                g = nx.Graph(target=target)
                m = Chem.MolFromSmiles(smiles)
                m = Chem.AddHs(m)

                fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
                factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
                feats = factory.GetFeaturesForMol(m)

                # Create nodes
                for i in range(0, m.GetNumAtoms()):
                    atom_i = m.GetAtomWithIdx(i)

                    g.add_node(i, 
                    NodeType="None", # all are not Supernode
                    AtomSymbol = atom_i.GetSymbol(), 
                    NumAtomic = atom_i.GetAtomicNum(), 
                    acceptor=0, 
                    donor=0,
                    IsAromatic=atom_i.GetIsAromatic(), 
                    FormalCharge = atom_i.GetFormalCharge(),
                    NumExplicit = atom_i.GetNumExplicitHs(),
                    NumImplicit = atom_i.GetNumImplicitHs(),
                    ChiralTag = atom_i.GetChiralTag(),
                    Hybridization = atom_i.GetHybridization(),
                    TotalNum=atom_i.GetTotalNumHs(),
                    coord=np.array(atom_properties[i][1:4]).astype(np.float)) 

                for i in range(0, len(feats)):
                    if feats[i].GetFamily() == 'Donor':
                        node_list = feats[i].GetAtomIds()
                        for i in node_list:
                            g.nodes[i]['donor'] = 1
                    elif feats[i].GetFamily() == 'Acceptor':
                        node_list = feats[i].GetAtomIds()
                        for i in node_list:
                            g.nodes[i]['acceptor'] = 1

                # Read Edges
                for i in range(0, m.GetNumAtoms()):
                    for j in range(0, m.GetNumAtoms()):
                        e_ij = m.GetBondBetweenAtoms(i, j)
                        if e_ij is not None:
                            g.add_edge(i, j, BondType=e_ij.GetBondType(), StereoType=e_ij.GetStereo(),
                                    distance=np.linalg.norm(g.nodes[i]['coord']-g.nodes[j]['coord']))
                        else:
                            # Unbonded
                            g.add_edge(i, j, BondType=None, StereoType=None,
                                    distance=np.linalg.norm(g.nodes[i]['coord'] - g.nodes[j]['coord']))
                g_list.append(g)
            return g_list , target

    def add_supernode(self, g):
        g = nx.disjoint_union(g[0] ,g[1])

        supernode_index = len(g.nodes())
        g.add_node(supernode_index, 
            NodeType="Supernode",
            AtomSymbol = "None", 
            NumAtomic = "None", 
            acceptor= 0, 
            donor= 0,
            IsAromatic= 0, 
            FormalCharge = 0,
            NumExplicit = 0,
            NumImplicit = 0,
            ChiralTag = "None",
            Hybridization = "None",
            TotalNum= 0,
            coord=np.zeros((1, 3))) 
        # add supernode bond type 
        for n, dn in g.nodes(data=True):
            for m, dm in g.nodes(data=True):
                if dn["NodeType"] == 'Supernode' or dm["NodeType"] == 'Supernode':
                    g.add_edge(n, m, BondType="Supernode", StereoType=None,
                        distance=np.linalg.norm(g.nodes[n]['coord']-g.nodes[m]['coord']))
        g.remove_edge(*(supernode_index, supernode_index))
        return g

    @classmethod
    def csv2xyz(cls, csvpath: str, xyzpath: str, **kwargs):
        if os.path.exists(xyzpath) is False:
            root = f"{xyzpath}-{uuid.uuid4()}"
            os.makedirs(root)
             
        # generate random id for new generation 
        with open(csvpath, 'r') as read_obj:
            csv_reader = reader(read_obj)
            for i, row in enumerate(csv_reader):
                # new .xyz for i-th reaction
                if i == 0:
                    pass
                else:
                    file_path = Path(root).joinpath("reaction-{:04d}.xyz".format(i))
                    content = ""
                    content+="{:06d}\n{:s}\t{:s}\t{:s}\n".format(i, row[0], row[1], row[2])
                    for mols in row[:3]:
                        try:
                            coor, elements = get_coords_from_smiles(mols, "suffix", "rdkit")
                        except:
                            print(f"{i} reactions has no 3d structure,  try again...")
                            coor, elements = get_coords_from_smiles(mols, "suffix", "rdkit")
                            # continue
                        content+="{:3d}\n".format(len(coor))
                        for koords, atoms in zip(coor, elements):
                            content+="{:s}\t{:06f}\t{:06f}\t{:06f}\n".format(atoms, koords[0], koords[1], koords[2])
                        content+="\n"
                    content+="{:s}\n".format(row[-2])
                    with open(file_path, 'w+') as xyz_file:
                        xyz_file.write(content)
        del mols, coor, elements, koords, atoms, content, file_path, i, row, root
        return 



