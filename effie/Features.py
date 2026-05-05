# -*- coding: utf-8 -*-
"""

"""
import torch
import pickle
from torch.utils.data import Dataset
import json
import tqdm


### Data loaders ###

class CATHDataset:
    """
    Loader and container class for the CATH 4.2 dataset downloaded
    from http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/.

    Has attributes `self.train`, `self.val`, `self.test`, each of which are
    JSON/dictionary-type datasets as described in README.md.

    :param path: path to chain_set.jsonl
    :param splits_path: path to chain_set_splits.json or equivalent.
    """

    def __init__(self, path, splits_path):
        with open(splits_path) as f:
            dataset_splits = json.load(f)
        train_list, val_list, test_list = dataset_splits["train"], dataset_splits["validation"], dataset_splits["test"]

        self.train, self.val, self.test = [], [], []

        with open(path) as f:
            lines = f.readlines()

        for line in tqdm.tqdm(lines):
            entry = json.loads(line)
            name = entry["name"]
            coords = entry["coords"]

            entry["coords"] = list(zip(coords["N"], coords["CA"], coords["C"], coords["O"]))

            if name in train_list:
                self.train.append(entry)
            elif name in val_list:
                self.val.append(entry)
            elif name in test_list:
                self.test.append(entry)


def save_obj(obj, name, folder):
    with open(folder + "/" + name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_object(name, folder):
    with open(folder + "/" + name, "rb") as f:
        return pickle.load(f)


### To handle missing residues ####


def missing_residues(crd):

    # missing = torch.sum(torch.tensor([torch.zeros(3) in crd[i, :4] for i in range(crd.shape[0])])).item()
    nb_var = crd.shape[0]
    missing = torch.stack([torch.sum(torch.isnan(crd[i])) > 0 for i in range(nb_var)])

    return missing


def remove_N_C_ter(crd, seq, chain_idx=None, tm=None, return_cut = False):
    """
    Remove missing residues at N- and C-ter in sequence and coordinates
    """

    zero = torch.zeros((4, 3))
    cut = [[], []] #to remember which aa have been cut

    while torch.sum(torch.abs(crd[0, :4])) == 0 or torch.isnan(torch.sum(torch.abs(crd[0, :4]))):  # N-ter
        crd = crd[1:]
        cut[0].append(seq[0])
        seq = seq[1:]
        if chain_idx is not None:
            chain_idx = chain_idx[1:]
        if tm is not None:
            tm = tm[1:]

    while torch.sum(torch.abs(crd[-1, :4])) == 0 or torch.isnan(torch.sum(torch.abs(crd[-1, :4]))):  # C-ter
        crd = crd[:-1]
        cut[1].append(seq[-1])
        seq = seq[:-1]
        if chain_idx is not None:
            chain_idx = chain_idx[:-1]
        if tm is not None:
            tm = tm[:-1]

    cut[1] = [cut[1][i] for i in range(-1, -len(cut[1])-1, -1)]
    cut = [''.join(cut[0]), ''.join(cut[1])]

    if return_cut:
        return (crd, seq, chain_idx, tm, cut)
    else:
        return (crd, seq, chain_idx, tm)
    
    
def list_TM():
    
    L_TM = []
    f = open("../features/PDBTM/pdbtm_all.list")
    for l in f.readlines():
        L_TM.append(l[:4]+'.'+l.strip()[-1])
    f.close()
    
    return L_TM

    
    
### Computing dihedral angles in a parallzelized fashion ###
def normalize_batch_vector(b):
    
    nb_var = b.shape[0]
    return b/torch.linalg.norm(b, dim = 1).unsqueeze(-1).expand(nb_var, 3)

def dihedral_parallel(plane):
    
    """Praxeolitic formula (from wikipedia)
    Input: n*4*3"""

    nb_var = plane.shape[0]
    b0 = -(plane[:, 1] - plane[:, 0])
    b1 = plane[:, 2] - plane[:, 1]
    b2 = plane[:, 3] - plane[:, 2]
    b1 = normalize_batch_vector(b1)

    v = b0 - torch.matmul(b0, b1.T).diag().unsqueeze(-1).expand(nb_var, 3)*b1
    w = b2 - torch.matmul(b2, b1.T).diag().unsqueeze(-1).expand(nb_var, 3)*b1

    x = torch.matmul(v, w.T).diag()
    y = (torch.cross(b1, v)@w.T).diag()

    return torch.atan2(y, x)

def dihedral(crd):
    """
    Input backbone coordinates (torch tensor of size nb_var,3,4)
    Output Tensor of the cos and sin of phi, psi, omega angles for eash residue (size nb_var*6)
    """

    nb_var = crd.shape[0]
    crd = torch.cat((crd[:, :4], torch.zeros((1, 4, 3)).to(crd.device)))
    N, Ca, C, O = crd[:, 0].unsqueeze(1), crd[:, 1].unsqueeze(1), crd[:, 2].unsqueeze(1), crd[:, 3].unsqueeze(1)
    
    phi_plane = torch.cat((C[0:nb_var], N[1:nb_var+1], Ca[1:nb_var+1], C[1:nb_var+1]), 
                          dim = 1) #plane (C, N, Cα, C)
    psi_plane = torch.cat((N[0:nb_var], Ca[0:nb_var], C[0:nb_var], N[1:nb_var+1]), 
                          dim = 1) #plane (N, Cα, C, N)
    omega_plane = torch.cat((Ca[0:nb_var], C[0:nb_var], N[1:nb_var+1], Ca[1:nb_var+1]), 
                            dim = 1) #plane (Cα, C, N, Cα)
    
    dihedral_angles = torch.stack([dihedral_parallel(phi_plane), 
                                   dihedral_parallel(psi_plane), 
                                   dihedral_parallel(omega_plane)]).T
    dihedral_angles = torch.cat((torch.cos(dihedral_angles), torch.sin(dihedral_angles)), dim=1)
    dihedral_angles = torch.nan_to_num(dihedral_angles, nan=0.0)
    
    return dihedral_angles

def calc_angle(vector_a, vector_b):
    
    angle = torch.acos(torch.matmul(normalize_batch_vector(vector_a),
                        normalize_batch_vector(vector_b).T).diag())
    
    return torch.nan_to_num(angle, nan=0.0)

def bond_angles(crd):
    
    """
    Input coordinates Tensor (shape n*4*3) 
    Output cos, sin of bond angles (shape n*6)
    """
    
    nb_var = crd.shape[0]
    crd = torch.cat((crd[:, :4], torch.zeros((1, 4, 3)).to(crd.device)))
    
    N_Ca = crd[:nb_var, 1] - crd[:nb_var, 0]
    Ca_C = crd[:nb_var, 2] - crd[:nb_var, 1]
    C_N = crd[1:nb_var+1, 0] - crd[:nb_var, 2]

    alpha = calc_angle(-N_Ca, Ca_C) #angle N, Ca, C
    beta = calc_angle(C_N, -N_Ca) #angle C, N, Ca
    gamma = calc_angle(Ca_C, C_N) #angle Ca, C, N
    
    bond_ang = torch.stack([alpha, beta, gamma]).T
    bond_ang = torch.cat((torch.cos(bond_ang), torch.sin(bond_ang)), dim=1)
    bond_ang = torch.nan_to_num(bond_ang, nan=0.0)
    
    return bond_ang
