# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn.functional as F
import numpy as np
from os import listdir
import pickle
from random import sample

        
        
def new_PLL(W, idx_pairs, y, val = False, nb_neigh = 0, missing = None, 
            tm = None, tm_weight = 1, unary_costs = None):
    
    nb_var, max_neigh, nb_aa = W.shape
    if nb_var < max_neigh:
        W=W[:, :nb_var]
        idx_pairs=idx_pairs[:, :nb_var]
        max_neigh=nb_var
    nb_aa = int(nb_aa**0.5)
    W = W.reshape(nb_var, max_neigh, nb_aa, nb_aa)

    L_cost = W[torch.arange(nb_var)[:, None], #on all the residues
               torch.arange(max_neigh)[None, :], #on all the neighbours
               :, 
               y[idx_pairs[torch.arange(nb_var)]] #true identity of each neighbours of each residue
              ]
    
    if nb_neigh >0:
        samp = sample([i for i in range(max_neigh)]*nb_var, nb_neigh*nb_var) #random choice
        samp = torch.tensor(samp).reshape(nb_var, -1).to(W.device)
        neigh = torch.ones(L_cost.shape[0], L_cost.shape[1]).to(W.device)
        neigh[torch.arange(nb_var)[:, None], samp] = 0
        L_cost *= neigh.unsqueeze(-1).expand(nb_var, max_neigh, nb_aa)
    
    costs_per_value = torch.sum(L_cost, dim=1)
    if unary_costs is not None:
        costs_per_value += unary_costs

    lsm = F.log_softmax(-costs_per_value, dim=-1)
    val_lsm = lsm[torch.arange(nb_var), y]
    #if missing is not None:
     #   val_lsm = val_lsm[~missing]
    
    if val:
        _, idx = torch.min(costs_per_value, dim=1)
        #correct = y - idx == 0

        if missing is not None:
            #missing = missing.reshape(1, -1)
            acc = torch.sum((idx[~missing] == y[~missing])) / torch.sum(~missing)
            npll = -torch.sum(val_lsm[~missing])
        else:
            acc = torch.sum(y - idx == 0) / nb_var
            npll = -torch.sum(val_lsm)
            
        return (acc, npll)
    
    if tm is None:
        return torch.sum(val_lsm)
    else: 
        return torch.sum(val_lsm[tm])*tm_penalty + torch.sum(val_lsm[~np.array(tm)])
        
        
def val_metrics(W, y):

    nb_var = y.shape[1]
    p1var = torch.stack([PLL_1term(W, y, i, val=True) for i in range(nb_var)])
    y_pred = p1var[:, 0].T
    acc = torch.sum(y - y_pred == 0) / nb_var

    PLL = -p1var[:, 1].T

    return (acc, torch.sum(PLL))




def pearson(L1, L2):
    cov = np.cov(L1, L2)
    if np.sum(cov == 0) == 4:
        return None
    else:
        pearson = cov[0, 1]/(cov[0,0]**0.5*cov[1, 1]**0.5)
        return pearson

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(obj, filename, folder = ''):
    with open(folder + filename, 'wb') as file:
        pickle.dump(obj, file)
        
def load(filename, folder = ''): 
    with open(folder + filename, 'rb') as file:
        return pickle.load(file) 
        
def load_BLOSUM():
    
    """
    Reads the .txt conaining BLOSSUM
    And returns a numpy array of shape 20*20
    """

    file = open('BLOSUM62.txt', 'r')
    L = file.readlines()
    file.close()

    i = 0
    while i < len(L):
        l = L[i]
        if l[0]=="#":
            L.remove(l)
        else:
            i+=1

    BLOSUM = []
    for l in L[1:21]:
        M = []
        for e in l[1:].strip().split(" "):
            if len(e)>0:
                M.append(int(e))
        BLOSUM.append(M)
    BLOSUM = np.array(BLOSUM)[:, :20]
    
    return BLOSUM
    
    
def full_W(W, idx_pairs):

    """
    Input the optimized representation of the CFN (W, idx pairs)
    output the full version of shape n*n*20*20
    """
    
    nb_var, max_neigh, _ = W.shape
    if nb_var<max_neigh:
        W = W[:, :nb_var]
        idx_pairs = idx_pairs[:, :nb_var]
    W_full = torch.zeros(nb_var, nb_var, 400).to(W.device)
    W_full[torch.arange(nb_var)[:, None], idx_pairs[None, :]]=W[torch.arange(nb_var)]
    W_full = W_full.reshape(nb_var, nb_var, 20, 20)
    for i in range(nb_var):
        for j in range(i, nb_var):
            if torch.sum(W_full[j,i] != W_full[i,j].T)>0:
                if torch.sum(W_full[i, j]) == 0:
                    W_full[i, j] = W_full[j,i].T.clone()
    W_full = W_full.reshape(nb_var, nb_var, 20*20)
    
    return W_full
