import torch
import torch.nn as nn
import numpy as np

from .gMLP import *
from .ResNet import *

class Features(nn.Module):
    def __init__(
        self,
        nb_kernel=16, 
        num_pos_embedding=16
    ):
        super(Features, self).__init__()
        self.nb_kernel = nb_kernel
        self.num_pos_embedding = num_pos_embedding
            
    def _grbf_encod(self, d, dmax = 20):
    
        """distance encoding with Gaussian rdf
        Input distance d with shape (nb_var*nb_var, n)
        """
        
        dmin = 0
        nb_pairs, nb_dist = d.shape
        dd = d.unsqueeze(-1).expand(nb_pairs, nb_dist, self.nb_kernel)
        dd = dd - torch.arange(dmin, dmax * (1 + 1 / (self.nb_kernel - 1)), 
                               (dmax - dmin) / (self.nb_kernel - 1)).reshape(1, -1).to(d.device)
        dd = torch.exp(-torch.pow(dd, 2))
        
        return dd
    
    def _positional_embeddings(self, d, period_range=[2, 1000]): 
    # Adapted from https://github.com/jingraham/neurips19-graph-protein-design

        frequency = torch.exp(torch.arange(0, self.num_pos_embedding, 2, dtype=torch.float32, device=d.device) 
                              * -(np.log(10000.0) / self.num_pos_embedding))
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E
        
    def forward(self, crd, max_neigh=64):
    
        """
        Compute the features for the neural net.
        Input: coordinates of the protein (tensor of shape n*4*3)
               max number of neighbours to consider (int, default 64)
        Output: the sorted distances d (tensor of shape n*max_neigh)
                the distances dist with GRBF encodding (tensor of shape n*max_neigh*400)
                the positional embedding (tensor of shape n*max_neigh*16)
                the indices of the chosen neighbours idx_pairs (tensor of shape n*max_neigh)
        """
    
        ### feature extraction ###
        b = crd[:, 1] - crd[:, 0]
        c = crd[:, 2] - crd[:, 1]
        a = torch.cross(b, c, dim = -1)
        Cb = -0.58273431 * a + 0.56802827*b - 0.54067466*c + crd[:, 1]

        nb_var = crd.shape[0]
        nb_neigh = min(max_neigh, nb_var) #nb_var-1 if remove current residue

        crd = torch.cat((crd, Cb.unsqueeze(1)), dim = 1)
        #residue distance (ie distance between Ca)
        d = torch.norm(crd[:, 1].unsqueeze(1).expand(nb_var, nb_var, 3) 
             - crd[:, 1].unsqueeze(1).expand(nb_var, nb_var, 3).transpose(0,1), dim = -1)

        d[torch.arange(nb_var), torch.arange(nb_var)]=-1
             
        d, idx = torch.sort(d, dim = -1) #range neighbour from closest to furthest
        d[:,0] = 0
        idx_pairs = idx[:, :nb_neigh] #remove the current residue(first of the list)

        #For each pair of residue, distances between N, Ca, C, O and Cb (25 distances)
        #sort before computing distances:
        #instead of computing nb_var*nb_var*25 distances, only nb_var*seq_len*25
        current_res = crd[torch.arange(nb_var)].unsqueeze(
            -2).expand(nb_var,5,5,3).unsqueeze(1).expand(nb_var,nb_neigh,5,5,3)
        neigh_res = crd[idx_pairs[torch.arange(nb_var)]].unsqueeze(-2).expand(
            nb_var,nb_neigh,5,5,3).transpose(2,3)
        dist = torch.norm(current_res-neigh_res, dim = -1)
        dist = self._grbf_encod(dist.reshape(nb_var*nb_neigh, 25)).reshape(nb_var, nb_neigh, -1)
        dist = torch.nan_to_num(dist, 0.0) #nan are at the end of the sort
        d = torch.nan_to_num(d, 0.0)
        d = d[:, 0:nb_neigh]

        #positional encoding
        pos = self._positional_embeddings(torch.abs(
            torch.arange(nb_var).unsqueeze(1).expand(nb_var, nb_neigh).to(crd.device)-idx_pairs))
        
        return d, dist, pos, idx_pairs


class CbNet_v3(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        depth_gMLP=3,
        ff_mult=2,
        seq_len=32,
        gMLP_output_dim = 32,
        hidden=256,  # for resnet
        nblocks=10,
        block_size=2,
        nb_it = 6,
        nb_aa=20,
        multichain = False,
        unary = False,
        nblocks_unary=2,
        transmembrane = False
    ):

        super().__init__()

        self.nb_kernel = 16
        self.num_pos_embedding = 16
        self.seq_len = seq_len
        self.nb_aa = nb_aa
        self.multichain = multichain
        self.unary = unary
        self.transmembrane = False
        
        self.extract_ft_from_crd = Features(nb_kernel = self.nb_kernel, 
                                            num_pos_embedding = self.num_pos_embedding)
        
        self.input_dim = (25*self.nb_kernel+self.num_pos_embedding) #+self.multichain

        self.embed_dim = embed_dim
 
        #"""
        self.input_resnet = ResMLP(self.embed_dim,
                        self.input_dim,
                        hidden_size=hidden,
                        nblocks=nblocks,
                        block_size=block_size,
                        BatchNorm=False)
        #"""

        #self.input_layer = nn.Linear(self.input_dim, self.embed_dim)
        #self.input_layer.apply(weights_init)

        self.input_resnet1 = ResMLP(
                        self.embed_dim,
                        3*self.embed_dim,
                        hidden_size=hidden,
                        nblocks=nblocks,
                        block_size=block_size,
                        BatchNorm=False)

        self.gmlp1 = gMLP(
            output_dim = self.embed_dim,
            input_dim = self.embed_dim,  # input dim
            depth=depth_gMLP,
            ff_mult=ff_mult,  # the hidden size in GMLP block is ff_mult*output_dim
            seq_len=self.seq_len,  # nb10_max is 45
            circulant_matrix=False,
        )
        self.input_size = (self.embed_dim*4 + self.nb_kernel*25 
                           + self.num_pos_embedding + self.multichain) 
            
        self.input_resnet.apply(weights_init)
        self.input_resnet1.apply(weights_init)
        self.nb_it = nb_it
        
    
        self.resnet = ResMLP(self.nb_aa ** 2,
                        self.input_size,
                        hidden_size=hidden,
                        nblocks=nblocks,
                        block_size=block_size,
                        BatchNorm=False)
        self.resnet.apply(weights_init)

        if self.unary:
            self.input_size_unary = self.embed_dim 
            self.resnet_unary = ResMLP(self.nb_aa,
                    self.input_size_unary,
                    hidden_size=hidden,
                    nblocks=nblocks_unary,
                    block_size=block_size,
                    BatchNorm=False)
        
    
    def forward(self, crd, thresh=15, missing = None, chain_idx = None, tm = None):
    
        """
        Through the neural net
        Input: coordinates (tensor n*4*3), whose features will be calculated
               thresh: distance cut-off in Angstrum to decide which pairs to keep (int or None) 
               max_neigh: max number of neighbours to consider (int, default 64)
               missing: Boolean tensor (shape n) indicating which residues have missing coordinates
               chain_idx: list or tensor indicating the chain of each residue (shape n)
               tm: list of Boolean indicating whether a residue within a membrane (len n)
               
        Output: the tensor out_complete containing all the costs functions (shape n*max_neigh*400)
                the indices of the chosen neighbours idx_pairs (tensor of shape n*max_neigh)
                if the neural net is intialized with unary = True, the tensor of unary costs (shape n*20)
        """


        
        nb_var_init = crd.shape[0]
        max_neigh = self.seq_len
        device = crd.device
        if nb_var_init < self.seq_len: #padding necessary
            pad = torch.tensor([float('nan')]*(self.seq_len-nb_var_init)*4*3, device=device).reshape(-1, 4, 3)
            crd = torch.cat((crd, pad))
            nb_var = crd.shape[0]
            
        else:
            nb_var = nb_var_init
            
        nb_neigh = min(max_neigh, nb_var)
        d, dist, pos, idx_pairs = self.extract_ft_from_crd(crd, max_neigh)  
        #Features: grbf encodding of distances & positional encodding
        raw_ft = torch.cat((dist, pos), dim=-1)
 
        if self.multichain:
            if not isinstance(chain_idx, torch.Tensor):
                chain_idx = torch.tensor(chain_idx).to(crd.device)

            if nb_var_init<nb_var:
                chain_idx = torch.cat((chain_idx, torch.tensor([float('nan')]*(nb_var-nb_var_init), 
                                                               device=device)))
            current_chain = chain_idx[torch.arange(nb_var)].unsqueeze(1).expand(nb_var, max_neigh)
            chain_indicator = (chain_idx[idx_pairs[torch.arange(nb_var)]]==current_chain).unsqueeze(-1)     

        if self.transmembrane:
            if tm is None:
                tm = torch.zeros(nb_var).to(crd.device)
            else:
                tm = torch.tensor(tm).to(crd.device)
                
            if nb_var_init<nb_var:
                tm = torch.cat((tm, torch.tensor([float('nan')]*(nb_var-nb_var_init), 
                                                               device=device)))
                
            current_tm = tm.unsqueeze(1).expand(nb_var, nb_neigh).unsqueeze(-1)
            neigh_tm = tm[idx_pairs[torch.arange(nb_var)]].unsqueeze(-1)
            tm_pair = torch.cat((current_tm, neigh_tm), dim = -1)
            ft = torch.cat((ft, tm_pair[:, :self.seq_len]), dim = -1)
        
        ### Through Neural nets ###
        ft = self.input_resnet(raw_ft[:, :self.seq_len] )
        inp = ft
        
        for i in range(self.nb_it):
            ft_var = ft[idx_pairs[:,:self.seq_len],0] 
            coordwhints = torch.cat((ft,ft_var,inp),dim=-1)
            embedding = self.input_resnet1(coordwhints)
            nft = self.gmlp1(embedding)
            ft = nft

        ### Predicting unary cost ###
        if self.unary:
            if self.transmembrane:
                unary_costs = self.resnet_unary(torch.cat((ft[:,0], tm.reshape(-1, 1)), dim = -1))
            else:
                unary_costs = self.resnet_unary(ft[:,0])

        ### Predicting binary matrices ###
        #ft = torch.nn.functional.pad(ft,(0,0,0,1))
        #Inverse of idx_pairs (all the residues to which n is a neighbour)
        inv_idx_pairs = torch.zeros((nb_var,nb_var), dtype = torch.int, device = crd.device)
        
        #print(nb_var, idx_pairs[:, :nb_var].shape)
        inv_idx_pairs[torch.arange(nb_var, device = crd.device)[:,None],idx_pairs[:, :nb_var]] = torch.arange(
            nb_neigh, dtype = torch.int, device = crd.device)[None,:]


        current_env = ft[:,0].unsqueeze(1).expand(nb_var, max_neigh, ft.shape[-1])
        current_env_pair = ft[:,:self.seq_len]
        neigh_env = ft[idx_pairs,0]
        neigh_env_pair = ft[idx_pairs.long(), inv_idx_pairs[torch.arange(nb_var, device = device)[:,None],idx_pairs].long()]
        
        out = torch.cat((current_env, current_env_pair, neigh_env, neigh_env_pair, raw_ft), dim = -1)

        if self.multichain:
            out = torch.cat((out, chain_indicator), dim = -1)    
            
        if self.transmembrane:
            out = torch.cat((out, tm_pair), dim = -1)
        
        #choose which neighbours to keep: d<thresh and i<j
        # if i is neigh of j but j is not neigh of i, keep (j,i)
        idx_flat = torch.cat((
            torch.arange(nb_var, device = crd.device).unsqueeze(1).expand(nb_var, max_neigh).reshape(1, -1),
            idx_pairs.reshape(1, -1)), dim = 0)
        to_keep = (d<thresh).flatten() 
                    
        idx_flat = idx_flat[:,to_keep]
        trans = idx_flat[0]>idx_flat[1]
        idx_flat[:,trans] = idx_flat[:,trans].flip(0)
        idx_unique, inverse_indices = torch.unique(idx_flat, sorted=False, return_inverse = True,dim=1)
        idx_unique_pairs = inv_idx_pairs[idx_unique[0],idx_unique[1]]

        out = out.reshape(nb_var,max_neigh,out.shape[-1])
        out = out[idx_unique[0],idx_unique_pairs.type(torch.LongTensor)]
        out = out.reshape(-1, out.shape[-1])
        out = self.resnet(out) #through tne neural net

        out = out[inverse_indices]
        out[trans] = out[trans].reshape(-1,self.nb_aa, self.nb_aa).transpose(1,2).reshape(
            -1,self.nb_aa*self.nb_aa)
        out_complete = torch.zeros((nb_var*max_neigh,self.nb_aa* self.nb_aa),device = crd.device)
        out_complete[to_keep] = out
        out_complete = out_complete.reshape(nb_var,max_neigh,-1)
        
        out_complete[:,0] = 0
       
        ### Remove prediction for pairs containing missing crd
        missing = torch.isnan(torch.sum(crd.reshape(-1, 4*3), dim =1))
        missing_pair = missing[idx_pairs[torch.arange(nb_var)]] #remove pair where j is missing
        missing_pair[missing] = True #remove pair where i is missing
        out_complete[missing_pair] = 0

        if nb_var_init<self.seq_len:
            out_complete = out_complete[:nb_var_init]
            idx_pairs = idx_pairs[:nb_var_init]
        if self.unary:
            unary_costs[missing] = 0
            unary_costs = unary_costs[:nb_var_init]
            return out_complete, idx_pairs, unary_costs
        else:
            return out_complete, idx_pairs
