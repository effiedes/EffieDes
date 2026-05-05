# -*- coding: utf-8 -*-
"""

"""
import torch
import pandas as pd
import torch.nn as nn

class PDB_parser(nn.Module):
    def __init__(
        self,
    ):
        super(PDB_parser, self).__init__()
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12, 'X': 20}

        self.letter3to1 = {'CYS' : 'C', 'ASP' : 'D', 'SER' : 'S', 'GLN' : 'Q', 
                           'LYS' : 'K','ILE' : 'I', 'PRO' : 'P', 'THR' : 'T', 
                           'PHE' : 'F', 'ALA' : 'A', 'GLY' : 'G', 'HIS' : 'H', 
                           'GLU' : 'E', 'LEU' : 'L', 'ARG' : 'R', 'TRP' : 'W', 
                           'VAL' : 'V', 'ASN' : 'N', 'TYR' : 'Y', 'MET' : 'M',
                           'LYN' : 'K', 'XXX': 'X'}
        
    def _pdb_plouf(self, line):
        
        """ To parse a single line of a PDB"""

        atom = line[0:6]
        atom_nb = line[6:11]
        atom_type = line[12:16]
        alt_location = line[16:17]
        res_name = line[17:20]
        chain = line[21:22]
        res_seq_nb = line[22:26]
        insert_res = line[26:27]
        x = line[30:38]
        y = line[38:46]
        z = line[46:54]
        occupancy = line[54:60]
        T_factor = line[60:66]
        element_symbol = line[76:78]
        atom_charge = line[78:80]

        return [atom_nb, atom_type, res_name, res_seq_nb, x, y, z, occupancy, chain]
    
    def _PDB2df(self, filename, path):
        
        """Read PDB file and returns it as a data frame
        Extract only N, C, CA and O atoms"""
        
        file = open(path + filename, 'r')
        L = file.readlines()
        file.close()
        
        df = pd.DataFrame(
            columns=['id_atom', 'atom_type', 'AA', 'id_aa', 
                     'x', 'y', 'z', 'occupancy', 'chain'])
        for i in range(len(L)): #-1):
            if 'TER' not in  L[i] and L[i][:4] == 'ATOM':
                df = pd.concat([df, pd.Series(self._pdb_plouf(L[i]), index = df.columns).to_frame().T], 
                               ignore_index=True)

        df["id_atom"] = pd.to_numeric(df["id_atom"])
        df["id_aa"] = pd.to_numeric(df["id_aa"])
        df["x"] = pd.to_numeric(df["x"])
        df["y"] = pd.to_numeric(df["y"])
        df["z"] = pd.to_numeric(df["z"])
        df["occupancy"] = pd.to_numeric(df["occupancy"])

        atom = df['atom_type']
        atom = atom.apply(lambda x: x.strip())
        df = df[(atom == 'N') | (atom =='CA') | (atom =='C') | (atom == 'O')]
        
        return df

    def _choose_atom(self, df):

        """ In case of accupancy <1, choose the most likely atom"""
        
        nb_var = df["id_aa"].iat[-1] - df["id_aa"].iat[1] + 1
        start_num = df["id_aa"].iat[1] #numerotation in the PDB file
        chains = df["chain"].unique()
        num_chains = len(chains)
        
        for i in range(start_num,  start_num + nb_var):
            for c in chains:
                if len(df[(df["id_aa"] == i) & (df["chain"] == c)]) > 4:
                    atoms = df[df["id_aa"]==i]
                    atoms["atom_type"] = atoms["atom_type"].apply(
                        lambda x: x.strip())
                    for a in ['N', 'CA', 'C', 'O']:
                        idx = atoms[atoms["atom_type"]==a].sort_values(
                            'occupancy', ascending=False).index[1:]
                        df = df.drop(idx)
        return df
    
    def _fill_missing_atoms(self, df):
        
        """ If some atoms at a position are missing, 
        the coordinates are filled with 0.
        If a full atom is missing, it will not work."""

        nb_var = df["id_aa"].iat[-1] - df["id_aa"].iat[1] + 1
        start_num = df["id_aa"].iat[1] #numerotation in the PDB file
        chains = df["chain"].unique()
        num_chains = len(chains)
        
        missing = []
        for i in range(start_num,  df["id_aa"].iat[-1] +1):
            for c in chains:
                if len(df[(df["id_aa"] == i) & (df["chain"] == c)]) > 4:

                    #Finding which atom is missing
                    missing_atom = []
                    atom = df[df["id_aa"] == i+start_num]["atom_type"]
                    atom = atom.apply(lambda x: x.strip())
                    if 'N' not in atom.unique():
                        missing_atom.append(0)
                    if 'CA' not in atom.unique():
                        missing_atom.append(1)
                    if 'C' not in atom.unique():
                        missing_atom.append(2)
                    if 'O' not in atom.unique():
                        missing_atom.append(3)

                    missing.append([i-start_num, missing_atom])

        #inserting unknown atoms at the right place
        for res, atoms in missing:
            for a in atoms:
                insert = res*4 + a
                crd = torch.cat((crd[:insert], 
                                 torch.zeros((1, 3)), crd[insert:]), dim = 0)
                
        return df
        
            
    def forward(self, filename, path):
        
        df = self._PDB2df(filename, path)
        
        if df["occupancy"].min()<1:
            df = self._choose_atom(df)
    
        crd = torch.cat((torch.Tensor(df['x'].values).reshape(-1, 1),
           torch.Tensor(df['y'].values).reshape(-1, 1),
           torch.Tensor(df['z'].values).reshape(-1, 1)), dim = 1)
        
        df = self._fill_missing_atoms(df)
        
        chains = df["chain"].unique()
        nb_var = 0
        for chain in chains:
            nb_var += len(df[df["chain"]==chain]["id_aa"].unique())
        
        if crd.shape[0] != nb_var*4:
            print("Missing atoms detected: !missing residues will be ignored!")
        crd = crd.reshape(-1, 4, 3)
    
        seq = df['AA'].values[0:-1:4]
        int_seq = torch.Tensor([self.letter_to_num[self.letter3to1[a]] for a in seq])
        seq = [self.letter3to1[a] for a in seq]
        missing = torch.zeros_like(int_seq).type(torch.BoolTensor)
        
        dict_chain, i = dict(), 0
        num_chains = df["chain"].nunique()
        for c in chains:
            dict_chain[c] = i
            i+=1
        chain_idx = torch.Tensor([dict_chain[aa] for aa in df["chain"].values[0:-1:4]])

        dico = dict()
        dico["int_seq"] = int_seq.type(torch.LongTensor)
        dico["seq"] = seq
        dico["missing"] = missing
        dico["coordinates"] = crd
        dico["chain_idx"] = chain_idx
        dico["num_chains"] = num_chains
        dico["start_num"] = df["id_aa"].iat[1]
        
        assert dico["coordinates"].shape[0] == dico["int_seq"].shape[0], "Sequence length mismatch coordinates length. Check no atom is missing"
        
        return dico
