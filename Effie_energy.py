# Marianne Defresne @ INSA 2022

import torch

from effie.utils import count_parameters, full_W
from effie.PDB_parser import PDB_parser
from effie.utils_design import seq_to_pred, calc_E
from effie.Load_model import Load_model

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("-f", "--filename", required=True,
	help="Name of the PDB file")
ap.add_argument("-s", "--sequence", required=True,
	help="Name of the fasta file")  
ap.add_argument("-p", "--path", type=str, default = '',
	help="Path to the PDB and fasta files (features dict will be saved here)")
ap.add_argument("-v", "--version", type=int, default = 2,
	help="Effie version (2 or 3)")
ap.add_argument("-mp", "--model_path", type=str, default = "Model/",
	help="Path to the model weights")
ap.add_argument("-n", "--noise", type=float, default = 0, 
	help="Level of noise (defaults is 0. Also 0.02 or O.2)")
args = vars(ap.parse_args())


#loading model
device = torch.device("cpu")
version, noise = args["version"], args["noise"]
model_name = "PLL_opti_multi" if version == 2 else "PLL_optiR_multi"
if noise != 0:
    model_name += "_noise" + "".join(str(noise).split("."))
unary = False
multichain = True
model_loader = Load_model(version="v"+str(version), multi = multichain, tm = False, unary = unary)
model = model_loader(model_name, model_path=args["model_path"], device=device)
print("Model loaded, ",end="", flush = True)
print(f'{count_parameters(model)} parameters')


# Loading data (if features are not yet extracted, it does it)
path = args["path"]
filename = args["filename"]
fasta = args["sequence"]
device = torch.device("cpu")

                                            
model.eval() 
parser = PDB_parser()
thresh = 15
        
with torch.no_grad():

    dico = parser(filename, path)
    crd, seq, chain_idx = dico["coordinates"], dico["int_seq"], dico["chain_idx"]
    missing = dico["missing"].to(device)
    y = seq.type(torch.LongTensor).to(device)
    nb_var_tot = crd.shape[0]
    W, idx_pairs = model(crd, thresh = thresh, chain_idx=chain_idx)
    W = full_W(W, idx_pairs)
    
    #reading fasta file
    file = open(path + fasta, 'r')
    L = file.readlines()
    file.close()  
    
    #Note: the fasta parser fails when sequence on several lines
    for i in range(len(L)):
        if L[i][0] == '>':
            l = L[i+1]
            l = l.replace("\n", "")
            seq = l.replace("-", "")
            num_chain = nb_var_tot//len(seq)
            seq *= num_chain #repeat to reach the nb of var in the PDB
            pred = seq_to_pred(seq)
            E = calc_E(W, pred).item()
            
            #E inter & intra
            num_chain_pdb = dico["num_chains"]
            nb_var_tot = crd.shape[0]
            blocks = []
            for j in range(num_chain_pdb):
                nb_var_chain = torch.sum(chain_idx==j)
                blocks.append(torch.ones((nb_var_chain, nb_var_chain)))
            blocks = torch.block_diag(*blocks).unsqueeze(-1).expand(nb_var_tot, nb_var_tot, W.shape[-1])
            W = W.reshape(nb_var_tot, nb_var_tot, -1)
            W_intra = W*blocks
            W_inter = W*(1-blocks)
            
            E_intra = calc_E(W_intra, pred).item()
            E_inter = calc_E(W_inter, pred).item()
            
            print(f'seq {1+i//2}\t score: {round(E,3)}\t inter {round(E_inter,3)}\t intra {round(E_intra,3)}')
                       
            with open(path + fasta[:-6] + '.txt', 'a') as file:
                file.write(L[i])
                file.write(f"With {model_name}, total: {E}, inter: {E_inter}, intra: {E_intra}"+ '\n')
