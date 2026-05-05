# Marianne Defresne @ INSA 2022-23
# Thomas Schiex @ INRAE 2024-26

import numpy as np
import torch
from os import listdir
import os

from effie.utils import count_parameters
from effie.Solver import make_CFN, add_hints, LR_BCD
from effie.utils_design import read_resfile, num_to_letter, amino_acids1
from effie.PDB_parser import PDB_parser
from effie.Load_model import Load_model

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("-p", "--path", type=str, required=True,
	help="Path to folder containing the PDB files (features dict will be saved here)")   
ap.add_argument("-e", "--exact", type=int, default=1, 
	help="Whether to solve exactly with toulbar2 (1) or not (0)")
ap.add_argument("-b", "--bb_noise", type=float, default=0, 
	help="Standard deviation of the noise to apply to the input backbone (default is 0)")
ap.add_argument("-v", "--version", type=int, default = 2,
	help="Effie version (2 or 3)")
ap.add_argument("-mp", "--model_path", type=str, default = "Model/",
	help="Path to the model weights")
ap.add_argument("-s", "--save", type=str, default = None,
        help="Filename of a CFN/WCSP model file")
ap.add_argument("-n", "--noise", type=float, default = 0, 
	help="Level of training noise (defaults is 0. Also 0.02 or 0.2)")
args = vars(ap.parse_args())


resfile = None
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
print("Model loaded, ",end='',flush=True)
print(f'{count_parameters(model)} parameters.')

model.eval() 
thresh = 15

parser = PDB_parser()
domains = [num_to_letter[i] for i in range(20)]
path = args["path"]

fns = []
nats = []

for filename in sorted(listdir(path)):
    # Loading data (if features are not yet extracted, it does it)     
    if filename[-4:] == '.pdb':
    
        fns.append(filename[:-4])
        dico = parser(filename, path)
        print(f'Loading file {filename}', end="",flush=True)
        crd, seq, chain_idx = dico["coordinates"], dico["int_seq"], dico["chain_idx"]
        missing = dico["missing"].to(device)
        nb_var = crd.shape[0]
        chains_start = []
        init_chain = -1
        for i,chain in enumerate(dico["chain_idx"]):
            if chain != init_chain: 
                chains_start.append(i)
                init_chain = chain
        
        if args["bb_noise"] > 0:
            noise = torch.randn(*crd.shape).to(device)
            crd += noise*args["bb_noise"]
                
        W, idx_pairs = model(crd, thresh = thresh, chain_idx=chain_idx)
        W_full = torch.zeros(nb_var, nb_var, 400).to(W.device)
        W_full[torch.arange(nb_var)[:, None], idx_pairs[None, :]]=W[torch.arange(nb_var)]
        W = W_full
                                  
        
        if filename[:-3] + "weight" in listdir(path):
            f = open(path + filename[:-3] + "weight", 'r')
            weight = float(f.readline().strip())
            f.close()
        else:
            weight = 1.0
        print(f', weight {weight}.')    
        try:
            if W_msd.shape == W.shape:
                W_msd += W*weight
                y = seq.type(torch.LongTensor).to(device)
            
            # Multi-state design with apo and holo 
            # the n residues of apo form correspond to the first n residues of the holo form
            else:
                if W_msd.shape[1] < W.shape[1]:
                    W_msd, W = W, W_msd # so that W_msd has the biggest shape (apo form)
                    y = seq.type(torch.LongTensor).to(device)

                nb_var_apo = W.shape[1]
                W_msd[torch.arange(nb_var_apo)[:, None], torch.arange(nb_var_apo)[None, :]] += W*weight

        except:
            W_msd = W*weight
            y = seq.type(torch.LongTensor).to(device)
        
        if os.path.exists(os.path.join(path,filename[:-4]+".resfile")):
            resfilename = filename[:-4]+".resfile"
            resfile = read_resfile(os.path.join(path,resfilename))

W_msd = W_msd.squeeze().detach().cpu().numpy()
y = y.flatten().detach().cpu().numpy()
nb_var = W_msd.shape[0]

### Mutable region and resfile
yl = y.tolist()
print(f'Using {fns[-1]} as native.')

# default is ALLAA
hint = [list(range(20)) for aa in yl]

if resfile:
    print(f'Using {resfilename}.')

    for pos,chain,con in zip(*resfile):
        rpos = pos-dico["start_num"]+chains_start[ord(chain)-65]
        content = con
        if con != -1:
            hint[rpos] = con
        else:
            hint[rpos] = [int(y[rpos])] 
else:
    resfile = None
    print("No resfile found. Full redesign.")

### Defining & solving the CFN
if args["exact"]==1 :

    print("Creating the Cost Function Network model.")
    Problem = make_CFN(W_msd, domains=amino_acids1)
    if resfile:
        add_hints(Problem, nb_var, hint)
    if (args["save"] is not None):
        Problem.Dump(args["save"])
    print("Exact solving with HBFS, wait a minute...")
    pred = Problem.Solve()
    pred = pred[0]
    NSR = np.sum(np.array(pred)[:nb_var] == y[:nb_var])/nb_var 

else:
    print("Approximate solving with LR-BCD.")
    NSR, pred = LR_BCD(W_msd, y, missing = None, hint = hint if resfile is not None else None, nb_pred_seq = 20, filename = 'NSR_Cb')

pred_seq = ''
for a in np.array(pred):
    pred_seq += num_to_letter[a]
E = pred[1]

filenames = "+".join(fns)
with open(path + filenames + "-design.fasta", "a") as f:
    f.write('>' + "".join(filename.split('.')[:-1])  + " ; with model " + model_name + " (NSR:" + str(np.round(NSR*100, 2)) 
            + " , E: " + str(E) + ") \n")
    f.write(pred_seq + '\n')

print(f'NSR: {np.round(NSR*100, 2)}%\tSequence: {pred_seq}')

