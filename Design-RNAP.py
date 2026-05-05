# Marianne Defresne @ INSA 2022

import numpy as np
import torch
from os import listdir

from effie.utils import count_parameters
from effie.Solver import make_CFN
from effie.utils_design import read_resfile, num_to_letter, symmetrize_CFN
from effie.PDB_parser import PDB_parser
from effie.Load_model import Load_model

from argparse import ArgumentParser
ap = ArgumentParser()
ap.add_argument("-p", "--path", type=str, default='RNAP-example/',
	help="Path to folder containing the PDB files (features dict will be saved here)")
ap.add_argument("-d", "--delta_E", type=float, default=3.0,
    help="Maximum score gap with the optimum score (default is 3.0)")
ap.add_argument("-m", "--max_type", type=int, default=7,
    help="Maximum number of types of amino acid that can be used in the design (default is 7)")
ap.add_argument("-b", "--bb_noise", type=float, default=0, 
	help="Standard deviation of the noise to apply to the input backbone (default is 0)")
ap.add_argument("-v", "--version", type=int, default = 2,
	help="Effie version (2 or 3)")
ap.add_argument("-mp", "--model_path", type=str, default = "Model/",
	help="Path to the model weights")
ap.add_argument("-n", "--noise", type=float, default = 0, 
	help="Level of training noise (defaults is 0. Also 0.02 or 0.2)")
args = vars(ap.parse_args())

device = torch.device("cpu")
version, noise = args["version"], args["noise"]
model_name = "PLL_opti_multi" if version == 2 else "PLL_optiR_multi"
if noise != 0:
    model_name += "_noise" + "".join(str(noise).split("."))
model_loader = Load_model(version="v"+str(version), multi = True, tm = False, unary = False)
model = model_loader(model_name, model_path=args["model_path"], device=device)
print("Model loaded, ",end='',flush=True)
print(f'{count_parameters(model)} parameters.')

model.eval()

parser = PDB_parser()
path = args["path"]

filename = "dpbbss.pdb"  # RNA polymerase double-psi beta barrel core
   
dico = parser(filename, path)
crd, seq, chain_idx = dico["coordinates"], dico["int_seq"], dico["chain_idx"]
native = seq.type(torch.LongTensor).to(device)
missing = dico["missing"].to(device)
nb_var = crd.shape[0]

if args["bb_noise"] > 0:
    noise = torch.randn(*crd.shape).to(device)
    crd += noise*args["bb_noise"]

W, idx_pairs = model(crd, thresh = 15, chain_idx=chain_idx)
W_full = torch.zeros(nb_var, nb_var, 20 * 20).to(W.device)
W_full[torch.arange(nb_var)[:, None], idx_pairs[None, :]]=W[torch.arange(nb_var)]
W = W_full

deltaE = args["delta_E"]
max_type = args["max_type"]
num_chains = dico["num_chains"]
print("Order", num_chains, "symetry assumed.")
nb_var = W.shape[1] // num_chains
var_names = ["X" + str(i) for i in range(1, nb_var + 1)]

W_sym = symmetrize_CFN(W, list("A" * num_chains))
nb_var_sym = W_sym.shape[0]
assert(nb_var_sym == nb_var), "Error in symmetrization of W"
Problem = make_CFN(W_sym.detach().cpu().numpy(), var_names=var_names, domains=[num_to_letter[i] for i in range(20)])

# Add extra Boolean variables to detect if a given amino acid type is used at least once in the design
for aa in range(20):
    var_name = "used_" + num_to_letter[aa]
    Problem.AddVariable(var_name, [0, 1])  # 0: not used, 1: used
    # Add constraints linking the new variable to the amino acid assignments
    # Forces used to be set to 1 if the given amino acid is assigned to the given variable
    pairs = [999999999 if (xi == aa and used == 0) else 0 for xi in range(20) for used in range(2)]
    for i in range(nb_var):        
        Problem.AddFunction([i, nb_var + aa], pairs)

Problem.AddGeneralizedLinearConstraint([(nb_var + aa, 1, 1) for aa in range(20)], "<=", max_type)

native = native.flatten().detach().cpu().numpy()

print("Finding best solution (if any). Wait a minute...")
solution = Problem.Solve()
if solution is None:
    print("No solution found.")
    exit()

E = solution[1]
print(f'Optimum has score E = {E}\n')
Emax = E+deltaE

Problem.SetUB(Emax)
print(f'Finding all solutions with score below {Emax}, wait a minute...')
Problem.Solve(allSolutions = 1000000)

solutions = Problem.GetSolutions()
# sort the solutions, with pairs (score, solution), with increasing score
sorted_solutions = sorted(solutions, key=lambda x: x[0])
trimed_solutions = [solution for solution in sorted_solutions if solution[0] <= Emax]
designs_seen = set()

with open(path + "".join(filename.split('.')[:-1]) + "-enum.fasta", "a") as f:
    for (score,design) in trimed_solutions:
        if tuple(design) in designs_seen:
            continue
        designs_seen.add(tuple(design))
        pred = design[:nb_var]
        NSR = np.sum(np.array(pred)[:nb_var] == native[:nb_var])/nb_var 
        pred_seq = ''.join([num_to_letter[a] for a in pred])
        print(f'Score {score}\tNSR: {np.round(NSR*100, 2)}%\tSequence: {pred_seq}:{pred_seq}') 
        f.write('>' + "".join(filename.split('.')[:-1])  + " ; with model " + model_name + " (NSR:" + str(np.round(NSR*100, 2)) + " , E: " + str(score) + ")\n")
        f.write(pred_seq + '\n')

print(f'\n{len(designs_seen)} solution(s) found.')
