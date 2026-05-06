# Marianne Defresne @ INSA 2022-23
# Thomas Schiex @ INRAE 2024-26

import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser

from effie.utils import count_parameters
from effie.Solver import make_CFN
from effie.utils_design import num_to_letter, symmetrize_CFN, amino_acids1
from effie.PDB_parser import PDB_parser
from effie.Load_model import Load_model

from effie.app_utils import load_effie_model, apply_bb_noise, get_full_W

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Protein design for RNAP with symmetry and diversity constraints.")
    parser.add_argument("-p", "--path", type=str, default='RNAP-example/',
                      help="Path to folder containing the PDB files")
    parser.add_argument("-d", "--delta_E", type=float, default=3.0,
                      help="Maximum score gap with the optimum score (default is 3.0)")
    parser.add_argument("-m", "--max_type", type=int, default=7,
                      help="Maximum number of types of amino acid that can be used (default is 7)")
    parser.add_argument("-b", "--bb_noise", type=float, default=0.0,
                      help="Standard deviation of noise to apply to input backbone")
    parser.add_argument("-v", "--version", type=int, default=2,
                      help="Effie version (2 or 3)")
    parser.add_argument("-mp", "--model_path", type=str, default="Model/",
                      help="Path to the model weights")
    parser.add_argument("-n", "--noise", type=float, default=0.0,
                      help="Level of training noise (default 0)")
    return parser.parse_args()

def setup_problem(args, dico, model, device):
    """Set up the CFN problem with symmetry and amino acid type constraints."""
    crd = dico["coordinates"].to(device)
    chain_idx = dico["chain_idx"]
    
    crd = apply_bb_noise(crd, args.bb_noise, device)
    W = get_full_W(model, crd, chain_idx, thresh=15, device=device)

    num_chains = dico["num_chains"]
    print(f"Order {num_chains} symetry assumed.")
    
    # Symmetrization
    nb_var = W.shape[1] // num_chains
    var_names = [f"X{i+1}" for i in range(nb_var)]
    
    W_sym = symmetrize_CFN(W, list("A" * num_chains))
    nb_var_sym = W_sym.shape[0]
    assert nb_var_sym == nb_var, "Error in symmetrization of W"
    
    problem = make_CFN(W_sym.detach().cpu().numpy(), var_names=var_names, domains=amino_acids1)
    
    # Add extra Boolean variables to detect if a given amino acid type is used at least once
    for aa in range(20):
        var_name = f"used_{num_to_letter[aa]}"
        problem.AddVariable(var_name, [0, 1])
        # Force 'used' to 1 if aa is assigned to any variable
        pairs = [999999999 if (xi == aa and used == 0) else 0 for xi in range(20) for used in range(2)]
        for i in range(nb_var):
            problem.AddFunction([i, nb_var + aa], pairs)
            
    problem.AddGeneralizedLinearConstraint([(nb_var + aa, 1, 1) for aa in range(20)], "<=", args.max_type)
    
    return problem, nb_var

def main():
    args = parse_args()
    base_path = Path(args.path)
    device = torch.device("cpu")
    
    # Load model
    model, model_name = load_effie_model(args, device)
    model.eval()
    
    # Parse PDB
    filename = "dpbbss.pdb"
    parser = PDB_parser()
    dico = parser(filename, str(base_path) + "/")
    native = dico["int_seq"].type(torch.LongTensor).to(device).flatten().detach().cpu().numpy()
    
    # Setup CFN
    problem, nb_var = setup_problem(args, dico, model, device)
    
    # Solve for optimum
    print("Finding best solution (if any). Wait a minute...")
    opt_solution = problem.Solve()
    if opt_solution is None:
        print("No solution found.")
        return

    best_score = opt_solution[1]
    print(f"Optimum has score E = {best_score}\n")
    
    # Enumerate solutions within delta_E
    e_max = best_score + args.delta_E
    problem.SetUB(e_max)
    print(f"Finding all solutions with score below {e_max}, wait a minute...")
    problem.Solve(allSolutions=1000000)
    
    solutions = problem.GetSolutions()
    sorted_solutions = sorted(solutions, key=lambda x: x[0])
    trimmed_solutions = [sol for sol in sorted_solutions if sol[0] <= e_max]
    designs_seen = set()
    
    # Output to file
    output_fasta = base_path / f"{Path(filename).stem}-enum.fasta"
    with open(output_fasta, "a") as f:
        for score, design in trimmed_solutions:
            design_tuple = tuple(design)
            if design_tuple in designs_seen:
                continue
            designs_seen.add(design_tuple)
            
            pred = design[:nb_var]
            nsr = np.sum(np.array(pred)[:nb_var] == native[:nb_var]) / nb_var
            nsr_pct = np.round(nsr * 100, 2)
            pred_seq = "".join([num_to_letter[a] for a in pred])
            
            print(f"Score {score}\tNSR: {nsr_pct}%\tSequence: {pred_seq}:{pred_seq}")
            
            header = f">{Path(filename).stem} ; with model {model_name} (NSR:{nsr_pct} , E: {score})"
            f.write(f"{header}\n{pred_seq}\n")
            
    print(f"\n{len(designs_seen)} solution(s) found.")

if __name__ == "__main__":
    main()
