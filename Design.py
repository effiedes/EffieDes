# Marianne Defresne @ INSA 2022-23
# Thomas Schiex @ INRAE 2024-26

import numpy as np
import torch
from pathlib import Path
from argparse import ArgumentParser

from effie.utils import count_parameters
from effie.Solver import make_CFN, add_hints, LR_BCD
from effie.utils_design import read_resfile, num_to_letter, amino_acids1
from effie.PDB_parser import PDB_parser
from effie.Load_model import Load_model

from effie.app_utils import load_effie_model, get_chain_starts, apply_bb_noise, get_full_W

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Protein design using Effie models.")
    parser.add_argument("-p", "--path", type=str, required=True,
                      help="Path to folder containing the PDB files")
    parser.add_argument("-e", "--exact", type=int, default=1,
                      help="Whether to solve exactly with toulbar2 (1) or not (0)")
    parser.add_argument("-b", "--bb_noise", type=float, default=0.0,
                      help="Standard deviation of noise to apply to input backbone")
    parser.add_argument("-v", "--version", type=int, default=2,
                      help="Effie version (2 or 3)")
    parser.add_argument("-mp", "--model_path", type=str, default="Model/",
                      help="Path to the model weights")
    parser.add_argument("-s", "--save", type=str, default=None,
                      help="Filename of a CFN/WCSP model file to save")
    parser.add_argument("-n", "--noise", type=float, default=0.0,
                      help="Level of training noise (default 0)")
    return parser.parse_args()

def process_pdb_files(path: Path, model, device, bb_noise_std: float):
    """Process all PDB files in the path and accumulate weights for MSD."""
    parser = PDB_parser()
    thresh = 15
    W_msd = None
    y_native = None
    processed_filenames = []
    chains_start = []
    dico_native = None
    resfile_info = None

    for pdb_file in sorted(path.glob("*.pdb")):
        filename = pdb_file.name
        processed_filenames.append(pdb_file.stem)
        
        print(f"Loading file {filename}", end="", flush=True)
        
        dico = parser(filename, str(path) + "/")
        crd = dico["coordinates"].to(device)
        seq = dico["int_seq"].to(device)
        chain_idx = dico["chain_idx"]
        
        # Calculate chain starts
        current_chains_start = get_chain_starts(chain_idx)
        
        crd = apply_bb_noise(crd, bb_noise_std, device)
        W = get_full_W(model, crd, chain_idx, thresh=thresh, device=device)

        # Weight handling
        weight_file = path / (pdb_file.stem + ".weight")
        weight = 1.0
        if weight_file.exists():
            with open(weight_file, 'r') as f:
                weight = float(f.readline().strip())
        print(f", weight {weight}.")

        # Accumulate for Multi-State Design
        if W_msd is None:
            W_msd = W * weight
            y_native = seq.type(torch.LongTensor).to(device)
        else:
            if W_msd.shape[1] == W.shape[1]:
                W_msd += W * weight
                y_native = seq.type(torch.LongTensor).to(device)
            elif W_msd.shape[1] < W.shape[1]:
                # Current PDB is larger, swap and update
                W_msd_new = W * weight
                nb_var_old = W_msd.shape[1]
                W_msd_new[torch.arange(nb_var_old)[:, None], torch.arange(nb_var_old)[None, :]] += W_msd
                W_msd = W_msd_new
                y_native = seq.type(torch.LongTensor).to(device)
            else:
                # Current PDB is smaller, add to top-left of W_msd
                nb_var_curr = W.shape[1]
                W_msd[torch.arange(nb_var_curr)[:, None], torch.arange(nb_var_curr)[None, :]] += W * weight

        # Original code used dico and chains_start from the last PDB processed
        dico_native = dico
        chains_start = current_chains_start
        
        # Check for resfile
        resfile_path = path / (pdb_file.stem + ".resfile")
        if resfile_path.exists():
            resfile_info = resfile_path

    if W_msd is None:
        raise ValueError(f"No PDB files found in {path}")

    W_msd = W_msd.squeeze().detach().cpu().numpy()
    y_native = y_native.flatten().detach().cpu().numpy()
    
    return W_msd, y_native, processed_filenames, chains_start, dico_native, resfile_info

def process_resfile(resfile_path: Path, y_native, dico_native, chains_start):
    """Parse resfile and create constraints (hints)."""
    if resfile_path is None:
        print("No resfile found. Full redesign.")
        return None, [list(range(20)) for _ in range(len(y_native))]

    print(f"Using {resfile_path.name}.")
    resfile_data = read_resfile(str(resfile_path))
    hint = [list(range(20)) for _ in range(len(y_native))]
    
    start_num = dico_native["start_num"]
    for pos, chain, con in zip(*resfile_data):
        # rpos = pos - start_num + offset
        # offset depends on chain letter ('A'=65, 'B'=66...)
        chain_offset = chains_start[ord(chain) - 65]
        rpos = pos - start_num + chain_offset
        
        if con != -1:
            hint[rpos] = con
        else:
            hint[rpos] = [int(y_native[rpos])]
            
    return resfile_data, hint

def solve_design(args, W_msd, y_native, hint, resfile_exists):
    """Solve the design problem using either toulbar2 or LR-BCD."""
    nb_var = W_msd.shape[0]
    
    if args.exact == 1:
        print("Creating the Cost Function Network model.")
        problem = make_CFN(W_msd, domains=amino_acids1)
        if resfile_exists:
            add_hints(problem, nb_var, hint)
        
        if args.save:
            problem.Dump(args.save)
            
        print("Exact solving with HBFS, wait a minute...")
        pred_full = problem.Solve()
        pred = pred_full[0]
        # Replicate original "bug": energy is set to the second residue index
        energy = pred[1]
        nsr = np.sum(np.array(pred)[:nb_var] == y_native[:nb_var]) / nb_var
    else:
        print("Approximate solving with LR-BCD.")
        nsr, pred = LR_BCD(W_msd, y_native, missing=None, 
                          hint=hint if resfile_exists else None, 
                          nb_pred_seq=20, filename='NSR_Cb')
        energy = pred[1]

    return nsr, pred, energy

def main():
    args = parse_args()
    base_path = Path(args.path)
    device = torch.device("cpu")
    
    # Load model
    model, model_name = load_effie_model(args, device)
    model.eval()
    
    # Process PDBs for MSD
    W_msd, y_native, filenames, chains_start, dico_native, resfile_path = process_pdb_files(
        base_path, model, device, args.bb_noise
    )
    
    # Process Resfile
    last_filename = filenames[-1]
    print(f"Using {last_filename} as native.")
    resfile_data, hint = process_resfile(resfile_path, y_native, dico_native, chains_start)
    resfile_exists = resfile_data is not None
    
    # Solve
    nsr, pred, energy = solve_design(args, W_msd, y_native, hint, resfile_exists)
    
    # Output results
    pred_seq = "".join([num_to_letter[a] for a in np.array(pred)])
    nsr_pct = np.round(nsr * 100, 2)
    
    print(f"NSR: {nsr_pct}%\tSequence: {pred_seq}")
    
    # Save to FASTA
    combined_filenames = "+".join(filenames)
    output_fasta = base_path / f"{combined_filenames}-design.fasta"
    with open(output_fasta, "a") as f:
        header = f">{last_filename} ; with model {model_name} (NSR:{nsr_pct}, E: {energy})"
        f.write(f"{header}\n{pred_seq}\n")

if __name__ == "__main__":
    main()
