# Marianne Defresne @ INSA 2022-23
# Thomas Schiex @ INRAE 2024-26

import torch
from pathlib import Path
from argparse import ArgumentParser
from typing import Iterator, Tuple

from effie.utils import count_parameters, full_W
from effie.PDB_parser import PDB_parser
from effie.utils_design import seq_to_pred, calc_E
from effie.Load_model import Load_model

def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Calculate energy breakdown for a PDB and FASTA sequences.")
    parser.add_argument("-f", "--filename", required=True,
                      help="Name of the PDB file in the design folder")
    parser.add_argument("-s", "--sequence", required=True,
                      help="Name of the FASTA file in the design folder")
    parser.add_argument("-p", "--path", type=str, default="",
                      help="Path to the PDB and FASTA files folder (features dict will be saved here)")
    parser.add_argument("-v", "--version", type=int, default=2,
                      help="Effie version (2 or 3)")
    parser.add_argument("-mp", "--model_path", type=str, default="Model/",
                      help="Path to the model weights")
    parser.add_argument("-n", "--noise", type=float, default=0.0,
                      help="Level of noise (default is 0. Also 0.02 or 0.2)")
    return parser.parse_args()

def load_effie_model(args, device):
    """Initialize and load the Effie model."""
    version = args.version
    noise = args.noise
    model_name = "PLL_opti_multi" if version == 2 else "PLL_optiR_multi"
    
    if noise != 0:
        # Replicates original logic: "noise" + "".join(str(noise).split("."))
        noise_suffix = "".join(str(noise).split("."))
        model_name += f"_noise{noise_suffix}"
    
    # Configuration based on original script
    unary = False
    multichain = True
    
    model_loader = Load_model(version=f"v{version}", multi=multichain, tm=False, unary=unary)
    model = model_loader(model_name, model_path=args.model_path, device=device)
    
    print(f"Model loaded, {count_parameters(model)} parameters")
    return model, model_name

def parse_fasta(fasta_path: Path) -> Iterator[Tuple[str, str]]:
    """Robustly parse a FASTA file, handling multiline sequences."""
    header = None
    sequence_parts = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    yield header, "".join(sequence_parts)
                header = line
                sequence_parts = []
            else:
                sequence_parts.append(line.replace("-", ""))
        
        if header:
            yield header, "".join(sequence_parts)

def calculate_energy_breakdown(W, pred, chain_idx, num_chains):
    """Calculate total, intra-chain, and inter-chain energies."""
    nb_var_tot = W.shape[0]
    
    # Calculate blocks mask for intra-chain interactions
    blocks_list = []
    for j in range(num_chains):
        nb_var_chain = torch.sum(chain_idx == j)
        blocks_list.append(torch.ones((nb_var_chain, nb_var_chain)))
    
    # Create the full mask: 1 for intra, 0 for inter
    blocks = torch.block_diag(*blocks_list).unsqueeze(-1).expand(nb_var_tot, nb_var_tot, W.shape[-1])
    
    # Calculate energy components
    E_total = calc_E(W, pred).item()
    
    W_intra = W * blocks
    W_inter = W * (1 - blocks)
    
    E_intra = calc_E(W_intra, pred).item()
    E_inter = calc_E(W_inter, pred).item()
    
    return E_total, E_intra, E_inter

def main():
    args = parse_args()
    base_path = Path(args.path)
    pdb_file = base_path / args.filename
    fasta_file = base_path / args.sequence
    
    device = torch.device("cpu")
    
    # Load model
    model, model_name = load_effie_model(args, device)
    model.eval()
    
    # Load PDB and extract features
    parser = PDB_parser()
    thresh = 15
    
    with torch.no_grad():
        # dico = parser(filename, path) -> parser expects string path
        dico = parser(str(pdb_file.name), str(pdb_file.parent) + "/")
        crd = dico["coordinates"]
        chain_idx = dico["chain_idx"]
        num_chains_pdb = dico["num_chains"]
        nb_var_tot = crd.shape[0]
        
        # Get weight matrix from model
        W, idx_pairs = model(crd, thresh=thresh, chain_idx=chain_idx)
        W = full_W(W, idx_pairs)
        W = W.reshape(nb_var_tot, nb_var_tot, -1)
        
        output_file_path = base_path / f"{fasta_file.stem}.txt"
        
        # Process sequences
        for i, (header, seq) in enumerate(parse_fasta(fasta_file)):
            # Repeat sequence if necessary (original logic)
            num_repeats = nb_var_tot // len(seq)
            full_seq = seq * num_repeats
            
            pred = seq_to_pred(full_seq)
            
            E_total, E_intra, E_inter = calculate_energy_breakdown(W, pred, chain_idx, num_chains_pdb)
            
            print(f"seq {i+1}\t score: {E_total:.3f}\t inter {E_inter:.3f}\t intra {E_intra:.3f}")
            
            with open(output_file_path, 'a') as f:
                f.write(f"{header}\n")
                f.write(f"With {model_name}, total: {E_total}, inter: {E_inter}, intra: {E_intra}\n")

if __name__ == "__main__":
    main()
