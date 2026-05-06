import torch
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List
from effie.Load_model import Load_model
from effie.utils import count_parameters, full_W

def load_effie_model(args, device):
    """Initialize and load the Effie model based on command line arguments."""
    version = getattr(args, 'version', 2)
    noise = getattr(args, 'noise', 0.0)
    model_path = getattr(args, 'model_path', 'Model/')
    
    model_name = "PLL_opti_multi" if version == 2 else "PLL_optiR_multi"
    
    if noise != 0:
        noise_suffix = "".join(str(noise).split("."))
        model_name += f"_noise{noise_suffix}"
    
    model_loader = Load_model(version=f"v{version}", multi=True, tm=False, unary=False)
    model = model_loader(model_name, model_path=model_path, device=device)
    
    print(f"Model loaded, {count_parameters(model)} parameters.")
    return model, model_name

def get_chain_starts(chain_idx: List[int]) -> List[int]:
    """Identify the starting index of each chain."""
    chains_start = []
    last_chain = -1
    for i, chain in enumerate(chain_idx):
        if chain != last_chain:
            chains_start.append(i)
            last_chain = chain
    return chains_start

def apply_bb_noise(crd: torch.Tensor, noise_std: float, device: torch.device) -> torch.Tensor:
    """Apply Gaussian noise to backbone coordinates."""
    if noise_std > 0:
        noise = torch.randn(*crd.shape).to(device)
        return crd + noise * noise_std
    return crd

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

def get_full_W(model, crd, chain_idx, thresh=15, device=None):
    """Run the model and return the full dense weight matrix."""
    if device is None:
        device = crd.device
        
    with torch.no_grad():
        W, idx_pairs = model(crd, thresh=thresh, chain_idx=chain_idx)
        W = full_W(W, idx_pairs)
        # Ensure it's in the expected shape (N, N, 400)
        nb_var = W.shape[0]
        return W.reshape(nb_var, nb_var, -1)
