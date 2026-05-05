# EffieDes, a powerful neuro-symbolic architecture for protein design

- Deep Learning to generate backbone-conditioned sequence fitness landscapes (as Potts models)
- Automated Reasoning to systematically and rigorously explore them

&rarr; retraining-free complex conditioning (multi-state, symetries, sequence composition)  
&rarr; seamless integration with other Potts models (derived from MSAs (evolution), experimental data, or physics-based decomposable energy models)

---

The EffieDes package relies on 'uv' for python dependencies management.
Please install 'uv' as follows:

`curl -LsSf https://astral.sh/uv/install.sh | sh`

or visit [Astral UV GitHub](https://github.com/astral-sh/uv) for alternate installation methods.

## Running single or multi-state design

Single and multi-state sequence design is achieved using the `Design.py` script. 
All processed files must be in the same folder:

- PDB file(s) (if several, multi-state design is performed).
- Weight files (only for multi-state-design, each must have the same name as the corresponding .pdb, with a `.weight` extension. Negative weights are possible for negative design).
- An optional resfile (`.resfile`): Rosetta-like syntax, ALLAA is default, no insertion code, one position per line, chains are assumed to be 'A', 'B',... successively in the PDB file. With no resfile, full redesign is performed.

Options:
- `-p (--path)` path of the input files folder, ends with a '/'
- `-e (--exact)` 1 (default) to run with pytoulbar2, 0 with [LR-BCD](https://github.com/ValDurante/LR-BCD) (that has to be installed first)
- `-v (--version)` Effie version (2, default, or 3)
- `-n (--noise)` level of training noise (0 by default, or 0.02 (v2) or 0.2 or 0.5 (v3))
- `-b (--bb_noise)` Standard deviation of the noise to apply to the input backbone (default is 0)
- `-mp (--model_path)` path of the model (default "/Model")
- `-s (--save)` Filename to save the CFN/WCSP model (default None) a .cfn/.wcsp suffix should be used.

### Example (bi-state design of the double-psi-beta-barrel of the RNA polymerase)

`uv run Design.py --path RNAP-example/`

Output: 

```
Model loaded, 3170640 parameters.
Loading file dpbb2.pdb, weight 0.75.
Loading file dpbbss.pdb, weight 1.0.
Using dpbbss as native.
Using dpbb2.resfile.
Creating the Cost Function Network model.
Exact solving with HBFS, wait a minute...
NSR: 54.76%     Sequence: KVIAKVKKAREEDKGKNVVRINEELMKKIGVKEGDIVEIKPVSVKAKVKKAREEDKGKNVVRINEELMKKIGVKEGDEVEMKKV
```

## Challenging constrained symmetric design (RNAP Double-Psi-Beta-Barrel with few amino-acid types)

Enumerates all sequences using less than `max_type` different types of amino acids within a `delta_E` score threshold of the optimum design satisfying this contraint.
Exploits the capacities of neuro-symbolic AI to exhaustively enumerate sequences and contrain design with complex requirements. See `Design-RNAP.py`.

Options:
- `-p (--path)` path of the PDB file (defaults to the provided RNAP-example folder)
- `-m (--max_type)` maximum number of different amino acid types used in the design
- `-d (--delta_E)` maximum difference of score with the optimum design with maximum `max_type` different amino acid types used.
- `-v (--version)` Effie version (2, default, or 3)
- `-n (--noise)` level of training noise (0 by default, or 0.02 (v2) or 0.2 or 0.5 (v3))
- `-b (--bb_noise)` Standard deviation of the noise to apply to the input backbone (default is 0)
- `-mp (--model_path)` path of the model (default "/Model")

Example to list all sequences using less than 7 different amino acid types, within 3.0 Effie score units of the constrained optimum. 

`uv run Design-RNAP.py -m 7 -d 3.0`

Output:

```
Model loaded, 3170640 parameters.
Order 2 symetry assumed.
Finding best solution (if any). Wait a minute...
Optimum has score E = -423.916

Finding all solutions with score below -420.916, wait a minute...
Score -423.916  NSR: 59.52%     Sequence: AVRARVVAAREEDRGRDAVRVDEETRARVGVEEGDVVEVRAV:AVRARVVAAREEDRGRDAVRVDEETRARVGVEEGDVVEVRAV
Score -423.891  NSR: 57.14%     Sequence: AVRARVVAAREEDRGRDAVRVDEETRRRVGVEEGDVVEVRAV:AVRARVVAAREEDRGRDAVRVDEETRRRVGVEEGDVVEVRAV
Score -423.852  NSR: 59.52%     Sequence: AVRARVVAAREEDRGRDAVRVDEETRARVGVAEGDVVEVRAV:AVRARVVAAREEDRGRDAVRVDEETRARVGVAEGDVVEVRAV
...
Score -420.919  NSR: 59.52%     Sequence: AVRARVVAAREEDRGRDAVRVDEATRRAVGVREGDVVEVRAV:AVRARVVAAREEDRGRDAVRVDEATRRAVGVREGDVVEVRAV
Score -420.918  NSR: 59.52%     Sequence: AVVARVVAAREEDRGRDAVRVDEATRARVGVAEGDTVRVEAV:AVVARVVAAREEDRGRDAVRVDEATRARVGVAEGDTVRVEAV
Score -420.917  NSR: 61.9%      Sequence: AVTARVVAARAEDRGRDAVRVDEATRRAVGVAEGDVVEVRAV:AVTARVVAARAEDRGRDAVRVDEATRRAVGVAEGDVVEVRAV

1401 solutions found.
```

## Assessing the score of sequences on a given backbone

- Create a folder containing the PDB and a text file with the sequences to rank on the target structure.
- Caution: each sequence in the sequence file should be on a single line 
- Output: a `.txt` file with the energies of each sequence

Options:
- `-p (--path)` path to the PDB and FASTA files folder
- `-f (--filename)` filename of the PDB input file
- `-s (--sequence)` filename of the FASTA input sequence
- `-v (--version)` version of Effie (2, default, or 3)
- `-n (--noise)` noise (0 by default, or 0.02 (v2) or 0.2 or 0.5 (v3))
- `-mp (--model_path)` path of the model (default "/Model")

### Application to 5 sequences of the previous design

`uv run Effie_energy.py -p RNAP-example/ -f dpbbss.pdb -s ScoreMe.fasta -v 2 -n 0`

Output:

```
Model loaded, 3170640 parameters
seq 1    score: -420.937         inter -221.465  intra -199.472
seq 2    score: -420.927         inter -222.532  intra -198.396
seq 3    score: -420.931         inter -221.997  intra -198.935
seq 4    score: -420.928         inter -221.945  intra -198.983
seq 5    score: -420.927         inter -221.517  intra -199.409
```
