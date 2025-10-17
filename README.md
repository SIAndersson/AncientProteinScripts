## Ancient Protein Design Scripts

A comprehensive pipeline for designing and analyzing proteins using an ancient (reduced) amino acid alphabet. This repository contains scripts for protein structure generation, sequence design with restricted amino acid sets, structural validation, as well as standard 20-amino acid designs (for comparison).

The general pipeline is:

1. Generates protein structures using RFDiffusion with scaffold guidance
2. Designs sequences with ProteinMPNN restricted to 10 amino acids (excluding N, K, Q, R, C, H, F, M, Y, W)
3. Validates designs through structural metrics (TM-score, RMSD, DSSP)
4. Predicts structures using ESMFold

### Repo structure

```
├── run_design_pipeline.py # Main pipeline: structure generation & sequence design
├── analysis_pipeline.py # Filtering, structural analysis, and visualization
├── redesign_20.py # Redesign with 20 amino acids (assumes structs have been generated)
├── pyproject.toml # Python dependencies
└── README.md # This file
```

### Installation

#### 1. Python dependencies

All dependencies for the project are managed using the package manager `uv`. Please install `uv` according to the instructions on the [website](https://docs.astral.sh/uv/getting-started/installation/).

After installing `uv`, you should be able to run `uv sync` in the root of the repository in order to install all relevant packages. You can then either activate the environment in the `.venv` folder or run the files with `uv run [script]`.

#### 2. External dependencies

This project has multiple external dependencies that will need to be installed separately.

##### RFDiffusion

For protein structure generation:

```bash
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion
# Follow installation instructions in RFdiffusion repository
```

##### ProteinMPNN

We use the version that comes with RFdiffusion. Can also be installed separately, but then requires changing the paths in the scripts.

##### USalign

For TM-score and RMSD calculations. Install using the [Github instructions](https://github.com/pylelab/USalign).

##### DSSP (mkdssp)

Used for protein secondary structure analysis. Please note that there are many different versions of DSSP. Use whatever you have available. The most recent version of DSSP can be found [here](https://github.com/PDB-REDO/dssp).

#### 3. Database requirements

Download the SCOPe database:

- SCOPe structures: [https://scop.berkeley.edu/downloads/](https://scop.berkeley.edu/downloads/)
- ASTRAL FASTA: Required for sequence extraction

### Usage

#### 1. Design pipeline

Generate protein structures and design sequences with restricted amino acids:

```bash
python run_design_pipeline.py \
    --scope_database /path/to/scope/database/ \
    --out_path /path/to/output/ \
    --astral_path /path/to/astral.fa \
    --rfdiffusion_path /path/to/RFdiffusion
```

Parameters:

- `--scope_database`: Path to SCOPe structure database
- `--out_path`: Output directory for generated structures and sequences
- `--astral_path`: Path to ASTRAL FASTA file for sequence extraction
- `--rfdiffusion_path`: Path to RFdiffusion repository

#### 2. Analysis pipeline

Validate and filter designed structures:

```bash
python analysis_pipeline.py \
    --data_path /path/to/protein/evolution/data/ \
    --output_prefix protein_evo \
    --gpu_id 0 \
    --usalign_path USalign \
    --tm_threshold 0.5 \
    --rmsd_threshold 5.0
```

Parameters:

- `--data_path`: Path to directory containing designed structures
- `--output_prefix`: Prefix for output files and plots
- `--gpu_id`: GPU device ID for ESMFold predictions
- `--usalign_path`: Path to USalign executable
- `--tm_threshold`: TM-score threshold for acceptance (default: 0.5)
- `--rmsd_threshold`: RMSD threshold in Ångströms (default: 5.0)
- `--skip_structure_validation`: Skip ESMFold validation step

#### 3. Redesign with 20 amino acids

Redesign accepted structures using all 20 amino acids for comparison:

```bash
python redesign_20.py \
    --input_csv protein_evo_results.csv \
    --output_dir /path/to/redesign/output/ \
    --rfdiffusion_path /path/to/RFdiffusion
```

This uses the csv generated from the previous steps to access the templates. In the end, only templates that have the correct secondary structure are considered for the comparison in order to keep it fair.

### Output data structure

```
output_path/
├── TIM-barrel_c.1/
│   ├── d1a53a_.pdb                 # Template structure
│   ├── d1a53a_/                    # Design directory
│   │   ├── test_d1a53a_*.pdb       # Generated structures
│   │   ├── seqs/                   # ProteinMPNN sequences
│   │   │   └── *.fa                # FASTA files with scores
│   │   └── *_ESMfold.pdb          # ESMFold predictions
│   └── SStruct/                    # Secondary structure files
├── Rossman-fold_c.2/
│   └── ...
└── ...
```
