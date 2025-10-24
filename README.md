## Ancient Protein Design Scripts

A comprehensive pipeline for designing and analyzing proteins using an ancient (reduced) amino acid alphabet. This repository contains scripts for protein structure generation, sequence design with restricted amino acid sets, structural validation, as well as standard 20-amino acid designs (for comparison).

The general pipeline is:

1. Generates protein structures using RFDiffusion with scaffold guidance
2. Designs sequences with ProteinMPNN restricted to 10 amino acids (excluding N, K, Q, R, C, H, F, M, Y, W)
3. Validates designs through structural metrics (TM-score, RMSD, DSSP)
4. Predicts structures using ESMFold
5. Simulation of mutational robustness of the folds either with rectricted 10 amino acids or all 20

### Repo structure

```
├── run_design_pipeline.py # Main pipeline: structure generation & sequence design
├── analysis_pipeline.py # Filtering, structural analysis, and visualization
├── redesign_20.py # Redesign with 20 amino acids (assumes structs have been generated)
├── mutations_simulation_10E.py # Mutational robustness simulations script with 10 amino acids
├── mutations_simulation_20F.py #  Mutational robustness simulations script with 20 amino acids
├── pyproject.toml # Python dependencies
├── raw_results.zip # Folder with raw results from mutation simulations
├── statistics.zip # Statistical comparison results between groups sompared in the paper
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

##### TMalign

Download from [Zhang Lab](https://zhanggroup.org/TM-align/) and ensure it's in your PATH. This package is used in mutation simulation script, if not running these simulations, can be omitted.

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

#### 4. Simulation of mutational robustness of the fold

Generate random substitutions of AAs in original structure and predict of the mutated structure with ESMFold. Calculate RMSD and TM-score against original .pdb structure. Script mutations_simulation_10E.py is generating random substitutions of amino acids from 10 subset, mutations_simulation_20F.py uses full 20 alphabet for simulation, description below is for 10E script, but they both work with the same logic.

```bash
python mutations_simulation_10E.py \
    --pdb pdb_file.pdb \
    --fasta fasta_file.fasta \
    --outdir /path/to/output/directory \
    --n_runs 20 \
    --max_workers 1
```
Parameters:

- `--pdb`: Original PDB structure
- `--fasta`: Fasta file with the original sequence
- `--outdir`: Directory for storing the outputs
- `--n_runs`: Number of independent simulation runs
- `--max_workers`: Maximum number of parallel workers

## Output Structure

```
output_directory/
├── run_1/
│   ├── mutant_1.pdb
│   ├── mutant_2.pdb
│   ├── ...
│   └── mutation_results.csv
├── run_2/
│   ├── mutant_1.pdb
│   ├── mutant_2.pdb
│   ├── ...
│   └── mutation_results.csv
├── ...
└── aggregated_results.csv
```
### Output Files

- **Individual PDB files**: `mutant_N.pdb` - 3D structures for each mutation count
- **Per-run CSV**: `mutation_results.csv` - Detailed results for each run
- **Aggregated CSV**: `aggregated_results.csv` - Combined results from all runs

### CSV Columns

- `n_mutations`: Number of mutations introduced
- `mutations`: List of (position, original_aa, new_aa) tuples
- `mutant_sequence`: Full mutated protein sequence
- `tm_score`: TM-score from TMalign comparison
- `rmsd`: Root Mean Square Deviation from Biopython
- `run`: Run number (in aggregated results)

## Raw Results

The `raw_results/` directory contains CSV files with detailed simulation outputs from running the mutation simulation scripts on selected proteins. These files represent the raw data produced by the simulation process and include:

### File Naming Convention
Files follow the pattern: `raw_results_{Protein}_{Type}_{Alphabet}_{Runs}.csv`

- **Protein**: `Ap3`, `Fd3`, or `Rn2` (protein identifiers)
- **Type**: `Design_10`, `Design_20`, or `WT` (design variants or wild-type)
- **Alphabet**: `All_20` or `Ancient_10` (amino acid alphabet used)

### Content Structure
Each raw results file contains:
- Individual mutation results from multiple simulation runs
- TM-scores and RMSD values for structural comparisons
- Detailed mutation information (position, original amino acid, new amino acid)
- Complete mutant sequences and structural predictions

## Statistics

The `statistics/` directory contains statistical analysis files that compare simulation results across different experimental conditions:

### Available Statistical Comparisons

1. **`statistical_comparison_proteins_all_20_alphabet.csv`**
   - Statistical analysis of simulations using the full 20-amino acid alphabet

2. **`statistical_comparison_proteins_ancient_10_alphabet.csv`**
   - Statistical analysis of simulations using the limited 10-amino acid alphabet

3. **`statistical_comparison_two_alphabets.csv`**
   - Direct statistical comparison between 10-amino acid and 20-amino acid alphabet simulations


### Memory Management
- Scripts include aggressive GPU memory cleanup
- Sequential processing (`--max_workers 1`) recommended for memory-constrained systems
- Each run loads the ESMFold model independently

### Computational Requirements
- **GPU**: CUDA-compatible GPU strongly recommended
- **RAM**: 8GB+ recommended for typical proteins
- **Storage**: ~1-10MB per mutation depending on protein size
