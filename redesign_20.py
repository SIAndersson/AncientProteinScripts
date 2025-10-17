import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
import subprocess
import re
import itertools
from itertools import islice
from tqdm import tqdm
from Bio import PDB
from Bio.PDB import PDBParser
import argparse

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import warnings
import biotite.structure.io as bsio
import tempfile

warnings.filterwarnings("ignore")

# Set visual standards
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "axes.linewidth": 0.5,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    }
)


# Set colour palette to colorblind-friendly colours
colours = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
sns.set_palette(colours)

usalign_path = str(Path.home() / "USalign")

# ESMFold atom order mapping (14 atoms per residue)
ATOM_NAMES = [
    "N",
    "CA",
    "C",
    "O",
    "CB",
    "OG",
    "OG1",
    "CG",
    "CG1",
    "CG2",
    "CD",
    "CD1",
    "CD2",
    "CE",
]
STANDARD_ATOMS = ["N", "CA", "C", "O", "CB"]

print("Loading ESM model...")
model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
print("ESM model loaded successfully.")


def extract_csv(path):
    df = pd.read_csv(path)
    return df


def get_fresh_df(df):
    template_list = df["Path"].tolist()
    class_list = df["Class"].tolist()

    df_fresh = pd.DataFrame({"Template": template_list, "Fold": class_list})
    return df_fresh


def esm_inference(sequence=None):
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to("cuda:0")

    with torch.no_grad():
        outputs = model(**inputs)

    esm_plddt = outputs.plddt.cpu().numpy()
    positions = outputs.positions.cpu().numpy()  # Shape: [batch, seq_len, 14, 3]

    return esm_plddt, positions


def extract_ca_atoms_from_positions(positions):
    """
    Extract C-alpha atoms from ESMFold positions
    ESMFold atom order: N, CA, C, O, CB, SG, etc.
    C-alpha is typically at index 1
    """
    # Remove batch dimension and extract CA atoms (index 1)
    ca_coords = positions[0, 0, :, 1, :]  # Shape: [seq_len, 3]
    return ca_coords


def write_coords_to_pdb(coords, sequence, filename):
    """
    Write coordinates to PDB format for TM-score calculation
    """
    with open(filename, "w") as f:
        f.write("REMARK Generated structure\n")
        for i, (coord, aa) in enumerate(zip(coords, sequence)):
            f.write(
                f"ATOM  {i + 1:5d}  CA  {aa} A{i + 1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00 50.00           C\n"
            )
        f.write("END\n")


def calculate_tm_score(predicted_pdb, reference_pdb):
    """
    Calculate TM-score using TMscore executable
    You need to install TMscore: https://zhanggroup.org/TM-score/
    """
    try:
        result = subprocess.run(
            [usalign_path, predicted_pdb, reference_pdb, "-TMscore", "0"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout
        tm_match = re.search(
            r"TM-score=\s+([0-9.]+) \(normalized by length of Structure_2",
            output,
        )
        tm_score = float(tm_match.group(1)) if tm_match else None

        # Parse RMSD
        rmsd_match = re.search(r"RMSD=\s+([0-9.]+)", output)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None

        return tm_score, rmsd

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running TMscore: {e}")
        print("Make sure TMscore is installed and in PATH")
        return None


def compare_structures(sequence, reference_pdb_file):
    """
    Main function to compare predicted and reference structures
    """
    # Get prediction
    plddt, positions = esm_inference(sequence)

    # Extract C-alpha coordinates from prediction
    pred_ca_coords = extract_ca_atoms_from_positions(positions)

    # Calculate TM-score (requires external TMscore program)
    tm_score = None
    rmsd = None
    try:
        # Create temporary PDB file for predicted structure
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as tmp_pred:
            write_coords_to_pdb(pred_ca_coords, sequence, tmp_pred.name)
            tm_score, rmsd = calculate_tm_score(tmp_pred.name, reference_pdb_file)
            # print(f"TM-score: {tm_score}, RMSD: {rmsd}")
            os.unlink(tmp_pred.name)  # Clean up
    except Exception as e:
        print(f"TM-score calculation failed: {e}")

    return tm_score, rmsd, plddt.mean() * 100  # Return mean PLDDT as percentage


def get_best_seq(fasta_path):
    with open(fasta_path) as f:
        temp_seq = []
        temp_score = []
        for line in islice(f, 2, None):
            result = re.findall("global_score=\d*\.?\d*", line)
            if len(result) == 0:
                temp_seq.append(line.replace("\n", ""))
            else:
                sc = result[0].replace("global_score=", "")
                temp_score.append(float(sc))
    f.close()

    seq = np.array(temp_seq, dtype=str)
    score = np.array(temp_score)

    best_i = np.argmin(score)
    best_seq = seq[best_i]

    return best_seq, score[best_i]


def run_redesign(df, out_dir, rfdiffusion_path):
    if not Path(out_dir).exists():
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    best_seqs = []
    best_scores = []
    esm_plddts = []
    tm_scores = []
    rmsds = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Redesigning"):
        template = row["Template"]
        class_name = row["Fold"]

        odir = out_dir + "/" + class_name + "/"
        if not Path(odir).exists():
            Path(odir).mkdir(parents=True, exist_ok=True)

        # Simulate redesign process
        tqdm.write(f"Redesigning {template} for class {class_name}")

        design_command = (
            f"python {rfdiffusion_path}/sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/protein_mpnn_run.py "
            f"--num_seq_per_target=20 --batch_size=10 --out_folder={odir} --pdb_path={template}"
        )

        with open(os.path.join(odir, "design.log"), "w") as log_file:
            subprocess.run(
                design_command, shell=True, stdout=log_file, stderr=subprocess.STDOUT
            )

        # Get best sequence
        fasta_path = odir + "seqs/" + Path(template).stem + ".fa"
        best_seq, best_score = get_best_seq(fasta_path)

        # Run ESM inference
        tm_score, rmsd, plddt = compare_structures(best_seq, template)
        esm_plddts.append(plddt)
        tm_scores.append(tm_score)
        rmsds.append(rmsd)

        torch.cuda.empty_cache()

        best_seqs.append(best_seq)
        best_scores.append(best_score)

    df["Designed Sequence"] = best_seqs
    df["MPNN Score"] = best_scores
    df["ESM PLDDT"] = esm_plddts
    df["TM-score"] = tm_scores
    df["RMSD"] = rmsds

    return df


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Protein redesign pipeline with ESMFold evaluation"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file containing protein structures",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for redesign results",
    )
    parser.add_argument(
        "--rfdiffusion_path",
        type=str,
        required=True,
        help="Path to RFDiffusion installation directory",
    )
    args = parser.parse_args()
    return args.csv_path, args.out_dir, args.rfdiffusion_path


if __name__ == "__main__":
    # Parse command-line arguments
    csv_path, out_dir, rfdiffusion_path = parse_arguments()

    df = extract_csv(csv_path)
    print("Initial DataFrame:")
    print("=" * 50)
    print(df.head())

    df_fresh = get_fresh_df(df)
    print("Fresh DataFrame:")
    print("=" * 50)
    print(df_fresh.head())

    # Run redesign process
    print("\nStarting redesign process...")
    print("=" * 50)
    df_redesign = run_redesign(df_fresh, out_dir, rfdiffusion_path)

    # Save the updated DataFrame
    output_csv = Path(out_dir) / "protein_evo_results_redesign.csv"
    df_redesign.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")