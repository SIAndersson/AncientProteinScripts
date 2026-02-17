import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
import subprocess
import re
from itertools import islice
from tqdm import tqdm
import argparse
from biotite.structure import lddt
import biotite.structure.io.pdb as pdb
import shlex

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import warnings
import tempfile

warnings.filterwarnings("ignore")


def get_fresh_df(df):
    """Extract template paths and fold class labels into a clean DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Path' and 'Class' columns.

    Returns:
        pd.DataFrame: DataFrame with 'Template' and 'Fold' columns.
    """
    template_list = df["Path"].tolist()
    class_list = df["Class"].tolist()

    df_fresh = pd.DataFrame({"Template": template_list, "Fold": class_list})
    return df_fresh


def initialize_esmfold_model(gpu_id=0):
    """Load the ESMFold model and tokenizer onto the specified device.

    Args:
        gpu_id (int): CUDA device ID. Falls back to CPU if CUDA is unavailable.

    Returns:
        tuple[EsmForProteinFolding, AutoTokenizer]: Loaded model and tokenizer.
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESMFold on {device}...")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model = model.eval()

    return model, tokenizer


def esmfold_inference(sequence, model, tokenizer):
    """Run a single ESMFold forward pass for a protein sequence.

    Args:
        sequence (str): Amino acid sequence in single-letter code.
        model (EsmForProteinFolding): Loaded ESMFold model.
        tokenizer (AutoTokenizer): Corresponding ESMFold tokenizer.

    Returns:
        tuple[np.ndarray, np.ndarray, EsmForProteinFoldingOutput]: Per-residue pLDDT array,
            atom positions array of shape (8, batch, seq_len, 14, 3),
            and the raw model output object.
    """
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    esm_plddt = outputs.plddt.cpu().numpy()
    positions = outputs.positions.cpu().numpy()

    return esm_plddt, positions, outputs


def write_coords_to_pdb(model, outputs_dict, filename):
    """Write an ESMFold prediction to a PDB file.

    Args:
        model (EsmForProteinFolding): ESMFold model used for inference.
        outputs_dict (EsmForProteinFoldingOutput): Raw output from ``esmfold_inference``.
        filename (str or Path): Destination path for the output PDB file.
    """
    pdb_string = model.output_to_pdb(outputs_dict)

    # Check if pdb_string is a string or a list of strings
    if isinstance(pdb_string, list):
        pdb_string = pdb_string[0]

    if len(pdb_string) == 0:
        print("No PDB string generated")
        return

    with open(filename, "w") as f:
        f.write(pdb_string)
        

def calculate_lddt(predicted_pdb, reference_pdb):
    """Calculate the lDDT score between a predicted and a reference structure.

    Only Cα atoms are compared to handle structures with different atom counts.

    Args:
        predicted_pdb (str or Path): Path to the predicted structure PDB file.
        reference_pdb (str or Path): Path to the reference structure PDB file.

    Returns:
        float | None: lDDT score on a 0–1 scale, or None if calculation fails.
    """
    try:
        with warnings.catch_warnings():
            # Ignore Biotite warnings about atom guessing
            warnings.filterwarnings("ignore", message=".*elements were guessed.*")
            # Load structures using Biotite
            ref_file = pdb.PDBFile.read(reference_pdb)
            pred_file = pdb.PDBFile.read(predicted_pdb)

            # Get atom arrays
            ref_structure = ref_file.get_structure(model=1)
            pred_structure = pred_file.get_structure(model=1)

        # Filter to CA atoms only to ensure matching atom counts
        # Sometimes predicted models have an atom mismatch
        ref_ca = ref_structure[ref_structure.atom_name == "CA"]
        pred_ca = pred_structure[pred_structure.atom_name == "CA"]

        # Verify we have matching numbers of CA atoms
        if len(ref_ca) != len(pred_ca):
            print(
                f"Warning: Mismatched CA atom counts - ref: {len(ref_ca)}, pred: {len(pred_ca)}"
            )
            return None

        # Calculate global lDDT score on CA atoms
        lddt_score = lddt(ref_ca, pred_ca)

        return lddt_score

    except Exception as e:
        print(f"Error calculating LDDT: {e}")
        return None


def calculate_tm_score(usalign_path, predicted_pdb, reference_pdb):
    """Calculate TM-score and RMSD between two structures using USalign.

    The TM-score is normalised by the length of the reference structure
    (Structure_2 in USalign convention).

    Args:
        usalign_path (str or Path): Path to the USalign executable.
        predicted_pdb (str or Path): Path to the predicted structure PDB file.
        reference_pdb (str or Path): Path to the reference structure PDB file.

    Returns:
        tuple[float | None, float | None]: (tm_score, rmsd), or (None, None)
            if USalign fails or is not found.
    """
    try:
        result = subprocess.run(
            [str(usalign_path), str(predicted_pdb), str(reference_pdb), "-TMscore", "0"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = result.stdout

        # Parse TM-score (normalized by length of Structure_2, i.e., reference)
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
        print(f"Error running USalign: {e}")
        print("Make sure USalign is installed and in PATH")
        return None, None


def compare_structures(sequence, reference_pdb_file, usalign_path, model, tokenizer):
    """Predict a structure with ESMFold and compare it to a reference.

    Args:
        sequence (str): Amino acid sequence to fold.
        reference_pdb_file (str or Path): Path to the reference PDB file.
        usalign_path (str or Path): Path to the USalign executable.
        model (EsmForProteinFolding): Loaded ESMFold model.
        tokenizer (AutoTokenizer): ESMFold tokenizer.

    Returns:
        tuple[float | None, float | None, float, float | None]: A tuple of:
            - tm_score: TM-score between predicted and reference structures.
            - rmsd: RMSD in Angstroms.
            - plddt: Mean pLDDT score scaled to 0-100.
            - lddt: lDDT score on Calpha atoms (0-1 scale).
    """
    # Get prediction
    plddt, positions, model_outputs = esmfold_inference(sequence, model, tokenizer)

    # Calculate TM-score
    tm_score = None
    rmsd = None
    try:
        # Create temporary PDB file for predicted structure
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdb", delete=False
        ) as tmp_pred:
            tmp_pred_path = Path(tmp_pred.name)
            write_coords_to_pdb(model, model_outputs, tmp_pred_path)
            tm_score, rmsd = calculate_tm_score(usalign_path, tmp_pred_path, reference_pdb_file)
            lddt = calculate_lddt(tmp_pred_path, reference_pdb_file)
            tmp_pred_path.unlink()  # Clean up
    except Exception as e:
        print(f"TM-score calculation failed: {e}")

    return tm_score, rmsd, plddt.mean() * 100, lddt


def get_best_seq(fasta_path):
    """Return the sequence with the lowest global score from a ProteinMPNN FASTA.

    Args:
        fasta_path (str or Path): Path to the ProteinMPNN-generated FASTA file.

    Returns:
        tuple[str, float]: A tuple of:
            - best_seq: Amino acid sequence with the lowest global score.
            - best_score: Corresponding global score.
    """
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


def run_redesign(df, out_dir, rfdiffusion_path, rfdiffusion_python_path, usalign_path, model, tokenizer):
    """Run ProteinMPNN sequence redesign and ESMFold evaluation for each structure.

    For each entry in the DataFrame, runs ProteinMPNN to generate candidate sequences,
    selects the best-scoring sequence, and evaluates it with ESMFold against the
    original structure.

    Args:
        df (pd.DataFrame): DataFrame with 'Template' (PDB path) and 'Fold'
            (class label) columns.
        out_dir (str or Path): Root output directory; per-class subdirectories
            are created automatically.
        rfdiffusion_path (str or Path): Path to the RFDiffusion installation
            directory.
        rfdiffusion_python_path (str or Path): Path to the RFDiffusion Python
            environment.
        usalign_path (str or Path): Path to the USalign executable.
        model (EsmForProteinFolding): Loaded ESMFold model.
        tokenizer (AutoTokenizer): ESMFold tokenizer.

    Returns:
        pd.DataFrame: Input DataFrame with appended columns: 'Designed Sequence',
            'MPNN Score', 'ESM PLDDT', 'TM-score', 'RMSD', and 'lDDT'.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_seqs = []
    best_scores = []
    esm_plddts = []
    tm_scores = []
    rmsds = []
    lddts = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Redesigning"):
        template = Path(row["Template"])
        class_name = row["Fold"]

        odir = out_dir / class_name
        odir.mkdir(parents=True, exist_ok=True)

        # Print redesign progress/information
        tqdm.write(f"Redesigning {template} for class {class_name}")

        design_command = (
            f"{rfdiffusion_python_path}/bin/python {rfdiffusion_path}/sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/protein_mpnn_run.py "
            f"--num_seq_per_target=20 --batch_size=10 --out_folder={shlex.quote(str(odir))} --pdb_path={shlex.quote(str(template))}"
        )

        with open(odir / "design.log", "w") as log_file:
            subprocess.run(
                design_command, shell=True, stdout=log_file, stderr=subprocess.STDOUT
            )

        # Get best sequence
        fasta_path = odir / "seqs" / f"{template.stem}.fa"
        best_seq, best_score = get_best_seq(fasta_path)

        # Run ESM inference
        tm_score, rmsd, plddt, lddt = compare_structures(best_seq, template, usalign_path, model, tokenizer)
        esm_plddts.append(plddt)
        tm_scores.append(tm_score)
        rmsds.append(rmsd)
        lddts.append(lddt)

        torch.cuda.empty_cache()

        best_seqs.append(best_seq)
        best_scores.append(best_score)

    df["Designed Sequence"] = best_seqs
    df["MPNN Score"] = best_scores
    df["ESM PLDDT"] = esm_plddts
    df["TM-score"] = tm_scores
    df["RMSD"] = rmsds
    df["lDDT"] = lddts

    return df


def parse_arguments():
    """
    Parse command-line arguments for the redesign pipeline.
    """
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
        "--usalign_path",
        type=str,
        required=False,
        default="USalign",
        help="Path to USalign executable.",
    )
    parser.add_argument(
        "--rfdiffusion_path",
        type=str,
        required=True,
        help="Path to RFDiffusion installation directory",
    )
    parser.add_argument(
        "--rfdiffusion_python_path",
        type=str,
        required=True,
        help="Path to RFdiffusion Python environment.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        required=False,
        default=0,
        help="Default GPU device ID to use.",
    )
    args = parser.parse_args()
    return Path(args.csv_path), Path(args.out_dir), Path(args.rfdiffusion_path), Path(args.rfdiffusion_python_path), Path(args.usalign_path), args.gpu_id


if __name__ == "__main__":
    # Parse command-line arguments
    csv_path, out_dir, rfdiffusion_path, rfdiffusion_python_path, usalign_path, gpu_id = parse_arguments()

    df = pd.read_csv(csv_path)
    print("Initial DataFrame:")
    print("=" * 50)
    print(df.head())

    df_fresh = get_fresh_df(df)
    print("Fresh DataFrame:")
    print("=" * 50)
    print(df_fresh.head())

    # Initialize ESMFold model
    model, tokenizer = initialize_esmfold_model(gpu_id)

    # Run redesign process
    print("\nStarting redesign process...")
    print("=" * 50)
    df_redesign = run_redesign(df_fresh, out_dir, rfdiffusion_path, usalign_path, model, tokenizer)

    # Save the updated DataFrame
    output_csv = out_dir / "protein_evo_results_redesign.csv"
    df_redesign.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")