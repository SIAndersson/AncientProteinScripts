import argparse
import glob
import os
import re
import subprocess
import tempfile
import textwrap
from itertools import islice
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio.PDB import DSSP, PDBIO, PDBParser
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

# Constants
AA_THREE_LETTER = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]

FOLD_NAME_DICT = {
    "1": "TIM-barrel_c.1",
    "2": "Rossman-fold_c.2",
    "37": "P-fold_Hydrolase_c.37",
    "58": "Ferredoxin_d.58",
    "4": "DNA_RNA-binding_3-helical_a.4",
    "23": "Flavodoxin-like_c.23",
    "55": "Ribonuclease_H-like_motif_c.55",
    "40": "OB-fold_greek-key_b.40",
    "66": "Nucleoside_Hydrolase_c.66",
}

DSSP_SIMPLIFICATION = {
    "H": "H",
    "G": "H",
    "I": "H",
    "P": "H",  # Helices
    "E": "E",
    "B": "E",  # Strands
    "T": "C",
    "S": "C",
    "C": "C",
    "-": "C",  # Coils
}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate and analyze protein design results."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to protein evolution data directory.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=False,
        default="protein_evo",
        help="Prefix for output files.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        required=False,
        default=0,
        help="GPU device ID to use.",
    )
    parser.add_argument(
        "--usalign_path",
        type=str,
        required=False,
        default="USalign",
        help="Path to USalign executable.",
    )
    parser.add_argument(
        "--skip_structure_validation",
        action="store_true",
        help="Skip ESMFold structure validation step.",
    )
    parser.add_argument(
        "--tm_threshold",
        type=float,
        required=False,
        default=0.5,
        help="TM-score threshold for accepting structures.",
    )
    parser.add_argument(
        "--rmsd_threshold",
        type=float,
        required=False,
        default=5.0,
        help="RMSD threshold for accepting structures.",
    )

    return parser.parse_args()


# ============================================================================
# Utility Functions
# ============================================================================


def flatten_chain(matrix):
    """Flatten a nested list structure."""
    import itertools

    return list(itertools.chain.from_iterable(matrix))


def wrap_labels(ax, width, break_long_words=True):
    """Wrap x-axis labels of a matplotlib Axes object."""
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


# ============================================================================
# Structure Analysis Functions
# ============================================================================


def calculate_tm_score(usalign_path, predicted_pdb, reference_pdb):
    """
    Calculate TM-score and RMSD between two protein structures using US-align.

    Args:
        usalign_path (str): Path to USalign executable.
        predicted_pdb (str): Path to predicted structure PDB file.
        reference_pdb (str): Path to reference structure PDB file.

    Returns:
        tuple: (tm_score, rmsd) or (None, None) if calculation fails.
    """
    try:
        result = subprocess.run(
            [usalign_path, predicted_pdb, reference_pdb, "-TMscore", "0"],
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


# ============================================================================
# Secondary Structure Functions
# ============================================================================


def shorten_secondary_structure(ss):
    """Remove consecutive duplicate characters from secondary structure string."""
    shortened_ss = ""
    prev_char = ""
    for char in ss:
        if char != prev_char:
            shortened_ss += char
        prev_char = char
    return shortened_ss


def get_dssp(pdb_file, simplify=False):
    """
    Calculate secondary structure using DSSP.

    Args:
        pdb_file (str): Path to PDB file.
        simplify (bool): If True, return simplified structure with consecutive duplicates removed.

    Returns:
        str: Secondary structure string.
    """
    # Load structure
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)

    header = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"

    # Use temporary file with automatic cleanup
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_file:
        tmp_filename = tmp_file.name

        # Write header and structure to temporary file
        tmp_file.write(header)
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(tmp_file)

    try:
        # Calculate DSSP
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", tmp_filename)
        model = structure[0]
        dssp = DSSP(model, tmp_filename)

        # Extract secondary structure
        secondary_structure = "".join([dssp[key][2] for key in dssp.keys()])

        # Simplify to three classifications (H, E, C)
        simplified_string = "".join(
            DSSP_SIMPLIFICATION[symbol] for symbol in secondary_structure
        )

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

    if simplify:
        shortened_ss = shorten_secondary_structure(simplified_string)
        return shortened_ss.replace("C", "")
    else:
        return simplified_string


# ============================================================================
# Sequence Processing Functions
# ============================================================================


def get_best_sequences(path):
    """
    Extract best sequences from ProteinMPNN output files.

    Args:
        path (str): Path to directory containing .fa files or path to specific .fa file.

    Returns:
        tuple: Arrays of (sequences, names, scores).
    """
    if path.endswith(".pdb"):
        temp = path.split("/")
        path3 = "/".join(temp[:-1]) + "/seqs/" + temp[-1].replace(".pdb", ".fa")
        files = [path3]
    else:
        files = glob.glob(path + "/*.fa")

    seq = []
    score = []
    pname = []

    for fname in files:
        filename, ext = os.path.splitext(fname)
        filename = os.path.basename(fname)
        if ext == ".fa":
            pname.append(filename.replace(ext, ""))
            with open(fname) as f:
                temp_seq = []
                temp_score = []
                for line in islice(f, 2, None):
                    result = re.findall(r"global_score=\d*\.?\d*", line)
                    if len(result) == 0:
                        temp_seq.append(line.replace("\n", ""))
                    else:
                        sc = result[0].replace("global_score=", "")
                        temp_score.append(float(sc))
            seq.append(np.array(temp_seq, dtype=str))
            score.append(np.array(temp_score))

    return np.array(seq), np.array(pname), np.array(score)


def get_best_sequence_for_file(path):
    """
    Get the single best sequence from a file based on ProteinMPNN score.

    Args:
        path (str): Path to sequences.

    Returns:
        tuple: (best_sequence, best_pdb_filename)
    """
    seqs, pname, score = get_best_sequences(path)

    best_i = np.argmin(score)
    best_seq = seqs[best_i]
    best_name = pname[best_i]

    return best_seq, best_name.replace(".fa", ".pdb")


# ============================================================================
# ESMFold Structure Prediction
# ============================================================================


def initialize_esmfold_model(gpu_id=0):
    """
    Initialize ESMFold model with modern transformers API.

    Args:
        gpu_id (int): GPU device ID.

    Returns:
        tuple: (model, tokenizer)
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model = model.to(device)
    model = model.eval()

    return model, tokenizer


def esmfold_inference(sequence, model, tokenizer):
    """
    Run ESMFold inference on a sequence.

    Args:
        sequence (str): Protein sequence.
        model: ESMFold model.
        tokenizer: ESMFold tokenizer.

    Returns:
        tuple: (plddt, positions) arrays from ESMFold output.
    """
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    esm_plddt = outputs.plddt.cpu().numpy()
    positions = outputs.positions.cpu().numpy()  # Shape: [batch, seq_len, 14, 3]

    return esm_plddt, positions


def extract_ca_atoms_from_positions(positions):
    """
    Extract C-alpha atoms from ESMFold positions.
    ESMFold atom order: N, CA, C, O, CB, SG, etc.
    C-alpha is typically at index 1.

    Args:
        positions (np.ndarray): Position array from ESMFold output.

    Returns:
        np.ndarray: C-alpha coordinates, shape [seq_len, 3].
    """
    # Remove batch dimension and extract CA atoms (index 1)
    ca_coords = positions[0, :, 1, :]  # Shape: [seq_len, 3]
    return ca_coords


def write_coords_to_pdb(coords, sequence, filename):
    """
    Write coordinates to PDB format for TM-score calculation.

    Args:
        coords (np.ndarray): Coordinates array, shape [seq_len, 3].
        sequence (str): Protein sequence.
        filename (str): Output PDB filename.
    """
    # Standard amino acid three-letter codes
    aa_map = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
    }

    with open(filename, "w") as f:
        f.write("REMARK Generated by ESMFold\n")
        for i, (coord, aa) in enumerate(zip(coords, sequence)):
            aa_3letter = aa_map.get(aa, "UNK")
            f.write(
                f"ATOM  {i + 1:5d}  CA  {aa_3letter} A{i + 1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00 50.00           C\n"
            )
        f.write("END\n")


def predict_structure_esmfold(sequence, model, tokenizer, output_path):
    """
    Predict protein structure using ESMFold and save to PDB file.

    Args:
        sequence (str): Protein sequence.
        model: ESMFold model.
        tokenizer: ESMFold tokenizer.
        output_path (str): Path to save output PDB.

    Returns:
        float: Mean pLDDT score.
    """
    # Get prediction
    plddt, positions = esmfold_inference(sequence, model, tokenizer)

    # Extract C-alpha coordinates
    ca_coords = extract_ca_atoms_from_positions(positions)

    # Write to PDB file
    write_coords_to_pdb(ca_coords, sequence, output_path)

    # Calculate mean pLDDT (convert to percentage scale)
    mean_plddt = plddt.mean()

    return mean_plddt


def validate_structure_with_esmfold(path, model, tokenizer, usalign_path):
    """
    Validate a designed structure by predicting with ESMFold and comparing DSSP and TM-score.

    Args:
        path (str): Path to designed structure PDB file.
        model: ESMFold model.
        tokenizer: ESMFold tokenizer.
        usalign_path (str): Path to USalign executable.

    Returns:
        dict: Validation results including pLDDT, DSSP match, TM-score, etc.
    """
    seq, pname, score = get_best_sequences(path)

    minargs = np.argmin(score, axis=1)
    best_seqs = [seq[i, j] for i, j in enumerate(minargs)]

    results = {}

    for sequence, name in zip(best_seqs, pname):
        # Predict structure and get mean pLDDT
        plddt, positions = esmfold_inference(sequence, model, tokenizer)
        mean_plddt = plddt.mean()

        # Extract C-alpha coordinates
        ca_coords = extract_ca_atoms_from_positions(positions)

        # Save predicted structure to temporary file and compare with template
        base_pdb = str(Path(path).parent) + ".pdb"

        # Use temporary file for ESMFold prediction
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_ESMfold.pdb", delete=False
        ) as tmp_file:
            output_path = tmp_file.name
            write_coords_to_pdb(ca_coords, sequence, output_path)

        try:
            # Compare DSSP
            dssp_base = get_dssp(base_pdb, simplify=True)
            new_dssp = get_dssp(output_path, simplify=True)

            dssp_match = dssp_base.count("H") == new_dssp.count(
                "H"
            ) and dssp_base.count("E") == new_dssp.count("E")

            # Calculate TM-score and RMSD
            tm_score, rmsd = calculate_tm_score(usalign_path, output_path, base_pdb)

            # Save to permanent location with proper name
            final_output_path = path[:-5] + name + "_ESMfold.pdb"
            write_coords_to_pdb(ca_coords, sequence, final_output_path)

            results = {
                "Path": path,
                "Class": path.split("/")[-3],
                "Sequence": sequence,
                "pLDDT": mean_plddt,
                "TM_score": tm_score,
                "RMSD": rmsd,
                "ESM_path": final_output_path,
                "DSSP": "True" if dssp_match else "False",
                "Target_DSSP": dssp_base,
                "Predicted_DSSP": new_dssp,
            }
        finally:
            # Clean up temporary file
            if os.path.exists(output_path):
                os.remove(output_path)

    return results


# ============================================================================
# Data Collection and Analysis
# ============================================================================


def collect_designed_structures(data_path):
    """
    Collect all designed protein structures from the data directory.

    Args:
        data_path (str): Path to protein evolution data directory.

    Returns:
        list: List of PDB file paths that have corresponding design directories.
    """
    fold_types = list(FOLD_NAME_DICT.values())
    all_files = []

    for fold in fold_types:
        pathdir = os.path.join(data_path, fold)

        if not os.path.exists(pathdir):
            continue

        files = glob.glob(os.path.join(pathdir, "*.pdb"))
        file_exist = [
            f
            for f in files
            if os.path.exists(os.path.join(pathdir, os.path.basename(f)[:-4]))
        ]

        print(
            f"Fold {fold}: {len(file_exist)} valid designs found "
            f"({len(files) - len(file_exist)} invalid)"
        )

        all_files.append(file_exist)

    return flatten_chain(all_files)


def analyze_designed_structures(
    template_files, output_csv, usalign_path, tm_threshold=0.5, rmsd_threshold=5.0
):
    """
    Analyze designed structures using TM-score, RMSD, and DSSP metrics.

    Args:
        template_files (list): List of template PDB file paths.
        output_csv (str): Path to save results CSV.
        usalign_path (str): Path to USalign executable.
        tm_threshold (float): TM-score threshold for acceptance.
        rmsd_threshold (float): RMSD threshold for acceptance.

    Returns:
        pd.DataFrame: Analysis results.
    """
    tm_scores = []
    rmsds = []
    dssp_bool = []
    classes = []
    accept_arr = []
    filepaths = []
    og_dssp = []
    calc_dssp = []

    for template in tqdm(template_files, desc="Analyzing structures"):
        pathdir = template[:-4]
        pdbfiles = glob.glob(os.path.join(pathdir, "*.pdb"))

        dssp_base = get_dssp(template, simplify=True)

        for newpdb in pdbfiles:
            # Calculate TM-score and RMSD
            tm_score, rmsd = calculate_tm_score(usalign_path, newpdb, template)
            new_dssp = get_dssp(newpdb, simplify=True)

            # Check acceptance criteria
            dssp_match = dssp_base.count("H") == new_dssp.count(
                "H"
            ) and dssp_base.count("E") == new_dssp.count("E")

            # Accept if both DSSP matches and metrics are good
            accepted = (
                dssp_match
                and (tm_score is not None and tm_score >= tm_threshold)
                and (rmsd is not None and rmsd <= rmsd_threshold)
            )
            """
            Note that the accepted column is very heavily filtered when you use both TM-score and RMSD. In reality, just the DSSP match is considered for structures at this stage, and TM-score and RMSD become more relevant later in the analysis (the part that is mostly manual work and inspection). At this point, we just want to know if the secondary structure is preserved (indicating the correct fold), so the DSSP match is the key criterion. Predicting the structure at the later stage is more important than how close it is to the original template, as long as the overall fold is maintained. This is also an interesting thing to visualise, as it shows the difference in the different metrics and their impact on acceptance.
            """

            # Store results
            tm_scores.append(tm_score if tm_score is not None else np.nan)
            rmsds.append(rmsd if rmsd is not None else np.nan)
            dssp_bool.append("True" if dssp_match else "False")
            accept_arr.append("True" if accepted else "False")
            og_dssp.append(dssp_base)
            calc_dssp.append(new_dssp)

            namelist = newpdb.split("/")
            classes.append(namelist[-3])
            filepaths.append(newpdb)

    df = pd.DataFrame(
        {
            "TM_score": tm_scores,
            "RMSD": rmsds,
            "DSSP": dssp_bool,
            "Fold": classes,
            "Accepted": accept_arr,
            "Paths": filepaths,
            "Target_DSSP": og_dssp,
            "New_DSSP": calc_dssp,
        }
    )

    df.to_csv(output_csv, index=False)

    return df


def validate_structures_with_esmfold(df_accept, output_csv, usalign_path, gpu_id=0):
    """
    Validate accepted structures using ESMFold predictions.

    Args:
        df_accept (pd.DataFrame): DataFrame of accepted structures.
        output_csv (str): Path to save validation results.
        usalign_path (str): Path to USalign executable.
        gpu_id (int): GPU device ID.

    Returns:
        pd.DataFrame: Validation results.
    """
    model, tokenizer = initialize_esmfold_model(gpu_id)

    dssp_paths = df_accept["Paths"].to_numpy()
    validation_results = []

    for path in tqdm(dssp_paths, desc="ESMFold validation"):
        try:
            result = validate_structure_with_esmfold(
                path, model, tokenizer, usalign_path
            )
            validation_results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    df_struct = pd.DataFrame(validation_results)
    df_struct.to_csv(output_csv, index=False)

    return df_struct


# ============================================================================
# Visualization Functions
# ============================================================================


def plot_structure_analysis(df, output_prefix, tm_threshold=0.5, rmsd_threshold=5.0):
    """
    Create comprehensive plots of structure analysis results.

    Args:
        df (pd.DataFrame): Analysis results dataframe.
        output_prefix (str): Prefix for output file names.
        tm_threshold (float): TM-score threshold used for acceptance.
        rmsd_threshold (float): RMSD threshold used for acceptance.
    """
    sns.set_theme()

    # Plot 1: TM-score, RMSD distributions with acceptance
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))

    sns.histplot(data=df, x="TM_score", hue="Accepted", kde=True, ax=ax1)
    ax1.set_xlabel("TM-score")
    ax1.text(
        0.2,
        df.shape[0] * 0.4,
        f"{sum(df['TM_score'] >= tm_threshold)} accepted on TM-score",
        bbox=dict(facecolor="red", alpha=0.5),
    )

    sns.histplot(data=df, x="RMSD", hue="Accepted", kde=True, ax=ax2)
    ax2.set_xlabel("RMSD (Å)")
    ax2.text(
        df["RMSD"].quantile(0.75, interpolation="nearest"),
        df.shape[0] * 0.2,
        f"{sum(df['RMSD'] <= rmsd_threshold)} accepted on RMSD",
        bbox=dict(facecolor="blue", alpha=0.5),
    )

    sns.histplot(data=df, x="RMSD", hue="DSSP", kde=True, ax=ax3)
    ax3.set_xlabel("RMSD (Å)")
    ax3.text(
        df["RMSD"].quantile(0.75, interpolation="nearest"),
        df.shape[0] * 0.15,
        f"{sum(df['DSSP'] == 'True')} accepted on DSSP",
        bbox=dict(facecolor="green", alpha=0.5),
    )

    plt.savefig(f"{output_prefix}_TM_RMSD_distributions.pdf", bbox_inches="tight")
    plt.close()

    # Plot 2: Fold class distributions
    fig, ax = plt.subplots(2, 2, figsize=(20, 20))

    sns.histplot(data=df, x="Fold", hue="Accepted", kde=False, ax=ax[0, 0])
    sns.histplot(data=df, x="Fold", hue="DSSP", kde=False, ax=ax[0, 1])
    sns.histplot(data=df, x="TM_score", hue="Fold", kde=True, ax=ax[1, 0])
    sns.histplot(data=df, x="RMSD", hue="Fold", kde=True, ax=ax[1, 1])

    wrap_labels(ax[0, 0], 12)
    wrap_labels(ax[0, 1], 12)

    plt.savefig(f"{output_prefix}_class_distributions.pdf", bbox_inches="tight")
    plt.close()

    # Plot 3: RMSD by fold and DSSP
    g = sns.displot(data=df, x="RMSD", hue="Fold", col="DSSP")
    plt.savefig(f"{output_prefix}_RMSD_fold_distribution.pdf", bbox_inches="tight")
    plt.close()

    # Plot 4: RMSD vs TM-score scatter
    f, axs = plt.subplots(1, 2, figsize=(15, 10), gridspec_kw=dict(width_ratios=[4, 3]))
    sns.scatterplot(data=df, x="RMSD", y="TM_score", hue="Fold", ax=axs[0])
    axs[0].set_xlabel("RMSD (Å)")
    axs[0].set_ylabel("TM-score")
    sns.histplot(
        data=df, x="Fold", hue="Fold", shrink=0.8, alpha=0.8, legend=False, ax=axs[1]
    )
    wrap_labels(axs[1], 12)
    f.tight_layout()
    plt.savefig(f"{output_prefix}_RMSD_vs_TMscore_counts.pdf", bbox_inches="tight")
    plt.close()

    # Plot 5: DSSP acceptance by fold
    g = sns.displot(data=df, x="DSSP", col="Fold", facet_kws={"sharey": False})
    plt.savefig(f"{output_prefix}_accepted_folds.pdf", bbox_inches="tight")
    plt.close()


def plot_esmfold_validation(df_struct, output_prefix):
    """
    Create plots for ESMFold validation results.

    Args:
        df_struct (pd.DataFrame): ESMFold validation results.
        output_prefix (str): Prefix for output file names.
    """
    sns.set_theme()

    # pLDDT by class and DSSP
    g = sns.displot(
        data=df_struct, x="pLDDT", col="Class", row="DSSP", facet_kws={"sharey": False}
    )
    plt.savefig(f"{output_prefix}_pLDDT_validation_class.pdf", bbox_inches="tight")
    plt.close()

    # TM-score distribution for ESMFold predictions
    if "TM_score" in df_struct.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.histplot(data=df_struct, x="TM_score", hue="Class", kde=True, ax=ax1)
        ax1.set_xlabel("TM-score (ESMFold vs Template)")

        sns.scatterplot(data=df_struct, x="pLDDT", y="TM_score", hue="Class", ax=ax2)
        ax2.set_xlabel("pLDDT")
        ax2.set_ylabel("TM-score")

        plt.tight_layout()
        plt.savefig(
            f"{output_prefix}_ESMFold_TMscore_analysis.pdf", bbox_inches="tight"
        )
        plt.close()


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Main pipeline for protein design validation and analysis."""
    args = parse_arguments()

    # Stage 1: Analyze designed structures
    results_csv = f"{args.output_prefix}_results.csv"

    if not os.path.exists(results_csv):
        print("Analyzing designed structures...")
        template_files = collect_designed_structures(args.data_path)
        df = analyze_designed_structures(
            template_files,
            results_csv,
            args.usalign_path,
            args.tm_threshold,
            args.rmsd_threshold,
        )
        print(f"Analysis complete. Results saved to {results_csv}")
    else:
        print(f"Loading existing analysis from {results_csv}")
        df = pd.read_csv(results_csv)

    # Create analysis plots
    print("Generating analysis plots...")
    plot_structure_analysis(
        df, args.output_prefix, args.tm_threshold, args.rmsd_threshold
    )

    # Stage 2: ESMFold validation (optional)
    if not args.skip_structure_validation:
        validation_csv = f"{args.output_prefix}_structure_evaluation.csv"

        if not os.path.exists(validation_csv):
            print("Performing ESMFold structure validation...")
            df_accept = df[df["DSSP"] == "True"]
            df_struct = validate_structures_with_esmfold(
                df_accept, validation_csv, args.usalign_path, args.gpu_id
            )
            print(f"Validation complete. Results saved to {validation_csv}")
        else:
            print(f"Loading existing validation from {validation_csv}")
            df_struct = pd.read_csv(validation_csv)

        # Create validation plots
        print("Generating validation plots...")
        plot_esmfold_validation(df_struct, args.output_prefix)

    print("Pipeline complete!")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total structures analyzed: {len(df)}")
    print(
        f"Accepted structures (DSSP + TM-score + RMSD): {sum(df['Accepted'] == 'True')}"
    )
    print(f"DSSP matches: {sum(df['DSSP'] == 'True')}")
    print(f"TM-score ≥ {args.tm_threshold}: {sum(df['TM_score'] >= args.tm_threshold)}")
    print(f"RMSD ≤ {args.rmsd_threshold}Å: {sum(df['RMSD'] <= args.rmsd_threshold)}")
    print(f"Mean TM-score: {df['TM_score'].mean():.3f} ± {df['TM_score'].std():.3f}")
    print(f"Mean RMSD: {df['RMSD'].mean():.3f} ± {df['RMSD'].std():.3f} Å")
    print("=" * 60)


if __name__ == "__main__":
    main()
