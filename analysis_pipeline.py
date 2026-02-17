import argparse
import glob
import multiprocessing
import os
import re
import subprocess
import tempfile
import textwrap
import warnings
from functools import partial
from itertools import islice
from pathlib import Path

import biotite.structure.io.pdb as pdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Bio.PDB import DSSP, PDBIO, PDBParser
from biotite.structure import lddt
from tqdm import tqdm
from transformers import AutoTokenizer, EsmForProteinFolding

torch.backends.cuda.matmul.allow_tf32 = True

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
    "3": "FAD_NAD(P)-binding_domain_c.3",
    "23": "Flavodoxin-like_c.23",
    "26": "Adenine_nucleotide_alpha_hydrolase-like_c.26",
    "36": "Thiamin_diphosphate-binding_fold_c.36",
    "37": "P-fold_Hydrolase_c.37",
    "55": "Ribonuclease_H-like_motif_c.55",
    "66": "Nucleoside_Hydrolase_c.66",
    "67": "PLP-dependent_transferase-like_c.67",
    "94": "Periplasmic_binding_protein-like_II_c.94",
    "58": "Ferredoxin_d.58",
    "4": "DNA_RNA-binding_3-helical_a.4",
    "40": "OB-fold_greek-key_b.40",
    "84": "Barrel-sandwich_hybrid_b.84",
}

SHORT_FOLD_NAME_DICT = {
    "TIM-barrel_c.1": "TIM-barrel",
    "Rossman-fold_c.2": "Rossman-fold",
    "FAD_NAD(P)-binding_domain_c.3": "FAD/NAD(P)-binding",
    "Flavodoxin-like_c.23": "Flavodoxin-like",
    "Adenine_nucleotide_alpha_hydrolase-like_c.26": "Adenine-nucleotide",
    "Thiamin_diphosphate-binding_fold_c.36": "Thiamin-diphosphate",
    "P-fold_Hydrolase_c.37": "P-fold",
    "Ribonuclease_H-like_motif_c.55": "Ribonuclease-H",
    "Nucleoside_Hydrolase_c.66": "Nucleoside-H",
    "Periplasmic_binding_protein-like_II_c.94": "Periplasmic-BP-II",
    "Ferredoxin_d.58": "Ferredoxin",
    "DNA_RNA-binding_3-helical_a.4": "DNA/RNA-3helix",
    "OB-fold_greek-key_b.40": "OB-fold",
    "Barrel-sandwich_hybrid_b.84": "Barrel-sandwich",
}

DSSP_SIMPLIFICATION = {
    "H": "H",  # Helix
    "G": "H",  # 3-10 helix
    "I": "H",  # Pi helix
    "P": "H",  # Poly-proline II helix (Treat as Helix)
    "E": "E",  # Extended strand (Beta sheet/strand)
    "B": "E",  # Isolated beta-bridge residue
    "T": "C",  # Turn
    "S": "C",  # Bend
    "C": "C",  # Coil or unassigned region
    "-": "C",  # Treat "-" as Coil (unassigned or gap)
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
        help="Default GPU device ID to use.",
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


def calculate_lddt(predicted_pdb, reference_pdb):
    """
    Calculate local Distance Difference Test (lDDT) score between two structures using Biotite.
    Filters to Cα atoms only to handle cases where structures have different atom counts.

    Args:
        predicted_pdb (str): Path to predicted structure PDB file.
        reference_pdb (str): Path to reference structure PDB file.

    Returns:
        float: lDDT score (0-1 scale) or None if calculation fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*elements were guessed.*")
            # Load structures using Biotite
            ref_file = pdb.PDBFile.read(reference_pdb)
            pred_file = pdb.PDBFile.read(predicted_pdb)

            # Get atom arrays (first model)
            ref_structure = ref_file.get_structure(model=1)
            pred_structure = pred_file.get_structure(model=1)

        # Filter to CA atoms only to ensure matching atom counts
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


def initialize_esmfold_model(gpu_id=0):
    """
    Initialize ESMFold model with modern transformers API.

    Args:
        gpu_id (int): GPU device ID.

    Returns:
        tuple: (model, tokenizer)
    """
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESMFold on {device}...")

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
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    esm_plddt = outputs.plddt.cpu().numpy()
    positions = outputs.positions.cpu().numpy()  # Shape: [8, batch, seq_len, 14, 3]

    return esm_plddt, positions, outputs


def write_coords_to_pdb(model, outputs_dict, filename):
    """
    Write backbone coordinates to PDB file.

    Args:
        model (ESMForProteinFolding): ESMFold model
        outputs_dict (dict): Raw output dict from ESMFold
        filename (str): Output PDB filename
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
    plddt, positions, model_outputs = esmfold_inference(sequence, model, tokenizer)

    # Write to PDB file
    write_coords_to_pdb(model, model_outputs, output_path)

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

        plddt, positions, model_outputs = esmfold_inference(sequence, model, tokenizer)
        mean_plddt = plddt.mean()
        if mean_plddt < 1.0:
            mean_plddt *= 100.0  # Convert to percentage scale if needed

        # Save predicted structure to temporary file and compare with template
        base_pdb = str(Path(path).parent) + ".pdb"

        # Use temporary file for ESMFold prediction
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_ESMfold.pdb", delete=False
        ) as tmp_file:
            output_path = tmp_file.name
            write_coords_to_pdb(model, model_outputs, output_path)

            try:
                # Compare DSSP
                dssp_base = get_dssp(base_pdb, simplify=True)
                new_dssp = get_dssp(output_path, simplify=True)

                # Might be issue with temporary file, try again with final output path
                if len(new_dssp) == 0:
                    # Save to permanent location with proper name
                    final_output_path = path[:-5] + name + "_ESMfold.pdb"
                    write_coords_to_pdb(model, model_outputs, final_output_path)
                    new_dssp = get_dssp(final_output_path, simplify=True)
                    os.remove(final_output_path)

                dssp_match = dssp_base.count("H") == new_dssp.count(
                    "H"
                ) and dssp_base.count("E") == new_dssp.count("E")

                # Calculate TM-score and RMSD
                tm_score, rmsd = calculate_tm_score(usalign_path, output_path, base_pdb)

                # Save to permanent location with proper name
                # final_output_path = path[:-5] + name + "_ESMfold.pdb"
                # write_coords_to_pdb(model, model_outputs, final_output_path)

                results = {
                    "Path": path,
                    "Class": path.split("/")[-3],
                    "Sequence": sequence,
                    "pLDDT": mean_plddt,
                    "TM_score": tm_score,
                    "RMSD": rmsd,
                    # "ESM_path": final_output_path,
                    "DSSP": "True" if dssp_match else "False",
                    "Target_DSSP": dssp_base,
                    "Predicted_DSSP": new_dssp,
                }
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        # Clean up temporary file
        if os.path.exists(output_path):
            os.remove(output_path)

    return results


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


def _process_single_template(template_path, usalign_path, tm_threshold, rmsd_threshold):
    """
    Helper function to process a single template and its associated designs.
    Running this in parallel avoids the GIL bottleneck.
    """
    results = []

    # Get template DSSP once
    try:
        dssp_base = get_dssp(template_path, simplify=True)
    except Exception:
        # If template fails, skip its designs
        return results

    pathdir = template_path[:-4]
    pdbfiles = glob.glob(os.path.join(pathdir, "*.pdb"))
    pdbfiles = [f for f in pdbfiles if "ESMfold" not in os.path.basename(f)]

    for newpdb in pdbfiles:
        try:
            # Calculate TM-score and RMSD
            tm_score, rmsd = calculate_tm_score(usalign_path, newpdb, template_path)
            new_dssp = get_dssp(newpdb, simplify=True)

            # Calculate LDDT
            lddt_score = calculate_lddt(newpdb, template_path)

            # Check acceptance criteria
            dssp_match = dssp_base.count("H") == new_dssp.count(
                "H"
            ) and dssp_base.count("E") == new_dssp.count("E")

            accepted = (
                dssp_match
                and (tm_score is not None and tm_score >= tm_threshold)
                and (rmsd is not None and rmsd <= rmsd_threshold)
            )

            """
            Note that the accepted column is very heavily filtered when you use both TM-score and RMSD. In reality, just the DSSP match is considered for structures at this stage, and TM-score and RMSD become more relevant later in the analysis (the part that is mostly manual work and inspection). At this point, we just want to know if the secondary structure is preserved (indicating the correct fold), so the DSSP match is the key criterion. Predicting the structure at the later stage is more important than how close it is to the original template, as long as the overall fold is maintained. This is also an interesting thing to visualise, as it shows the difference in the different metrics and their impact on acceptance.
            """

            namelist = newpdb.split("/")

            # Store result as a dictionary/row
            results.append(
                {
                    "TM_score": tm_score if tm_score is not None else np.nan,
                    "RMSD": rmsd if rmsd is not None else np.nan,
                    "LDDT": lddt_score if lddt_score is not None else np.nan,
                    "DSSP": "True" if dssp_match else "False",
                    "Fold": namelist[-3],
                    "Short_Fold": SHORT_FOLD_NAME_DICT.get(namelist[-3], namelist[-3]),
                    "Accepted": "True" if accepted else "False",
                    "Paths": newpdb,
                    "Target_DSSP": dssp_base,
                    "New_DSSP": new_dssp,
                }
            )
        except Exception as e:
            print(f"Error processing {newpdb}: {e}")
            continue

    return results


def analyze_designed_structures(
    template_files, output_csv, usalign_path, tm_threshold=0.5, rmsd_threshold=5.0
):
    """
    Analyze designed structures using TM-score, RMSD, and DSSP metrics.
    Parallelized for performance.
    """
    # Create a partial function with fixed arguments
    process_func = partial(
        _process_single_template,
        usalign_path=usalign_path,
        tm_threshold=tm_threshold,
        rmsd_threshold=rmsd_threshold,
    )

    # Determine number of CPUs (leave one free for system responsiveness)
    num_workers = max(1, multiprocessing.cpu_count() - 1)

    all_results = []

    print(f"Analyzing structures using {num_workers} cores...")

    # process_map or Pool.imap can be used. Using Pool for standard library support.
    with multiprocessing.Pool(processes=num_workers) as pool:
        # imap_unordered is generally faster if order doesn't strictly matter
        # (though we just collect results anyway)
        results_iterator = pool.imap_unordered(process_func, template_files)

        # Use tqdm to track progress
        for batch_result in tqdm(
            results_iterator, total=len(template_files), desc="Analyzing structures"
        ):
            all_results.extend(batch_result)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(all_results)

    # Ensure columns are in the specific order expected by downstream functions
    # (Optional, but good for consistency with original code)
    if not df.empty:
        cols = [
            "TM_score",
            "RMSD",
            "LDDT",
            "DSSP",
            "Fold",
            "Short_Fold",
            "Accepted",
            "Paths",
            "Target_DSSP",
            "New_DSSP",
        ]
        df = df[cols]

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

    if not validation_results:
        print("No validation results to save.")
        df_struct = pd.DataFrame()
    else:
        df_struct = pd.DataFrame(validation_results)
        df_struct.to_csv(output_csv, index=False)

    return df_struct


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
        data=df,
        x="Fold",
        hue="Fold",
        shrink=0.8,
        alpha=0.8,
        legend=False,
        ax=axs[1],
    )
    wrap_labels(axs[1], 12)
    f.tight_layout()
    plt.savefig(f"{output_prefix}_RMSD_vs_TMscore_counts.pdf", bbox_inches="tight")
    plt.close()

    # Plot 5: DSSP acceptance by fold
    g = sns.displot(data=df, x="DSSP", col="Short_Fold", facet_kws={"sharey": False})
    plt.savefig(f"{output_prefix}_accepted_folds.pdf", bbox_inches="tight")
    plt.close()

    # Plot 5: TM-score by fold and DSSP
    g = sns.displot(data=df, x="TM_score", hue="Short_Fold", col="DSSP")
    plt.savefig(f"{output_prefix}_TMscore_fold_distribution.pdf", bbox_inches="tight")
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


def plot_comprehensive_overview(
    df, output_prefix, tm_threshold=0.5, rmsd_threshold=5.0
):
    """
    Create comprehensive overview visualizations optimized for multiple fold classes.

    Args:
        df (pd.DataFrame): Analysis results dataframe.
        output_prefix (str): Prefix for output file names.
        tm_threshold (float): TM-score threshold used for acceptance.
        rmsd_threshold (float): RMSD threshold used for acceptance.
    """
    sns.set_theme(style="whitegrid")

    # Calculate summary statistics by fold
    summary_stats = (
        df.groupby("Short_Fold")
        .agg(
            {
                "TM_score": ["mean", "std", "count"],
                "RMSD": ["mean", "std"],
                "DSSP": lambda x: (x == "True").sum(),
                "Accepted": lambda x: (x == "True").sum(),
            }
        )
        .round(3)
    )

    summary_stats.columns = [
        "_".join(col).strip() for col in summary_stats.columns.values
    ]
    summary_stats["DSSP_rate"] = (
        summary_stats["DSSP_<lambda>"] / summary_stats["TM_score_count"]
    )
    summary_stats["Accept_rate"] = (
        summary_stats["Accepted_<lambda>"] / summary_stats["TM_score_count"]
    )
    summary_stats["TM_pass_rate"] = df.groupby("Short_Fold")["TM_score"].apply(
        lambda x: (x >= tm_threshold).sum() / len(x)
    )
    summary_stats["RMSD_pass_rate"] = df.groupby("Short_Fold")["RMSD"].apply(
        lambda x: (x <= rmsd_threshold).sum() / len(x)
    )

    # Sort by total count for consistent ordering
    summary_stats = summary_stats.sort_values("TM_score_count", ascending=False)
    fold_order = summary_stats.index.tolist()

    # ===== PLOT 1: Summary Heatmap =====
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Heatmap of rates
    rate_data = summary_stats[
        ["DSSP_rate", "Accept_rate", "TM_pass_rate", "RMSD_pass_rate"]
    ].T
    rate_data.index = [
        "DSSP Match\nRate",
        "Overall\nAccept Rate",
        f"TM-score ≥{tm_threshold}\nRate",
        f"RMSD ≤{rmsd_threshold}Å\nRate",
    ]

    sns.heatmap(
        rate_data,
        annot=True,
        fmt=".2%",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Success Rate"},
        ax=axes[0],
        linewidths=0.5,
    )
    axes[0].set_title("Success Rates by Fold Class", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Metric", fontsize=12)
    axes[0].set_xlabel("Fold Class", fontsize=12)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

    # Heatmap of mean values
    mean_data = summary_stats[["TM_score_mean", "RMSD_mean"]].T
    mean_data.index = ["Mean\nTM-score", "Mean\nRMSD (Å)"]

    sns.heatmap(
        mean_data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar_kws={"label": "Mean Value"},
        ax=axes[1],
        linewidths=0.5,
    )
    axes[1].set_title(
        "Mean Metric Values by Fold Class", fontsize=14, fontweight="bold"
    )
    axes[1].set_ylabel("Metric", fontsize=12)
    axes[1].set_xlabel("Fold Class", fontsize=12)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_overview_heatmap.pdf", bbox_inches="tight")
    plt.close()

    # ===== PLOT 2: Performance Dashboard =====
    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.45)

    # Sample counts
    ax1 = fig.add_subplot(gs[0, :])
    counts_df = summary_stats[
        ["TM_score_count", "DSSP_<lambda>", "Accepted_<lambda>"]
    ].copy()
    counts_df.columns = ["Total Structures", "DSSP Matches", "Fully Accepted"]
    counts_df.plot(kind="bar", ax=ax1, width=0.8)
    ax1.set_title("Structure Counts by Fold Class", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_xlabel("Fold Class", fontsize=12)
    ax1.legend(title="Category", loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    wrap_labels(ax1, 12)
    # plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # TM-score box plot
    ax2 = fig.add_subplot(gs[1, :2])
    df_sorted = df.copy()
    df_sorted["Short_Fold"] = pd.Categorical(
        df_sorted["Short_Fold"], categories=fold_order, ordered=True
    )
    sns.boxplot(data=df_sorted, x="Short_Fold", y="TM_score", ax=ax2, palette="Set2")
    ax2.axhline(
        y=tm_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({tm_threshold})",
    )
    ax2.set_title("TM-score Distribution by Fold", fontsize=14, fontweight="bold")
    ax2.set_ylabel("TM-score", fontsize=12)
    ax2.set_xlabel("Fold Class", fontsize=12)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    wrap_labels(ax2, 12)
    # plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

    # TM-score box plot
    ax3 = fig.add_subplot(gs[1, 2])
    sns.violinplot(data=df_sorted, x="DSSP", y="TM_score", ax=ax3, palette="muted")
    ax3.axhline(
        y=tm_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({tm_threshold})",
    )
    ax3.set_title("TM-score by DSSP Match", fontsize=14, fontweight="bold")
    ax3.set_ylabel("TM-score", fontsize=12)
    ax3.set_xlabel("DSSP Match", fontsize=12)
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Success rates bar chart
    ax4 = fig.add_subplot(gs[2, :2])
    rate_plot_data = summary_stats[
        ["DSSP_rate", "TM_pass_rate", "RMSD_pass_rate", "Accept_rate"]
    ].copy()
    rate_plot_data.columns = [
        "DSSP Match",
        f"TM≥{tm_threshold}",
        f"RMSD≤{rmsd_threshold}Å",
        "All Criteria",
    ]
    rate_plot_data.plot(kind="bar", ax=ax4, width=0.8)
    ax4.set_title(
        "Pass Rates by Fold Class and Criterion", fontsize=14, fontweight="bold"
    )
    ax4.set_ylabel("Pass Rate", fontsize=12)
    ax4.set_xlabel("Fold Class", fontsize=12)
    ax4.set_ylim([0, 1.05])
    ax4.legend(
        title="Criterion",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=4,
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
    )
    ax4.grid(axis="y", alpha=0.3)
    wrap_labels(ax4, 12)
    # plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

    # Metric correlation scatter
    ax5 = fig.add_subplot(gs[2, 2])
    scatter = ax5.scatter(
        df["TM_score"],
        df["RMSD"],
        c=df["Short_Fold"].astype("category").cat.codes,
        alpha=0.5,
        cmap="tab20",
        s=30,
    )
    ax5.axvline(x=tm_threshold, color="r", linestyle="--", linewidth=1, alpha=0.7)
    ax5.axhline(y=rmsd_threshold, color="r", linestyle="--", linewidth=1, alpha=0.7)
    ax5.set_xlabel("TM-score", fontsize=12)
    ax5.set_ylabel("RMSD (Å)", fontsize=12)
    ax5.set_title(
        "TM-score vs RMSD\n(coloured by fold)", fontsize=14, fontweight="bold"
    )
    ax5.grid(alpha=0.3)

    plt.savefig(f"{output_prefix}_overview_dashboard.pdf", bbox_inches="tight")
    plt.close()

    # ===== PLOT 3: Detailed Fold Comparison =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # TM-score by fold with individual points
    ax = axes[0, 0]
    sns.stripplot(
        data=df_sorted,
        x="Short_Fold",
        y="TM_score",
        hue="DSSP",
        alpha=0.6,
        size=3,
        ax=ax,
        dodge=True,
    )
    sns.boxplot(
        data=df_sorted,
        x="Short_Fold",
        y="TM_score",
        ax=ax,
        showfliers=False,
        boxprops=dict(alpha=0.3),
        width=0.6,
    )
    ax.axhline(
        y=tm_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({tm_threshold})",
    )
    ax.set_title(
        "TM-score Distribution by Fold (with DSSP status)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("TM-score", fontsize=11)
    ax.set_xlabel("")
    ax.legend(title="DSSP Match", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    # RMSD by fold
    ax = axes[0, 1]
    sns.stripplot(
        data=df_sorted,
        x="Short_Fold",
        y="RMSD",
        hue="DSSP",
        alpha=0.6,
        size=3,
        ax=ax,
        dodge=True,
    )
    sns.boxplot(
        data=df_sorted,
        x="Short_Fold",
        y="RMSD",
        ax=ax,
        showfliers=False,
        boxprops=dict(alpha=0.3),
        width=0.6,
    )
    ax.axhline(
        y=rmsd_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({rmsd_threshold}Å)",
    )
    ax.set_title(
        "RMSD Distribution by Fold (with DSSP status)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("RMSD (Å)", fontsize=11)
    ax.set_xlabel("")
    ax.legend(title="DSSP Match", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    # Acceptance breakdown
    ax = axes[1, 0]
    accept_summary = df.groupby(["Short_Fold", "Accepted"]).size().unstack(fill_value=0)
    accept_summary_pct = accept_summary.div(accept_summary.sum(axis=1), axis=0) * 100
    accept_summary_pct = accept_summary_pct.reindex(fold_order)
    accept_summary_pct.plot(
        kind="barh", stacked=True, ax=ax, color=["#d62728", "#2ca02c"], width=0.8
    )
    ax.set_title("Acceptance Status by Fold (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_ylabel("Fold Class", fontsize=11)
    ax.legend(
        title="Accepted",
        labels=["False", "True"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.grid(axis="x", alpha=0.3)

    # DSSP breakdown
    ax = axes[1, 1]
    dssp_summary = df.groupby(["Short_Fold", "DSSP"]).size().unstack(fill_value=0)
    dssp_summary_pct = dssp_summary.div(dssp_summary.sum(axis=1), axis=0) * 100
    dssp_summary_pct = dssp_summary_pct.reindex(fold_order)
    dssp_summary_pct.plot(
        kind="barh", stacked=True, ax=ax, color=["#ff7f0e", "#1f77b4"], width=0.8
    )
    ax.set_title("DSSP Match Status by Fold (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_ylabel("Fold Class", fontsize=11)
    ax.legend(
        title="DSSP Match",
        labels=["False", "True"],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_prefix}_overview_detailed_comparison.pdf", bbox_inches="tight"
    )
    plt.close()

    # ===== PLOT 4: Summary Statistics Table =====
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = summary_stats[
        [
            "TM_score_count",
            "TM_score_mean",
            "TM_score_std",
            "RMSD_mean",
            "RMSD_std",
            "DSSP_rate",
            "Accept_rate",
        ]
    ].copy()
    table_data.columns = [
        "N",
        "TM\nMean",
        "TM\nStd",
        "RMSD\nMean",
        "RMSD\nStd",
        "DSSP\nRate",
        "Accept\nRate",
    ]
    table_data["DSSP\nRate"] = table_data["DSSP\nRate"].apply(lambda x: f"{x:.1%}")
    table_data["Accept\nRate"] = table_data["Accept\nRate"].apply(lambda x: f"{x:.1%}")
    table_data = table_data.round(3)

    # Create table
    table = ax.table(
        cellText=table_data.values,
        rowLabels=table_data.index,
        colLabels=table_data.columns,
        cellLoc="center",
        rowLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(table_data.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(len(table_data)):
        table[(i + 1, -1)].set_facecolor("#D9E1F2")
        table[(i + 1, -1)].set_text_props(weight="bold")

    ax.set_title(
        "Summary Statistics by Fold Class", fontsize=16, fontweight="bold", pad=20
    )

    plt.savefig(f"{output_prefix}_overview_summary_table.pdf", bbox_inches="tight")
    plt.close()

    print(
        f"\nComprehensive overview plots saved with prefix: {output_prefix}_overview_*"
    )


def plot_esmfold_comprehensive_analysis(
    df_original, df_esmfold, output_prefix, tm_threshold=0.5, rmsd_threshold=5.0
):
    """
    Create comprehensive ESMFold validation visualizations with comparison to original metrics.

    Args:
        df_original (pd.DataFrame): Original analysis results
        df_esmfold (pd.DataFrame): ESMFold validation results
        output_prefix (str): Prefix for output filenames
        tm_threshold (float): TM-score threshold
        rmsd_threshold (float): RMSD threshold
    """
    # Make copies to avoid modifying originals
    df_esmfold = df_esmfold.copy()
    df_original = df_original.copy()

    # Ensure consistent column naming
    if "Class" in df_esmfold.columns:
        df_esmfold["Fold"] = df_esmfold["Class"]

    # Ensure Short_Fold exists in both dataframes
    if "Short_Fold" not in df_original.columns:
        df_original["Short_Fold"] = df_original["Fold"].map(SHORT_FOLD_NAME_DICT)
        df_original["Short_Fold"] = df_original["Short_Fold"].fillna(
            df_original["Fold"]
        )

    if "Short_Fold" not in df_esmfold.columns:
        df_esmfold["Short_Fold"] = df_esmfold["Fold"].map(SHORT_FOLD_NAME_DICT)
        df_esmfold["Short_Fold"] = df_esmfold["Short_Fold"].fillna(df_esmfold["Fold"])

    # Debug info
    print(f"\ndf_original shape: {df_original.shape}")
    print(f"df_esmfold shape: {df_esmfold.shape}")
    print(f"Unique folds in df_original: {df_original['Fold'].nunique()}")
    print(f"Unique folds in df_esmfold: {df_esmfold['Fold'].nunique()}")

    # Try to merge on Path/Paths first (more accurate)
    if "Path" in df_esmfold.columns and "Paths" in df_original.columns:
        print("Merging on Path/Paths columns...")
        # Only select the columns we need from df_original to avoid conflicts
        df_merged = df_esmfold.merge(
            df_original[
                ["Paths", "TM_score", "RMSD"]
            ],  # Removed "Short_Fold" from here
            left_on="Path",
            right_on="Paths",
            how="left",
            suffixes=("_esmfold", "_original"),
        )
        # Use Short_Fold from df_esmfold (which is already in df_merged)
        df_merged["Fold_Name"] = df_merged["Short_Fold"]
    else:
        # Fallback: aggregate original data by fold and merge
        print("Warning: Path columns not found. Aggregating by fold...")
        df_original_agg = (
            df_original.groupby("Fold")
            .agg({"TM_score": "mean", "RMSD": "mean", "Short_Fold": "first"})
            .reset_index()
        )

        df_merged = df_esmfold.merge(
            df_original_agg,
            on="Fold",
            how="left",
            suffixes=("_esmfold", "_original"),
        )
        # Handle potential suffix on Short_Fold
        if "Short_Fold_esmfold" in df_merged.columns:
            df_merged["Fold_Name"] = df_merged["Short_Fold_esmfold"]
        elif "Short_Fold" in df_merged.columns:
            df_merged["Fold_Name"] = df_merged["Short_Fold"]
        else:
            df_merged["Fold_Name"] = df_merged["Fold"].map(SHORT_FOLD_NAME_DICT)

    print(f"df_merged shape: {df_merged.shape}")
    print(f"Columns in df_merged: {df_merged.columns.tolist()}")
    print(
        f"Missing values:\n{df_merged[['TM_score_esmfold', 'TM_score_original', 'RMSD_esmfold', 'RMSD_original']].isnull().sum()}"
    )

    # Handle missing values
    if df_merged[["TM_score_original", "RMSD_original"]].isnull().any().any():
        print("\nWarning: Some ESMFold results could not be matched to original data.")
        print("Dropping unmatched rows...")
        df_merged = df_merged.dropna(subset=["TM_score_original", "RMSD_original"])
        print(f"Remaining rows: {len(df_merged)}")

    if len(df_merged) == 0:
        print("Error: No matching data found between df_original and df_esmfold!")
        return

    # Ensure Fold_Name exists and has no missing values
    if "Fold_Name" not in df_merged.columns or df_merged["Fold_Name"].isnull().any():
        print("Warning: Fold_Name missing or has nulls, recreating from Fold column...")
        df_merged["Fold_Name"] = df_merged["Fold"].map(SHORT_FOLD_NAME_DICT)
        df_merged["Fold_Name"] = df_merged["Fold_Name"].fillna(df_merged["Fold"])

    # Sort by fold for consistent ordering
    fold_order = sorted(df_merged["Fold_Name"].unique())
    df_sorted = df_merged.copy()
    df_sorted["Fold_Name"] = pd.Categorical(
        df_sorted["Fold_Name"], categories=fold_order, ordered=True
    )
    df_sorted = df_sorted.sort_values("Fold_Name")

    # ===== CREATE DELTA COLUMNS HERE (before any plotting) =====
    df_sorted["TM_delta"] = (
        df_sorted["TM_score_esmfold"] - df_sorted["TM_score_original"]
    )
    df_sorted["RMSD_delta"] = df_sorted["RMSD_esmfold"] - df_sorted["RMSD_original"]

    # Also update df_merged for consistency
    df_merged["TM_delta"] = (
        df_merged["TM_score_esmfold"] - df_merged["TM_score_original"]
    )
    df_merged["RMSD_delta"] = df_merged["RMSD_esmfold"] - df_merged["RMSD_original"]

    # ===== PLOT 1: ESMFold Metrics Overview Dashboard =====
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # TM-score distribution (ESMFold)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(
        df_merged["TM_score_esmfold"],
        bins=30,
        alpha=0.7,
        color="#2E86AB",
        edgecolor="black",
    )
    ax1.axvline(
        tm_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({tm_threshold})",
    )
    ax1.set_xlabel("TM-score (ESMFold)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("ESMFold TM-score Distribution", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # RMSD distribution (ESMFold)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        df_merged["RMSD_esmfold"],
        bins=30,
        alpha=0.7,
        color="#A23B72",
        edgecolor="black",
    )
    ax2.axvline(
        rmsd_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({rmsd_threshold}Å)",
    )
    ax2.set_xlabel("RMSD (Å) (ESMFold)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("ESMFold RMSD Distribution", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # pLDDT distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df_merged["pLDDT"], bins=30, alpha=0.7, color="#F18F01", edgecolor="black")
    ax3.axvline(
        70, color="orange", linestyle="--", linewidth=2, label="Good quality (70)"
    )
    ax3.axvline(
        90, color="green", linestyle="--", linewidth=2, label="High quality (90)"
    )
    ax3.set_xlabel("pLDDT", fontsize=11)
    ax3.set_ylabel("Count", fontsize=11)
    ax3.set_title("ESMFold Confidence (pLDDT)", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # TM-score comparison (Original vs ESMFold)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(
        df_merged["TM_score_original"],
        df_merged["TM_score_esmfold"],
        c=df_merged["pLDDT"],
        cmap="viridis",
        alpha=0.6,
        s=50,
    )
    ax4.plot([0, 1], [0, 1], "r--", linewidth=2, label="y=x")
    ax4.axhline(tm_threshold, color="gray", linestyle=":", alpha=0.5)
    ax4.axvline(tm_threshold, color="gray", linestyle=":", alpha=0.5)
    ax4.set_xlabel("Original TM-score", fontsize=11)
    ax4.set_ylabel("ESMFold TM-score", fontsize=11)
    ax4.set_title(
        "TM-score: Original vs ESMFold\n(coloured by pLDDT)",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar(scatter, ax=ax4, label="pLDDT")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # RMSD comparison (Original vs ESMFold)
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(
        df_merged["RMSD_original"],
        df_merged["RMSD_esmfold"],
        c=df_merged["pLDDT"],
        cmap="plasma",
        alpha=0.6,
        s=50,
    )
    max_rmsd = max(df_merged["RMSD_original"].max(), df_merged["RMSD_esmfold"].max())
    ax5.plot([0, max_rmsd], [0, max_rmsd], "r--", linewidth=2, label="y=x")
    ax5.axhline(rmsd_threshold, color="gray", linestyle=":", alpha=0.5)
    ax5.axvline(rmsd_threshold, color="gray", linestyle=":", alpha=0.5)
    ax5.set_xlabel("Original RMSD (Å)", fontsize=11)
    ax5.set_ylabel("ESMFold RMSD (Å)", fontsize=11)
    ax5.set_title(
        "RMSD: Original vs ESMFold\n(coloured by pLDDT)", fontsize=12, fontweight="bold"
    )
    plt.colorbar(scatter, ax=ax5, label="pLDDT")
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Correlation between pLDDT and TM-score
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(
        df_merged["pLDDT"],
        df_merged["TM_score_esmfold"],
        c=df_merged["RMSD_esmfold"],
        cmap="coolwarm_r",
        alpha=0.6,
        s=50,
    )
    ax6.set_xlabel("pLDDT", fontsize=11)
    ax6.set_ylabel("ESMFold TM-score", fontsize=11)
    ax6.set_title(
        "Confidence vs Accuracy\n(coloured by RMSD)", fontsize=12, fontweight="bold"
    )
    plt.colorbar(scatter, ax=ax6, label="RMSD (Å)")
    ax6.grid(alpha=0.3)

    # Metric deltas (ESMFold - Original)
    ax7 = fig.add_subplot(gs[2, 0])
    df_merged["TM_delta"] = (
        df_merged["TM_score_esmfold"] - df_merged["TM_score_original"]
    )
    ax7.hist(
        df_merged["TM_delta"], bins=30, alpha=0.7, color="#06A77D", edgecolor="black"
    )
    ax7.axvline(0, color="r", linestyle="--", linewidth=2, label="No change")
    ax7.set_xlabel("ΔTM-score (ESMFold - Original)", fontsize=11)
    ax7.set_ylabel("Count", fontsize=11)
    ax7.set_title("TM-score Change Distribution", fontsize=12, fontweight="bold")
    ax7.legend()
    ax7.grid(alpha=0.3)

    # RMSD deltas
    ax8 = fig.add_subplot(gs[2, 1])
    df_merged["RMSD_delta"] = df_merged["RMSD_esmfold"] - df_merged["RMSD_original"]
    ax8.hist(
        df_merged["RMSD_delta"], bins=30, alpha=0.7, color="#D62828", edgecolor="black"
    )
    ax8.axvline(0, color="r", linestyle="--", linewidth=2, label="No change")
    ax8.set_xlabel("ΔRMSD (ESMFold - Original) (Å)", fontsize=11)
    ax8.set_ylabel("Count", fontsize=11)
    ax8.set_title("RMSD Change Distribution", fontsize=12, fontweight="bold")
    ax8.legend()
    ax8.grid(alpha=0.3)

    # Summary metrics comparison
    ax9 = fig.add_subplot(gs[2, 2])
    metrics_comparison = pd.DataFrame(
        {
            "Original": [
                df_merged["TM_score_original"].mean(),
                df_merged["RMSD_original"].mean(),
            ],
            "ESMFold": [
                df_merged["TM_score_esmfold"].mean(),
                df_merged["RMSD_esmfold"].mean(),
            ],
        },
        index=["TM-score\n(mean)", "RMSD\n(mean, Å)"],
    )

    metrics_comparison.plot(kind="bar", ax=ax9, width=0.7, color=["#4E598C", "#FF6B35"])
    ax9.set_title("Mean Metrics Comparison", fontsize=12, fontweight="bold")
    ax9.set_ylabel("Value", fontsize=11)
    ax9.set_xlabel("")
    ax9.legend(title="Method", fontsize=10)
    ax9.grid(axis="y", alpha=0.3)
    plt.setp(ax9.get_xticklabels(), rotation=0)

    plt.savefig(
        f"{output_prefix}_esmfold_overview_dashboard.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # ===== PLOT 2: Per-Fold ESMFold Analysis =====
    n_folds = len(fold_order)
    n_cols = 3
    n_rows = int(np.ceil(n_folds / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_folds > 1 else [axes]

    for idx, fold_name in enumerate(fold_order):
        ax = axes[idx]
        fold_data = df_sorted[df_sorted["Fold_Name"] == fold_name]

        # Create violin plot with strip overlay
        parts = ax.violinplot(
            [fold_data["TM_score_original"], fold_data["TM_score_esmfold"]],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
        )

        # Color the violins
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#4E598C", "#FF6B35"][i])
            pc.set_alpha(0.6)

        # Add individual points
        ax.scatter(
            [1] * len(fold_data),
            fold_data["TM_score_original"],
            alpha=0.4,
            s=30,
            color="#4E598C",
        )
        ax.scatter(
            [2] * len(fold_data),
            fold_data["TM_score_esmfold"],
            alpha=0.4,
            s=30,
            color="#FF6B35",
        )

        ax.axhline(tm_threshold, color="r", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Original", "ESMFold"], fontsize=10)
        ax.set_ylabel("TM-score", fontsize=10)
        ax.set_title(
            f"{fold_name}\n(n={len(fold_data)})", fontsize=11, fontweight="bold"
        )
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.05])

    # Hide unused subplots
    for idx in range(n_folds, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{output_prefix}_esmfold_per_fold_comparison.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # ===== PLOT 3: Detailed Metric Comparison by Fold =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # TM-score comparison boxplot
    ax = axes[0, 0]
    data_for_box = []
    labels_for_box = []
    colors_for_box = []

    for fold_name in fold_order:
        fold_data = df_sorted[df_sorted["Fold_Name"] == fold_name]
        data_for_box.extend(
            [fold_data["TM_score_original"], fold_data["TM_score_esmfold"]]
        )
        labels_for_box.extend([f"{fold_name[:20]}\nOrig", f"{fold_name[:20]}\nESM"])
        colors_for_box.extend(["#4E598C", "#FF6B35"])

    bp = ax.boxplot(
        data_for_box,
        labels=labels_for_box,
        patch_artist=True,
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(
        tm_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({tm_threshold})",
    )
    ax.set_ylabel("TM-score", fontsize=11)
    ax.set_title(
        "TM-score by Fold: Original vs ESMFold", fontsize=12, fontweight="bold"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # RMSD comparison boxplot
    ax = axes[0, 1]
    data_for_box = []
    labels_for_box = []

    for fold_name in fold_order:
        fold_data = df_sorted[df_sorted["Fold_Name"] == fold_name]
        data_for_box.extend([fold_data["RMSD_original"], fold_data["RMSD_esmfold"]])
        labels_for_box.extend([f"{fold_name[:20]}\nOrig", f"{fold_name[:20]}\nESM"])

    bp = ax.boxplot(
        data_for_box,
        labels=labels_for_box,
        patch_artist=True,
        showfliers=False,
    )

    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(
        rmsd_threshold,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({rmsd_threshold}Å)",
    )
    ax.set_ylabel("RMSD (Å)", fontsize=11)
    ax.set_title("RMSD by Fold: Original vs ESMFold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    # pLDDT distribution by fold
    ax = axes[1, 0]
    sns.violinplot(
        data=df_sorted,
        x="Fold_Name",
        y="pLDDT",
        ax=ax,
        inner="box",
        palette="muted",
    )
    ax.axhline(
        70, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Good (70)"
    )
    ax.axhline(
        90, color="green", linestyle="--", linewidth=1, alpha=0.7, label="High (90)"
    )
    ax.set_ylabel("pLDDT", fontsize=11)
    ax.set_xlabel("")
    ax.set_title("ESMFold Confidence (pLDDT) by Fold", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)

    # Metric improvement analysis
    ax = axes[1, 1]
    improvement_summary = (
        df_sorted.groupby("Fold_Name")
        .agg(
            {
                "TM_delta": "mean",
                "RMSD_delta": "mean",
            }
        )
        .reset_index()
    )

    x = np.arange(len(improvement_summary))
    width = 0.35

    ax.bar(
        x - width / 2,
        improvement_summary["TM_delta"],
        width,
        label="ΔTM-score",
        color="#06A77D",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        improvement_summary["RMSD_delta"],
        width,
        label="ΔRMSD",
        color="#D62828",
        alpha=0.8,
    )

    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    ax.set_ylabel("Mean Change", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(
        "Mean Metric Changes by Fold\n(ESMFold - Original)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        improvement_summary["Fold_Name"], rotation=45, ha="right", fontsize=9
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_prefix}_esmfold_detailed_comparison.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # ===== PLOT 4: Summary Statistics Table =====
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.axis("tight")
    ax.axis("off")

    # Calculate summary statistics
    summary_by_fold = (
        df_sorted.groupby("Fold_Name")
        .agg(
            {
                "TM_score_original": ["mean", "std"],
                "TM_score_esmfold": ["mean", "std"],
                "RMSD_original": ["mean", "std"],
                "RMSD_esmfold": ["mean", "std"],
                "pLDDT": ["mean", "std"],
                "TM_delta": "mean",
                "RMSD_delta": "mean",
            }
        )
        .round(3)
    )

    summary_by_fold.columns = [
        "TM Orig\nMean",
        "TM Orig\nStd",
        "TM ESM\nMean",
        "TM ESM\nStd",
        "RMSD Orig\nMean",
        "RMSD Orig\nStd",
        "RMSD ESM\nMean",
        "RMSD ESM\nStd",
        "pLDDT\nMean",
        "pLDDT\nStd",
        "ΔTM\nMean",
        "ΔRMSD\nMean",
    ]

    # Add sample counts
    counts = df_sorted.groupby("Fold_Name").size()
    summary_by_fold.insert(0, "N", counts.values)

    # Create table
    table = ax.table(
        cellText=summary_by_fold.values,
        rowLabels=summary_by_fold.index,
        colLabels=summary_by_fold.columns,
        cellLoc="center",
        rowLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)

    # Style header
    for i in range(len(summary_by_fold.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style row labels
    for i in range(len(summary_by_fold)):
        table[(i + 1, -1)].set_facecolor("#D9E1F2")
        table[(i + 1, -1)].set_text_props(weight="bold", fontsize=8)

    # Color cells based on values
    for i in range(len(summary_by_fold)):
        # Highlight good TM-scores (ESMFold)
        tm_esm_val = summary_by_fold.iloc[i]["TM ESM\nMean"]
        if tm_esm_val >= tm_threshold:
            table[(i + 1, 3)].set_facecolor("#C6EFCE")  # Light green

        # Highlight good RMSD (ESMFold)
        rmsd_esm_val = summary_by_fold.iloc[i]["RMSD ESM\nMean"]
        if rmsd_esm_val <= rmsd_threshold:
            table[(i + 1, 7)].set_facecolor("#C6EFCE")  # Light green

        # Highlight improvements in TM-score
        tm_delta_val = summary_by_fold.iloc[i]["ΔTM\nMean"]
        if tm_delta_val > 0:
            table[(i + 1, 11)].set_facecolor("#C6EFCE")  # Light green
        elif tm_delta_val < 0:
            table[(i + 1, 11)].set_facecolor("#FFC7CE")  # Light red

    ax.set_title(
        "ESMFold Validation Summary Statistics by Fold Class",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(
        f"{output_prefix}_esmfold_summary_table.pdf", bbox_inches="tight", dpi=300
    )
    plt.close()

    # Print summary
    print(f"\n{'=' * 80}")
    print("ESMFold VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total structures validated: {len(df_merged)}")
    print("\nOriginal metrics:")
    print(
        f"  Mean TM-score: {df_merged['TM_score_original'].mean():.3f} ± {df_merged['TM_score_original'].std():.3f}"
    )
    print(
        f"  Mean RMSD: {df_merged['RMSD_original'].mean():.3f} ± {df_merged['RMSD_original'].std():.3f} Å"
    )
    print("\nESMFold metrics:")
    print(
        f"  Mean TM-score: {df_merged['TM_score_esmfold'].mean():.3f} ± {df_merged['TM_score_esmfold'].std():.3f}"
    )
    print(
        f"  Mean RMSD: {df_merged['RMSD_esmfold'].mean():.3f} ± {df_merged['RMSD_esmfold'].std():.3f} Å"
    )
    print(
        f"  Mean pLDDT: {df_merged['pLDDT'].mean():.1f} ± {df_merged['pLDDT'].std():.1f}"
    )
    print("\nMetric changes (ESMFold - Original):")
    print(
        f"  Mean ΔTM-score: {df_merged['TM_delta'].mean():.3f} ± {df_merged['TM_delta'].std():.3f}"
    )
    print(
        f"  Mean ΔRMSD: {df_merged['RMSD_delta'].mean():.3f} ± {df_merged['RMSD_delta'].std():.3f} Å"
    )
    print("\nAcceptance rates (using ESMFold metrics):")
    tm_accept = (df_merged["TM_score_esmfold"] >= tm_threshold).sum()
    rmsd_accept = (df_merged["RMSD_esmfold"] <= rmsd_threshold).sum()
    both_accept = (
        (df_merged["TM_score_esmfold"] >= tm_threshold)
        & (df_merged["RMSD_esmfold"] <= rmsd_threshold)
    ).sum()
    print(
        f"  TM-score ≥ {tm_threshold}: {tm_accept}/{len(df_merged)} ({100 * tm_accept / len(df_merged):.1f}%)"
    )
    print(
        f"  RMSD ≤ {rmsd_threshold}Å: {rmsd_accept}/{len(df_merged)} ({100 * rmsd_accept / len(df_merged):.1f}%)"
    )
    print(
        f"  Both criteria: {both_accept}/{len(df_merged)} ({100 * both_accept / len(df_merged):.1f}%)"
    )
    print(f"{'=' * 80}\n")

    print(
        f"ESMFold comprehensive analysis plots saved with prefix: {output_prefix}_esmfold_*"
    )


def add_polyalanine_count(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    new_col: str = "num_polyA",
    min_len: int = 5,
) -> pd.DataFrame:
    """
    Add a column counting polyalanine regions (contiguous runs of A).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    sequence_col : str
        Name of the column containing sequences.
    new_col : str
        Name of the output column.
    min_len : int
        Minimum run length to qualify as a polyalanine region.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added column counting polyalanine regions.
    """
    pattern = re.compile(f"A{{{min_len},}}")

    def count_polyA(seq):
        if not isinstance(seq, str):
            return 0
        return len(pattern.findall(seq))

    df[new_col] = df[sequence_col].apply(count_polyA)
    return df


def main():
    """Main pipeline for protein design validation and analysis."""
    args = parse_arguments()

    # Make sure USalign path is valid
    if not Path(args.usalign_path).exists():
        raise FileNotFoundError(f"USalign path not found: {args.usalign_path}")

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

        # Create analysis plots
        print("Generating analysis plots...")
        plot_structure_analysis(
            df, args.output_prefix, args.tm_threshold, args.rmsd_threshold
        )

        # Create comprehensive overview plots
        print("Generating comprehensive overview plots...")
        plot_comprehensive_overview(
            df, args.output_prefix, args.tm_threshold, args.rmsd_threshold
        )
    else:
        print(f"Loading existing analysis from {results_csv}")
        df = pd.read_csv(results_csv)

    df["DSSP"] = df["DSSP"].astype(bool)

    # Stage 2: ESMFold validation (optional)
    if not args.skip_structure_validation:
        validation_csv = f"{args.output_prefix}_structure_evaluation.csv"

        if not os.path.exists(validation_csv):
            print("Performing ESMFold structure validation...")
            df_accept = df[df["DSSP"] == True]
            df_struct = validate_structures_with_esmfold(
                df, validation_csv, args.usalign_path, args.gpu_id
            )
            print(f"Validation complete. Results saved to {validation_csv}")
        else:
            print(f"Loading existing validation from {validation_csv}")
            df_struct = pd.read_csv(validation_csv)

        # Create validation plots
        if "Short_Fold" not in df_struct.columns:
            df_struct["Short_Fold"] = df_struct["Class"].apply(
                lambda x: SHORT_FOLD_NAME_DICT.get(x, x)
            )

        print("Calculating polyalanine counts...")
        df_struct = add_polyalanine_count(df_struct)

        print("Generating validation plots...")
        plot_esmfold_validation(df_struct, args.output_prefix)

        # Create comprehensive ESMFold analysis plots
        print("Generating comprehensive ESMFold analysis plots...")
        plot_esmfold_comprehensive_analysis(
            df, df_struct, args.output_prefix, args.tm_threshold, args.rmsd_threshold
        )

    print("Pipeline complete!")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total structures analyzed: {len(df)}")
    print(
        f"Accepted structures (DSSP + TM-score + RMSD): {sum(df['Accepted'] == True)}"
    )
    print(f"DSSP matches: {sum(df['DSSP'] == True)}")
    print(f"TM-score ≥ {args.tm_threshold}: {sum(df['TM_score'] >= args.tm_threshold)}")
    print(f"RMSD ≤ {args.rmsd_threshold}Å: {sum(df['RMSD'] <= args.rmsd_threshold)}")
    print(f"Mean TM-score: {df['TM_score'].mean():.3f} ± {df['TM_score'].std():.3f}")
    print(f"Mean RMSD: {df['RMSD'].mean():.3f} ± {df['RMSD'].std():.3f} Å")
    print(
        f"Mean polyalanine count (more than 5 A in a row): {df_struct['num_polyA'].mean():.3f} ± {df_struct['num_polyA'].std():.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
