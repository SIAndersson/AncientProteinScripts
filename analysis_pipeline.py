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
    """Parse command line arguments for the protein design validation pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate and analyse designed protein structures against reference templates. "
            "Computes TM-score, RMSD, lDDT, and DSSP secondary-structure metrics, and "
            "optionally runs ESMFold structure prediction for independent validation."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "Root directory of the protein evolution data. Expected layout: "
            "<data_path>/<fold_name>/<template>.pdb with a matching subdirectory "
            "<data_path>/<fold_name>/<template>/ containing designed PDB files."
        ),
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=False,
        default="protein_evo",
        help=(
            "Prefix applied to all output files (CSVs and PDFs). "
            "Defaults to 'protein_evo'."
        ),
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        required=False,
        default=0,
        help="CUDA device ID to use for ESMFold inference. Defaults to 0.",
    )
    parser.add_argument(
        "--usalign_path",
        type=str,
        required=False,
        default="USalign",
        help=(
            "Path to the USalign executable. Defaults to 'USalign', "
            "assuming it is available on PATH."
        ),
    )
    parser.add_argument(
        "--skip_structure_validation",
        action="store_true",
        help=(
            "Skip the ESMFold structure-prediction validation stage."
        ),
    )
    parser.add_argument(
        "--tm_threshold",
        type=float,
        required=False,
        default=0.5,
        help=(
            "Minimum TM-score (0–1) required to accept a designed structure. "
            "Structures below this value are flagged as rejected. Defaults to 0.5."
        ),
    )
    parser.add_argument(
        "--rmsd_threshold",
        type=float,
        required=False,
        default=5.0,
        help=(
            "Maximum Cα RMSD (Å) allowed for a designed structure to be accepted. "
            "Structures above this value are flagged as rejected. Defaults to 5.0."
        ),
    )

    return parser.parse_args()


def flatten_chain(matrix):
    """Flatten a list of lists into a single list.

    Args:
        matrix (list[list]): Nested list to flatten.

    Returns:
        list: Flattened list.
    """
    import itertools

    return list(itertools.chain.from_iterable(matrix))


def wrap_labels(ax, width, break_long_words=True):
    """Wrap x-axis tick labels of a matplotlib Axes to a given character width.

    Args:
        ax (matplotlib.axes.Axes): Axes whose x-tick labels will be wrapped.
        width (int): Maximum line width in characters.
        break_long_words (bool): Whether to break words longer than ``width``.
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


def calculate_lddt(predicted_pdb, reference_pdb):
    """Calculate the lDDT score between a predicted and a reference structure.

    Only Cα atoms are compared to handle structures with different atom counts.

    Args:
        predicted_pdb (str): Path to the predicted structure PDB file.
        reference_pdb (str): Path to the reference structure PDB file.

    Returns:
        float | None: lDDT score on a 0–1 scale, or None if calculation fails.
    """
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*elements were guessed.*")
            ref_file = pdb.PDBFile.read(reference_pdb)
            pred_file = pdb.PDBFile.read(predicted_pdb)

            ref_structure = ref_file.get_structure(model=1)
            pred_structure = pred_file.get_structure(model=1)

        ref_ca = ref_structure[ref_structure.atom_name == "CA"]
        pred_ca = pred_structure[pred_structure.atom_name == "CA"]

        if len(ref_ca) != len(pred_ca):
            print(
                f"Warning: Mismatched CA atom counts - ref: {len(ref_ca)}, pred: {len(pred_ca)}"
            )
            return None

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

        rmsd_match = re.search(r"RMSD=\s+([0-9.]+)", output)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None

        return tm_score, rmsd

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error running USalign: {e}")
        print("Make sure USalign is installed and in PATH")
        return None, None


def shorten_secondary_structure(ss):
    """Remove consecutive duplicate characters from a secondary structure string.

    Args:
        ss (str): Secondary structure string (e.g. "HHHEEECCC").

    Returns:
        str: String with consecutive duplicate characters collapsed to one.
    """
    shortened_ss = ""
    prev_char = ""
    for char in ss:
        if char != prev_char:
            shortened_ss += char
        prev_char = char
    return shortened_ss


def get_dssp(pdb_file, simplify=False):
    """Compute secondary structure assignments for a PDB file using DSSP.

    Secondary structure symbols are mapped to three classes:
    H (helix), E (strand), C (coil/other) via ``DSSP_SIMPLIFICATION``.

    Args:
        pdb_file (str): Path to the input PDB file.
        simplify (bool): If True, collapse consecutive duplicate symbols and
            remove coil ('C') characters, returning a compact motif string.

    Returns:
        str: Secondary structure string using the simplified H/E/C alphabet,
            optionally compacted when ``simplify=True``.
    """
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)

    header = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp_file:
        tmp_filename = tmp_file.name
        tmp_file.write(header)
        pdb_io = PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(tmp_file)

    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", tmp_filename)

        model = structure[0]
        dssp = DSSP(model, tmp_filename)

        secondary_structure = "".join([dssp[key][2] for key in dssp.keys()])

        simplified_string = "".join(
            DSSP_SIMPLIFICATION[symbol] for symbol in secondary_structure
        )

    finally:
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

    if simplify:
        shortened_ss = shorten_secondary_structure(simplified_string)
        return shortened_ss.replace("C", "")
    else:
        return simplified_string


def get_best_sequences(path):
    """Load sequences and ProteinMPNN global scores from FASTA files.

    Args:
        path (str or Path): Path to a directory containing ``.fa`` files, or path to a
            single ``.pdb`` file whose corresponding sequence file is resolved
            automatically.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of
            (sequences, protein names, scores), one entry per FASTA file.
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
    """Return the single best-scoring sequence from a ProteinMPNN output file.

    The best sequence is determined by the minimum ProteinMPNN global score.

    Args:
        path (str or Path): Path to a sequence file or directory (see ``get_best_sequences``).

    Returns:
        tuple[str, str]: (best_sequence, corresponding_pdb_filename).
    """
    seqs, pname, score = get_best_sequences(path)

    best_i = np.argmin(score)
    best_seq = seqs[best_i]
    best_name = pname[best_i]

    return best_seq, best_name.replace(".fa", ".pdb")


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

    if isinstance(pdb_string, list):
        pdb_string = pdb_string[0]

    if len(pdb_string) == 0:
        print("No PDB string generated")
        return

    with open(filename, "w") as f:
        f.write(pdb_string)


def predict_structure_esmfold(sequence, model, tokenizer, output_path):
    """Predict a protein structure with ESMFold and save it to a PDB file.

    Args:
        sequence (str): Amino acid sequence in single-letter code.
        model (EsmForProteinFolding): Loaded ESMFold model.
        tokenizer (AutoTokenizer): Corresponding ESMFold tokenizer.
        output_path (str): Path at which to save the predicted PDB file.

    Returns:
        float: Mean per-residue pLDDT score.
    """
    plddt, positions, model_outputs = esmfold_inference(sequence, model, tokenizer)
    write_coords_to_pdb(model, model_outputs, output_path)
    mean_plddt = plddt.mean()
    return mean_plddt


def validate_structure_with_esmfold(path, model, tokenizer, usalign_path):
    """Validate a designed structure by predicting it with ESMFold and comparing metrics.

    For the best-scoring sequence in ``path``, predicts a structure with ESMFold
    and compares it to the template PDB using DSSP secondary structure matching
    and TM-score / RMSD (via USalign).

    Args:
        path (str): Path to the ProteinMPNN sequence directory for a given design.
        model (EsmForProteinFolding): Loaded ESMFold model.
        tokenizer (AutoTokenizer): Corresponding ESMFold tokenizer.
        usalign_path (str): Path to the USalign executable.

    Returns:
        dict: Validation results with keys: Path, Class, Sequence, pLDDT,
            TM_score, RMSD, DSSP, Target_DSSP, Predicted_DSSP.
    """
    seq, pname, score = get_best_sequences(path)

    minargs = np.argmin(score, axis=1)
    best_seqs = [seq[i, j] for i, j in enumerate(minargs)]

    results = {}

    for sequence, name in zip(best_seqs, pname):
        plddt, positions, model_outputs = esmfold_inference(sequence, model, tokenizer)
        mean_plddt = plddt.mean()
        if mean_plddt < 1.0:
            mean_plddt *= 100.0

        base_pdb = str(Path(path).parent) + ".pdb"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_ESMfold.pdb", delete=False
        ) as tmp_file:
            output_path = tmp_file.name
            write_coords_to_pdb(model, model_outputs, output_path)

            try:
                dssp_base = get_dssp(base_pdb, simplify=True)
                new_dssp = get_dssp(output_path, simplify=True)

                if len(new_dssp) == 0:
                    final_output_path = path[:-5] + name + "_ESMfold.pdb"
                    write_coords_to_pdb(model, model_outputs, final_output_path)
                    new_dssp = get_dssp(final_output_path, simplify=True)
                    os.remove(final_output_path)

                dssp_match = dssp_base.count("H") == new_dssp.count(
                    "H"
                ) and dssp_base.count("E") == new_dssp.count("E")

                tm_score, rmsd = calculate_tm_score(usalign_path, output_path, base_pdb)

                results = {
                    "Path": path,
                    "Class": path.split("/")[-3],
                    "Sequence": sequence,
                    "pLDDT": mean_plddt,
                    "TM_score": tm_score,
                    "RMSD": rmsd,
                    "DSSP": "True" if dssp_match else "False",
                    "Target_DSSP": dssp_base,
                    "Predicted_DSSP": new_dssp,
                }
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue

        if os.path.exists(output_path):
            os.remove(output_path)

    return results


def collect_designed_structures(data_path):
    """Find all designed structure PDB files that have a corresponding design directory.

    Iterates over all fold types defined in ``FOLD_NAME_DICT`` and returns PDB
    files for which a matching subdirectory of designed structures exists.

    Args:
        data_path (str): Root directory of the protein evolution data.

    Returns:
        list[str]: Flat list of valid template PDB file paths.
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
    """Process one template and all its associated designed structures.

    Computes TM-score, RMSD, lDDT, and DSSP for each design relative to the
    template, and evaluates acceptance based on the supplied thresholds.

    Note:
        The ``Accepted`` flag requires all three criteria (DSSP match,
        TM-score ≥ threshold, RMSD ≤ threshold) to be True simultaneously.
        In practice, DSSP matching is the primary filter at this stage; TM-score
        and RMSD are used for downstream inspection and visualisation.

    Args:
        template_path (str): Path to the template PDB file.
        usalign_path (str): Path to the USalign executable.
        tm_threshold (float): Minimum TM-score for acceptance.
        rmsd_threshold (float): Maximum RMSD (Å) for acceptance.

    Returns:
        list[dict]: One result dictionary per designed structure, containing
            TM_score, RMSD, LDDT, DSSP, Fold, Short_Fold, Accepted, Paths,
            Target_DSSP, and New_DSSP fields.
    """
    results = []

    try:
        dssp_base = get_dssp(template_path, simplify=True)
    except Exception:
        return results

    pathdir = template_path[:-4]
    pdbfiles = glob.glob(os.path.join(pathdir, "*.pdb"))
    pdbfiles = [f for f in pdbfiles if "ESMfold" not in os.path.basename(f)]

    for newpdb in pdbfiles:
        try:
            tm_score, rmsd = calculate_tm_score(usalign_path, newpdb, template_path)
            new_dssp = get_dssp(newpdb, simplify=True)
            lddt_score = calculate_lddt(newpdb, template_path)

            dssp_match = dssp_base.count("H") == new_dssp.count(
                "H"
            ) and dssp_base.count("E") == new_dssp.count("E")

            accepted = (
                dssp_match
                and (tm_score is not None and tm_score >= tm_threshold)
                and (rmsd is not None and rmsd <= rmsd_threshold)
            )

            namelist = newpdb.split("/")

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
    """Analyse all designed structures in parallel and save results to CSV.

    Distributes template processing across CPU workers using
    ``multiprocessing.Pool``. Results are collected into a DataFrame and
    written to ``output_csv``.

    Args:
        template_files (list[str]): List of template PDB file paths to process.
        output_csv (str): Path for the output CSV file.
        usalign_path (str): Path to the USalign executable.
        tm_threshold (float): Minimum TM-score threshold for acceptance.
        rmsd_threshold (float): Maximum RMSD (Å) threshold for acceptance.

    Returns:
        pd.DataFrame: Results with columns TM_score, RMSD, LDDT, DSSP, Fold,
            Short_Fold, Accepted, Paths, Target_DSSP, New_DSSP.
    """
    process_func = partial(
        _process_single_template,
        usalign_path=usalign_path,
        tm_threshold=tm_threshold,
        rmsd_threshold=rmsd_threshold,
    )

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    all_results = []

    print(f"Analyzing structures using {num_workers} cores...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(process_func, template_files)

        for batch_result in tqdm(
            results_iterator, total=len(template_files), desc="Analyzing structures"
        ):
            all_results.extend(batch_result)

    df = pd.DataFrame(all_results)

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


def add_polyalanine_count(
    df: pd.DataFrame,
    sequence_col: str = "Sequence",
    new_col: str = "num_polyA",
    min_len: int = 4,
) -> pd.DataFrame:
    """Count contiguous polyalanine runs in protein sequences.

    Adds a new column to ``df`` recording the number of runs of consecutive
    alanine residues of at least ``min_len`` in each sequence.

    Args:
        df (pd.DataFrame): Input DataFrame.
        sequence_col (str): Column name containing amino acid sequences.
        new_col (str): Name for the new count column.
        min_len (int): Minimum run length to qualify as a polyalanine region.

    Returns:
        pd.DataFrame: Input DataFrame with ``new_col`` added.
    """
    pattern = re.compile(f"A{{{min_len},}}")

    def count_polyA(seq):
        if not isinstance(seq, str):
            return 0
        return len(pattern.findall(seq))

    df[new_col] = df[sequence_col].apply(count_polyA)
    return df


def validate_structures_with_esmfold(df_accept, output_csv, usalign_path, gpu_id=0):
    """Run ESMFold validation on a set of accepted structures.

    For each structure in ``df_accept``, predicts the sequence with ESMFold and
    computes pLDDT, TM-score, RMSD, and DSSP agreement relative to the template.

    Args:
        df_accept (pd.DataFrame): DataFrame of structures to validate;
            must contain a 'Paths' column.
        output_csv (str): Path for the output validation CSV file.
        usalign_path (str): Path to the USalign executable.
        gpu_id (int): CUDA device ID for ESMFold inference.

    Returns:
        pd.DataFrame: Validation results, one row per structure.
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
        
        if "Short_Fold" not in df_struct.columns:
            df_struct["Short_Fold"] = df_struct["Class"].apply(
                lambda x: SHORT_FOLD_NAME_DICT.get(x, x)
            )

        print("Calculating polyalanine counts...")
        df_struct = add_polyalanine_count(df_struct)
        
        df_struct.to_csv(output_csv, index=False)

    return df_struct


def main():
    args = parse_arguments()

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


    print("Pipeline complete!")

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
        f"Mean polyalanine count (more than 4 A in a row): {df_struct['num_polyA'].mean():.3f} ± {df_struct['num_polyA'].std():.3f}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()