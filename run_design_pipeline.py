import argparse
import re
import shlex
import subprocess
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Classes used in the initial study (50–100 residues)
INITIAL_CLASSES = [37, 4, 1, 2, 58, 23, 55, 40, 66]
INITIAL_LENGTH_RANGE = (50, 100)

# Additional classes introduced in the extension study (50–150 residues)
EXTENSION_CLASSES = [3, 26, 36, 67, 94, 84]
ALL_CLASSES = INITIAL_CLASSES + EXTENSION_CLASSES
EXTENSION_LENGTH_RANGE = (50, 150)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scaffold-guided protein structure and sequence design pipeline using RFdiffusion and ProteinMPNN."
    )

    parser.add_argument(
        "--scope_database",
        type=str,
        required=True,
        help=(
            "Path to the SCOPe structure database directory. "
            "Expects PDB-style subdirectory layout (e.g. db/ab/d1abc__.ent)."
        ),
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Root output directory. Subdirectories are created per fold class.",
    )
    parser.add_argument(
        "--astral_path",
        type=str,
        required=True,
        help="Path to the SCOPe ASTRAL FASTA file used to extract sequences and class annotations.",
    )
    parser.add_argument(
        "--rfdiffusion_path",
        type=str,
        required=True,
        help="Path to the RFdiffusion repository root (contains scripts/ and helper_scripts/).",
    )
    parser.add_argument(
        "--rfdiffusion_python_path",
        type=str,
        required=True,
        help="Path to the Python environment used to run RFdiffusion (e.g. /path/to/conda/envs/rfdiffusion).",
    )
    parser.add_argument(
        "--extension",
        action="store_true",
        default=False,
        help=(
            "Run the extension study instead of the initial study. "
            f"Uses length range {EXTENSION_LENGTH_RANGE[0]}–{EXTENSION_LENGTH_RANGE[1]} residues "
            f"and fold classes {EXTENSION_CLASSES} (excludes initial classes {INITIAL_CLASSES}, "
            "which were already designed at lengths 50–100)."
        ),
    )

    args = parser.parse_args()

    return (
        Path(args.scope_database),
        Path(args.out_path),
        Path(args.astral_path),
        Path(args.rfdiffusion_path),
        Path(args.rfdiffusion_python_path),
        args.extension,
    )


def extract_atoms_from_model(input_pdb_file, output_pdb_file, target_model_id):
    """Extract ATOM records for a single chain from a PDB file.

    Reads until the second MODEL record (if present) and writes only ATOM
    lines whose chain identifier matches ``target_model_id``.

    Args:
        input_pdb_file (Path): Input PDB file to read.
        output_pdb_file (Path): Output PDB file to write.
        target_model_id (str): Chain ID to extract (e.g. ``"A"``).
    """
    with open(input_pdb_file, "r") as input_file:
        with open(output_pdb_file, "w") as output_file:
            for line in input_file:
                if line.startswith("MODEL") and "1" not in line:
                    break
                if line.startswith("ATOM") and line[21] == target_model_id:
                    output_file.write(line)


def sort_list(list1, list2):
    """Sort list1 by the lengths of the corresponding elements in list2.

    Args:
        list1 (list): List to sort.
        list2 (list): List whose element lengths determine sort order.

    Returns:
        list: Elements of list1 sorted by ascending length of the paired
        element in list2.
    """
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs, key=len)]
    return z


def extract_scope(path, length_min, length_max):
    """Parse an ASTRAL FASTA file and return sequences within a length range.

    Sequences are matched against a fixed set of SCOPe fold classes across the
    a, b, c, and d hierarchies. Results are sorted by ascending
    sequence length.

    Args:
        path (Path): Path to the ASTRAL FASTA file.
        length_min (int): Minimum sequence length (inclusive).
        length_max (int): Maximum sequence length (inclusive).

    Returns:
        pd.DataFrame: DataFrame with columns:

            - Length (int): Sequence length.
            - Classes (int): SCOPe class number.
            - Seqs (str): Amino acid sequence.
            - Names (str): Human-readable fold class name.
            - Subclass (str): Full SCOPe subclass identifier (e.g. ``c.1.2.3``).
            - SCOPe name (str): Seven-character SCOPe domain identifier.
    """
    name_dict = {
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

    with open(path) as f:
        temp_string = []
        temp_seqs = []
        temp_class = []
        temp_sub = []
        temp_name = []
        correct_match = False
        for line in f:
            if correct_match and not line.startswith(">"):
                temp_string.append(line.strip())
                continue
            else:
                if len(temp_string) != 0:
                    temp = " ".join(temp_string).strip().upper()
                    temp_seqs.append(re.sub(r"[\n\t\s]*", "", temp))
                    temp_string = []
                correct_match = False
                result1 = re.findall(
                    "c\.(1|2|3|23|26|36|37|55|66|67|94)\.\d*\.\d*", line
                )
                result2 = re.findall("d\.(58)\.\d*\.\d*", line)
                result3 = re.findall("a\.(4)\.\d*\.\d*", line)
                result4 = re.findall("b\.(40|84)\.\d*\.\d*", line)
                if len(result1) != 0:
                    temp_class.append(result1[0])
                    correct_match = True
                elif len(result2) != 0:
                    temp_class.append(result2[0])
                    correct_match = True
                elif len(result3) != 0:
                    temp_class.append(result3[0])
                    correct_match = True
                elif len(result4) != 0:
                    temp_class.append(result4[0])
                    correct_match = True
                result1 = re.findall(
                    "c\.(1|2|3|23|26|36|37|55|66|67|94)\.\d*\.\d*", line
                )
                result2 = re.findall("d\.58\.\d*\.\d*", line)
                result3 = re.findall("a\.4\.\d*\.\d*", line)
                result4 = re.findall("b\.(40|84)\.\d*\.\d*", line)
                if len(result1) != 0:
                    temp_sub.append(
                        re.search(
                            "c\.(1|2|3|23|26|36|37|55|66|67|94)\.\d*\.\d*", line
                        ).group()
                    )
                elif len(result2) != 0:
                    temp_sub.append(result2[0])
                elif len(result3) != 0:
                    temp_sub.append(result3[0])
                elif len(result4) != 0:
                    temp_sub.append(result4[0])
                if correct_match and line.startswith(">"):
                    temp_name.append(line[1:8])

    f.close()

    seqs = np.array(temp_seqs)
    classes = np.array(temp_class)
    subclass = np.array(temp_sub)
    names = np.array(temp_name)

    sort_seqs = np.array(sorted(seqs, key=len))
    args = []
    lens = []
    for x in sort_seqs:
        temp = np.argwhere(seqs == x)
        args.append(temp[0][0])
        lens.append(len(x))

    classnames = []
    for x in classes[args]:
        classnames.append(name_dict[x])

    df = pd.DataFrame(
        {
            "Length": lens,
            "Classes": classes[args].astype(np.int64),
            "Seqs": sort_seqs,
            "Names": classnames,
            "Subclass": subclass[args],
            "SCOPe name": names[args],
        }
    )

    return df[(df["Length"] >= length_min) & (df["Length"] <= length_max)]


def validate_sequences(outdir):
    """Check that all designed sequences contain only the permitted amino acids.

    Reads .fa files from outdir/seqs/, skipping the first two lines of
    each file (header and native sequence). A sequence is considered invalid if
    it contains any amino acid in the biased-against set
    {N, K, Q, R, C, H, F, M, Y, W}.

    Args:
        outdir (Path): Root output directory containing the seqs/ subdirectory.

    Returns:
        bool: True if every sequence in every file passes validation,
        False otherwise.
    """
    bad_aa = ["N", "K", "Q", "R", "C", "H", "F", "M", "Y", "W"]

    seqs_path = outdir / "seqs"
    files = list(seqs_path.glob("*.fa"))

    seq = []
    score = []
    pname = []

    for fname in files:
        if fname.suffix == ".fa":
            pname.append(fname.stem)
            with open(fname) as f:
                temp_seq = []
                temp_score = []
                for line in islice(f, 2, None):
                    result = re.findall("global_score=\d*\.?\d*", line)
                    if len(result) == 0:
                        temp_seq.append(line.replace("\n", ""))
                    else:
                        sc = result[0].replace("global_score=", "")
                        temp_score.append(float(sc))
            seq.append(np.array(temp_seq, dtype=str))
            score.append(np.array(temp_score))
            f.close()

    seq = np.array(seq)
    score = np.array(score)
    pname = np.array(pname)

    bool_arr = []

    for ss in seq:
        temp_bool = []
        for s in ss:
            arr = [1 for e in bad_aa if e in s]
            if len(arr) == 0:
                temp_bool.append(True)
            else:
                temp_bool.append(False)
        bool_arr.append(np.array(temp_bool, dtype=bool))

    bool_arr = np.array(bool_arr)

    return np.all(bool_arr == True)


def print_class_distribution(df):
    """Print a summary table of sequence counts per SCOPe fold class.

    Args:
        df (pd.DataFrame): DataFrame with a Names column containing fold
            class labels.
    """
    print("\n" + "=" * 70)
    print("SCOPe Class Distribution")
    print("=" * 70)

    class_counts = df["Names"].value_counts().sort_index()
    total_sequences = len(df)

    for class_name, count in class_counts.items():
        percentage = (count / total_sequences) * 100
        print(f"{class_name:50s}: {count:4d} ({percentage:5.1f}%)")

    print("-" * 70)
    print(f"{'Total sequences':50s}: {total_sequences:4d}")
    print("=" * 70 + "\n")


def remove_designed_sequences(df, extension=False):
    """Remove sequences that were already designed in the initial study.

    In the extension study, sequences with lengths 50–100 belonging to the
    initial fold classes (INITIAL_CLASSES) are excluded to avoid
    redundant computation. No filtering is applied for the initial study.

    Args:
        df (pd.DataFrame): DataFrame with Length and Classes columns.
        extension (bool): If True, apply the extension-study filter.
            Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if extension:
        mask_to_remove = (
            (df["Length"] >= 50)
            & (df["Length"] <= 100)
            & (df["Classes"].isin(INITIAL_CLASSES))
        )
        return df[~mask_to_remove]
    else:
        return df


def main():
    scopepath, out_prefix, astral_path, rfdiffusion_path, rfdiffusion_python_path, extension = (
        parse_arguments()
    )

    if extension:
        length_min, length_max = EXTENSION_LENGTH_RANGE
        target_classes = ALL_CLASSES
        print(
            f"Running EXTENSION study: "
            f"length range {length_min}–{length_max} residues, "
            f"classes {target_classes}"
        )
    else:
        length_min, length_max = INITIAL_LENGTH_RANGE
        target_classes = INITIAL_CLASSES
        print(
            f"Running INITIAL study: "
            f"length range {length_min}–{length_max} residues, "
            f"classes {target_classes}"
        )

    df = extract_scope(astral_path, length_min, length_max)
    df = remove_designed_sequences(df, extension=extension)

    print_class_distribution(df)

    scope_names = df["SCOPe name"].tolist()
    scope_class = df["Names"].tolist()

    basepdb = []
    inputpdb = []
    outscaff = []
    outpath = []
    outdir = []

    for name, cl in zip(scope_names, scope_class):
        basepdb.append(scopepath / name[2:4] / f"{name}.ent")

        temp_dir = out_prefix / cl

        try:
            if not temp_dir.exists():
                temp_dir.mkdir(parents=True, exist_ok=True)
                print(f"Directory '{temp_dir}' created successfully")
        except OSError:
            print(f"Directory '{temp_dir}' can not be created")

        inputpdb.append(temp_dir / f"{name}.pdb")
        outscaff.append(temp_dir / "SStruct" / name)
        outpath.append(temp_dir / name / f"test_{name}")
        outdir.append(temp_dir / name)

    for bpdb, ipdb, oscaff, opath, odir in tqdm(
        zip(basepdb, inputpdb, outscaff, outpath, outdir),
        total=len(basepdb),
        desc="Designing proteins",
    ):
        if odir.is_dir():
            print(f"Skipping {odir}, output already exists.")
            continue
        else:
            if bpdb.is_file():
                extract_atoms_from_model(bpdb, ipdb, "A")
            else:
                continue

        # STRUCTURE DESIGN
        command1 = (
            f"{rfdiffusion_python_path}/bin/python {rfdiffusion_path}/helper_scripts/make_secstruc_adj.py --input_pdb "
            + shlex.quote(str(ipdb))
            + " --out_dir "
            + shlex.quote(str(oscaff))
        )

        # Need to escape parentheses for Hydra config parsing
        opath_escaped = (
            str(opath).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )
        oscaff_escaped = (
            str(oscaff).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        )

        command2 = (
            f"{rfdiffusion_python_path}/bin/python {rfdiffusion_path}/scripts/run_inference.py "
            f"'inference.output_prefix={opath_escaped}' "
            f"scaffoldguided.scaffoldguided=True "
            f"scaffoldguided.target_pdb=False "
            f"'scaffoldguided.scaffold_dir={oscaff_escaped}'"
        )

        subprocess.run(command1, shell=True)
        subprocess.run(command2, shell=True)

        # SEQUENCE DESIGN
        bias_command = (
            f"{rfdiffusion_python_path}/bin/python {rfdiffusion_path}/sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/helper_scripts/make_bias_AA.py --output_path "
            + shlex.quote(str(odir / "bias.jsonl"))
            + " --AA_list 'N K Q R C H F M Y W' --bias_list '-10 -10 -10 -10 -10 -10 -10 -10 -10 -10'"
        )

        subprocess.run(bias_command, shell=True)

        json_command = (
            f"{rfdiffusion_python_path}/bin/python {rfdiffusion_path}/sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path "
            + shlex.quote(str(odir))
            + " --output_path "
            + shlex.quote(str(odir / "test.jsonl"))
        )

        subprocess.run(json_command, shell=True)

        command3 = (
            f"{rfdiffusion_python_path}/bin/python {rfdiffusion_path}/sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/protein_mpnn_run.py --num_seq_per_target=20 --batch_size=10 --out_folder="
            + shlex.quote(str(odir))
            + " --jsonl_path="
            + shlex.quote(str(odir / "test.jsonl"))
            + " --bias_AA_jsonl "
            + shlex.quote(str(odir / "bias.jsonl"))
        )

        subprocess.run(command3, shell=True)

    valid = validate_sequences(out_prefix)

    if valid:
        print("All sequences are valid.")
    else:
        print("Some sequences are invalid.")

    print("Design pipeline finished.")


if __name__ == "__main__":
    main()