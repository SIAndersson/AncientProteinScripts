import argparse
import re
import shlex
import subprocess
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scope_database",
        type=str,
        required=True,
        help="Path to SCOPe structure database.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Where to store the output files.",
    )
    parser.add_argument(
        "--astral_path",
        type=str,
        required=True,
        help="Path to SCOPe ASTRAL fasta file.",
    )
    parser.add_argument(
        "--rfdiffusion_path",
        type=str,
        required=True,
        help="Path to RFdiffusion repository.",
    )
    parser.add_argument(
        "--rfdiffusion_python_path",
        type=str,
        required=True,
        help="Path to RFdiffusion Python environment.",
    )

    args = parser.parse_args()

    # Convert all paths to Path objects
    return (
        Path(args.scope_database),
        Path(args.out_path),
        Path(args.astral_path),
        Path(args.rfdiffusion_path),
        Path(args.rfdiffusion_python_path),
    )


def extract_atoms_from_model(input_pdb_file, output_pdb_file, target_model_id):
    """
    Extracts atoms from a specific model in a PDB file and writes them to a new PDB file.
    Args:
        input_pdb_file (Path): Path to the input PDB file.
        output_pdb_file (Path): Path to the output PDB file where the extracted atoms will be written.
        target_model_id (str): The model ID from which atoms should be extracted.
    Returns:
        None
    """

    with open(input_pdb_file, "r") as input_file:
        with open(output_pdb_file, "w") as output_file:
            for line in input_file:
                if line.startswith("MODEL") and "1" not in line:
                    break
                if line.startswith("ATOM") and line[21] == target_model_id:
                    # Extract only the necessary information from the ATOM line
                    output_file.write(line)


def sort_list(list1, list2):
    """
    Sorts list1 based on the lengths of the elements in list2.
    Args:
        list1 (list): The list to be sorted.
        list2 (list): The list whose element lengths determine the sort order.
    Returns:
        list: A new list containing the elements of list1 sorted based on the lengths of the corresponding elements in list2.
    """

    zipped_pairs = zip(list2, list1)

    z = [x for _, x in sorted(zipped_pairs, key=len)]

    return z


def extract_scope(path):
    """
    Extracts and processes protein sequences and their classifications from a specified file.
    This function reads a file containing protein sequences and their SCOPe classifications,
    extracts relevant sequences and their associated metadata, and returns a DataFrame
    containing sequences with lengths between 50 and 100 amino acids.
    Args:
        path (Path): Path to the ASTRAL fasta file.
    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'Length': Length of the protein sequences.
            - 'Classes': Class identifiers of the protein sequences.
            - 'Seqs': The protein sequences.
            - 'Names': Class names derived from the class identifiers.
            - 'Subclass': Subclass identifiers of the protein sequences.
            - 'SCOPe name': Names of the protein sequences from the file.
    """

    matches = [
        "c.1.",
        "c.2.",
        "c.3.",
        "c.23.",
        "c.26.",
        "c.36.",
        "c.37.",
        "c.55.",
        "c.66.",
        "c.67.",
        "c.94.",
        "d.58",
        "a.4",
        "b.40",
        "b.84",
    ]

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

    res = min(temp_seqs, key=len)
    arg = np.argwhere(seqs == res)

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

    df_short = df[df["Length"] <= 150]

    return df_short[df_short["Length"] >= 50]


def validate_sequences(outdir):
    """
    Validates the generated sequences by checking if they are valid and only have the expected amino acids.
    Args:
        outdir (Path): The directory where the generated sequences are stored.
    Returns:
        bool : True if all sequences are valid, False otherwise.
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

    # See if all sequences are valid (e.g. only the desired amino acids are present)
    return np.all(bool_arr == True)


def print_class_distribution(df):
    """
    Prints the number of sequences found for each SCOPe class.

    Args:
        df (pd.DataFrame): DataFrame containing the sequences with 'Names' column for class names.

    Returns:
        None
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


def remove_designed_sequences(df):
    """
    Removes sequences that have already been designed based on specific criteria.

    Filters out sequences with lengths between 50 and 100 (inclusive) for the
    following classes that have already been designed:
    - c.37 (P-fold_Hydrolase_c.37)
    - a.4 (DNA_RNA-binding_3-helical_a.4)
    - c.1 (TIM-barrel_c.1)
    - c.2 (Rossman-fold_c.2)
    - d.58 (Ferredoxin_d.58)
    - c.23 (Flavodoxin-like_c.23)
    - c.55 (Ribonuclease_H-like_motif_c.55)
    - b.40 (OB-fold_greek-key_b.40)
    - c.66 (Nucleoside_Hydrolase_c.66)

    Args:
        df (pd.DataFrame): DataFrame containing sequences with 'Length' and 'Classes' columns.

    Returns:
        pd.DataFrame: Filtered DataFrame with already-designed sequences removed.
    """
    # Classes that have already been designed
    designed_classes = [37, 4, 1, 2, 58, 23, 55, 40, 66]

    # Create mask for sequences to remove (length 50-100 AND in designed classes)
    mask_to_remove = (
        (df["Length"] >= 50)
        & (df["Length"] <= 100)
        & (df["Classes"].isin(designed_classes))
    )

    # Return dataframe with these sequences removed
    return df[~mask_to_remove]


def main():
    scopepath, out_prefix, astral_path, rfdiffusion_path, rfdiffusion_python_path = (
        parse_arguments()
    )
    df = extract_scope(astral_path)
    df = remove_designed_sequences(df)

    # Print class distribution before starting design loop
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
        # Skip if output directory already exists
        if odir.is_dir():
            print(f"Skipping {odir}, output already exists.")
            continue
        else:
            if bpdb.is_file():
                extract_atoms_from_model(bpdb, ipdb, "A")
            # Skip if file does not exist in database
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

    # Validate sequences
    valid = validate_sequences(out_prefix)

    if valid:
        print("All sequences are valid.")
    else:
        print("Some sequences are invalid.")

    print("Design pipeline finished.")


if __name__ == "__main__":
    main()
