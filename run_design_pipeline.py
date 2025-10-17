import argparse
import glob
import os
import re
import subprocess
from itertools import islice

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scope_database",
        type=str,
        required=False,
        default="/home/shared/databases/SCOPe/unrelaxed/native/",
        help="Path to SCOPe structure database.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=False,
        default="/mnt/nasdata/sofia/Protein_Evo/",
        help="Where to store the output files.",
    )
    parser.add_argument(
        "--astral_path",
        type=str,
        required=False,
        default="/home/sofia/RFDiffusion/RFdiffusion/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa",
        help="Path to SCOPe ASTRAL fasta file.",
    )

    args = parser.parse_args()

    return args.scope_database, args.out_path, args.astral_path


def extract_atoms_from_model(input_pdb_file, output_pdb_file, target_model_id):
    """
    Extracts atoms from a specific model in a PDB file and writes them to a new PDB file.
    Args:
        input_pdb_file (str): Path to the input PDB file.
        output_pdb_file (str): Path to the output PDB file where the extracted atoms will be written.
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
    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - 'Length': Length of the protein sequences.
            - 'Classes': Class identifiers of the protein sequences.
            - 'Seqs': The protein sequences.
            - 'Names': Class names derived from the class identifiers.
            - 'Subclass': Subclass identifiers of the protein sequences.
            - 'SCOPe name': Names of the protein sequences from the file.
    """

    matches = ["c.1.", "c.2.", "c.37.", "d.58", "a.4", "c.23", "c.55", "b.40", "c.66"]

    name_dict = {
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
                result1 = re.findall("c\.(1|2|23|37|55|66)\.\d*\.\d*", line)
                result2 = re.findall("d\.(58)\.\d*\.\d*", line)
                result3 = re.findall("a\.(4)\.\d*\.\d*", line)
                result4 = re.findall("b\.(40)\.\d*\.\d*", line)
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
                result1 = re.findall("c\.(1|2|23|37|55|66)\.\d*\.\d*", line)
                result2 = re.findall("d\.58\.\d*\.\d*", line)
                result3 = re.findall("a\.4\.\d*\.\d*", line)
                result4 = re.findall("b\.40\.\d*\.\d*", line)
                if len(result1) != 0:
                    temp_sub.append(
                        re.search("c\.(1|2|23|37|55|66)\.\d*\.\d*", line).group()
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

    df_short = df[df["Length"] <= 100]

    return df_short[df_short["Length"] >= 50]


def validate_sequences(outdir):
    """
    Validates the generated sequences by checking if they are valid and only have the expected amino acids.
    Args:
        outdir (str): The directory where the generated sequences are stored.
    Returns:
        bool : True if all sequences are valid, False otherwise.
    """

    bad_aa = ["N", "K", "Q", "R", "C", "H", "F", "M", "Y", "W"]

    path = outdir + "/seqs/"

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


def main():
    scopepath, out_prefix, astral_path = parse_arguments()
    df = extract_scope(astral_path)

    scope_names = df["SCOPe name"].tolist()
    scope_class = df["Names"].tolist()

    basepdb = []
    inputpdb = []
    outscaff = []
    outpath = []
    outdir = []

    for name, cl in zip(scope_names, scope_class):
        basepdb.append(scopepath + name[2:4] + "/" + name + ".ent")

        temp_dir = out_prefix + cl

        try:
            os.makedirs(temp_dir, exist_ok=True)
            print("Directory '%s' created successfully" % temp_dir)
        except OSError:
            print("Directory '%s' can not be created" % temp_dir)

        inputpdb.append(out_prefix + cl + "/" + name + ".pdb")
        outscaff.append(out_prefix + cl + "/SStruct/" + name)
        outpath.append(out_prefix + cl + "/" + name + "/test_" + name)
        outdir.append(out_prefix + cl + "/" + name)

    for bpdb, ipdb, oscaff, opath, odir in tqdm(
        zip(basepdb, inputpdb, outscaff, outpath, outdir),
        total=len(basepdb),
        desc="Designing proteins",
    ):
        # Skip if file already exists
        if os.path.isfile(ipdb):
            continue
        else:
            if os.path.isfile(bpdb):
                extract_atoms_from_model(bpdb, ipdb, "A")
            # Skip if file does not exist in database
            else:
                continue

        # STRUCTURE DESIGN
        command1 = (
            "./helper_scripts/make_secstruc_adj.py --input_pdb "
            + ipdb
            + " --out_dir "
            + oscaff
        )
        command2 = (
            "./scripts/run_inference.py inference.output_prefix="
            + opath
            + " scaffoldguided.scaffoldguided=True scaffoldguided.target_pdb=False scaffoldguided.scaffold_dir="
            + oscaff
        )

        subprocess.run(command1, shell=True)
        subprocess.run(command2, shell=True)

        # SEQUENCE DESIGN
        bias_command = (
            "python ./sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/helper_scripts/make_bias_AA.py --output_path "
            + odir
            + "/bias.jsonl \
            --AA_list 'N K Q R C H F M Y W'\
            --bias_list '-10 -10 -10 -10 -10 -10 -10 -10 -10 -10'"
        )

        subprocess.run(bias_command, shell=True)

        json_command = (
            "python ./sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/helper_scripts/parse_multiple_chains.py --input_path "
            + odir
            + " --output_path "
            + odir
            + "/test.jsonl"
        )

        subprocess.run(json_command, shell=True)

        command3 = (
            "python ./sequence_design/dl_binder_design/mpnn_fr/ProteinMPNN/protein_mpnn_run.py --num_seq_per_target=20 --batch_size=10 --out_folder="
            + odir
            + " --jsonl_path="
            + odir
            + "/test.jsonl  --bias_AA_jsonl "
            + odir
            + "/bias.jsonl"
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
