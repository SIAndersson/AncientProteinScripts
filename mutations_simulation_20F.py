import torch
import esm
import argparse
import pandas as pd
from Bio import SeqIO
import random
import os
import subprocess
import concurrent.futures
import glob
import gc
from Bio.PDB import PDBParser, Superimposer

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def mutate_sequence(seq, n_mutations):
    seq = list(seq)
    positions = random.sample(range(len(seq)), n_mutations)
    mutations = []
    for pos in positions:
        original = seq[pos]
        choices = [aa for aa in AMINO_ACIDS if aa != original]
        new_aa = random.choice(choices)
        seq[pos] = new_aa
        mutations.append((pos, original, new_aa))
    return ''.join(seq), mutations

def calculate_rmsd(pdb1, pdb2):
    parser = PDBParser(QUIET=True)
    structure1 = parser.get_structure('ref', pdb1)
    structure2 = parser.get_structure('alt', pdb2)
    atoms1 = [atom for atom in structure1[0].get_atoms() if atom.get_id() == 'CA']
    atoms2 = [atom for atom in structure2[0].get_atoms() if atom.get_id() == 'CA']
    if len(atoms1) != len(atoms2):
        raise ValueError("Structures have different number of CA atoms.")
    sup = Superimposer()
    sup.set_atoms(atoms1, atoms2)
    rmsd = sup.rms
    return rmsd

def run_tmalign(pdb1, pdb2):
    try:
        result = subprocess.run(["TMalign", pdb1, pdb2], capture_output=True, text=True, check=True)
        output = result.stdout
        tm_score = None
        for line in output.splitlines():
            if line.strip().startswith("TM-score="):
                tm_score = float(line.split()[1])
        return tm_score
    except Exception as e:
        print(f"TMalign failed: {e}")
        return None

def run_simulation(pdb_file, fasta_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    # Load model only when needed and ensure it's on GPU
    try:
        model = esm.pretrained.esmfold_v1()
        model = model.eval().cuda()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    record = SeqIO.read(fasta_file, "fasta")
    wildtype_sequence = str(record.seq)
    results = []
    
    for n_mut in range(1, len(wildtype_sequence)+1):
        try:
            mut_seq, mutations = mutate_sequence(wildtype_sequence, n_mut)
            pdb_out = os.path.join(outdir, f"mutant_{n_mut}.pdb")
            
            # Clear cache before inference
            torch.cuda.empty_cache()
            gc.collect()
            
            with torch.no_grad():
                output = model.infer_pdb(mut_seq)
            
            with open(pdb_out, "w") as f:
                f.write(output)
            
            mutation_info = {
                'n_mutations': n_mut,
                'mutations': mutations,
                'mutant_sequence': mut_seq
            }
            
            # Calculate TM-score using TMalign
            tm_score = run_tmalign(pdb_file, pdb_out)
            
            # Calculate RMSD using Biopython
            try:
                rmsd = calculate_rmsd(pdb_file, pdb_out)
            except Exception as e:
                print(f"RMSD calculation failed: {e}")
                rmsd = None
            
            mutation_info['tm_score'] = tm_score
            mutation_info['rmsd'] = rmsd
            results.append(mutation_info)
            print(f"[{outdir}] Mutations: {mutations}, TM-score: {tm_score}, RMSD: {rmsd}")
            
            # Clear cache after each mutation
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing mutation {n_mut} in {outdir}: {e}")
            continue
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(outdir, "mutation_results.csv"), index=False)
    
    # Aggressive memory cleanup
    del model, df, results, wildtype_sequence, record
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Mutation simulation script")
    parser.add_argument('--pdb', type=str, required=True, help='Path to wild type PDB file')
    parser.add_argument('--fasta', type=str, required=True, help='Path to wild type FASTA file')
    parser.add_argument('--outdir', type=str, default="mutant_structures", help='Output directory')
    parser.add_argument('--n_runs', type=int, default=20, help='Number of parallel runs')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of parallel workers')
    args = parser.parse_args()
    
    base_outdir = args.outdir
    n_runs = args.n_runs
    max_workers = args.max_workers
    pdb_file = args.pdb
    fasta_file = args.fasta
    
    os.makedirs(base_outdir, exist_ok=True)
    run_args = [(pdb_file, fasta_file, os.path.join(base_outdir, f"run_{i+1}")) for i in range(n_runs)]
    
    # If max_workers is 1, run sequentially to avoid memory issues
    if max_workers == 1:
        print("Running simulations sequentially to avoid GPU memory conflicts...")
        for i, arg in enumerate(run_args):
            print(f"Running simulation {i+1}/{n_runs}")
            try:
                run_simulation(*arg)
            except Exception as exc:
                print(f"Run {i+1} failed: {exc}")
    else:
        # Use process pool with limited workers
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_runs, max_workers)) as executor:
            futures = [executor.submit(run_simulation, *arg) for arg in run_args]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"A run failed: {exc}")
    
    # Aggregate results from all runs
    all_csvs = glob.glob(os.path.join(base_outdir, "run_*/mutation_results.csv"))
    dfs = []
    for i, csv_file in enumerate(all_csvs):
        try:
            df = pd.read_csv(csv_file)
            df['run'] = i + 1
            dfs.append(df)
        except Exception as e:
            print(f"Failed to read {csv_file}: {e}")
    
    if dfs:
        agg_df = pd.concat(dfs, ignore_index=True)
        agg_df.to_csv(os.path.join(base_outdir, "aggregated_results.csv"), index=False)
        print(f"Aggregated results saved to {os.path.join(base_outdir, 'aggregated_results.csv')}")
    else:
        print("No results to aggregate.")

if __name__ == "__main__":
    main()