import sys
import os
import math
import subprocess
import re
import argparse
import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import QuantileTransformer


def run_nwalign(query_seq_file: str, target_seq_file: str, nwalign_path: str) -> Tuple[str, str]:
    try:
        result = subprocess.run(
            [nwalign_path, query_seq_file, target_seq_file],
            capture_output=True,
            text=True,
            check=True
        )
        output_lines = result.stdout.split('\n')
        aligned_seq1 = ""
        aligned_seq2 = ""
        match_line_found = False
        in_sequence_section = False

        for line in output_lines:
            stripped_line = line.strip()
            if not stripped_line:
                continue
            if "Sequence identity:" in stripped_line:
                in_sequence_section = True
                continue
            if not in_sequence_section:
                continue
            if any(c in stripped_line for c in [":", "."]):
                match_line_found = True
                continue
            if not aligned_seq1 and not match_line_found:
                aligned_seq1 = stripped_line
            elif match_line_found and not aligned_seq2:
                aligned_seq2 = stripped_line

        if not aligned_seq1 or not aligned_seq2:
            raise ValueError("Failed to parse alignment sequences from NWalign output")
        return aligned_seq1, aligned_seq2

    except Exception as e:
        raise ValueError(f"NWalign failed: {str(e)}")


def read_fasta_sequence(fasta_file: str) -> str:
    sequence = ""
    with open(fasta_file) as f:
        for line in f:
            if not line.startswith(">"):
                sequence += line.strip()
    return sequence


def map_scores_to_target(query_scores: List[float], aligned_query: str, aligned_target: str) -> List[float]:
    target_scores = [0.0] * len(aligned_target.replace("-", ""))
    target_scores_idx = 0
    query_scores_idx = 0

    for q, t in zip(aligned_query, aligned_target):
        if q != "-" and t != "-":
            if query_scores_idx < len(query_scores) and target_scores_idx < len(target_scores):
                target_scores[target_scores_idx] = query_scores[query_scores_idx]
            query_scores_idx += 1
            target_scores_idx += 1
        elif q != "-" and t == "-":
            query_scores_idx += 1
        elif q == "-" and t != "-":
            target_scores_idx += 1

    return target_scores


def find_existing_pdb_protein(input_file: str, afdb_pdb_dir: str, mode: str) -> Optional[str]:
    protein_ids = []
    seen = set()

    print(f"Parsing input file: {input_file}, mode: {mode}")

    try:
        with open(input_file, 'r', errors='replace') as f:
            for line in f:
                pid = None
                match = re.search(r"afdb_([^\s]+)", line)
                if match:
                    pid = match.group(1)

                if pid and pid not in seen:
                    protein_ids.append(pid)
                    seen.add(pid)

        if not protein_ids:
            print(f"Error: No protein ID matching format (afdb_XXX) found in file {input_file}")
            return None

        for protein_id in protein_ids:
            dir1 = protein_id[0:2]
            dir2 = protein_id[2:4]
            dir3 = protein_id[4:6]
            pdb_file = f"{afdb_pdb_dir}/{dir1}/{dir2}/{dir3}/{protein_id}.pdb"
            if os.path.exists(pdb_file):
                print(f"[{mode}] Found first existing PDB file: {protein_id}")
                return protein_id

        print(f"Error: None of the protein IDs found in file {input_file} exist in the PDB library")
        return None

    except IOError as e:
        print(f"Cannot read input file: {e}")
        return None


def extract_sequence_from_pdb(pdb_file: str) -> str:
    sequence = ""
    aa_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    try:
        with open(pdb_file) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    res_name = line[17:20].strip()
                    sequence += aa_dict.get(res_name, 'X')
    except IOError:
        raise ValueError("Cannot read PDB file")
    return sequence


def save_sequence_to_fasta(sequence: str, output_file: str, header: str = ">sequence") -> None:
    with open(output_file, 'w') as f:
        f.write(f"{header}\n{sequence}\n")


def process_rmsd_files(rmsd_dir: str, temp_output: str) -> None:
    if not os.path.exists(rmsd_dir):
        return
    files = [f for f in os.listdir(rmsd_dir) if f.endswith('.rmsd')]
    for file in files:
        file_path = os.path.join(rmsd_dir, file)
        rm_file = False
        try:
            with open(file_path) as f, open(temp_output, 'a+') as outfile:
                for line in f:
                    if line.startswith("Chain_1TM-score="):
                        tmscore = float(line.split()[1])
                        outfile.write(f"{file[:-5]}\t{tmscore}\n")
                        if tmscore < 0.3:
                            rm_file = True
            if rm_file:
                os.remove(file_path)
        except IOError:
            pass


def calculate_alignment_scores(rmsd_dir: str, seq_length: int, sigma: float = 1.0) -> tuple:
    res_scores = [0.0] * seq_length
    num_files = 0
    if not os.path.exists(rmsd_dir):
        return res_scores, 0

    for file in os.listdir(rmsd_dir):
        if not file.endswith('.rmsd') or "temp" in file or "fasta" in file:
            continue
        file_path = os.path.join(rmsd_dir, file)
        try:
            with open(file_path) as f:
                content = f.read()
                if "align_result:" not in content:
                    continue
            num_files += 1
            with open(file_path) as f:
                t1, t2, t3 = " ", " ", " "
                align_i = 0
                stop_read = False
                res_rmsd = []
                for line in f:
                    if line.startswith("RMSD"):
                        stop_read = True
                    if line.startswith("align_result:"):
                        align_i = 1
                    if not stop_read and line.strip() and not line.startswith("Chain"):
                        try:
                            res_rmsd.append(float(line.split()[1]))
                        except Exception:
                            pass
                    if align_i == 2:
                        t1 = line.strip()
                    elif align_i == 3:
                        t2 = line.strip()
                    elif align_i == 4:
                        t3 = line.strip()
                    if align_i >= 1:
                        align_i += 1

                res_idx = 0
                align_idx = 0
                iter_len = min(len(t1), len(t3))
                for i in range(iter_len):
                    if t1[i] != "-":
                        if t3[i] != "-":
                            if align_idx < len(res_rmsd):
                                di = res_rmsd[align_idx]
                                score = math.exp(-(di ** 2) / (2 * sigma ** 2))
                                if res_idx < len(res_scores):
                                    res_scores[res_idx] += score
                                align_idx += 1
                        res_idx += 1
        except Exception:
            continue
    return res_scores, num_files


def write_results(output_file: str, scores: List[float], num_files: int) -> None:
    try:
        if num_files == 0:
            print("Warning: No valid alignment files for score calculation, outputting all-zero results.")
            normalized_scores = [0.0] * len(scores)
        else:
            scores_arr = np.array(scores).reshape(-1, 1)
            qt = QuantileTransformer(output_distribution='uniform', random_state=42)
            normalized_scores = qt.fit_transform(scores_arr).flatten()

        with open(output_file, 'w') as outfile:
            for i, score in enumerate(normalized_scores):
                outfile.write(f"{i}\t{score:.3f}\n")
        print(f"Scores normalized and written to {output_file}")
    except Exception as e:
        print(f"Error writing results: {e}")


def merge_msta_files(output_file: str, input_files: List[str]) -> None:
    if not input_files:
        print("Error: No input files for merging.")
        sys.exit(1)

    print(f"Merging {len(input_files)} potential MSTA files into {output_file} ...")

    valid_columns_data = []

    zero_eps = 1e-6

    for file_path in input_files:
        current_data = {}
        has_content = False
        n_scores = 0
        n_zero = 0

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue

                    try:
                        idx = int(parts[0])
                        score_str = parts[1]
                        score_val = float(score_str)
                    except Exception:
                        continue

                    current_data[idx] = score_str
                    has_content = True

                    n_scores += 1
                    if abs(score_val) <= zero_eps:
                        n_zero += 1

        except IOError as e:
            print(f"Warning: Could not read file {file_path}: {e}")
            continue

        if not has_content or n_scores == 0:
            print(f"Skipping file {os.path.basename(file_path)}: file is empty.")
            continue

        zero_fraction = n_zero / n_scores

        if n_zero > (n_scores / 2.0):
            print(
                f"Skipping file {os.path.basename(file_path)}: "
                f"too many zero scores "
                f"({n_zero}/{n_scores}, {zero_fraction:.3f} > 0.5)."
            )
            continue

        valid_columns_data.append(current_data)

    if not valid_columns_data:
        print("Error: No valid data columns found. All inputs were empty or had more than half zero scores.")
        sys.exit(1)

    print(f"Merging {len(valid_columns_data)} valid columns...")

    all_indices = set()
    for col_data in valid_columns_data:
        all_indices.update(col_data.keys())

    if not all_indices:
        print("Error: No indices found in valid data.")
        sys.exit(1)

    max_index = max(all_indices)

    try:
        with open(output_file, 'w') as f:
            for i in range(max_index + 1):
                row = [str(i)]
                for col_data in valid_columns_data:
                    row.append(col_data.get(i, "0.000"))
                f.write("\t".join(row) + "\n")
        print("Merge successful.")
    except IOError as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--mode', type=str, choices=['m8', 'hmmer', 'sup'])
    group.add_argument('--merge_mode', action='store_true')

    parser.add_argument('--input_file', type=str)
    parser.add_argument('--RMSD_path', type=str)
    parser.add_argument('--seq_path', type=str)
    parser.add_argument('--msta_outpath', type=str)
    parser.add_argument('--afdb_pdb_dir', type=str)
    parser.add_argument('--nwalign_path', type=str)
    parser.add_argument('--merge_files', type=str, nargs='+')

    args = parser.parse_args()

    if args.merge_mode:
        merge_msta_files(args.msta_outpath, args.merge_files)
        sys.exit(0)

    if args.mode not in {'m8', 'hmmer', 'sup'}:
        print(f"Unsupported mode: {args.mode}")
        sys.exit(1)

    try:
        target_seq = read_fasta_sequence(args.seq_path)
    except IOError:
        sys.exit(1)

    protein_id = find_existing_pdb_protein(args.input_file, args.afdb_pdb_dir, args.mode)
    if not protein_id:
        print("Program terminated: Cannot determine reference protein.")
        write_results(args.msta_outpath, [0.0] * len(target_seq), 0)
        sys.exit(0)

    if len(protein_id) >= 6:
        dir1, dir2, dir3 = protein_id[0:2], protein_id[2:4], protein_id[4:6]
        pdb_file = f"{args.afdb_pdb_dir}/{dir1}/{dir2}/{dir3}/{protein_id}.pdb"
    else:
        pdb_file = f"{args.afdb_pdb_dir}/{protein_id}.pdb"

    try:
        query_seq = extract_sequence_from_pdb(pdb_file)
        if not os.path.exists(args.RMSD_path):
            os.makedirs(args.RMSD_path, exist_ok=True)
        query_fasta = os.path.join(args.RMSD_path, "query_temp.fasta")
        save_sequence_to_fasta(query_seq, query_fasta, f">{protein_id}")
        aligned_query, aligned_target = run_nwalign(query_fasta, args.seq_path, args.nwalign_path)

        temp_output = f"{args.RMSD_path}/temp_tmscore"
        process_rmsd_files(args.RMSD_path, temp_output)

        query_scores, num_files = calculate_alignment_scores(args.RMSD_path, len(query_seq))
        target_scores = map_scores_to_target(query_scores, aligned_query, aligned_target)
        write_results(args.msta_outpath, target_scores, num_files)
    except Exception as e:
        print(f"Error during processing: {e}")
        write_results(args.msta_outpath, [0.0] * len(target_seq), 0)


if __name__ == "__main__":
    main()
