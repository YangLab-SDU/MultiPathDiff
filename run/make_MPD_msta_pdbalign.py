import os
import re
import argparse
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import requests


AFDB_DOWNLOAD_ROOT = "https://alphafold.ebi.ac.uk/files"


def run_tmalign(query_pdb, target_pdb, output_file, tmalign_path):
    cmd = f"{tmalign_path} {query_pdb} {target_pdb} > {output_file}"
    os.system(cmd)


def parse_hmmer_file(file_path: str) -> Tuple[List[str], List[str]]:
    protein_info = {}
    current_round = 1
    first_round_proteins = []
    round_identified = False

    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            if line.startswith("@@ Round:"):
                match = re.search(r"Round:\s+(\d+)", line)
                if match:
                    current_round = int(match.group(1))
                    round_identified = True

            match = re.search(r"afdb_([^\s]+)", line)
            if match:
                protein_id = match.group(1)
                if protein_id not in protein_info:
                    protein_info[protein_id] = current_round
                    if not round_identified or current_round == 1:
                        first_round_proteins.append(protein_id)

    all_proteins = list(protein_info.keys())
    return first_round_proteins, all_proteins


def parse_m8_like_file(file_path: str) -> Tuple[List[str], List[str]]:
    protein_ids = []
    seen = set()
    with open(file_path, 'r', errors='replace') as f:
        for line in f:
            match = re.search(r"afdb_([^\s]+)", line)
            if match:
                pid = match.group(1)
                if pid not in seen:
                    protein_ids.append(pid)
                    seen.add(pid)
    return protein_ids, protein_ids


def get_pdb_path(base_dir: str, protein_id: str) -> Optional[str]:
    if not protein_id or len(protein_id) < 6:
        return None
    dir1 = protein_id[0:2]
    dir2 = protein_id[2:4]
    dir3 = protein_id[4:6]
    return os.path.join(base_dir, dir1, dir2, dir3, f"{protein_id}.pdb")


def build_afdb_download_urls(protein_id: str) -> List[str]:
    """
    Try several likely AlphaFold DB filename patterns.
    Prefer v4 first, then fall back to v6.
    Local saved filename is always just {protein_id}.pdb under the 3-level subdirs.
    """
    candidates = [
        f"AF-{protein_id}-F1-model_v4.pdb",
        f"AF-{protein_id}-F1-model_v6.pdb",
    ]
    seen = set()
    urls = []
    for name in candidates:
        if name not in seen:
            seen.add(name)
            urls.append(f"{AFDB_DOWNLOAD_ROOT}/{name}")
    return urls


def download_pdb_if_missing(protein_id: str, afdb_pdb_dir: str, timeout: int = 60) -> Optional[str]:
    pdb_path = get_pdb_path(afdb_pdb_dir, protein_id)
    if not pdb_path:
        return None

    if os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
        return pdb_path

    os.makedirs(os.path.dirname(pdb_path), exist_ok=True)
    urls = build_afdb_download_urls(protein_id)

    for url in urls:
        tmp_path = pdb_path + ".part"
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp_path, pdb_path)
            print(f"Downloaded missing PDB: {protein_id} from {os.path.basename(url)} -> {pdb_path}")
            return pdb_path
        except requests.RequestException as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"Warning: failed download attempt for {protein_id} using {os.path.basename(url)}: {e}")
            continue
        except OSError as e:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f"Warning: failed to save downloaded PDB for {protein_id} using {os.path.basename(url)}: {e}")
            continue

    return None


def ensure_pdb_available(protein_id: str, afdb_pdb_dir: str, timeout: int = 60) -> Optional[str]:
    pdb_path = get_pdb_path(afdb_pdb_dir, protein_id)
    if pdb_path and os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0:
        return pdb_path
    return download_pdb_if_missing(protein_id, afdb_pdb_dir, timeout=timeout)


def process_alignment(args):
    threads_per_task = args.threads_per_task
    input_file = args.input_file
    output_dir = args.output_dir
    afdb_pdb_dir = args.afdb_pdb_dir
    tmalign_path = args.tmalign_path
    mode = args.mode
    download_workers = max(1, args.download_workers)
    download_timeout = args.download_timeout

    os.makedirs(output_dir, exist_ok=True)

    if mode == 'hmmer':
        candidate_refs, all_proteins = parse_hmmer_file(input_file)
    elif mode in {'m8', 'sup'}:
        candidate_refs, all_proteins = parse_m8_like_file(input_file)
    else:
        print(f"Unknown mode: {mode}")
        return

    if len(all_proteins) < 2:
        print(f"Insufficient proteins found in {input_file} (at least 2 required, current: {len(all_proteins)})")
        return

    ref_protein = None
    ref_pdb_path = None

    for pid in candidate_refs:
        pdb_path = ensure_pdb_available(pid, afdb_pdb_dir, timeout=download_timeout)
        if pdb_path and os.path.exists(pdb_path):
            ref_protein = pid
            ref_pdb_path = pdb_path
            break

    if not ref_protein:
        print(f"No valid reference protein PDB file found or downloaded (among {len(candidate_refs)} candidates checked)")
        if len(candidate_refs) > 0:
            example_path = get_pdb_path(afdb_pdb_dir, candidate_refs[0])
            print(f"Debug: Checked path example: {example_path}")
        return

    print(f"[{mode}] Using reference protein: {ref_protein}")
    print(f"Total proteins to align: {len(all_proteins)}")

    target_ids = [pid for pid in all_proteins if pid != ref_protein]
    available_targets = {}

    def fetch_target(pid: str):
        return pid, ensure_pdb_available(pid, afdb_pdb_dir, timeout=download_timeout)

    if target_ids:
        with ThreadPoolExecutor(max_workers=min(download_workers, len(target_ids))) as ex:
            futures = [ex.submit(fetch_target, pid) for pid in target_ids]
            for fut in as_completed(futures):
                pid, target_pdb = fut.result()
                if target_pdb and os.path.exists(target_pdb):
                    available_targets[pid] = target_pdb
                else:
                    print(f"Warning: target PDB unavailable for {pid}, skipped.")

    tasks = []
    for pid in target_ids:
        target_pdb = available_targets.get(pid)
        if target_pdb:
            output_file = os.path.join(output_dir, f"{pid}.rmsd")
            tasks.append((ref_pdb_path, target_pdb, output_file, tmalign_path))

    if tasks:
        print(f"Starting alignment of {len(tasks)} structures (TM-align workers: {threads_per_task})...")
        with Pool(threads_per_task) as p:
            p.starmap(run_tmalign, tasks)

        success_count = 0
        for _, _, out_f, _ in tasks:
            if os.path.exists(out_f) and os.path.getsize(out_f) > 0:
                success_count += 1
        print(f"Alignment complete: {success_count}/{len(tasks)} RMSD files successfully generated.")
    else:
        print("No valid alignment tasks generated (target PDBs could not be found or downloaded).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process m8, hmmer, or sup files and calculate RMSD')
    parser.add_argument('--mode', type=str, choices=['m8', 'hmmer', 'sup'], required=True,
                        help='Input file type: m8, hmmer, or sup')
    parser.add_argument('--input_file', type=str, required=True, help='Input MSA file path (m8, hmmer, or sup)')
    parser.add_argument('--output_dir', type=str, required=True, help='RMSD output directory')
    parser.add_argument('--afdb_pdb_dir', type=str, required=True, help='Local AFDB PDB directory')
    parser.add_argument('--tmalign_path', type=str, required=True, help='TMalign executable path')
    parser.add_argument('--threads_per_task', type=int, default=10, help='Number of parallel TM-align workers')
    parser.add_argument('--download_workers', type=int, default=8, help='Number of parallel download workers')
    parser.add_argument('--download_timeout', type=int, default=60, help='HTTP timeout for PDB download (seconds)')

    args = parser.parse_args()
    process_alignment(args)
