#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map PDB IDs / PDB chains to UniProt accessions, then assign AFDB class labels.

This script combines:
  1) map_pdb_to_uniprot.py
  2) assign_uniprot_label_2.py

It does NOT write intermediate files such as:
  - uniprot_mapping.csv
  - uniprot_mapping.summary.csv

It only writes the final:
  - uniprot_mapping.with_label.csv

If afdb_labels cannot be determined, it is assigned to label 4
(cellular_organisms) by default.

Examples
--------
Single protein:
    python map_pdb_to_uniprot_with_label.py 1AB7_A \
      -l /path/to/merged_target_h_with_label.clean.txt \
      -o uniprot_mapping.with_label.csv

Multiple proteins:
    python map_pdb_to_uniprot_with_label.py -i 1AB7_A 1RIS_A 2ACY_A \
      -l /path/to/merged_target_h_with_label.clean.txt \
      -o uniprot_mapping.with_label.csv

Optional old list-file mode:
    python map_pdb_to_uniprot_with_label.py --input-list list.txt \
      -l /path/to/merged_target_h_with_label.clean.txt \
      -o uniprot_mapping.with_label.csv
"""

import argparse
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import requests


PDBe_SIFTS_URL = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"


# =============================================================================
# Part 1. PDB / chain -> UniProt mapping
# =============================================================================

def parse_pdb_chain_id(s):
    """
    Parse one PDB ID / PDB_CHAIN string.

    Supported formats:
        1AB7_A
        1AB7:A
        1AB7-A
        1AB7 A
        1AB7

    Returns:
        (original_id, pdb_id_lower, chain_or_None)
    """
    s = str(s).strip()
    m = re.match(r"^([0-9][A-Za-z0-9]{3})(?:[_:\-\s]+([A-Za-z0-9]+))?$", s)

    if m is None:
        raise ValueError(f"无法解析蛋白质名字/PDB_CHAIN: {s}")

    pdb_id = m.group(1).lower()
    chain = m.group(2)

    if chain is not None:
        chain = chain.strip()

    return s, pdb_id, chain


def read_pdb_chain_list(input_file):
    """
    Optional compatibility mode: read PDB IDs / PDB_CHAINs from list file.
    """
    items = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            try:
                items.append(parse_pdb_chain_id(s))
            except ValueError:
                print(f"[WARNING] 无法解析这一行，跳过: {s}")

    return items


def make_items_from_names(names):
    """
    Build items from protein names directly passed through command line.
    """
    items = []

    for name in names:
        try:
            items.append(parse_pdb_chain_id(name))
        except ValueError as e:
            print(f"[WARNING] {e}，跳过")

    return items


def request_json(session, url, retries=3, sleep=1.0):
    """
    JSON request with retry.
    """
    last_error = None

    for i in range(retries):
        try:
            r = session.get(url, timeout=30)

            if r.status_code == 404:
                return None

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_error = e
            time.sleep(sleep * (i + 1))

    raise RuntimeError(f"请求失败: {url}\nLast error: {last_error}")


def get_pdbe_uniprot_mapping(session, pdb_id, cache):
    """
    Get one PDB entry's UniProt mapping from PDBe/SIFTS.
    """
    if pdb_id in cache:
        return cache[pdb_id]

    url = PDBe_SIFTS_URL.format(pdb_id=pdb_id)
    data = request_json(session, url)

    if not data:
        cache[pdb_id] = {}
        return {}

    mapping = data.get(pdb_id, {})
    cache[pdb_id] = mapping
    return mapping


def get_residue_number(pos):
    """
    In PDBe mapping, start/end is usually a dict:
        {"residue_number": 1, "author_residue_number": 1, ...}
    """
    if not isinstance(pos, dict):
        return ""

    x = pos.get("residue_number", "")
    return "" if x is None else str(x)


def get_author_residue_number(pos):
    if not isinstance(pos, dict):
        return ""

    x = pos.get("author_residue_number", "")
    return "" if x is None else str(x)


def chain_matches(input_chain, mapping):
    """
    input_chain may correspond to either:
        chain_id       : PDB author chain ID
        struct_asym_id : mmCIF label asym ID

    Use both for robust matching.
    """
    if input_chain is None:
        return True

    chain_id = str(mapping.get("chain_id", ""))
    struct_asym_id = str(mapping.get("struct_asym_id", ""))

    return input_chain == chain_id or input_chain == struct_asym_id


def extract_uniprot_mappings(pdbe_mapping, input_chain):
    """
    Extract UniProt mapping records for the selected chain.
    """
    results = []

    uniprot_block = pdbe_mapping.get("UniProt", {})

    for accession, info in uniprot_block.items():
        protein_name_from_pdbe = info.get("name", "")

        for mp in info.get("mappings", []):
            if not chain_matches(input_chain, mp):
                continue

            start = mp.get("start", {})
            end = mp.get("end", {})

            results.append(
                {
                    "uniprot_accession": accession,
                    "pdbe_protein_name": protein_name_from_pdbe,
                    "chain_id": mp.get("chain_id", ""),
                    "struct_asym_id": mp.get("struct_asym_id", ""),
                    "entity_id": mp.get("entity_id", ""),
                    "unp_start": mp.get("unp_start", ""),
                    "unp_end": mp.get("unp_end", ""),
                    "pdb_start": get_residue_number(start),
                    "pdb_end": get_residue_number(end),
                    "auth_pdb_start": get_author_residue_number(start),
                    "auth_pdb_end": get_author_residue_number(end),
                    "identity": mp.get("identity", ""),
                    "coverage": mp.get("coverage", ""),
                }
            )

    return results


def get_uniprot_details(session, accession, cache):
    """
    Optionally fetch UniProt entry name, recommended protein name, and gene names.
    """
    if accession in cache:
        return cache[accession]

    detail = {
        "uniprot_entry_name": "",
        "uniprot_protein_name": "",
        "gene_names": "",
    }

    query_accessions = [accession]

    if "-" in accession:
        query_accessions.append(accession.split("-")[0])

    data = None
    for acc in query_accessions:
        url = UNIPROT_URL.format(accession=acc)
        data = request_json(session, url, retries=2, sleep=0.5)
        if data:
            break

    if data:
        detail["uniprot_entry_name"] = data.get("uniProtkbId", "")

        protein_desc = data.get("proteinDescription", {})
        rec_name = protein_desc.get("recommendedName", {})
        full_name = rec_name.get("fullName", {})
        detail["uniprot_protein_name"] = full_name.get("value", "")

        genes = []
        for g in data.get("genes", []):
            gene_name = g.get("geneName", {}).get("value", "")
            if gene_name:
                genes.append(gene_name)

        detail["gene_names"] = ";".join(sorted(set(genes)))

    cache[accession] = detail
    return detail


def build_uniprot_mapping_dataframe(items, no_uniprot_details=False):
    """
    Build mapping dataframe in memory.
    No intermediate mapping CSV is written.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "pdb-to-uniprot-mapper/1.0",
            "Accept": "application/json",
        }
    )

    pdbe_cache = {}
    uniprot_cache = {}
    detailed_rows = []

    for original_id, pdb_id, input_chain in items:
        print(f"[INFO] Processing {original_id}")

        try:
            pdbe_mapping = get_pdbe_uniprot_mapping(session, pdb_id, pdbe_cache)
            matched = extract_uniprot_mappings(pdbe_mapping, input_chain)

            if not matched:
                detailed_rows.append(
                    {
                        "input_id": original_id,
                        "pdb_id": pdb_id.upper(),
                        "input_chain": "" if input_chain is None else input_chain,
                        "status": "NO_UNIPROT_MAPPING_FOUND",
                        "uniprot_accession": "",
                        "uniprot_entry_name": "",
                        "uniprot_protein_name": "",
                        "gene_names": "",
                        "pdbe_protein_name": "",
                        "chain_id": "",
                        "struct_asym_id": "",
                        "entity_id": "",
                        "unp_start": "",
                        "unp_end": "",
                        "pdb_start": "",
                        "pdb_end": "",
                        "auth_pdb_start": "",
                        "auth_pdb_end": "",
                        "identity": "",
                        "coverage": "",
                    }
                )
                continue

            for mp in matched:
                detail = {
                    "uniprot_entry_name": "",
                    "uniprot_protein_name": "",
                    "gene_names": "",
                }

                if not no_uniprot_details:
                    detail = get_uniprot_details(
                        session,
                        mp["uniprot_accession"],
                        uniprot_cache,
                    )

                row = {
                    "input_id": original_id,
                    "pdb_id": pdb_id.upper(),
                    "input_chain": "" if input_chain is None else input_chain,
                    "status": "OK",
                    **mp,
                    **detail,
                }
                detailed_rows.append(row)

        except Exception as e:
            print(f"[ERROR] {original_id}: {e}")
            detailed_rows.append(
                {
                    "input_id": original_id,
                    "pdb_id": pdb_id.upper(),
                    "input_chain": "" if input_chain is None else input_chain,
                    "status": f"ERROR: {e}",
                    "uniprot_accession": "",
                    "uniprot_entry_name": "",
                    "uniprot_protein_name": "",
                    "gene_names": "",
                    "pdbe_protein_name": "",
                    "chain_id": "",
                    "struct_asym_id": "",
                    "entity_id": "",
                    "unp_start": "",
                    "unp_end": "",
                    "pdb_start": "",
                    "pdb_end": "",
                    "auth_pdb_start": "",
                    "auth_pdb_end": "",
                    "identity": "",
                    "coverage": "",
                }
            )

    return pd.DataFrame(detailed_rows)


# =============================================================================
# Part 2. UniProt accession -> AFDB label
# =============================================================================

def normalize_accession(acc):
    """
    Normalize UniProt accession.

    Input may be:
        P11540
        afdb_P11540
        P11540-2

    Returns:
        P11540 or P11540-2
    """
    if pd.isna(acc):
        return ""

    acc = str(acc).strip()

    if not acc:
        return ""

    if acc.startswith("afdb_"):
        acc = acc.replace("afdb_", "", 1)

    return acc


def get_accession_candidates(acc):
    """
    Generate candidate accessions for matching.

    Example:
        P11540   -> [P11540]
        P11540-2 -> [P11540-2, P11540]
    """
    acc = normalize_accession(acc)

    if not acc:
        return []

    candidates = [acc]

    if "-" in acc:
        base_acc = acc.split("-")[0]
        if base_acc not in candidates:
            candidates.append(base_acc)

    return candidates


def split_accessions(value):
    """
    Split accession field.

    Supports:
        P11540
        P11540;Q9XXX1
        P11540,Q9XXX1
        P11540 Q9XXX1
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if not value:
        return []

    for sep in [",", " ", "\t"]:
        value = value.replace(sep, ";")

    accs = []
    for x in value.split(";"):
        x = normalize_accession(x)
        if x and x not in accs:
            accs.append(x)

    return accs


def unique_join(values):
    out = []

    for v in values:
        if pd.isna(v):
            continue

        v = str(v).strip()

        if v and v not in out:
            out.append(v)

    return ";".join(out)


def detect_accession_column(df):
    """Detect UniProt accession column in dataframe."""
    if "uniprot_accessions" in df.columns:
        return "uniprot_accessions"
    if "uniprot_accession" in df.columns:
        return "uniprot_accession"
    raise ValueError("Dataframe 中没有找到 uniprot_accessions 或 uniprot_accession 列。")


def collect_required_candidates(df, acc_col):
    """
    Only collect accession candidates actually needed by the mapping dataframe.
    This avoids loading the whole huge label file into memory.
    """
    required = set()

    for _, row in df.iterrows():
        accessions = split_accessions(row.get(acc_col, ""))
        for acc in accessions:
            for cand in get_accession_candidates(acc):
                if cand:
                    required.add(cand)

    return required


def make_byte_ranges(file_size, num_workers):
    """
    Split a large file into byte ranges.
    """
    num_workers = max(1, int(num_workers))

    if file_size <= 0:
        return [(0, 0)]

    num_workers = min(num_workers, file_size)
    chunk_size = file_size // num_workers

    ranges = []
    start = 0
    for i in range(num_workers):
        if i == num_workers - 1:
            end = file_size
        else:
            end = (i + 1) * chunk_size
        ranges.append((start, end))
        start = end

    return ranges


def _scan_label_file_range(args):
    """
    Worker: scan selected byte range of label file.
    """
    label_file, start, end, required_candidates = args
    required_candidates = set(required_candidates)

    matches = {}
    checked_lines = 0

    with open(label_file, "rb") as f:
        f.seek(start)

        # If not at file start, skip the first potentially truncated line.
        if start > 0:
            f.readline()

        while True:
            pos = f.tell()
            if pos >= end:
                break

            line = f.readline()
            if not line:
                break

            line = line.strip()
            if not line or line.startswith(b"#"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            try:
                afdb_id = parts[0].decode("utf-8", errors="ignore")
                label = parts[1].decode("utf-8", errors="ignore")
            except Exception:
                continue

            acc = normalize_accession(afdb_id)
            if acc in required_candidates:
                matches[acc] = label

            checked_lines += 1

    return matches, checked_lines


def read_label_file_filtered_parallel(label_file, required_candidates, num_workers=1):
    """
    Multi-process scan of the label file.

    Returns:
        accession_to_label = {"Q53652": "1", "A0A0F0KGI4": "1", ...}
    """
    label_file = str(label_file)
    required_candidates = set(required_candidates)

    if not required_candidates:
        return {}

    file_size = os.path.getsize(label_file)
    ranges = make_byte_ranges(file_size, num_workers)
    real_workers = len(ranges)

    print(f"[INFO] Label file size: {file_size:,} bytes")
    print(f"[INFO] Required accession candidates: {len(required_candidates):,}")
    print(f"[INFO] Using {real_workers} CPU worker(s) to scan label file")

    tasks = [
        (label_file, start, end, required_candidates)
        for start, end in ranges
    ]

    accession_to_label = {}
    total_checked_lines = 0

    if real_workers == 1:
        results = [_scan_label_file_range(tasks[0])]
    else:
        with Pool(processes=real_workers) as pool:
            results = pool.map(_scan_label_file_range, tasks)

    for matches, checked_lines in results:
        accession_to_label.update(matches)
        total_checked_lines += checked_lines

    print(f"[INFO] Scanned approximately {total_checked_lines:,} valid label lines")
    print(
        f"[INFO] Matched {len(accession_to_label):,} / "
        f"{len(required_candidates):,} accession candidates"
    )

    return accession_to_label


def find_labels_for_accessions(accessions, accession_to_label):
    """
    Find class labels for a list of UniProt accessions.
    """
    matched_records = []
    not_found = []

    for acc in accessions:
        found = False

        for cand in get_accession_candidates(acc):
            if cand in accession_to_label:
                matched_records.append(
                    {
                        "input_accession": acc,
                        "matched_accession": cand,
                        "matched_afdb_id": f"afdb_{cand}",
                        "label": accession_to_label[cand],
                    }
                )
                found = True
                break

        if not found:
            not_found.append(acc)

    return matched_records, not_found


DEFAULT_UNKNOWN_LABEL = "4"
DEFAULT_UNKNOWN_TAXON = "cellular_organisms"


def assign_labels_to_dataframe(df, acc_col, accession_to_label):
    """
    Assign AFDB labels to each PDB-UniProt mapping row.

    If afdb_labels cannot be determined, assign it to label 4
    (cellular_organisms) by default. This covers cases such as:
        1) no UniProt accession was mapped from PDBe/SIFTS;
        2) UniProt accession exists but is not found in the label file;
        3) matched records exist but the final label string is empty.
    """
    all_labels = []
    all_matched_accessions = []
    all_matched_afdb_ids = []
    all_not_found = []
    all_label_status = []

    for _, row in df.iterrows():
        accessions = split_accessions(row.get(acc_col, ""))

        if not accessions:
            # 无法从 PDB 映射到 UniProt accession 时，默认归为第 4 类 cellular_organisms。
            all_labels.append(DEFAULT_UNKNOWN_LABEL)
            all_matched_accessions.append("")
            all_matched_afdb_ids.append("")
            all_not_found.append("")
            all_label_status.append(f"DEFAULT_TO_LABEL_{DEFAULT_UNKNOWN_LABEL}_{DEFAULT_UNKNOWN_TAXON}_NO_UNIPROT_ACCESSION")
            continue

        matched_records, not_found = find_labels_for_accessions(
            accessions,
            accession_to_label,
        )

        labels = unique_join([r["label"] for r in matched_records])
        matched_accessions = unique_join([r["matched_accession"] for r in matched_records])
        matched_afdb_ids = unique_join([r["matched_afdb_id"] for r in matched_records])
        not_found_str = ";".join(not_found)

        if not matched_records:
            # UniProt accession 存在，但在 AFDB label 文件中找不到时，默认归为第 4 类 cellular_organisms。
            labels = DEFAULT_UNKNOWN_LABEL
            status = f"DEFAULT_TO_LABEL_{DEFAULT_UNKNOWN_LABEL}_{DEFAULT_UNKNOWN_TAXON}_NOT_FOUND_IN_LABEL_FILE"
        elif not labels:
            # 极少数异常情况：有匹配记录但 label 为空，也默认归为第 4 类。
            labels = DEFAULT_UNKNOWN_LABEL
            status = f"DEFAULT_TO_LABEL_{DEFAULT_UNKNOWN_LABEL}_{DEFAULT_UNKNOWN_TAXON}_EMPTY_LABEL"
        elif len(labels.split(";")) == 1 and not not_found:
            status = "OK"
        elif len(labels.split(";")) == 1 and not_found:
            status = "PARTIAL_FOUND"
        elif len(labels.split(";")) > 1:
            status = "MULTIPLE_LABELS"
        else:
            labels = DEFAULT_UNKNOWN_LABEL
            status = f"DEFAULT_TO_LABEL_{DEFAULT_UNKNOWN_LABEL}_{DEFAULT_UNKNOWN_TAXON}_UNKNOWN"

        all_labels.append(labels)
        all_matched_accessions.append(matched_accessions)
        all_matched_afdb_ids.append(matched_afdb_ids)
        all_not_found.append(not_found_str)
        all_label_status.append(status)

    df["matched_afdb_accessions"] = all_matched_accessions
    df["matched_afdb_ids"] = all_matched_afdb_ids
    df["afdb_labels"] = all_labels
    df["uniprot_not_found_in_label_file"] = all_not_found
    df["label_status"] = all_label_status

    return df


def print_summary(df):
    print("\n[SUMMARY]")
    print(df["label_status"].value_counts(dropna=False))

    if "afdb_labels" in df.columns:
        print("\n[LABEL COUNTS]")
        label_count = {}

        for x in df["afdb_labels"]:
            if pd.isna(x) or str(x).strip() == "":
                continue

            for label in str(x).split(";"):
                label = label.strip()
                if not label:
                    continue
                label_count[label] = label_count.get(label, 0) + 1

        def sort_key(item):
            label = item[0]
            return int(label) if str(label).isdigit() else str(label)

        for label, count in sorted(label_count.items(), key=sort_key):
            print(f"Label {label}: {count}")


# =============================================================================
# Main
# =============================================================================

def main():
    default_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

    parser = argparse.ArgumentParser(
        description=(
            "Map PDB IDs / PDB chains to UniProt accessions, then assign AFDB "
            "labels in one step. Only the final .with_label.csv is written."
        )
    )

    parser.add_argument(
        "proteins",
        nargs="*",
        help="直接输入蛋白质名字/PDB_CHAIN，例如 1AB7_A、1AB7:A、1AB7-A、1AB7 A 或 1AB7",
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=[],
        help="直接输入一个或多个蛋白质名字/PDB_CHAIN，例如 -i 1AB7_A 或 -i 1AB7_A 1RIS_A",
    )

    parser.add_argument(
        "--input-list",
        default=None,
        help="可选：兼容旧 list.txt 输入，每行一个 PDB ID 或 PDB_CHAIN",
    )

    parser.add_argument(
        "-l",
        "--label_file",
        default="/storage/zhaokailong/Foldseek-Cluster/Cluster_LCA/new_class_lineage/AFDB_member_taxon_groups_10000000/classified_fasta_createdb/merged_target_h_with_label.clean.txt",
        help="afdb_accession 与类别对应文件，例如 merged_target_h_with_label.clean.txt",
    )

    parser.add_argument(
        "-o",
        "--output_csv",
        default="uniprot_mapping.with_label.csv",
        help="最终输出 CSV 文件；默认 uniprot_mapping.with_label.csv",
    )

    parser.add_argument(
        "-j",
        "--num_workers",
        type=int,
        default=default_workers,
        help="用于并行扫描 label 文件的 CPU 进程数。默认使用 SLURM_CPUS_PER_TASK；否则为 1。",
    )

    parser.add_argument(
        "--no-uniprot-details",
        action="store_true",
        help="只输出 UniProt accession，不额外查询 UniProt entry name / protein name / gene name，可加快运行。",
    )

    args = parser.parse_args()

    direct_names = []
    direct_names.extend(args.input)
    direct_names.extend(args.proteins)

    items = []
    if direct_names:
        items.extend(make_items_from_names(direct_names))

    if args.input_list is not None:
        items.extend(read_pdb_chain_list(args.input_list))

    if not items:
        raise SystemExit(
            "没有有效输入。示例：\n"
            "  python map_pdb_to_uniprot_with_label.py 1AB7_A -l merged_target_h_with_label.clean.txt\n"
            "  python map_pdb_to_uniprot_with_label.py -i 1AB7_A 1RIS_A -l merged_target_h_with_label.clean.txt\n"
            "  python map_pdb_to_uniprot_with_label.py --input-list list.txt -l merged_target_h_with_label.clean.txt"
        )

    label_file = Path(args.label_file)
    output_csv = Path(args.output_csv)

    if not label_file.exists():
        raise FileNotFoundError(f"找不到类别文件: {label_file}")

    # 1. Build PDB -> UniProt mapping in memory.
    df = build_uniprot_mapping_dataframe(
        items=items,
        no_uniprot_details=args.no_uniprot_details,
    )

    if df.empty:
        raise RuntimeError("没有生成任何 PDB-UniProt mapping 记录，请检查输入蛋白质名字。")

    # 2. Assign labels directly to this dataframe.
    acc_col = detect_accession_column(df)
    print(f"[INFO] Using accession column: {acc_col}")

    required_candidates = collect_required_candidates(df, acc_col)

    print(f"[INFO] Reading label file with multi-CPU scan: {label_file}")
    accession_to_label = read_label_file_filtered_parallel(
        label_file=label_file,
        required_candidates=required_candidates,
        num_workers=args.num_workers,
    )

    df = assign_labels_to_dataframe(df, acc_col, accession_to_label)

    # 3. Only write final with-label CSV.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[DONE] Saved final result to: {output_csv}")
    print_summary(df)


if __name__ == "__main__":
    main()
