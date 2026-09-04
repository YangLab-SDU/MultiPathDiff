#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


def unique_keep_order(items):
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def clean_token(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    if s.startswith(">"):
        s = s[1:]
    s = s.split()[0]
    s = s.split("/")[0]
    return s.strip()


def extract_core_id(raw: str):
    token = clean_token(raw)
    if not token:
        return None

    m = re.match(r"UniRef\d+_(\S+)", token)
    if m:
        return m.group(1)

    if token.startswith("afdb_"):
        return token[len("afdb_"):]

    if "|" in token:
        parts = token.split("|")
        if len(parts) >= 2 and parts[1].strip():
            return parts[1].strip()

    return token


def core_to_afdb(core_id: str):
    if not core_id:
        return None
    return f"afdb_{core_id}"


def parse_hhr_ids(hhr_file):
    if not os.path.isfile(hhr_file) or os.path.getsize(hhr_file) == 0:
        return []

    ids = []

    in_hit_table = False
    with open(hhr_file, "r", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")

            if s.startswith(" No Hit"):
                in_hit_table = True
                continue

            if in_hit_table:
                if not s.strip():
                    break
                m = re.match(r"^\s*\d+\s+(\S+)", s)
                if m:
                    ids.append(m.group(1))

    with open(hhr_file, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith(">"):
                token = clean_token(s[1:])
                if token:
                    ids.append(token)

    return unique_keep_order(ids)


def load_target_core_id_set(species_id_file):
    with open(species_id_file, "rb") as f:
        data = f.read()

    data = data.replace(b"\x00", b"\n").replace(b"\r", b"\n")
    text = data.decode("utf-8", errors="ignore")

    raw_ids = re.findall(r"afdb_[A-Za-z0-9_.-]+", text)

    core_ids = set()
    for rid in raw_ids:
        cid = extract_core_id(rid)
        if cid:
            core_ids.add(cid)

    return core_ids


def process_one_species(index, db_path, hh_core_ids, output_dir):
    db_name = os.path.basename(db_path)
    species_id_file = os.path.join(db_path, f"target_{db_name}_h")
    out_file = os.path.join(output_dir, f"msa_{index}.hhr")

    target_core_ids = load_target_core_id_set(species_id_file)
    kept_core_ids = [cid for cid in hh_core_ids if cid in target_core_ids]
    kept_core_ids = unique_keep_order(kept_core_ids)

    with open(out_file, "w") as fout:
        for cid in kept_core_ids:
            fout.write(f"seq\t{core_to_afdb(cid)}\n")

    return {
        "index": index,
        "db_name": db_name,
        "target_n": len(target_core_ids),
        "kept_n": len(kept_core_ids),
        "examples": kept_core_ids[:10],
        "out_file": out_file,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hhblits_hhr", required=True)
    parser.add_argument("--db_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    db_dirs = [
        os.path.join(args.db_root, x)
        for x in os.listdir(args.db_root)
        if os.path.isdir(os.path.join(args.db_root, x))
    ]
    db_dirs = sorted(db_dirs, key=lambda x: os.path.basename(x).lower())

    hh_raw_ids = parse_hhr_ids(args.hhblits_hhr)
    hh_core_ids = unique_keep_order([extract_core_id(x) for x in hh_raw_ids if extract_core_id(x)])

    print(f"[INFO] HH raw IDs   : {len(hh_raw_ids)}")
    print(f"[INFO] HH core IDs  : {len(hh_core_ids)}")
    print(f"[INFO] HH examples  : {hh_core_ids[:20]}")

    futures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        for i, db_path in enumerate(db_dirs, start=1):
            futures.append(ex.submit(process_one_species, i, db_path, hh_core_ids, args.output_dir))

        results = [f.result() for f in as_completed(futures)]

    for r in sorted(results, key=lambda x: x["index"]):
        print(
            f"[INFO] {r['index']:02d} {r['db_name']:<20} "
            f"target={r['target_n']:<8} kept={r['kept_n']:<6} "
            f"examples={r['examples']}"
        )


if __name__ == "__main__":
    main()