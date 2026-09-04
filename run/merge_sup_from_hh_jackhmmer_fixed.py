#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re


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


def load_target_core_id_set(species_id_file):
    core_ids = set()
    if not species_id_file or (not os.path.isfile(species_id_file)):
        return core_ids

    with open(species_id_file, "rb") as f:
        data = f.read()

    data = data.replace(b"\x00", b"\n").replace(b"\r", b"\n")
    text = data.decode("utf-8", errors="ignore")

    raw_ids = re.findall(r"afdb_[A-Za-z0-9_.-]+", text)

    for rid in raw_ids:
        cid = extract_core_id(rid)
        if cid:
            core_ids.add(cid)

    return core_ids


def parse_jackhmmer_ids(jackhmmer_file):
    if not jackhmmer_file or (not os.path.isfile(jackhmmer_file)) or os.path.getsize(jackhmmer_file) == 0:
        return []

    first_nonempty = ""
    with open(jackhmmer_file, "r", errors="replace") as f:
        for line in f:
            if line.strip():
                first_nonempty = line.strip()
                break

    # Stockholm
    if first_nonempty.startswith("# STOCKHOLM"):
        ids = []
        with open(jackhmmer_file, "r", errors="replace") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("//"):
                    continue
                parts = s.split()
                if len(parts) >= 2:
                    ids.append(parts[0])
        return unique_keep_order(ids)

    ids = []
    in_score_table = False

    with open(jackhmmer_file, "r", errors="replace") as f:
        for line in f:
            s = line.rstrip("\n")

            if re.search(r"\bSequence\s+Description\b", s):
                in_score_table = True
                continue

            if not in_score_table:
                continue

            if "Domain annotation for each sequence" in s:
                in_score_table = False
                continue

            stripped = s.strip()
            if not stripped:
                continue
            if stripped.startswith("-------"):
                continue
            if "inclusion threshold" in stripped:
                continue

            parts = stripped.split()
            if parts and parts[0] in {"+", "-", "?", "!"}:
                parts = parts[1:]

            if len(parts) < 9:
                continue

            seq_id = parts[8]
            if seq_id and seq_id not in {"Sequence", "Description"}:
                ids.append(seq_id)

    return unique_keep_order(ids)


def load_hh_species_core_ids(hh_species_file):
    ids = []
    if not hh_species_file or (not os.path.isfile(hh_species_file)) or os.path.getsize(hh_species_file) == 0:
        return ids

    with open(hh_species_file, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            cid = extract_core_id(parts[1])
            if cid:
                ids.append(cid)

    return unique_keep_order(ids)


def load_m8_core_ids(m8_file):
    ids = []
    if not m8_file or (not os.path.isfile(m8_file)) or os.path.getsize(m8_file) == 0:
        return ids

    with open(m8_file, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                continue
            cid = extract_core_id(parts[1])
            if cid:
                ids.append(cid)

    return unique_keep_order(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hh_species_file", required=True, help="per-species HHblits result")
    parser.add_argument("--output_sup", required=True, help="output .sup file")
    parser.add_argument("--jackhmmer_file", default="", help="jackhmmer .hmmer or .sto")
    parser.add_argument("--m8_file", default="", help="mmseqs .m8 file")
    parser.add_argument("--species_id_file", required=True, help="target_${DbName}_h")
    args = parser.parse_args()

    target_core_ids = load_target_core_id_set(args.species_id_file)

    # HHblits
    hh_core_ids = load_hh_species_core_ids(args.hh_species_file)
    if target_core_ids:
        hh_core_ids = [cid for cid in hh_core_ids if cid in target_core_ids]
        hh_core_ids = unique_keep_order(hh_core_ids)

    # JackHMMER
    jack_raw_ids = parse_jackhmmer_ids(args.jackhmmer_file) if args.jackhmmer_file else []
    jack_core_ids = []
    for rid in jack_raw_ids:
        cid = extract_core_id(rid)
        if cid and ((not target_core_ids) or (cid in target_core_ids)):
            jack_core_ids.append(cid)
    jack_core_ids = unique_keep_order(jack_core_ids)

    # MMseqs .m8
    m8_core_ids = load_m8_core_ids(args.m8_file)
    if target_core_ids:
        m8_core_ids = [cid for cid in m8_core_ids if cid in target_core_ids]
        m8_core_ids = unique_keep_order(m8_core_ids)

    merged_core_ids = unique_keep_order(m8_core_ids + jack_core_ids + hh_core_ids)

    outdir = os.path.dirname(args.output_sup)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.output_sup, "w") as fout:
        for cid in merged_core_ids:
            fout.write(f"seq\t{core_to_afdb(cid)}\n")

    print(f"[INFO] MMseqs IDs       : {len(m8_core_ids)}")
    print(f"[INFO] JackHMMER raw    : {len(jack_raw_ids)}")
    print(f"[INFO] JackHMMER kept   : {len(jack_core_ids)}")
    print(f"[INFO] HH species kept  : {len(hh_core_ids)}")
    print(f"[INFO] Final merged     : {len(merged_core_ids)}")
    print(f"[INFO] Output           : {args.output_sup}")


if __name__ == "__main__":
    main()