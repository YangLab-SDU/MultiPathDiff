#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Fixed labels for the 10 species-group databases
# -----------------------------------------------------------------------------

DEFAULT_LABEL_TO_TAXON = {
    1: "Actinomycetota",
    2: "Bacillota",
    3: "Bacteria",
    4: "cellular_organisms",
    5: "FCB_group",
    6: "Fungi",
    7: "Metazoa",
    8: "Pseudomonadati",
    9: "Pseudomonadota",
    10: "Streptophyta",
}


DEFAULT_LABEL_TO_TIERS: Dict[int, List[List[int]]] = {
    # Actinomycetota > Bacillota >
    # (Pseudomonadati = FCB_group = Pseudomonadota) >
    # (Fungi = Metazoa = Streptophyta) > Bacteria > cellular_organisms
    1: [[1], [2], [8, 5, 9], [6, 7, 10], [3], [4]],

    # Bacillota > Actinomycetota >
    # (Pseudomonadati = FCB_group = Pseudomonadota) >
    # (Fungi = Metazoa = Streptophyta) > Bacteria > cellular_organisms
    2: [[2], [1], [8, 5, 9], [6, 7, 10], [3], [4]],

    # Bacteria >
    # (Actinomycetota = Bacillota = Pseudomonadati = FCB_group = Pseudomonadota) >
    # (Fungi = Metazoa = Streptophyta) > cellular_organisms
    3: [[3], [1, 2, 8, 5, 9], [6, 7, 10], [4]],

    # cellular_organisms > all other nine groups equally
    4: [[4], [1, 2, 3, 5, 6, 7, 8, 9, 10]],

    # FCB_group > Pseudomonadota > (Actinomycetota = Bacillota) >
    # (Fungi = Metazoa = Streptophyta) > Pseudomonadati > Bacteria > cellular_organisms
    5: [[5], [9], [1, 2], [6, 7, 10], [8], [3], [4]],

    # Fungi > Metazoa > Streptophyta > all bacterial groups equally > cellular_organisms
    6: [[6], [7], [10], [1, 2, 3, 5, 8, 9], [4]],

    # Metazoa > Fungi > Streptophyta > all bacterial groups equally > cellular_organisms
    7: [[7], [6], [10], [1, 2, 3, 5, 8, 9], [4]],

    # Pseudomonadati > (FCB_group = Pseudomonadota) >
    # (Actinomycetota = Bacillota) > (Fungi = Metazoa = Streptophyta) >
    # Bacteria > cellular_organisms
    8: [[8], [5, 9], [1, 2], [6, 7, 10], [3], [4]],

    # Pseudomonadota > FCB_group > (Actinomycetota = Bacillota) >
    # (Fungi = Metazoa = Streptophyta) > Pseudomonadati > Bacteria > cellular_organisms
    9: [[9], [5], [1, 2], [6, 7, 10], [8], [3], [4]],

    # Streptophyta > (Fungi = Metazoa) > all bacterial groups equally > cellular_organisms
    10: [[10], [6, 7], [1, 2, 3, 5, 8, 9], [4]],
}


def validate_tier_definition() -> None:
    expected = set(DEFAULT_LABEL_TO_TAXON)
    if set(DEFAULT_LABEL_TO_TIERS) != expected:
        raise ValueError("Source labels in DEFAULT_LABEL_TO_TIERS must exactly match those in DEFAULT_LABEL_TO_TAXON.")

    for src_label, tiers in DEFAULT_LABEL_TO_TIERS.items():
        flat = [x for tier in tiers for x in tier]
        if len(flat) != len(set(flat)):
            raise ValueError(f"The tier definition for source label {src_label} contains duplicate labels: {flat}")
        if set(flat) != expected:
            missing = sorted(expected - set(flat))
            extra = sorted(set(flat) - expected)
            raise ValueError(
                f"The tier definition for source label {src_label} must cover all 10 labels; "
                f"missing={missing}, extra={extra}"
            )


validate_tier_definition()


def read_csv_auto(csv_file: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_file, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"Cannot read CSV file: {csv_file}; last error: {last_error}")


def find_column(df: pd.DataFrame, candidates: Iterable[str], fallback_first: bool = False) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if fallback_first:
        return df.columns[0]
    raise ValueError(
        f"None of the candidate columns were found: {list(candidates)}\n"
        f"Columns present in the file: {list(df.columns)}"
    )


def parse_labels(x) -> List[int]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []

    for sep in [",", " ", "\t"]:
        s = s.replace(sep, ";")

    labels = []
    for item in s.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            label = int(float(item))
        except ValueError:
            continue
        if 1 <= label <= 10 and label not in labels:
            labels.append(label)
    return labels


def parse_float(x) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v):
        return None
    return v


def read_msta_value_column(msta_file: Path) -> List[float]:
    values = []
    with open(msta_file, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            value = None
            for p in reversed(parts):
                value = parse_float(p)
                if value is not None:
                    break
            if value is not None:
                values.append(value)
    return values


def check_one_temp_msta(
    msta_file: Path,
    zero_fraction_threshold: float = 0.5,
    zero_eps: float = 1e-12,
) -> Dict[str, object]:
    if not msta_file.exists():
        return {
            "msta_file": str(msta_file),
            "exists": False,
            "n_residues": 0,
            "zero_count": 0,
            "zero_ratio": math.nan,
            "valid_msta": False,
            "invalid_reason": "MISSING_FILE",
        }

    values = read_msta_value_column(msta_file)
    n = len(values)
    if n == 0:
        return {
            "msta_file": str(msta_file),
            "exists": True,
            "n_residues": 0,
            "zero_count": 0,
            "zero_ratio": math.nan,
            "valid_msta": False,
            "invalid_reason": "EMPTY_OR_NO_NUMERIC_VALUE",
        }

    zero_count = sum(1 for v in values if abs(v) <= zero_eps)
    zero_ratio = zero_count / n
    valid = zero_ratio <= zero_fraction_threshold

    return {
        "msta_file": str(msta_file),
        "exists": True,
        "n_residues": n,
        "zero_count": zero_count,
        "zero_ratio": zero_ratio,
        "valid_msta": valid,
        "invalid_reason": "" if valid else f"ZERO_RATIO_GT_{zero_fraction_threshold}",
    }


def find_msta_dir(target_dir: Path, user_msta_dir: Optional[Path] = None) -> Path:
    if user_msta_dir is not None:
        return user_msta_dir
    candidate = target_dir / "msta"
    if candidate.exists():
        return candidate
    return target_dir


def collect_msta_validity(
    target: str,
    msta_dir: Path,
    zero_fraction_threshold: float,
    zero_eps: float,
) -> pd.DataFrame:
    rows = []
    for label in range(1, 11):
        taxon = DEFAULT_LABEL_TO_TAXON[label]
        msta_file = msta_dir / f"temp_msa_{label}.msta"
        info = check_one_temp_msta(
            msta_file=msta_file,
            zero_fraction_threshold=zero_fraction_threshold,
            zero_eps=zero_eps,
        )
        rows.append({"target": target, "label": label, "taxon": taxon, **info})
    return pd.DataFrame(rows)


def flatten_tiers(tiers: List[List[int]]) -> List[int]:
    return [label for tier in tiers for label in tier]


def filter_tiers(tiers: List[List[int]], valid_labels: List[int]) -> List[List[int]]:
    valid_set = set(valid_labels)
    filtered = []
    for tier in tiers:
        kept = [label for label in tier if label in valid_set]
        if kept:
            filtered.append(kept)
    return filtered


def format_tiers_labels(tiers: List[List[int]]) -> str:
    parts = []
    for tier in tiers:
        if len(tier) == 1:
            parts.append(str(tier[0]))
        else:
            parts.append("(" + " = ".join(map(str, tier)) + ")")
    return " > ".join(parts)


def format_tiers_taxa(tiers: List[List[int]]) -> str:
    parts = []
    for tier in tiers:
        names = [DEFAULT_LABEL_TO_TAXON[x] for x in tier]
        if len(names) == 1:
            parts.append(names[0])
        else:
            parts.append("(" + " = ".join(names) + ")")
    return " > ".join(parts)


def get_effective_tier_rank_by_label(tiers: List[List[int]]) -> Dict[int, int]:
    out = {}
    for tier_rank, tier in enumerate(tiers, start=1):
        for label in tier:
            out[label] = tier_rank
    return out


def tier_rank_power_probabilities(
    tiers: List[List[int]],
    rank_power: float = 1.0,
    normalize: bool = True,
) -> Dict[int, float]:
    if not tiers:
        return {}

    power = float(rank_power)
    if power < 0:
        raise ValueError("--probability_rank_power must be >= 0.")

    k = len(tiers)
    raw_probability_by_label: Dict[int, float] = {}

    for tier_rank, tier in enumerate(tiers, start=1):
        raw_weight = float(k - tier_rank + 1) ** power
        for label in tier:
            raw_probability_by_label[label] = raw_weight

    if not normalize:
        return raw_probability_by_label

    total = sum(raw_probability_by_label.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("The sum of tier probabilities is invalid. Check --probability_rank_power.")

    return {label: probability / total for label, probability in raw_probability_by_label.items()}


def combine_probabilities_for_labels(
    source_labels: List[int],
    filtered_tiers_by_source: Dict[int, List[List[int]]],
    rank_power: float,
) -> Dict[int, float]:
    probability_sum: Dict[int, float] = {}

    for src_label in source_labels:
        tiers = filtered_tiers_by_source.get(src_label, [])
        one_source_probabilities = tier_rank_power_probabilities(
            tiers,
            rank_power=rank_power,
            normalize=True,
        )
        for label, probability in one_source_probabilities.items():
            probability_sum[label] = probability_sum.get(label, 0.0) + probability

    total = sum(probability_sum.values())
    if total > 0:
        probability_sum = {label: probability / total for label, probability in probability_sum.items()}
    return probability_sum


def format_probability(x: float) -> str:
    return f"{x:.8f}"


def find_default_label_csv(target_dir: Path, output_dir: Path) -> Path:
    candidates = []
    search_dirs = [
        output_dir,
        target_dir,
        target_dir / "pathway_probability",
    ]
    seen_dirs = set()
    for directory in search_dirs:
        directory = Path(directory)
        if directory in seen_dirs:
            continue
        seen_dirs.add(directory)
        if directory.exists():
            candidates.extend(sorted(directory.glob("*with_label*.csv")))
            candidates.extend(sorted(directory.glob("*label*.csv")))
    if not candidates:
        raise FileNotFoundError(
            "No CSV containing afdb_labels was found. Specify one with --label_csv, for example uniprot_mapping.with_label.csv"
        )
    return candidates[0]


def get_target_source_labels(label_csv: Path, target: str) -> Tuple[List[int], pd.DataFrame, str, str]:
    df = read_csv_auto(label_csv)

    protein_col = find_column(
        df,
        candidates=["protein_name", "protein", "input_id", "pdb_id"],
        fallback_first=True,
    )
    label_col = find_column(
        df,
        candidates=["afdb_labels", "afdb_label", "label", "labels"],
        fallback_first=False,
    )

    df = df.copy()
    df[protein_col] = df[protein_col].astype(str).str.strip()

    sub = df[df[protein_col] == target]
    if sub.empty:
        if len(df) == 1:
            sub = df
        else:
            raise ValueError(f"Target protein {target} was not found in {label_csv}. Protein column: {protein_col}")

    labels = []
    for _, row in sub.iterrows():
        for label in parse_labels(row.get(label_col, "")):
            if label not in labels:
                labels.append(label)

    if not labels:
        raise ValueError(f"Target {target} has no valid afdb_labels in {label_csv}.")

    return labels, sub, protein_col, label_col

def process_one_target(args, target: str) -> None:
    root = Path(args.root)
    target_dir = root / target
    output_dir = Path(args.output_dir) if args.output_dir else target_dir / "pathway_probability"
    msta_dir = find_msta_dir(target_dir, Path(args.msta_dir) if args.msta_dir else None)

    output_dir.mkdir(parents=True, exist_ok=True)

    label_csv = Path(args.label_csv) if args.label_csv else find_default_label_csv(target_dir, output_dir)
    source_labels, label_rows, protein_col, label_col = get_target_source_labels(label_csv, target)

    print(f"\n[INFO] Target: {target}")
    print(f"[INFO] Label CSV: {label_csv}")
    print(f"[INFO] Protein column: {protein_col}")
    print(f"[INFO] afdb_labels column: {label_col}")
    print(f"[INFO] Source labels: {source_labels} -> {[DEFAULT_LABEL_TO_TAXON[x] for x in source_labels]}")
    print(f"[INFO] MSTA directory: {msta_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(
        f"[INFO] Probability rule: taxonomic_tier_rank_power, "
        f"probability_rank_power={args.probability_rank_power}"
    )
    print(
        f"[INFO] distance_threshold={args.distance_threshold} is accepted for compatibility "
        "but is not used in this taxonomy probability script."
    )

    validity_df = collect_msta_validity(
        target=target,
        msta_dir=msta_dir,
        zero_fraction_threshold=args.zero_fraction_threshold,
        zero_eps=args.zero_eps,
    )

    all_labels = list(range(1, 11))
    valid_labels = [
        int(r["label"])
        for _, r in validity_df.iterrows()
        if bool(r["valid_msta"])
    ]

    if len(valid_labels) == 0:
        raise ValueError(f"Target {target} has no valid temp_msa_X.msta files, so pathway probabilities cannot be assigned.")

    invalid_labels = [x for x in all_labels if x not in set(valid_labels)]

    label_to_path_index = {label: i for i, label in enumerate(valid_labels)}

    filtered_tiers_by_source = {
        src_label: filter_tiers(DEFAULT_LABEL_TO_TIERS[src_label], valid_labels)
        for src_label in DEFAULT_LABEL_TO_TIERS
    }

    fallback_rows = []
    for src_label in range(1, 11):
        all_tiers = DEFAULT_LABEL_TO_TIERS[src_label]
        filtered_tiers = filtered_tiers_by_source[src_label]
        flat_all = flatten_tiers(all_tiers)
        flat_filtered = flatten_tiers(filtered_tiers)
        removed = [x for x in flat_all if x not in set(flat_filtered)]

        fallback_rows.append(
            {
                "target": target,
                "source_label": src_label,
                "source_taxon": DEFAULT_LABEL_TO_TAXON[src_label],
                "fallback_order_all_labels": ";".join(map(str, flat_all)),
                "fallback_order_all_taxa": ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in flat_all),
                "fallback_order_filtered_labels": ";".join(map(str, flat_filtered)),
                "fallback_order_filtered_taxa": ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in flat_filtered),
                "fallback_tiers_all_labels": format_tiers_labels(all_tiers),
                "fallback_tiers_all_taxa": format_tiers_taxa(all_tiers),
                "fallback_tiers_filtered_labels": format_tiers_labels(filtered_tiers),
                "fallback_tiers_filtered_taxa": format_tiers_taxa(filtered_tiers),
                "n_tiers_all": len(all_tiers),
                "n_tiers_filtered": len(filtered_tiers),
                "removed_invalid_labels": ";".join(map(str, removed)),
                "removed_invalid_taxa": ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in removed),
                "valid_labels_used_for_probability": ";".join(map(str, valid_labels)),
                "invalid_labels_without_probability": ";".join(map(str, invalid_labels)),
                "probability_rule": "taxonomic_tier_rank_power",
                "probability_rank_power": args.probability_rank_power,
                "distance_threshold": args.distance_threshold,
                "note": (
                    "Equal probabilities are assigned within each taxonomic proximity tier. "
                    "Invalid MSTA labels are removed before probability assignment, and empty tiers are dropped. "
                    "Flattened fallback_order columns are retained only for compatibility and do not imply "
                    "within-tier ranking. distance_threshold is kept only for command compatibility."
                ),
            }
        )

    fallback_df = pd.DataFrame(fallback_rows)
    fallback_csv = output_dir / "fallback_order.filtered.csv"
    fallback_df.to_csv(fallback_csv, index=False)

    probabilities = combine_probabilities_for_labels(
        source_labels=source_labels,
        filtered_tiers_by_source=filtered_tiers_by_source,
        rank_power=args.probability_rank_power,
    )

    if not probabilities:
        raise ValueError(
            f"Target {target} with source_labels={source_labels} has no candidate pathways after invalid MSTA files are filtered."
        )

    tier_rank_by_source = {
        src_label: get_effective_tier_rank_by_label(filtered_tiers_by_source[src_label])
        for src_label in source_labels
    }

    rows = []
    source_labels_set = set(source_labels)
    source_label_str = ";".join(map(str, source_labels))
    source_taxon_str = ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in source_labels)

    for label in valid_labels:
        probability = probabilities.get(label, 0.0)
        taxon = DEFAULT_LABEL_TO_TAXON[label]
        validity_row = validity_df[validity_df["label"] == label].iloc[0].to_dict()

        path_index = label_to_path_index[label]
        batch_name = f"batch{path_index}"

        assignments = []
        for src_label in source_labels:
            tier_rank = tier_rank_by_source.get(src_label, {}).get(label)
            if tier_rank is not None:
                assignments.append(f"{src_label}:{tier_rank}")

        single_source_tier = None
        if len(source_labels) == 1:
            single_source_tier = tier_rank_by_source[source_labels[0]].get(label)

        rows.append(
            {
                "target": target,
                "path_index": path_index,
                "batch_name": batch_name,
                "label": label,
                "taxon": taxon,
                "probability": probability,
                "taxonomic_tier": single_source_tier,
                "source_tier_assignments": ";".join(assignments),
                "probability_rule": "taxonomic_tier_rank_power",
                "probability_rank_power": args.probability_rank_power,
                "distance_threshold": args.distance_threshold,
                "is_source_label": label in source_labels_set,
                "source_labels": source_label_str,
                "source_taxa": source_taxon_str,
                "msta_file": validity_row["msta_file"],
                "zero_count": validity_row["zero_count"],
                "n_residues": validity_row["n_residues"],
                "zero_ratio": validity_row["zero_ratio"],
                "invalid_reason": validity_row.get("invalid_reason", ""),
            }
        )

    rows.sort(key=lambda r: (-r["probability"], int(r["path_index"])))

    current_rank = 0
    previous_probability = None
    for row in rows:
        probability = float(row["probability"])
        if previous_probability is None or not math.isclose(probability, previous_probability, rel_tol=1e-12, abs_tol=1e-15):
            current_rank += 1
            previous_probability = probability
        row["rank"] = current_rank
        row["probability"] = format_probability(probability)
        if isinstance(row["zero_ratio"], float) and not math.isnan(row["zero_ratio"]):
            row["zero_ratio"] = f"{row['zero_ratio']:.6f}"

    out_cols = [
        "target", "rank", "path_index", "batch_name",
        "label", "taxon", "probability", "taxonomic_tier", "source_tier_assignments",
        "probability_rule", "probability_rank_power", "distance_threshold",
        "is_source_label", "source_labels", "source_taxa",
        "msta_file", "zero_count", "n_residues", "zero_ratio",
        "invalid_reason",
    ]

    out_df = pd.DataFrame(rows, columns=out_cols)
    out_csv = output_dir / "pathway_probability.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"[DONE] Saved: {out_csv}")
    print(f"[DONE] Saved: {fallback_csv}")
    print("[INFO] Equal taxonomic tiers retain exactly equal pathway probabilities.")
    print("[INFO] Invalid paths are not assigned probabilities and are not written to pathway_probability.csv.")
    print("\n[PREVIEW]")
    print(out_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign pathway probabilities using predefined taxonomic proximity tiers. "
            "Pathways in the same tier receive exactly equal probabilities. "
            "Invalid temp_msa_X.msta files are removed before probability assignment."
        )
    )

    parser.add_argument(
        "target_positional",
        nargs="?",
        default=None,
        help="Target protein name, for example 1AB7_A. It may also be specified with --target.",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target protein name, for example 1AB7_A.",
    )
    parser.add_argument(
        "--root",
        default="/storage/zhaokailong/FPdiffusion/test_mult_pathway/Fold_order_testdata/test_data_0513_msa_new_db",
        help="Root directory containing multiple-pathway results.",
    )
    parser.add_argument(
        "--label_csv",
        default=None,
        help=(
            "CSV containing afdb_labels. If omitted, the script searches the output directory, "
            "<target>/pathway_probability, and <target> for *with_label*.csv or *label*.csv."
        ),
    )
    parser.add_argument(
        "--msta_dir",
        default=None,
        help=(
            "Directory containing temp_msa_X.msta files. If omitted, the script uses <root>/<target>/msta, "
            "or <root>/<target> if the msta directory does not exist."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Default: <root>/<target>/pathway_probability.",
    )
    parser.add_argument(
        "--zero_fraction_threshold",
        type=float,
        default=0.5,
        help="A temp_msa_X.msta file is invalid when its zero-value fraction exceeds this threshold. Default: 0.5.",
    )
    parser.add_argument(
        "--zero_eps",
        type=float,
        default=1e-12,
        help="Floating-point tolerance used to identify zero values. Default: 1e-12.",
    )
    parser.add_argument(
        "--probability_rank_power",
        type=float,
        default=1.0,
        help=(
            "Power applied to taxonomic tier-rank weights. The default 1.0 produces a linear decrease across tiers; "
            "all pathways in the same tier receive identical weights."
        ),
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.35,
        help=(
            "Compatibility option retained from older commands. This script assigns taxonomy probabilities but does not cluster, "
            "so this value is not used in the calculation. Default: 0.35."
        ),
    )

    parser.add_argument("--method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--decay", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    target = args.target or args.target_positional
    if not target:
        raise ValueError("Specify a target protein, for example: python pathway_probability.py 1AB7_A")

    process_one_target(args, target.strip())


if __name__ == "__main__":
    main()
