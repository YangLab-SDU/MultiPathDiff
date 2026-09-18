#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据目标蛋白所属物种类别，为多路径预测结果分配 pathway score。

本版本使用预定义的 taxonomic proximity tiers，而不是强制把 10 个物种组
排成严格的 1-10 顺序。同一个 tier 内的物种组获得完全相同的 pathway score。

核心逻辑
--------
1. 从包含 afdb_labels 的 CSV 中读取目标蛋白的 label。
2. 检查每个 temp_msa_X.msta 是否有效。0 值残基比例 > 0.5 时视为无效。
3. 根据目标 label 选择固定的 taxonomic proximity tiers：
   - 目标自身 species group 优先；
   - 具体、近缘 lineage 优先；
   - 在当前简化 taxonomy 中无法进一步区分的 groups 放在同一 tier；
   - broad ancestral databases 放在较低优先级。
4. 删除无效 MSTA 对应的 label。若一个 tier 删除后为空，则该 tier 不参与本次计分。
5. 对剩余非空 tiers 按 tier rank 分配线性递减权重：
       raw_weight(tier t) = (K - t + 1) ** probability_rank_power
   其中 K 为过滤后非空 tier 数，t 从 1 开始。
   同一个 tier 中所有 pathway 使用相同 raw_weight，然后在所有有效 pathway 上归一化：
       score_i = raw_weight(tier_i) / sum_j raw_weight(tier_j)
   默认 probability_rank_power = 1.0。
6. 如果一个蛋白具有多个 afdb_labels，则分别计算每个 source label 的 tier score，
   将各 source label 对同一路径的 score 相加后再次归一化。

输出文件
--------
默认输出到：
    <root>/<target>/pathway_score/pathway_score.csv
    <root>/<target>/pathway_score/fallback_order.filtered.csv

注意
----
- pathway_score.csv 中 score 相同的路径会保留完全相同的数值。
- rank 使用 dense rank：相同 score 具有相同 rank。
- path_index 与真实生成出来的 sample 编号一致，只由有效 label 的原始编号顺序决定。
"""

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# 10 个 species-group databases 的固定 label
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


# -----------------------------------------------------------------------------
# 固定 taxonomic proximity tiers
#
# 每个 source label 对应一个从高到低的 tier 列表。
# 同一个子列表中的 labels 属于同一 tier，获得完全相同的 pathway score。
# broad ancestral databases 被保留为较低优先级 fallback。
# -----------------------------------------------------------------------------

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
    """确保每个 source label 的 tiers 恰好覆盖 1..10，且没有重复。"""
    expected = set(DEFAULT_LABEL_TO_TAXON)
    if set(DEFAULT_LABEL_TO_TIERS) != expected:
        raise ValueError("DEFAULT_LABEL_TO_TIERS 的 source labels 必须与 DEFAULT_LABEL_TO_TAXON 完全一致。")

    for src_label, tiers in DEFAULT_LABEL_TO_TIERS.items():
        flat = [x for tier in tiers for x in tier]
        if len(flat) != len(set(flat)):
            raise ValueError(f"source label {src_label} 的 tier 定义中存在重复 label: {flat}")
        if set(flat) != expected:
            missing = sorted(expected - set(flat))
            extra = sorted(set(flat) - expected)
            raise ValueError(
                f"source label {src_label} 的 tier 定义必须覆盖全部 10 个 labels; "
                f"missing={missing}, extra={extra}"
            )


validate_tier_definition()


# -----------------------------------------------------------------------------
# 基础读取与解析函数
# -----------------------------------------------------------------------------

def read_csv_auto(csv_file: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_file, encoding=enc)
        except UnicodeDecodeError as e:
            last_error = e
    raise RuntimeError(f"无法读取 CSV 文件: {csv_file}, last error: {last_error}")


def find_column(df: pd.DataFrame, candidates: Iterable[str], fallback_first: bool = False) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if fallback_first:
        return df.columns[0]
    raise ValueError(
        f"没有找到目标列，候选列名为: {list(candidates)}\n"
        f"当前文件列名为: {list(df.columns)}"
    )


def parse_labels(x) -> List[int]:
    """支持 2、2;3、2,3、2 3 等格式。"""
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


# -----------------------------------------------------------------------------
# MSTA 有效性判断
# -----------------------------------------------------------------------------

def read_msta_value_column(msta_file: Path) -> List[float]:
    """读取 temp_msa_X.msta 中每行最后一个可转成 float 的字段。"""
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
    """
    判断单个 temp_msa_X.msta 是否有效。
    只有 zero_ratio > threshold 时才无效；zero_ratio == threshold 仍然有效。
    """
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


# -----------------------------------------------------------------------------
# Taxonomic tier 处理
# -----------------------------------------------------------------------------

def flatten_tiers(tiers: List[List[int]]) -> List[int]:
    return [label for tier in tiers for label in tier]


def filter_tiers(tiers: List[List[int]], valid_labels: List[int]) -> List[List[int]]:
    """
    从每个 tier 中删除无效 labels，并删除过滤后为空的 tier。
    同一 tier 内剩余 labels 仍保持完全同权。
    """
    valid_set = set(valid_labels)
    filtered = []
    for tier in tiers:
        kept = [label for label in tier if label in valid_set]
        if kept:
            filtered.append(kept)
    return filtered


def format_tiers_labels(tiers: List[List[int]]) -> str:
    """例如 [[7],[6],[10],[1,2,3]] -> '7 > 6 > 10 > (1 = 2 = 3)'。"""
    parts = []
    for tier in tiers:
        if len(tier) == 1:
            parts.append(str(tier[0]))
        else:
            parts.append("(" + " = ".join(map(str, tier)) + ")")
    return " > ".join(parts)


def format_tiers_taxa(tiers: List[List[int]]) -> str:
    """把 tier labels 格式化为带 = 和 > 的 taxon 字符串。"""
    parts = []
    for tier in tiers:
        names = [DEFAULT_LABEL_TO_TAXON[x] for x in tier]
        if len(names) == 1:
            parts.append(names[0])
        else:
            parts.append("(" + " = ".join(names) + ")")
    return " > ".join(parts)


def get_effective_tier_rank_by_label(tiers: List[List[int]]) -> Dict[int, int]:
    """返回过滤后非空 tiers 中，每个 label 的 effective tier rank（从 1 开始）。"""
    out = {}
    for tier_rank, tier in enumerate(tiers, start=1):
        for label in tier:
            out[label] = tier_rank
    return out


# -----------------------------------------------------------------------------
# 分数分配
# -----------------------------------------------------------------------------

def tier_rank_power_scores(
    tiers: List[List[int]],
    rank_power: float = 1.0,
    normalize: bool = True,
) -> Dict[int, float]:
    """
    按 taxonomic tier rank 分配 score。

    若过滤后共有 K 个非空 tiers，则第 t 个 tier 的每条 pathway 均获得：
        raw_weight(t) = (K - t + 1) ** rank_power

    注意：同一 tier 的每条 pathway 使用完全相同的 raw_weight，
    不会因为该 tier 中 pathway 数量不同而在 tier 内再次均分。

    normalize=True 时，在所有有效 pathway 上归一化，使 score 总和为 1。
    """
    if not tiers:
        return {}

    power = float(rank_power)
    if power < 0:
        raise ValueError("--probability_rank_power 必须 >= 0。")

    k = len(tiers)
    raw_score_by_label: Dict[int, float] = {}

    for tier_rank, tier in enumerate(tiers, start=1):
        raw_weight = float(k - tier_rank + 1) ** power
        for label in tier:
            raw_score_by_label[label] = raw_weight

    if not normalize:
        return raw_score_by_label

    total = sum(raw_score_by_label.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("tier score 总和无效，请检查 --probability_rank_power。")

    return {label: score / total for label, score in raw_score_by_label.items()}


def combine_scores_for_labels(
    source_labels: List[int],
    filtered_tiers_by_source: Dict[int, List[List[int]]],
    rank_power: float,
) -> Dict[int, float]:
    """
    单个 source label：直接按其 filtered tiers 生成同-tier同分的 pathway score。

    多个 source labels：分别生成一套归一化 tier score，再按 label 相加，
    最后对总 score 再归一化。这样保留每个 source label 的等权贡献。
    """
    score_sum: Dict[int, float] = {}

    for src_label in source_labels:
        tiers = filtered_tiers_by_source.get(src_label, [])
        one_source_scores = tier_rank_power_scores(
            tiers,
            rank_power=rank_power,
            normalize=True,
        )
        for label, score in one_source_scores.items():
            score_sum[label] = score_sum.get(label, 0.0) + score

    total = sum(score_sum.values())
    if total > 0:
        score_sum = {label: score / total for label, score in score_sum.items()}
    return score_sum


def format_score(x: float) -> str:
    return f"{x:.8f}"


# -----------------------------------------------------------------------------
# afdb_labels 读取
# -----------------------------------------------------------------------------

def find_default_label_csv(target_dir: Path, output_dir: Path) -> Path:
    candidates = []
    search_dirs = [
        output_dir,
        target_dir,
        target_dir / "pathway_probability",
        target_dir / "pathway_score",
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
            "没有找到包含 afdb_labels 的 CSV。请用 --label_csv 指定，例如 uniprot_mapping.with_label.csv"
        )
    return candidates[0]


def get_target_source_labels(label_csv: Path, target: str) -> Tuple[List[int], pd.DataFrame, str, str]:
    df = read_csv_auto(label_csv)

    protein_col = find_column(
        df,
        candidates=["protein_name", "protein", "蛋白质名", "蛋白质名称", "input_id", "pdb_id"],
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
            raise ValueError(f"在 {label_csv} 中找不到目标蛋白 {target}。蛋白列为 {protein_col}")

    labels = []
    for _, row in sub.iterrows():
        for label in parse_labels(row.get(label_col, "")):
            if label not in labels:
                labels.append(label)

    if not labels:
        raise ValueError(f"{target} 在 {label_csv} 中没有有效 afdb_labels。")

    return labels, sub, protein_col, label_col


# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def process_one_target(args, target: str) -> None:
    root = Path(args.root)
    target_dir = root / target
    output_dir = Path(args.output_dir) if args.output_dir else target_dir / "pathway_score"
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
        f"[INFO] Score rule: taxonomic_tier_rank_power, "
        f"probability_rank_power={args.probability_rank_power}"
    )
    print(
        f"[INFO] distance_threshold={args.distance_threshold} is accepted for compatibility "
        "but is not used in this taxonomy score script."
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
        raise ValueError(f"{target} 没有任何有效 temp_msa_X.msta，无法分配 pathway score。")

    invalid_labels = [x for x in all_labels if x not in set(valid_labels)]

    # path_index 与真实生成 sample 编号一致。
    label_to_path_index = {label: i for i, label in enumerate(valid_labels)}

    # 对每个 source label 的固定 tiers 删除无效 MSTA，并删除空 tier。
    filtered_tiers_by_source = {
        src_label: filter_tiers(DEFAULT_LABEL_TO_TIERS[src_label], valid_labels)
        for src_label in DEFAULT_LABEL_TO_TIERS
    }

    # 保存完整和过滤后的 tier 定义。
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
                # 以下 4 列保留旧的 flattened 格式，方便兼容和查看。
                # 注意：同一 tier 内的先后顺序不代表 score 差异。
                "fallback_order_all_labels": ";".join(map(str, flat_all)),
                "fallback_order_all_taxa": ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in flat_all),
                "fallback_order_filtered_labels": ";".join(map(str, flat_filtered)),
                "fallback_order_filtered_taxa": ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in flat_filtered),
                # 新增明确表示同 tier 并列关系的列。
                "fallback_tiers_all_labels": format_tiers_labels(all_tiers),
                "fallback_tiers_all_taxa": format_tiers_taxa(all_tiers),
                "fallback_tiers_filtered_labels": format_tiers_labels(filtered_tiers),
                "fallback_tiers_filtered_taxa": format_tiers_taxa(filtered_tiers),
                "n_tiers_all": len(all_tiers),
                "n_tiers_filtered": len(filtered_tiers),
                "removed_invalid_labels": ";".join(map(str, removed)),
                "removed_invalid_taxa": ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in removed),
                "valid_labels_used_for_score": ";".join(map(str, valid_labels)),
                "invalid_labels_not_scored": ";".join(map(str, invalid_labels)),
                "score_rule": "taxonomic_tier_rank_power",
                "probability_rank_power": args.probability_rank_power,
                "distance_threshold": args.distance_threshold,
                "note": (
                    "Equal scores are assigned within each taxonomic proximity tier. "
                    "Invalid MSTA labels are removed before score assignment, and empty tiers are dropped. "
                    "Flattened fallback_order columns are retained only for compatibility and do not imply "
                    "within-tier ranking. distance_threshold is kept only for command compatibility."
                ),
            }
        )

    fallback_df = pd.DataFrame(fallback_rows)
    fallback_csv = output_dir / "fallback_order.filtered.csv"
    fallback_df.to_csv(fallback_csv, index=False)

    scores = combine_scores_for_labels(
        source_labels=source_labels,
        filtered_tiers_by_source=filtered_tiers_by_source,
        rank_power=args.probability_rank_power,
    )

    if not scores:
        raise ValueError(
            f"{target} 的 source_labels={source_labels} 在过滤无效 MSTA 后没有可用候选路径。"
        )

    # 记录每个 label 在各 source label 下的 effective tier rank。
    tier_rank_by_source = {
        src_label: get_effective_tier_rank_by_label(filtered_tiers_by_source[src_label])
        for src_label in source_labels
    }

    rows = []
    source_labels_set = set(source_labels)
    source_label_str = ";".join(map(str, source_labels))
    source_taxon_str = ";".join(DEFAULT_LABEL_TO_TAXON[x] for x in source_labels)

    for label in valid_labels:
        score = scores.get(label, 0.0)
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
                "score": score,
                "taxonomic_tier": single_source_tier,
                "source_tier_assignments": ";".join(assignments),
                "score_rule": "taxonomic_tier_rank_power",
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

    # score 降序；同分时仅用 path_index 保证输出顺序稳定，不改变 score。
    rows.sort(key=lambda r: (-r["score"], int(r["path_index"])))

    # dense rank：同 score 得到相同 rank。
    current_rank = 0
    previous_score = None
    for row in rows:
        score = float(row["score"])
        if previous_score is None or not math.isclose(score, previous_score, rel_tol=1e-12, abs_tol=1e-15):
            current_rank += 1
            previous_score = score
        row["rank"] = current_rank
        row["score"] = format_score(score)
        if isinstance(row["zero_ratio"], float) and not math.isnan(row["zero_ratio"]):
            row["zero_ratio"] = f"{row['zero_ratio']:.6f}"

    out_cols = [
        "target", "rank", "path_index", "batch_name",
        "label", "taxon", "score", "taxonomic_tier", "source_tier_assignments",
        "score_rule", "probability_rank_power", "distance_threshold",
        "is_source_label", "source_labels", "source_taxa",
        "msta_file", "zero_count", "n_residues", "zero_ratio",
        "invalid_reason",
    ]

    out_df = pd.DataFrame(rows, columns=out_cols)
    out_csv = output_dir / "pathway_score.csv"
    out_df.to_csv(out_csv, index=False)

    print(f"[DONE] Saved: {out_csv}")
    print(f"[DONE] Saved: {fallback_csv}")
    print("[INFO] Equal taxonomic tiers retain exactly equal pathway scores.")
    print("[INFO] Invalid paths are not scored and are not written to pathway_score.csv.")
    print("\n[PREVIEW]")
    print(out_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Assign pathway scores using predefined taxonomic proximity tiers. "
            "Pathways in the same tier receive exactly equal scores. "
            "Invalid temp_msa_X.msta files are removed before score assignment."
        )
    )

    parser.add_argument(
        "target_positional",
        nargs="?",
        default=None,
        help="目标蛋白名称，例如 1AB7_A。也可以使用 --target 指定。",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="目标蛋白名称，例如 1AB7_A。",
    )
    parser.add_argument(
        "--root",
        default="/storage/zhaokailong/FPdiffusion/test_mult_pathway/Fold_order_testdata/test_data_0513_msa_new_db",
        help="多路径结果根目录。",
    )
    parser.add_argument(
        "--label_csv",
        default=None,
        help=(
            "包含 afdb_labels 的 CSV。若不指定，会在 <target>/pathway_probability、"
            "<target>/pathway_score 和 <target> 目录下自动查找 *with_label*.csv 或 *label*.csv。"
        ),
    )
    parser.add_argument(
        "--msta_dir",
        default=None,
        help=(
            "包含 temp_msa_X.msta 的目录。若不指定，默认使用 <root>/<target>/msta；"
            "如果不存在则使用 <root>/<target>。"
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="输出目录。若不指定，默认输出到 <root>/<target>/pathway_score。",
    )
    parser.add_argument(
        "--zero_fraction_threshold",
        type=float,
        default=0.5,
        help="0 值比例超过该阈值时，认为 temp_msa_X.msta 无效。默认 0.5。",
    )
    parser.add_argument(
        "--zero_eps",
        type=float,
        default=1e-12,
        help="判断是否为 0 的浮点误差阈值。默认 1e-12。",
    )
    parser.add_argument(
        "--probability_rank_power",
        type=float,
        default=1.0,
        help=(
            "taxonomic tier rank 权重的 power。默认 1.0，即不同 tier 线性递减；"
            "同一 tier 中所有 pathways 权重完全相同。"
        ),
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=0.35,
        help=(
            "兼容旧命令的参数；本脚本只做 taxonomy score 分配，不进行聚类，"
            "因此该参数不会参与计算。默认 0.35。"
        ),
    )

    # 兼容旧命令：保留但不再使用。
    parser.add_argument("--method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--decay", type=float, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    target = args.target or args.target_positional
    if not target:
        raise ValueError("请指定目标蛋白名称，例如：python assign_pathway_score_by_taxonomy_all10_tier_equal.py 1AB7_A")

    process_one_target(args, target.strip())


if __name__ == "__main__":
    main()
