#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import glob
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import linkage, fcluster, cut_tree
    from scipy.spatial.distance import squareform
except Exception as e:
    raise ImportError(
        "This script requires scipy for hierarchical clustering. Install it with: pip install scipy"
    ) from e


# ----------------------------- PDB parsing -----------------------------

def _parse_xyz_from_pdb_line(line):
    """Parse x, y, z from a PDB ATOM line."""
    try:
        return np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
    except Exception:
        nums = re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?",
            line[30:],
        )
        if len(nums) < 3:
            raise ValueError(f"Cannot parse xyz from line: {line.rstrip()}")
        return np.array(list(map(float, nums[:3])), dtype=np.float64)


def _parse_resseq_from_pdb_line(line, parts):
    """Parse residue number from a PDB ATOM line."""
    try:
        return int(line[22:26].strip())
    except Exception:
        for idx in [5, 4]:
            if idx < len(parts):
                m = re.search(r"-?\d+", parts[idx])
                if m:
                    return int(m.group())
    raise ValueError(f"Cannot parse residue number from line: {line.rstrip()}")


def read_ca_coords(pdb_file):
    """
    Read one CA coordinate per residue from the first MODEL of a PDB file.

    Returns:
        coords: np.ndarray, shape [L, 3]
        residue_info: [(chain_id, resseq, resname), ...]
    """
    coords = []
    residue_info = []
    seen = set()
    has_model_record = False
    in_first_model = False

    with open(pdb_file, "r", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            record = line[:6].strip()

            if record == "MODEL":
                has_model_record = True
                if not in_first_model:
                    in_first_model = True
                continue

            if record == "ENDMDL" and has_model_record:
                break

            if has_model_record and not in_first_model:
                continue

            if record != "ATOM":
                continue

            atom_name = line[12:16].strip()
            parts = line.split()
            if atom_name == "" and len(parts) >= 3:
                atom_name = parts[2]
            if atom_name != "CA":
                continue

            altloc = line[16:17].strip()
            if altloc not in ["", "A", "1"]:
                continue

            resname = line[17:20].strip()
            if resname == "" and len(parts) >= 4:
                resname = parts[3]

            chain_id = line[21:22].strip()
            if chain_id == "" and len(parts) >= 5:
                if not re.match(r"-?\d+", parts[4]):
                    chain_id = parts[4]
                else:
                    chain_id = "A"
            if chain_id == "":
                chain_id = "A"

            try:
                resseq = _parse_resseq_from_pdb_line(line, parts)
                xyz = _parse_xyz_from_pdb_line(line)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse {pdb_file}, line {line_no}: {line.rstrip()}"
                ) from e

            key = (chain_id, resseq, resname)
            if key in seen:
                continue
            seen.add(key)
            coords.append(xyz)
            residue_info.append(key)

    if len(coords) == 0:
        raise ValueError(f"No CA atoms found in {pdb_file}")

    return np.asarray(coords, dtype=np.float64), residue_info


def pairwise_dist(coords):
    """Pairwise Euclidean distance matrix for coords [L, 3]."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ----------------------------- global alignment -----------------------------

def kabsch_superpose(mobile, target):
    """
    Globally align mobile coordinates to target coordinates using Kabsch.

    Args:
        mobile: [L, 3], coordinates to move.
        target: [L, 3], reference coordinates.

    Returns:
        aligned_mobile: [L, 3]
    """
    mobile = np.asarray(mobile, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if mobile.shape != target.shape:
        raise ValueError(f"Shape mismatch for Kabsch: mobile={mobile.shape}, target={target.shape}")

    mob_cent = mobile.mean(axis=0)
    tar_cent = target.mean(axis=0)
    p = mobile - mob_cent
    q = target - tar_cent

    c = p.T @ q
    v, s, wt = np.linalg.svd(c)
    d = np.sign(np.linalg.det(v @ wt))
    dmat = np.diag([1.0, 1.0, d])
    r = v @ dmat @ wt
    return p @ r + tar_cent


def compute_aligned_ca_distance_curve(coords_stack, final_coords):
    """
    For each frame, globally align it to final_coords and compute per-residue CA distance.

    Args:
        coords_stack: [T, L, 3]
        final_coords: [L, 3]

    Returns:
        local_distances: [T, L]
    """
    T, L, _ = coords_stack.shape
    out = np.full((T, L), np.nan, dtype=np.float64)
    for t in range(T):
        aligned = kabsch_superpose(coords_stack[t], final_coords)
        out[t] = np.sqrt(np.sum(np.square(aligned - final_coords), axis=1))
    return out


# ----------------------------- file grouping -----------------------------

def natural_key(path):
    base = os.path.basename(str(path))
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", base)]


def parse_frame_name(path):
    """
    Parse names like model_299_sample2.pdb.
    Returns: model_index, sample_id
    """
    base = os.path.basename(path)
    m = re.match(r"model_(\d+)_sample(\d+)\.pdb$", base)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def collect_trajectories(frames_dir, pattern="model_*_sample*.pdb"):
    files = sorted(glob.glob(os.path.join(frames_dir, pattern)), key=natural_key)
    if len(files) == 0:
        raise FileNotFoundError(f"No PDB files found in {frames_dir} with pattern {pattern}")

    by_sample = defaultdict(list)
    skipped = []
    for f in files:
        parsed = parse_frame_name(f)
        if parsed is None:
            skipped.append(f)
            continue
        model_idx, sample_id = parsed
        by_sample[sample_id].append((model_idx, f))

    if len(by_sample) == 0:
        raise ValueError("No files matched the expected name format: model_<step>_sample<id>.pdb")

    trajectories = {}
    for sample_id, items in sorted(by_sample.items()):
        trajectories[sample_id] = sorted(items, key=lambda x: x[0])

    if skipped:
        print(f"WARNING: skipped {len(skipped)} files that do not match model_<step>_sample<id>.pdb")

    return trajectories


def load_trajectory_coords(items, expected_len):
    """
    Load one trajectory into arrays.

    Returns:
        model_indices: np.ndarray [T]
        coords_stack: np.ndarray [T, L, 3]
        frame_paths: list[str]
    """
    model_indices = []
    coords_list = []
    frame_paths = []

    for model_idx, pdb in items:
        coords, info = read_ca_coords(pdb)
        if coords.shape[0] != expected_len:
            raise ValueError(
                f"Residue number mismatch in {pdb}: frame has {coords.shape[0]} CA atoms, "
                f"reference final model has {expected_len}. This script expects the same residue order/length."
            )
        model_indices.append(model_idx)
        coords_list.append(coords)
        frame_paths.append(pdb)

    return np.asarray(model_indices, dtype=int), np.stack(coords_list, axis=0), frame_paths


def select_final_item(items, final_model_index=299):
    """Return (model_index, pdb_path) for the desired final model, falling back to max model index."""
    by_model = {int(m): p for m, p in items}
    if int(final_model_index) in by_model:
        return int(final_model_index), by_model[int(final_model_index)]
    model_idx, pdb = max(items, key=lambda x: x[0])
    return int(model_idx), pdb


def select_final_coords(model_indices, coords_stack, sample_id, final_model_index=299):
    """
    Select the final structure used as residue-folding reference.

    Prefer exact --final_model_index, usually 299. If it is absent, use the maximum
    model index available for that sample.
    """
    model_indices = np.asarray(model_indices, dtype=int)
    hit = np.where(model_indices == int(final_model_index))[0]
    if len(hit) > 0:
        idx = int(hit[0])
    else:
        idx = int(np.argmax(model_indices))
        print(
            f"WARNING: sample{sample_id} has no model_{final_model_index}; "
            f"using model_{int(model_indices[idx])} as final reference instead."
        )
    return coords_stack[idx].copy(), int(model_indices[idx])


def load_final_references(trajectories, final_model_index=299, reference_sample=None):
    """
    Load model_299_sampleX final structures from every sample.

    Returns:
        final_coords_by_sample: dict[sample_id] -> coords [L,3]
        final_info_by_sample: dict[sample_id] -> residue_info
        final_summary_df: DataFrame
        reference_sample_id: sample used for residue labels
    """
    final_coords_by_sample = {}
    final_info_by_sample = {}
    records = []

    sample_ids = sorted(trajectories.keys())
    if reference_sample is None:
        reference_sample_id = sample_ids[0]
    else:
        reference_sample_id = int(reference_sample)
        if reference_sample_id not in trajectories:
            raise ValueError(f"--reference_sample {reference_sample_id} was not found in frames_dir")

    expected_len = None
    ref_info = None

    for sid in sample_ids:
        model_idx, final_pdb = select_final_item(trajectories[sid], final_model_index=final_model_index)
        coords, info = read_ca_coords(final_pdb)
        if expected_len is None:
            expected_len = coords.shape[0]
        elif coords.shape[0] != expected_len:
            raise ValueError(
                f"Final model residue count mismatch: sample{sid} has {coords.shape[0]} CA atoms, "
                f"expected {expected_len}."
            )
        if sid == reference_sample_id:
            ref_info = info
        final_coords_by_sample[sid] = coords
        final_info_by_sample[sid] = info
        records.append({
            "sample": sid,
            "final_model_used": model_idx,
            "final_pdb": final_pdb,
            "n_residues": coords.shape[0],
            "used_for_residue_labels": sid == reference_sample_id,
        })

    if ref_info is None:
        ref_info = final_info_by_sample[reference_sample_id]

    return final_coords_by_sample, final_info_by_sample, pd.DataFrame(records), reference_sample_id


# ----------------------------- feature calculation -----------------------------

def first_persistent_time(values, model_indices, threshold, min_consecutive, greater_equal=True):
    """
    First model index at which each feature satisfies a threshold for min_consecutive frames.

    values: [T, F]
    Returns:
        times: [F], actual model index; if never satisfied, max(model_indices)+1
    """
    values = np.asarray(values, dtype=np.float64)
    T, F = values.shape
    miss_time = int(np.max(model_indices)) + 1
    times = np.full(F, miss_time, dtype=np.float64)

    if greater_equal:
        ok = values >= threshold
    else:
        ok = values <= threshold
    ok = np.where(np.isfinite(values), ok, False)

    m = max(1, int(min_consecutive))
    if T < m:
        return times

    for f in range(F):
        series = ok[:, f]
        for start in range(0, T - m + 1):
            if np.all(series[start:start + m]):
                times[f] = float(model_indices[start])
                break
    return times


def define_reference_contacts_from_finals(
    final_coords_by_sample,
    residue_labels,
    contact_cutoff=8.0,
    contact_min_seq_sep=4,
    mode="consensus",
    reference_sample=None,
    min_fraction=0.5,
):
    """
    Define reference contacts using model_299 final structures instead of native.

    mode="consensus" (default):
        A pair is used as a reference contact if it is present in at least
        min_fraction of sample final models. The reference distance is the median
        final-model distance across all samples.

    mode="first_sample":
        Use the smallest sample ID final model as the reference.

    mode="reference_sample":
        Use --reference_sample final model as the reference.
    """
    mode = str(mode).lower()
    sample_ids = sorted(final_coords_by_sample.keys())
    if len(sample_ids) == 0:
        raise ValueError("No final model references were loaded")

    L = next(iter(final_coords_by_sample.values())).shape[0]
    valid_pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            if abs(i - j) > int(contact_min_seq_sep):
                valid_pairs.append((i, j))
    valid_pairs = np.asarray(valid_pairs, dtype=int)
    if valid_pairs.size == 0:
        raise ValueError("No residue pairs after applying --contact_min_seq_sep")

    if mode in ["first", "first_sample"]:
        ref_sid = sample_ids[0]
        dist = pairwise_dist(final_coords_by_sample[ref_sid])
        pair_dist = dist[valid_pairs[:, 0], valid_pairs[:, 1]]
        keep = pair_dist <= float(contact_cutoff)
        source = f"first_sample:sample{ref_sid}"
        presence_fraction = np.where(keep, 1.0, 0.0)
        reference_dist = pair_dist
    elif mode in ["reference", "reference_sample"]:
        if reference_sample is None:
            ref_sid = sample_ids[0]
        else:
            ref_sid = int(reference_sample)
        if ref_sid not in final_coords_by_sample:
            raise ValueError(f"--reference_sample {ref_sid} was not found")
        dist = pairwise_dist(final_coords_by_sample[ref_sid])
        pair_dist = dist[valid_pairs[:, 0], valid_pairs[:, 1]]
        keep = pair_dist <= float(contact_cutoff)
        source = f"reference_sample:sample{ref_sid}"
        presence_fraction = np.where(keep, 1.0, 0.0)
        reference_dist = pair_dist
    elif mode == "consensus":
        all_pair_dist = []
        for sid in sample_ids:
            dist = pairwise_dist(final_coords_by_sample[sid])
            all_pair_dist.append(dist[valid_pairs[:, 0], valid_pairs[:, 1]])
        all_pair_dist = np.stack(all_pair_dist, axis=0)  # [Nsample, Npair]
        present = all_pair_dist <= float(contact_cutoff)
        presence_fraction = present.mean(axis=0)
        reference_dist = np.nanmedian(all_pair_dist, axis=0)
        keep = presence_fraction >= float(min_fraction)
        source = f"consensus_model_{len(sample_ids)}_final_structures_min_fraction_{min_fraction}"
    else:
        raise ValueError(
            "Unknown --contact_reference_mode. Use consensus, first_sample, or reference_sample."
        )

    contact_pairs = valid_pairs[keep]
    contact_ref_dist = reference_dist[keep]
    contact_presence_fraction = presence_fraction[keep]

    if len(contact_pairs) == 0:
        raise ValueError(
            "No reference contacts were found from model_299. Try increasing --contact_cutoff, "
            "decreasing --contact_min_seq_sep, or decreasing --contact_reference_min_fraction."
        )

    labels = [f"{residue_labels[i]}-{residue_labels[j]}" for i, j in contact_pairs]
    return contact_pairs, labels, contact_ref_dist, contact_presence_fraction, source


def compute_contact_distance_curve(coords_stack, contact_pairs):
    """
    Compute contact distances for a trajectory.

    coords_stack: [T, L, 3]
    contact_pairs: [C, 2]
    Returns:
        distances: [T, C]
    """
    idx_i = contact_pairs[:, 0]
    idx_j = contact_pairs[:, 1]
    diff = coords_stack[:, idx_i, :] - coords_stack[:, idx_j, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def compute_contact_formation_times(
    contact_distances,
    contact_ref_dist,
    model_indices,
    contact_scale=1.2,
    contact_offset=1.0,
    min_consecutive=5,
):
    """
    Contact is formed if:
        d_ij(t) <= contact_scale * d_ij(reference final model) + contact_offset
    for min_consecutive frames.
    """
    thresholds = contact_scale * contact_ref_dist + contact_offset
    contact_margin = thresholds[None, :] - contact_distances
    return first_persistent_time(
        values=contact_margin,
        model_indices=model_indices,
        threshold=0.0,
        min_consecutive=min_consecutive,
        greater_equal=True,
    )


# ----------------------------- distance calculation -----------------------------

def spearman_distance(a, b):
    """
    Spearman rank distance in [0, 1]: (1 - rho) / 2.
    Robust to constant vectors and NaNs.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 3:
        if mask.sum() > 0 and np.allclose(a[mask], b[mask], equal_nan=True):
            return 0.0
        return 0.5

    x = a[mask]
    y = b[mask]

    if np.nanstd(x) < 1e-12 and np.nanstd(y) < 1e-12:
        return 0.0 if np.allclose(x, y, equal_nan=True) else 0.5
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return 0.5

    rx = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)

    rho = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(rho):
        return 0.5

    d = (1.0 - rho) / 2.0
    return float(np.clip(d, 0.0, 1.0))


def resolve_n_jobs(n_jobs):
    """Resolve requested number of CPU processes."""
    if n_jobs is None or int(n_jobs) <= 0:
        return max(1, cpu_count() - 1)
    return max(1, int(n_jobs))


def _pair_distance_worker(task):
    """Worker for one pair of samples in the path-distance matrix."""
    a, b, sid_a, sid_b, residue_a, residue_b, contact_a, contact_b, w_res, w_contact = task
    d_res = spearman_distance(residue_a, residue_b)
    d_contact = spearman_distance(contact_a, contact_b)
    d_total = w_res * d_res + w_contact * d_contact
    row = {
        "sample_i": sid_a,
        "sample_j": sid_b,
        "D_total": d_total,
        "D_residue_ca_distance_folding_time": d_res,
        "D_contact_formation_time": d_contact,
        "w_residue": w_res,
        "w_contact": w_contact,
    }
    return a, b, d_total, row


def compute_path_distance_matrix(sample_ids, residue_times, contact_times, w_res=0.6, w_contact=0.4, n_jobs=1):
    """
    Pairwise pathway distance:
        D = w_res * D_residue + w_contact * D_contact
    where each component is Spearman rank distance of formation/folding times.
    """
    n = len(sample_ids)
    D = np.zeros((n, n), dtype=np.float64)

    w_sum = float(w_res + w_contact)
    if w_sum <= 0:
        raise ValueError("w_res + w_contact must be > 0")
    w_res = w_res / w_sum
    w_contact = w_contact / w_sum

    pair_tasks = []
    for a in range(n):
        for b in range(a + 1, n):
            sid_a = sample_ids[a]
            sid_b = sample_ids[b]
            pair_tasks.append((
                a,
                b,
                sid_a,
                sid_b,
                residue_times[sid_a],
                residue_times[sid_b],
                contact_times[sid_a],
                contact_times[sid_b],
                w_res,
                w_contact,
            ))

    n_jobs = resolve_n_jobs(n_jobs)
    if len(pair_tasks) == 0:
        return D, pd.DataFrame([])

    if n_jobs == 1:
        results = [_pair_distance_worker(t) for t in pair_tasks]
    else:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_pair_distance_worker, pair_tasks)

    rows = []
    for a, b, d_total, row in results:
        D[a, b] = D[b, a] = d_total
        rows.append(row)

    return D, pd.DataFrame(rows)


# ----------------------------- clustering -----------------------------

def reindex_labels(labels):
    """Reindex non-negative cluster labels to 0,1,2...; keep -1 as noise."""
    labels = np.asarray(labels, dtype=int)
    new_labels = np.full_like(labels, -1)
    unique = [x for x in sorted(set(labels.tolist())) if x != -1]
    mapping = {old: new for new, old in enumerate(unique)}
    for i, x in enumerate(labels):
        if x == -1:
            new_labels[i] = -1
        else:
            new_labels[i] = mapping[x]
    return new_labels


def auto_distance_threshold_from_tree(D, default_cutoff=0.35, min_gap=0.05):
    """Choose a distance threshold automatically from hierarchical merge heights."""
    n = D.shape[0]
    if n <= 1:
        return 0.0

    upper = D[np.triu_indices(n, k=1)]
    if len(upper) == 0 or np.nanmax(upper) < 1e-12:
        return 1e-8

    if n == 2:
        d = float(upper[0])
        return default_cutoff if d <= default_cutoff else d * 0.5

    Z = linkage(squareform(D, checks=False), method="average")
    heights = Z[:, 2]
    if len(heights) < 2:
        return default_cutoff

    gaps = np.diff(heights)
    j = int(np.argmax(gaps))
    if float(gaps[j]) >= min_gap:
        threshold = float((heights[j] + heights[j + 1]) / 2.0)
    else:
        threshold = float(default_cutoff)
    return threshold


def cluster_paths(
    D,
    method="agglomerative_auto",
    distance_threshold=None,
    default_cutoff=0.35,
    min_gap=0.05,
    min_cluster_size=3,
    min_samples=2,
    n_clusters=None,
):
    """
    Cluster pathways.

    If n_clusters is provided, the script uses hierarchical clustering and cuts
    the tree into a fixed number of clusters. This keeps the downstream probability-based cluster sorting and centroid selection unchanged.

    If n_clusters is not provided, the original behavior is kept:
        agglomerative_auto: hierarchical clustering with distance threshold
        hdbscan: use hdbscan if installed, otherwise fall back to agglomerative_auto
        dbscan: use sklearn DBSCAN with eps/distance threshold
    """
    n = D.shape[0]
    if n == 0:
        return np.array([], dtype=int), {"method_used": method, "threshold": np.nan, "n_clusters_used": 0}
    if n == 1:
        return np.array([0], dtype=int), {"method_used": method, "threshold": 0.0, "n_clusters_used": 1}

    if n_clusters is not None:
        k = int(n_clusters)
        if k <= 0:
            raise ValueError("--n_clusters must be a positive integer when provided")
        k_used = min(k, n)
        Z = linkage(squareform(D, checks=False), method="average")
        labels = cut_tree(Z, n_clusters=[k_used]).reshape(-1)
        labels = reindex_labels(labels)
        return labels, {
            "method_used": "agglomerative_fixed_n_clusters",
            "threshold": np.nan,
            "default_cutoff": default_cutoff,
            "min_gap": min_gap,
            "n_clusters_requested": k,
            "n_clusters_used": int(len(set(labels.tolist()) - {-1})),
        }

    method = method.lower()

    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore
            mcs = max(2, min(int(min_cluster_size), n))
            ms = max(1, min(int(min_samples), n - 1))
            clusterer = hdbscan.HDBSCAN(
                metric="precomputed",
                min_cluster_size=mcs,
                min_samples=ms,
            )
            labels = clusterer.fit_predict(D)
            if len(set(labels.tolist()) - {-1}) > 0:
                return reindex_labels(labels), {
                    "method_used": "hdbscan",
                    "threshold": np.nan,
                    "min_cluster_size": mcs,
                    "min_samples": ms,
                }
            print("WARNING: HDBSCAN returned only noise; falling back to agglomerative_auto")
        except Exception as e:
            print(f"WARNING: HDBSCAN unavailable/failed ({repr(e)}); falling back to agglomerative_auto")
        method = "agglomerative_auto"

    if method == "dbscan":
        try:
            from sklearn.cluster import DBSCAN
            if distance_threshold is None:
                eps = auto_distance_threshold_from_tree(D, default_cutoff=default_cutoff, min_gap=min_gap)
            else:
                eps = float(distance_threshold)
            ms = max(1, min(int(min_samples), n))
            labels = DBSCAN(eps=eps, min_samples=ms, metric="precomputed").fit_predict(D)
            if len(set(labels.tolist()) - {-1}) > 0:
                return reindex_labels(labels), {
                    "method_used": "dbscan",
                    "threshold": eps,
                    "min_samples": ms,
                }
            print("WARNING: DBSCAN returned only noise; falling back to agglomerative_auto")
        except Exception as e:
            print(f"WARNING: DBSCAN unavailable/failed ({repr(e)}); falling back to agglomerative_auto")
        method = "agglomerative_auto"

    if method != "agglomerative_auto":
        raise ValueError(f"Unknown cluster method: {method}")

    if distance_threshold is None:
        threshold = auto_distance_threshold_from_tree(D, default_cutoff=default_cutoff, min_gap=min_gap)
    else:
        threshold = float(distance_threshold)

    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=threshold, criterion="distance")
    labels = reindex_labels(labels)
    return labels, {
        "method_used": "agglomerative_auto",
        "threshold": threshold,
        "default_cutoff": default_cutoff,
        "min_gap": min_gap,
    }



def load_pathway_probability_csv(probability_csv, sample_ids=None):
    """
    Load pathway probabilities from pathway_probability.csv.

    Probabilities are read directly from the input CSV:
        - No rank-power reweighting is performed here.
        - The probability column is used directly for cluster sorting and centroid selection.

    Expected columns:
        path_index: numeric sample id; path_index=0 corresponds to sample0.
        probability: pathway probability produced by the taxonomy probability script.

    Rows with empty path_index or empty/non-finite probability are ignored.

    Returns:
        path_probability_by_sample: dict[str, float], e.g. {"sample0": 0.18}
        used_df: DataFrame with sample and probability columns for matched paths.
    """
    if probability_csv is None or str(probability_csv).strip() == "":
        return None, pd.DataFrame([])

    if not os.path.exists(probability_csv):
        raise FileNotFoundError(f"--pathway_probability_csv not found: {probability_csv}")

    df = pd.read_csv(probability_csv)
    if "path_index" not in df.columns:
        raise ValueError(f"{probability_csv} must contain a 'path_index' column")
    if "probability" not in df.columns:
        raise ValueError(f"{probability_csv} must contain a 'probability' column")

    sample_id_set = None
    if sample_ids is not None:
        sample_id_set = {int(x) for x in sample_ids}

    records = []
    probability_by_sample = defaultdict(float)

    for _, row in df.iterrows():
        if pd.isna(row.get("path_index")):
            continue
        if pd.isna(row.get("probability")):
            continue

        try:
            path_index = int(float(row["path_index"]))
            probability = float(row["probability"])
        except Exception:
            continue

        if not np.isfinite(probability):
            continue

        if sample_id_set is not None and path_index not in sample_id_set:
            continue

        sample_name = f"sample{path_index}"
        probability_by_sample[sample_name] += probability

        rec = row.to_dict()
        rec["path_index"] = path_index
        rec["sample"] = sample_name
        rec["probability"] = probability
        records.append(rec)

    used_df = pd.DataFrame(records)

    if len(used_df) == 0:
        print(f"WARNING: no valid pathway probability rows were loaded from {probability_csv}")
        return {}, pd.DataFrame([])

    # If a path_index appears more than once, sum probabilities.
    grouped = (
        used_df.groupby(["path_index", "sample"], as_index=False)
        .agg(path_probability=("probability", "sum"))
        .sort_values(["path_probability", "path_index"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped["probability_rank"] = np.arange(1, len(grouped) + 1, dtype=int)

    path_probability_by_sample = {
        str(row["sample"]): float(row["path_probability"])
        for _, row in grouped.iterrows()
    }

    return path_probability_by_sample, grouped


def reorder_labels_by_cluster_probability(labels, sample_ids, path_probability_by_sample=None):
    """
    Reindex non-noise cluster labels so cluster 0 has the largest total probability.
    If no pathway probabilities are provided, sort by cluster size instead.
    """
    labels = np.asarray(labels, dtype=int)
    sample_ids = list(sample_ids)
    if len(labels) == 0:
        return labels

    cluster_probabilities = []
    for lab in sorted(set(labels.tolist())):
        if lab == -1:
            continue
        idx = np.where(labels == lab)[0]
        members = [sample_ids[i] for i in idx]
        if path_probability_by_sample is None:
            probability = float(len(idx))
        else:
            probability = float(sum(path_probability_by_sample.get(m, 0.0) for m in members))
        cluster_probabilities.append((lab, probability, len(idx)))

    # Larger probability first; if tied, larger cluster size first; if still tied, old label order.
    cluster_probabilities = sorted(cluster_probabilities, key=lambda x: (-x[1], -x[2], x[0]))
    mapping = {old_lab: new_lab for new_lab, (old_lab, _, _) in enumerate(cluster_probabilities)}

    new_labels = np.full_like(labels, -1)
    for i, lab in enumerate(labels):
        if lab == -1:
            new_labels[i] = -1
        else:
            new_labels[i] = mapping[int(lab)]
    return new_labels


def summarize_clusters(sample_ids, labels, D, path_probability_by_sample=None):
    """
    Build pathway_clusters.csv and cluster_summary.csv.

    If path_probability_by_sample is provided:
        - cluster probability is the sum of all member path probabilities.
        - centroid_sample is the member path with the largest probability.
          If several members have the same probability, the one with the smallest
          within-cluster distance sum is selected.

    If path_probability_by_sample is not provided:
        - cluster probability is cluster occupancy by path count.
        - centroid_sample is the member path with the smallest within-cluster
          distance sum.
    """
    sample_ids = list(sample_ids)
    labels = np.asarray(labels, dtype=int)
    n = len(sample_ids)
    use_path_probability = path_probability_by_sample is not None

    cluster_records = []
    summary_records = []

    if use_path_probability:
        total_matched_probability = float(sum(path_probability_by_sample.get(s, 0.0) for s in sample_ids))
    else:
        total_matched_probability = np.nan

    for lab in sorted(set(labels.tolist())):
        idx = np.where(labels == lab)[0]
        members = [sample_ids[i] for i in idx]
        n_path_fraction = len(idx) / float(n) if n > 0 else np.nan

        if len(idx) == 1:
            sum_dist = np.array([0.0], dtype=np.float64)
            mean_within = 0.0
        else:
            subD = D[np.ix_(idx, idx)]
            sum_dist = subD.sum(axis=1)
            upper = subD[np.triu_indices(len(idx), k=1)]
            mean_within = float(np.mean(upper)) if len(upper) > 0 else 0.0

        member_probabilities = np.array(
            [path_probability_by_sample.get(m, 0.0) for m in members],
            dtype=np.float64,
        ) if use_path_probability else np.full(len(idx), np.nan, dtype=np.float64)

        if use_path_probability:
            cluster_probability = float(np.nansum(member_probabilities))
            max_probability = float(np.nanmax(member_probabilities)) if len(member_probabilities) > 0 else np.nan

            candidate_local = np.where(np.isclose(member_probabilities, max_probability, rtol=1e-12, atol=1e-15))[0]
            if len(candidate_local) == 1:
                centroid_local = int(candidate_local[0])
            else:
                # Tie-break: among equal-probability paths, use the one most central by distance.
                centroid_local = int(candidate_local[np.argmin(sum_dist[candidate_local])])

            centroid_idx = idx[centroid_local]
            centroid_sum_dist = float(sum_dist[centroid_local])
            selection_rule = "max_path_probability"
            cluster_probability_renormalized_valid = (
                cluster_probability / total_matched_probability
                if np.isfinite(total_matched_probability) and total_matched_probability > 0
                else np.nan
            )
        else:
            cluster_probability = n_path_fraction
            centroid_local = int(np.argmin(sum_dist)) if len(idx) > 0 else 0
            centroid_idx = idx[centroid_local]
            centroid_sum_dist = float(sum_dist[centroid_local]) if len(idx) > 0 else np.nan
            max_probability = np.nan
            selection_rule = "min_sum_within_cluster_distance"
            cluster_probability_renormalized_valid = np.nan

        centroid_sample = sample_ids[centroid_idx]
        is_noise = (lab == -1)

        summary_records.append({
            "cluster": lab,
            "is_noise": is_noise,
            "n_paths": len(idx),
            "n_path_fraction": n_path_fraction,
            "cluster_probability_sum": cluster_probability,
            "cluster_probability_renormalized_valid": cluster_probability_renormalized_valid,
            "total_matched_path_probability": total_matched_probability,
            "centroid_sample": centroid_sample,
            "centroid_path_probability": max_probability if use_path_probability else np.nan,
            "centroid_selection_rule": selection_rule,
            "mean_within_distance": mean_within,
            "centroid_sum_distance": centroid_sum_dist,
            "members": ";".join(map(str, members)),
        })

        for local_pos, i in enumerate(idx):
            sample = sample_ids[i]
            path_probability = float(member_probabilities[local_pos]) if use_path_probability else np.nan
            cluster_records.append({
                "sample": sample,
                "cluster": lab,
                "is_noise": is_noise,
                "path_probability": path_probability,
                "cluster_probability_sum": cluster_probability,
                "cluster_probability_renormalized_valid": cluster_probability_renormalized_valid,
                "n_path_fraction": n_path_fraction,
                "is_centroid": sample == centroid_sample,
                "centroid_selection_rule": selection_rule,
            })

    cluster_df = pd.DataFrame(cluster_records).sort_values(
        ["is_noise", "cluster", "path_probability", "sample"],
        ascending=[True, True, False, True],
    )
    summary_df = pd.DataFrame(summary_records).sort_values(
        ["is_noise", "cluster_probability_sum", "n_paths"],
        ascending=[True, False, False],
    )
    return cluster_df, summary_df



# ----------------------------- parallel sample worker -----------------------------

def process_one_sample_worker(task):
    """Worker for one trajectory/sample."""
    (
        sid,
        items,
        expected_len,
        residue_labels,
        contact_pairs,
        contact_ref_dist,
        final_model_index,
        ca_distance_cutoff,
        min_consecutive,
        contact_scale,
        contact_offset,
        save_ca_distance_curves,
        out_dir,
    ) = task

    model_indices, coords_stack, frame_paths = load_trajectory_coords(items, expected_len=expected_len)

    final_coords, final_model_used = select_final_coords(
        model_indices=model_indices,
        coords_stack=coords_stack,
        sample_id=sid,
        final_model_index=final_model_index,
    )

    ca_dist = compute_aligned_ca_distance_curve(coords_stack, final_coords)
    r_times = first_persistent_time(
        values=ca_dist,
        model_indices=model_indices,
        threshold=ca_distance_cutoff,
        min_consecutive=min_consecutive,
        greater_equal=False,
    )

    contact_dist = compute_contact_distance_curve(coords_stack, contact_pairs)
    c_times = compute_contact_formation_times(
        contact_distances=contact_dist,
        contact_ref_dist=contact_ref_dist,
        model_indices=model_indices,
        contact_scale=contact_scale,
        contact_offset=contact_offset,
        min_consecutive=min_consecutive,
    )

    summary = {
        "sample": sid,
        "n_frames": len(model_indices),
        "first_model": int(np.min(model_indices)),
        "last_model": int(np.max(model_indices)),
        "final_model_used_for_residue_reference": final_model_used,
        "n_residues": expected_len,
        "n_contacts": len(contact_ref_dist),
        "ca_distance_cutoff": ca_distance_cutoff,
        "mean_residue_folding_time": float(np.nanmean(r_times)),
        "median_residue_folding_time": float(np.nanmedian(r_times)),
        "mean_contact_formation_time": float(np.nanmean(c_times)),
        "median_contact_formation_time": float(np.nanmedian(c_times)),
        "mean_final_frame_ca_distance_to_self_after_alignment": float(np.nanmean(ca_dist[model_indices == final_model_used]))
        if np.any(model_indices == final_model_used) else np.nan,
    }

    if save_ca_distance_curves:
        ca_df = pd.DataFrame(ca_dist, index=model_indices, columns=residue_labels)
        ca_df.index.name = "model"
        ca_df.to_csv(
            os.path.join(out_dir, f"sample{sid}_aligned_ca_distance_to_model{final_model_used}.csv"),
            float_format="%.6f",
        )

    return sid, r_times, c_times, summary


# ----------------------------- main workflow -----------------------------


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)

    if args.native:
        print(f"WARNING: --native was provided but will be ignored in this no-native version: {args.native}")

    trajectories = collect_trajectories(args.frames_dir, pattern=args.pattern)
    sample_ids = sorted(trajectories.keys())

    path_probability_by_sample, pathway_probability_used_df = load_pathway_probability_csv(
        args.pathway_probability_csv,
        sample_ids=sample_ids,
    )
    if path_probability_by_sample is not None:
        matched_sum = sum(path_probability_by_sample.get(f"sample{sid}", 0.0) for sid in sample_ids)
        missing_samples = [f"sample{sid}" for sid in sample_ids if f"sample{sid}" not in path_probability_by_sample]
        print(f"Loaded pathway probabilities from: {args.pathway_probability_csv}")
        print("The probability column is used directly for cluster sorting and centroid selection.")
        print(f"Matched pathway probability sum over loaded samples: {matched_sum:.10f}")
        if missing_samples:
            print("WARNING: no probability found for these samples; their path_probability will be 0: " + ",".join(missing_samples))

    final_coords_by_sample, final_info_by_sample, final_summary_df, label_sample = load_final_references(
        trajectories=trajectories,
        final_model_index=args.final_model_index,
        reference_sample=args.reference_sample,
    )
    reference_info = final_info_by_sample[label_sample]
    L = final_coords_by_sample[label_sample].shape[0]
    residue_labels = [str(resseq) for chain, resseq, resname in reference_info]

    print(f"Frames dir: {args.frames_dir}")
    print(f"Number of trajectories/samples: {len(sample_ids)}")
    print(f"Samples: {sample_ids}")
    print(f"Reference final model index: model_{args.final_model_index}_sampleX if available")
    print(f"Residue labels taken from: sample{label_sample}")
    print(f"CA residues: {L}")
    print(f"Residue folding criterion: aligned CA distance <= {args.ca_distance_cutoff} Å")

    contact_pairs, contact_labels, contact_ref_dist, contact_presence_fraction, contact_source = define_reference_contacts_from_finals(
        final_coords_by_sample=final_coords_by_sample,
        residue_labels=residue_labels,
        contact_cutoff=args.contact_cutoff,
        contact_min_seq_sep=args.contact_min_seq_sep,
        mode=args.contact_reference_mode,
        reference_sample=args.reference_sample,
        min_fraction=args.contact_reference_min_fraction,
    )
    print(f"Reference contacts from model_{args.final_model_index}: {len(contact_labels)}")
    print(f"Contact reference source: {contact_source}")
    if not args.save_contact_outputs:
        print("Contact-related intermediate CSV files are not saved. Use --save_contact_outputs to save them.")

    if args.save_contact_outputs:
        contact_df = pd.DataFrame({
            "contact_label": contact_labels,
            "res_i_index0": contact_pairs[:, 0],
            "res_j_index0": contact_pairs[:, 1],
            "res_i_label": [residue_labels[i] for i in contact_pairs[:, 0]],
            "res_j_label": [residue_labels[j] for j in contact_pairs[:, 1]],
            "reference_ca_distance": contact_ref_dist,
            "presence_fraction_in_final_models": contact_presence_fraction,
            "contact_reference_source": contact_source,
        })
        contact_df.to_csv(os.path.join(args.out_dir, "reference_contacts.csv"), index=False, float_format="%.6f")
        # Compatibility with older downstream scripts that expect native_contacts.csv.
        contact_df.to_csv(os.path.join(args.out_dir, "native_contacts.csv"), index=False, float_format="%.6f")

    residue_times = {}
    contact_times = {}
    traj_summary = []

    n_jobs = resolve_n_jobs(args.n_jobs)
    print(f"CPU processes: {n_jobs}")

    sample_tasks = []
    for sid in sample_ids:
        items = trajectories[sid]
        print(f"Queued sample{sid}: {len(items)} frames")
        sample_tasks.append((
            sid,
            items,
            L,
            residue_labels,
            contact_pairs,
            contact_ref_dist,
            args.final_model_index,
            args.ca_distance_cutoff,
            args.min_consecutive,
            args.contact_scale,
            args.contact_offset,
            args.save_ca_distance_curves,
            args.out_dir,
        ))

    if n_jobs == 1:
        sample_results = [process_one_sample_worker(t) for t in sample_tasks]
    else:
        with Pool(processes=n_jobs) as pool:
            sample_results = pool.map(process_one_sample_worker, sample_tasks)

    sample_results = sorted(sample_results, key=lambda x: x[0])
    for sid, r_times, c_times, summary in sample_results:
        residue_times[sid] = r_times
        contact_times[sid] = c_times
        traj_summary.append(summary)

    residue_time_df = pd.DataFrame(
        [residue_times[sid] for sid in sample_ids],
        index=[f"sample{sid}" for sid in sample_ids],
        columns=residue_labels,
    )
    residue_time_df.index.name = "sample"
    residue_time_df.to_csv(os.path.join(args.out_dir, "residue_folding_time.csv"), float_format="%.6f")

    if args.save_contact_outputs:
        contact_time_df = pd.DataFrame(
            [contact_times[sid] for sid in sample_ids],
            index=[f"sample{sid}" for sid in sample_ids],
            columns=contact_labels,
        )
        contact_time_df.index.name = "sample"
        contact_time_df.to_csv(os.path.join(args.out_dir, "contact_formation_time.csv"), float_format="%.6f")

    D, component_df = compute_path_distance_matrix(
        sample_ids=sample_ids,
        residue_times=residue_times,
        contact_times=contact_times,
        w_res=args.w_residue,
        w_contact=args.w_contact,
        n_jobs=n_jobs,
    )

    D_df = pd.DataFrame(
        D,
        index=[f"sample{sid}" for sid in sample_ids],
        columns=[f"sample{sid}" for sid in sample_ids],
    )
    D_df.index.name = "sample"
    D_df.to_csv(os.path.join(args.out_dir, "path_distance_matrix.csv"), float_format="%.6f")
    component_df.to_csv(
        os.path.join(args.out_dir, "path_distance_components.csv"),
        index=False,
        float_format="%.6f",
    )

    labels, cluster_info = cluster_paths(
        D=D,
        method=args.cluster_method,
        distance_threshold=args.distance_threshold,
        default_cutoff=args.default_cutoff,
        min_gap=args.min_gap,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        n_clusters=args.n_clusters,
    )

    sample_names = [f"sample{sid}" for sid in sample_ids]
    labels = reorder_labels_by_cluster_probability(
        labels=labels,
        sample_ids=sample_names,
        path_probability_by_sample=path_probability_by_sample,
    )

    cluster_df, summary_df = summarize_clusters(
        sample_ids=sample_names,
        labels=labels,
        D=D,
        path_probability_by_sample=path_probability_by_sample,
    )

    # Keep clustering method metadata out of cluster_summary.csv.
    # It can remain in pathway_clusters.csv for debugging if needed.
    for k, v in cluster_info.items():
        cluster_df[k] = v

    cluster_df.to_csv(os.path.join(args.out_dir, "pathway_clusters.csv"), index=False, float_format="%.6f")
    summary_df.to_csv(os.path.join(args.out_dir, "cluster_summary.csv"), index=False, float_format="%.6f")

    top3_df = summary_df[summary_df["is_noise"] == False].copy().head(3)

    print("\nDone.")
    print(f"Cluster method info: {cluster_info}")
    print(f"Saved outputs to: {args.out_dir}")
    print("Top clusters:")
    if len(top3_df) > 0:
        print(top3_df[["cluster", "n_paths", "cluster_probability_sum", "centroid_sample", "members"]].to_string(index=False))
    else:
        print("No non-noise clusters found.")



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cluster multiple PathDiffusion folding trajectories using residue CA-distance folding time "
            "and model_299-based contact formation time. By default the number of clusters is not predefined; "
            "use --n_clusters to optionally force a fixed number of clusters. "
            "Use --n_jobs for multi-CPU acceleration. No native PDB is required. "
            "Pathway ranking/centroid selection can be guided directly by pathway_probability.csv."
        )
    )

    p.add_argument("--frames_dir", required=True, help="Directory containing model_<step>_sample<id>.pdb files")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--pattern", default="model_*_sample*.pdb", help="Input filename glob pattern")
    p.add_argument("--native", default=None, help="Ignored. Kept only for backward compatibility with older commands.")
    p.add_argument(
        "--pathway_probability_csv",
        default=None,
        help=(
            "Optional pathway_probability.csv. The script reads path_index and probability directly. "
            "path_index is matched to sample id, e.g. path_index=0 -> sample0. "
            "After clustering, cluster probability is the sum of member path probabilities, and "
            "the cluster centroid/medoid is the member path with the largest probability."
        ),
    )

    # final model references
    p.add_argument(
        "--final_model_index",
        type=int,
        default=299,
        help="Use model_<this>_sampleX as the final structure for each sample. Default: 299",
    )
    p.add_argument(
        "--reference_sample",
        type=int,
        default=None,
        help=(
            "Sample ID used for residue labels and, if --contact_reference_mode reference_sample, "
            "for defining reference contacts. Default: smallest sample ID."
        ),
    )

    # residue CA-distance folding time
    p.add_argument(
        "--ca_distance_cutoff",
        type=float,
        default=8.0,
        help="Residue is folded when post-alignment CA distance to its own final model is <= this cutoff in Å",
    )
    p.add_argument(
        "--save_ca_distance_curves",
        action="store_true",
        help="Save per-frame per-residue aligned CA-distance matrices for each sample",
    )

    # contact formation time
    p.add_argument("--contact_cutoff", type=float, default=8.0, help="CA-CA cutoff for defining contacts from final model(s)")
    p.add_argument("--contact_min_seq_sep", type=int, default=4, help="Use contacts with |i-j| > this value")
    p.add_argument("--contact_scale", type=float, default=1.2, help="Contact formed if d <= scale*d_reference + offset")
    p.add_argument("--contact_offset", type=float, default=1.0, help="Contact formed if d <= scale*d_reference + offset")
    p.add_argument(
        "--contact_reference_mode",
        choices=["consensus", "first_sample", "reference_sample"],
        default="consensus",
        help=(
            "How to define reference contacts without native. consensus: contacts present in a fraction of final models; "
            "first_sample: use smallest sample ID final model; reference_sample: use --reference_sample final model."
        ),
    )
    p.add_argument(
        "--contact_reference_min_fraction",
        type=float,
        default=0.5,
        help="For consensus mode, keep contacts present in at least this fraction of model_<final_model_index> structures.",
    )
    p.add_argument(
        "--save_contact_outputs",
        action="store_true",
        help=(
            "Save contact-related intermediate CSV files: reference_contacts.csv, native_contacts.csv, "
            "and contact_formation_time.csv. By default these files are not saved."
        ),
    )

    p.add_argument("--min_consecutive", type=int, default=5, help="Require this many consecutive frames to call folded/contact formed")

    # distance weights
    p.add_argument("--w_residue", type=float, default=0.6, help="Weight of residue folding-time distance")
    p.add_argument("--w_contact", type=float, default=0.4, help="Weight of contact formation-time distance")

    # clustering
    p.add_argument(
        "--cluster_method",
        choices=["agglomerative_auto", "hdbscan", "dbscan"],
        default="agglomerative_auto",
        help=(
            "Clustering method used when --n_clusters is not set. "
            "If --n_clusters is provided, agglomerative fixed-cluster cutting is used."
        ),
    )
    p.add_argument(
        "--distance_threshold",
        type=float,
        default=0.35,
        help="Optional distance threshold. If omitted, it is chosen automatically.",
    )
    p.add_argument(
        "--default_cutoff",
        type=float,
        default=0.35,
        help="Fallback distance cutoff for agglomerative_auto/dbscan auto threshold",
    )
    p.add_argument(
        "--min_gap",
        type=float,
        default=0.05,
        help="Minimum meaningful largest gap in hierarchical merge distances",
    )
    p.add_argument("--min_cluster_size", type=int, default=3, help="Used only by HDBSCAN")
    p.add_argument("--min_samples", type=int, default=2, help="Used by HDBSCAN/DBSCAN")
    p.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help=(
            "Optional fixed number of clusters. For example, --n_clusters 3 forces "
            "hierarchical clustering to cut the tree into 3 clusters. If omitted, "
            "the original distance-threshold/auto clustering behavior is used."
        ),
    )

    p.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of CPU processes for sample feature calculation and pairwise distances; <=0 uses cpu_count()-1",
    )

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
