# Pathway probability assignment and clustering

This directory contains the post-processing pipeline used to assign evolutionary
scores to MultiPathDiff trajectories and cluster trajectories with similar
folding patterns. The workflow has three steps:

1. Map a PDB chain to UniProt and its species-group label.
2. Assign a normalized pathway probability according to taxonomic proximity.
3. Cluster folding trajectories using residue folding times and contact
   formation times, then rank the clusters using the pathway scores.

## Files

| File | Description |
| --- | --- |
| `map_pdb_taxonomy.py` | Maps PDB chains to UniProt accessions through PDBe/SIFTS and assigns one or more AFDB species-group labels. Unmatched accessions default to label 4, `cellular_organisms`. |
| `pathway_probability.py` | Filters invalid MSTA files and assigns normalized pathway scores using predefined taxonomic-proximity tiers. |
| `pathways_cluster.py` | Extracts folding-time features, calculates pairwise pathway distances, clusters trajectories, and selects score-guided cluster representatives. |

## Requirements

- Python 3.8 or later
- NumPy
- pandas
- SciPy
- requests
- hdbscan, only when `--cluster_method hdbscan` is used

Install the main Python dependencies with:

```bash
pip install numpy pandas scipy requests
```

## Expected input layout

For each target, the pipeline expects a directory similar to:

```text
test_data/
└── 1AB7_A/
    ├── msta/
    │   ├── temp_msa_1.msta
    │   ├── temp_msa_2.msta
    │   └── ...
    ├── tran_out_pdb/
    │   ├── model_0_sample0.pdb
    │   ├── model_1_sample0.pdb
    │   ├── ...
    │   └── model_299_sample9.pdb
    └── pathway_cluster/
        └── uniprot_mapping.with_label.csv
```

`sampleX` must correspond to the valid MSTA labels in their original numeric
order. The scoring script writes this association as `path_index` in
`pathway_score.csv`, and the clustering script matches `path_index=X` to
`sampleX`.

## Usage

### 1. Map the target to a species-group label

This step requires internet access to the PDBe and UniProt APIs. It can be
skipped when `uniprot_mapping.with_label.csv` has already been generated.

```bash
python map_pdb_taxonomy.py 1AB7_A \
  --label_file /path/to/merged_target_h_with_label.clean.txt \
  --output_csv /path/to/1AB7_A/pathway_cluster/uniprot_mapping.with_label.csv \
  --num_workers 5
```

The label file links AFDB or UniProt accessions to the ten species-group labels
used by MultiPathDiff.

### 2. Assign pathway scores

```bash
python Pathway_probability.py 1AB7_A \
  --label_csv /path/to/1AB7_A/pathway_cluster/uniprot_mapping.with_label.csv \
  --root /path/to/test_data \
  --output_dir /path/to/1AB7_A/pathway_cluster \
  --probability_rank_power 1.0
```

An MSTA is excluded when more than 50% of its residue values are zero. The
remaining pathways are placed into predefined taxonomic-proximity tiers.
Pathways in the same tier receive the same raw weight. With `K` nonempty tiers,
the weight of tier `t` is:

```text
weight(t) = (K - t + 1) ^ probability_rank_power
```

The weights are normalized across all valid pathways. The main output is
`pathway_score.csv`. The script also writes
`fallback_order.filtered.csv`, which records the complete and filtered tier
definitions. `--distance_threshold` is retained only for compatibility and is
not used during score assignment.

### 3. Cluster and rank folding pathways

```bash
python pathways_cluster.py \
  --frames_dir /path/to/1AB7_A/tran_out_pdb \
  --out_dir /path/to/1AB7_A/pathway_cluster \
  --pathway_score_csv /path/to/1AB7_A/pathway_cluster/pathway_score.csv \
  --final_model_index 299 \
  --ca_distance_cutoff 8.0 \
  --min_consecutive 5 \
  --contact_cutoff 8.0 \
  --contact_reference_mode consensus \
  --contact_reference_min_fraction 0.5 \
  --cluster_method agglomerative_auto \
  --distance_threshold 0.35 \
  --w_residue 0.6 \
  --w_contact 0.4 \
  --n_jobs 10
```

No experimental native structure is required. Each trajectory is aligned to
its own final predicted structure. The pathway distance combines:

- per-residue folding times defined by aligned C-alpha distances to the final
  structure;
- contact formation times defined from the final predicted structures.

By default, reference contacts are retained when they occur in at least half of
the final structures. Cluster scores are the sums of their member pathway
scores. The highest-scoring member of each cluster is reported as its
representative.

The principal clustering outputs are:

- `residue_folding_time.csv`
- `path_distance_matrix.csv`
- `path_distance_components.csv`
- `pathway_clusters.csv`
- `cluster_summary.csv`

Add `--save_contact_outputs` to save the contact definitions and contact
formation-time matrix. Add `--save_ca_distance_curves` to save the aligned
C-alpha distance curve for every trajectory.
