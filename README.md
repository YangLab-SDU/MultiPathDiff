# MultiPathDiff

## Installation

MultiPathDiff uses the same software environment as [PathDiffusion](https://github.com/YangLab-SDU/PathDiffusion). 
The provided `env.yml` therefore retains the Conda environment name `pathdiffusion`. 
If the PathDiffusion environment is already installed, it can be used directly.

```bash
# clone project
git clone https://github.com/YangLab-SDU/MultiPathDiff.git
cd MultiPathDiff

# create conda environment
conda env create -f env.yml
conda activate pathdiffusion

# install OpenFold if it is not already installed
git clone https://github.com/aqlaboratory/openfold.git
python -m pip install --no-build-isolation --no-use-pep517 -e openfold
```


## Datasets Preparation

### 1. Force-supervised fine-tuning dataset

The fine-tuning dataset was constructed from the PDB-derived training dataset used in PathDiffusion. Protein chains were clustered at 70% sequence identity, and representative proteins with sequence lengths between 20 and 400 residues were retained.

For each protein, PathDiffusion was used to generate a 300-step folding trajectory. Sixteen conformations were randomly sampled from each trajectory, including eight from the early stage and eight from the late stage. The early-stage conformations were used to fine-tune the unconditional score model, whereas the late-stage conformations were used to fine-tune the sequence-conditional score model.

### 2. OpenMM force annotation

Following ConfDiff, molecular mechanics forces calculated with OpenMM were used as physical supervision for force-supervised fine-tuning in MultiPathDiff.

Force labels for the sampled conformations can be generated using:

```bash
python3 src/utils/protein/openmm_energy.py \
    --input-root /path/to/your/generated_samples \
    --output-root /path/to/your/output_dir
```

### 3. Species-group Databases

The ten species-group databases used in MultiPathDiff are:

- Actinomycetota
- Bacillota
- Pseudomonadota
- Pseudomonadati
- FCB group
- Fungi
- Metazoa
- Streptophyta
- Bacteria
- cellular organisms

The preprocessed databases can be downloaded from: http://yanglab.qd.sdu.edu.cn/PathDiffusion/

Each species-group database is provided as a separate `.tar` archive. Download and extract all ten databases before generating the species-specific MSAs and MSTAs.


## Training

The sequence-conditional and unconditional score models are fine-tuned separately using the force-supervised datasets described above. The pretrained PathDiffusion checkpoints are used to initialize the corresponding models.

### 1. Unconditional model

```bash
python3 train.py \
    --config-name force_uncond_tune \
    model.score_network.cond_ckpt_path=/path/to/uncond_model.ckpt \
    data.train_batch_size=4 \
    data.val_batch_size=4
```
The detailed training configuration can be found in `settings/force_uncond_tune.yaml`.

### 2. Sequence-conditional model
```bash
python3 train.py \
    --config-name force_cond_tune \
    model.score_network.cond_ckpt_path=/path/to/cond_model.ckpt \
    data.train_batch_size=4 \
    data.val_batch_size=4
```
The detailed training configuration can be found in `settings/force_cond_tune.yaml`.


## Inference (Sampling)

### Pipeline Automation

For a complete MultiPathDiff sampling pipeline, you can use the script provided in the `run/` directory.

Prepare the target protein sequence as `seq.fasta` in the corresponding input directory, and run:

```bash
bash run/MultiPathDiff.sh ./example 1AB7_A
```

## Contact
For questions or issues, please open an issue on this repository.
