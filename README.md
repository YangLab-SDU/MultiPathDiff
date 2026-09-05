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

### Species-group Databases

### MSA and MSTA Generation


## Training

We use Hydra for configuration management. All configuration files are located in the `settings/` directory.



\*\*1. Conditional Training (Fold-based)\*\*:

```bash

python train.py \\

&nbsp;   --config-name cond\_model \\

&nbsp;   task\_name=cond\_train \\

&nbsp;   data.train\_batch\_size=1 \\

&nbsp;   paths.output\_dir="./train\_model"

```

The detailed training configuration can be found in `settings/cond\_model.yaml`.



\*\*2. Unconditional Training (Disorder-based)\*\*:

```bash

python train.py \\

&nbsp;   --config-name uncond\_model \\

&nbsp;   task\_name=uncond\_train \\

&nbsp;   paths.output\_dir="./train\_model"

```

The detailed training configuration can be found in `settings/uncond\_model.yaml`.



## Inference (Sampling)

Use eval.py to generate protein structures. The inference pipeline typically uses Classifier-Free Guidance (CFG) combining both conditional and unconditional checkpoints.

\### Basic Command

```bash

python eval.py \\

&nbsp;   sampling=cfg\_inference \\

&nbsp;   model.stage=1 \\

&nbsp;   paths.output\_dir="./output/inference\_result" \\

&nbsp;   paths.guidance.cond\_ckpt="/path/to/cond\_model.ckpt" \\

&nbsp;   paths.guidance.uncond\_ckpt="/path/to/uncond\_model.ckpt" \\

&nbsp;   model.score\_network.msta\_dir="/path/to/msta\_dir" \\

&nbsp;   data.dataset.test\_gen\_dataset.csv\_path="/path/to/test\_data.csv"

```

### Pipeline Automation

For a complete pipeline (MSA and MSTA Generation -> ESM\_repr Generation -> Folding Pathway Sampling -> Pathway Movie Generation), you can use the scripts provided in the `run/` directory.

Example:

```bash

bash run/FPdiffusion.sh ./example 1AB7\_A

```



