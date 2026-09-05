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

### Force-supervised fine-tuning dataset

MultiPathDiff is built upon the pretrained score models of PathDiffusion. 
The dataset used for force-supervised fine-tuning was constructed from the
PDB-derived training dataset used in PathDiffusion.

Protein chains were first clustered at 70% sequence identity, and one
representative structure was retained from each cluster. Proteins with sequence
lengths between 20 and 400 residues were used for subsequent trajectory
generation.

For each protein, the pretrained PathDiffusion model was used to generate a
300-step folding trajectory from an unfolded-like conformation toward a folded
state. Sixteen conformations were randomly sampled from each trajectory, with
eight conformations selected from the early stage and eight from the late stage.

The early-stage conformations, which mainly represent unfolded or weakly
structured states, were used to fine-tune the unconditional score model.
The late-stage conformations, which contain more sequence-dependent and
native-like structural features, were used to fine-tune the
sequence-conditional score model.

Conformations for which trajectory generation or force calculation failed were
removed. After filtering, the dataset contained 40,132 representative proteins
and 642,112 sampled conformations. Proteins were divided into training and
validation sets at a ratio of 9:1. For each score model, this resulted in
288,936 training conformations and 32,120 validation conformations.

### OpenMM force annotation

Following the force-guided diffusion strategy used in ConfDiff, molecular
mechanics forces calculated with OpenMM were used as physical supervision for
fine-tuning. MultiPathDiff uses force labels only and does not use energy labels
for model training.

Force labels for the sampled conformations can be generated using:

```bash
python3 src/utils/protein/openmm_energy.py \
    --input-root /path/to/your/generated_samples \
    --output-root /path/to/your/output_dir


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



