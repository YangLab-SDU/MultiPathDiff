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

We train FPdiffusion using protein structures from the \[Protein Data Bank](https://www.rcsb.org/) (for conditional generation) and the IDRome database (for unconditional generation). Details on dataset preparation can be found in the datasets folder.

The following datasets and pre-computed representations are required:

1\. RCSB PDB Dataset: See `datasets/rcsb` for details on downloading and processing structured proteins. Once prepared, specify the `csv\_path` and `data\_dir` in the configuration file `settings/cond\_model.yaml`.

2\. MSA and MSTA Generation: After preparing the RCSB dataset, you must generate Multiple Sequence Alignments (MSA) and Multiple Structural Alignments (MSTA) for each protein. Scripts for this process are located in the `run/` directory. Once generated, specify the `msta\_dir` in the configuration file `settings/cond\_model.yaml`.

3\. ESMFold Representations: See `pretrain\_repr` for details on extracting embeddings. Once prepared, specify the data\_root in the configuration file `settings/cond\_model.yaml`.

4\. Disordered Protein Dataset: See `datasets/disorder` for details on preparing the IDP dataset. Once prepared, specify the `csv\_path` and `data\_dir` in the configuration file `settings/uncond\_model.yaml`.



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



