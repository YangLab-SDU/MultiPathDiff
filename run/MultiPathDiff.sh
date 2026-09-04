#!/bin/bash
set -e

# =============================================================================
# Configuration & Path Setup
# =============================================================================

# 1. Get the absolute path of the directory where this script is located.
# This script is assumed to be placed in: project_root/run/
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 2. Define the project root.
# Example:
#   project_root/
#   ├── pretrain_repr/
#   ├── run/
#   │   ├── MultiPathDiff.sh
#   │   └── model/
#   ├── settings/
#   ├── src/
#   └── train_model/
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Internal project paths are determined automatically.
BaseDir="$SCRIPT_DIR"
SampleDir="$PROJECT_ROOT"

# 3. Input arguments
if [ "$#" -lt 2 ]; then
    echo "Usage: bash $0 <Output_Directory> <Protein_Name> [Dataset_Dir] [Db_Root] [HHblits_DB] [Model_Dir]"
    echo "Example: bash $0 ./results 1abc /path/to/AFDB /path/to/classified_fasta_createdb /path/to/UniRef"
    exit 1
fi

OUTPUT_ROOT=$1
ProteinName=$2
TestDir="$OUTPUT_ROOT/$ProteinName"

# 4. External data paths
# Users should modify these default paths according to their local environment,
# or provide them directly as the 3rd-6th command-line arguments.
DatasetDir=${3:-"/path/to/AFDB"}
DbRoot=${4:-"/path/to/classified_fasta_createdb"}
HHblitsDB=${5:-"/path/to/UniRef"}

# Model files are included under run/model by default.
ToolDir=${6:-"$SCRIPT_DIR/model"}

# 5. Runtime settings
Threads=10

# Print configuration
echo "=========================================================="
echo "Running MultiPathDiff Pipeline"
echo "Project Root : $PROJECT_ROOT"
echo "Script Dir   : $SCRIPT_DIR"
echo "Output Dir   : $TestDir"
echo "Dataset Dir  : $DatasetDir"
echo "DB Root      : $DbRoot"
echo "HHblits DB   : $HHblitsDB"
echo "Model Dir    : $ToolDir"
echo "Protein Name : $ProteinName"
echo "=========================================================="

source ~/.bashrc
conda activate confdiff

#========================== Step 1: Multi-path MSA Construction ==========================
echo "--------------------------------------------------------"
echo ">>> [Step 1] Building multi-path MSA..."

python "$BaseDir/build_MPD_msa.py" \
    --test_dir "$TestDir" \
    --base_dir "$BaseDir" \
    --db_root "$DbRoot" \
    --hhblits_db "$HHblitsDB" \
    --query_fasta "$TestDir/seq.fasta" \
    --tools "mmseqs, jackhmmer, hhblits" \
    --threads "$Threads" \
    --min_hits 10 \
    --mmseqs_evalue 1 \
    --mmseqs_sensitivity 5.7 \
    --mmseqs_max_seqs 50000 \
    --mmseqs_gpu 1 \
    --jackhmmer_iterations 3 \
    --jackhmmer_evalue 1 \
    --hhblits_iterations 3 \
    --hhblits_evalue 1e-3 \
    --hhblits_conda_env hhsuite


if [ ! -f "$TestDir/msa/msa_build_manifest.tsv" ]; then
    echo "Error: MSA manifest not found: $TestDir/msa/msa_build_manifest.tsv"
    exit 1
fi


#========================== Step 2: Multi-path MSTA Generation ==========================
conda activate FPdiffusion
echo "--------------------------------------------------------"
echo ">>> [Step 2] Starting multi-path MSTA processing..."

MstaFinalFile="$TestDir/msta/${ProteinName}.msta"

if [ ! -f "$MstaFinalFile" ]; then
    mkdir -p "$TestDir/msta"

    TempMstaList=()
    TriedPathCount=0

    while IFS=$'\t' read -r Index DbName DbPath UseMode UseFile UseCount Used ToolsUsed; do
        if [ "$Used" != "1" ]; then
            echo "   -> Index $Index ($DbName): No valid MSA found, skipping."
            continue
        fi

        if [ "$UseFile" = "NA" ] || [ ! -s "$UseFile" ]; then
            echo "   -> Index $Index ($DbName): Invalid MSA file in manifest, skipping: $UseFile"
            continue
        fi

        echo "   -> Index $Index ($DbName): Using $UseMode file, number of entries: $UseCount"
        echo "      $UseFile"

        CurRmsdDir="$TestDir/msta/RMSD_${Index}"
        CurTempMsta="$TestDir/msta/temp_msa_${Index}.msta"

        mkdir -p "$CurRmsdDir"

        if [ -z "$(ls -A "$CurRmsdDir" 2>/dev/null)" ]; then
            python "$BaseDir/make_MPD_msta_pdbalign.py" \
                --mode "$UseMode" \
                --input_file "$UseFile" \
                --threads_per_task "$Threads" \
                --afdb_pdb_dir "$DatasetDir" \
                --tmalign_path "$ToolDir/TMalign_rmsd" \
                --output_dir "$CurRmsdDir"
        else
            echo "      RMSD_${Index} already exists, skipping structural alignment."
        fi

        # 2.2 Run MSTA scoring
        python "$BaseDir/make_MPD_msta_Fscore_Quantile.py" \
            --mode "$UseMode" \
            --input_file "$UseFile" \
            --RMSD_path "$CurRmsdDir" \
            --seq_path "$TestDir/seq.fasta" \
            --msta_outpath "$CurTempMsta" \
            --afdb_pdb_dir "$DatasetDir" \
            --nwalign_path "$ToolDir/NWalign"

        if [ -f "$CurTempMsta" ]; then
            TempMstaList+=("$CurTempMsta")
            TriedPathCount=$((TriedPathCount + 1))
        else
            echo "   -> Error: Failed to generate MSTA for Index $Index."
        fi

    done < <(tail -n +2 "$TestDir/msa/msa_build_manifest.tsv")

    # 2.3 Merge all temporary MSTA files
    if [ "$TriedPathCount" -gt 0 ]; then
        echo ">>> Merging MSTA files..."

        python "$BaseDir/make_MPD_msta_Fscore_Quantile.py" \
            --merge_mode \
            --msta_outpath "$MstaFinalFile" \
            --merge_files "${TempMstaList[@]}"
    else
        echo "Error: No MSTA files were generated!"
        exit 1
    fi
else
    echo ">>> MSTA already exists, skipping generation: $MstaFinalFile"
fi

if [ -f "$MstaFinalFile" ]; then
    FinalColumns=$(head -n 1 "$MstaFinalFile" | awk '{print NF-1}')
    echo ">>> Final number of valid paths (Samples): $FinalColumns"

    if [ "$FinalColumns" -eq 0 ]; then
        echo "Error: Number of valid columns after merging is 0!"
        exit 1
    fi
else
    echo "Error: Failed to calculate the number of columns after final merging!"
    exit 1
fi

#========================== Step 3: General Data Preparation (CSV) ==========================
echo "--------------------------------------------------------"
echo ">>> [Step 3] Generating CSV data..."

CsvFile="$TestDir/test_data.csv"

if [ ! -f "$CsvFile" ]; then
    python "$BaseDir/make_data_csv.py" \
        --fasta_file "$TestDir/seq.fasta" \
        --chain_name "$ProteinName" \
        --train_val_test test \
        --csv_file "$CsvFile" \
        --overwrite

    if [ ! -f "$CsvFile" ]; then
        echo "Error: CSV file generation failed!"
        exit 1
    fi
else
    echo "test_data.csv already exists, skipping."
fi

#========================== Step 4: Generate ESM_repr ==========================
echo "--------------------------------------------------------"
echo ">>> [Step 4] Generating ESM Representation..."

EsmReprCsv="$TestDir/seqres_and_index.csv"

if [ ! -f "$EsmReprCsv" ]; then
    cd "$SampleDir"

    mkdir -p "$TestDir/ESM_repr"

    NumGpu=1

    for GpuId in $(seq 0 $(($NumGpu-1))); do
        CUDA_VISIBLE_DEVICES=$GpuId python3 -m pretrain_repr.esmfold.ESM_repr \
            --input-csv-path "$CsvFile" \
            --output-dir "$TestDir/ESM_repr" \
            --esm-ckpt-fpath "$ToolDir/esmfold_3B_v1.pt" \
            --num-recycles 3 \
            --batch-size 1 \
            --num-workers "$NumGpu" \
            --worker-id "$GpuId" &
    done

    wait

    cat "$TestDir"/ESM_repr/seqres_and_index.worker*.csv >> "$TestDir/ESM_repr/seqres_and_index.csv"
    mv "$TestDir/ESM_repr/seqres_and_index.csv" "$EsmReprCsv"
    rm "$TestDir"/ESM_repr/seqres_and_index.worker*.csv

    if [ ! -f "$EsmReprCsv" ]; then
        echo "Error: ESM_repr generation failed!"
        exit 1
    fi

    echo "ESM_repr generation successful!"
else
    echo "ESM Representation already exists, skipping."
fi

#========================== Step 5: Folding Pathway Sampling (Multi-path) ==========================
echo "--------------------------------------------------------"
echo ">>> [Step 5] Sampling folding pathways (Samples=$FinalColumns)..."

#========================== Stage 1 ==========================
Stage1Dir="$TestDir/stage1"

if [ ! -d "$Stage1Dir" ] || [ -z "$(ls -A "$Stage1Dir" 2>/dev/null)" ]; then
    cd "$SampleDir"

    python3 eval.py \
        sampling=cfg_inference \
        data.repr_loader.data_root="$TestDir" \
        paths.guidance.cond_ckpt="$ToolDir/cond_model2.ckpt" \
        paths.guidance.uncond_ckpt="$ToolDir/uncond_model.ckpt" \
        paths.output_dir="$TestDir" \
        data.dataset.test_gen_dataset.csv_path="$CsvFile" \
        data.dataset.test_gen_dataset.num_samples="$FinalColumns" \
        model.score_network.cfg.clsfree_guidance_strength=1.0 \
        model.score_network.msta_dir="$TestDir/msta" \
        model.stage=1

    if [ ! -d "$Stage1Dir" ] || [ -z "$(ls -A "$Stage1Dir" 2>/dev/null)" ]; then
        echo "Error: Stage1 sampling failed!"
        exit 1
    fi

    echo "Stage1 sampling successful!"
else
    echo "Stage1 already exists, skipping."
fi

#========================== Stage 2 ==========================
Stage2Dir="$TestDir/stage2"

if [ ! -d "$Stage2Dir" ] || [ -z "$(ls -A "$Stage2Dir" 2>/dev/null)" ]; then
    cd "$SampleDir"

    python3 eval.py \
        sampling=force_guidance_inference \
        data.repr_loader.data_root="$TestDir" \
        paths.guidance.cond_ckpt="$ToolDir/finetune_conditional_model.ckpt" \
        paths.guidance.uncond_ckpt="$ToolDir/finetune_unconditional_model.ckpt" \
        paths.output_dir="$TestDir" \
        data.dataset.test_gen_dataset.csv_path="$CsvFile" \
        data.dataset.test_gen_dataset.num_samples="$FinalColumns" \
        model.score_network.cfg.clsfree_guidance_strength=1.0 \
        model.score_network.cfg.starting_steps=70 \
        model.score_network.cfg.final_steps=73 \
        model.score_network.msta_dir="$TestDir/msta" \
        model.stage=2 \
        model.score_network.cfg.force_guidance_strength=1.0

    if [ ! -d "$Stage2Dir" ] || [ -z "$(ls -A "$Stage2Dir" 2>/dev/null)" ]; then
        echo "Error: Stage2 sampling failed!"
        exit 1
    fi

    echo "Folding pathway sampling successful!"
else
    echo "Stage2 already exists, skipping."
fi

#========================== Step 6: Generate Pathway Movies ==========================
echo "--------------------------------------------------------"
echo ">>> [Step 6] Generating pathway movies..."

bash "$BaseDir/make_model_movie.sh" "$TestDir" "$BaseDir" "$FinalColumns"

echo ">>> All steps completed successfully!"