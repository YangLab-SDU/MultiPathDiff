#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Error: Missing arguments."
    echo "Usage: $0 <target_path> <base_dir> <num_samples>"
    exit 1
fi

target_path="$1"
base_dir="$2"
num_samples="$3"

if [ ! -d "$base_dir" ]; then
    echo "Error: BaseDir does not exist: $base_dir"
    exit 1
fi

output_pdb_dir="$target_path/stage2"
tran_files_dir="$target_path/tran_files"
tran_out_pdb_dir="$target_path/tran_out_pdb"
tmscore_path="$base_dir/model/TMscore"

if [ ! -f "$tmscore_path" ]; then
    echo "Error: TMscore tool not found: $tmscore_path"
    exit 1
fi

mkdir -p "$tran_files_dir" "$tran_out_pdb_dir" || { echo "Failed to create directories"; exit 1; }

cleanup() {
    # rm -rf "$tran_files_dir"
    echo "Done."
}
trap cleanup EXIT

echo "Processing $num_samples pathways (samples) based on input argument."

for (( s=0; s<num_samples; s++ )); do
    echo "========================================"
    echo "Processing Pathway Sample Index: $s"
    
    pdb_files=($(ls "$output_pdb_dir"/Step_*_sample${s}.pdb 2>/dev/null | sort -V))
    file_count=${#pdb_files[@]}
    
    if [ "$file_count" -eq 0 ]; then
        echo "Warning: No files found for sample $s (looking for Step_*_sample${s}.pdb), skipping."
        continue
    fi

    first_file="${pdb_files[0]}"
    first_index=$(basename "$first_file" | sed -n "s/^Step_\([0-9]*\)_sample${s}\.pdb$/\1/p")
    
    final_state="${pdb_files[-1]}"
    
    echo "  Total steps found: $file_count"
    echo "  Start frame: $(basename "$first_file") (Index: $first_index)"
    echo "  Final frame: $(basename "$final_state")"

    cp "$first_file" "$tran_out_pdb_dir/model_${first_index}_sample${s}.pdb" || { echo "Failed to copy initial file"; continue; }

    for ((i=0; i < file_count-1; i++)); do
        file1="${pdb_files[$i]}"
        file2="${pdb_files[$i+1]}"
        
        index2=$(basename "$file2" | sed -n "s/^Step_\([0-9]*\)_sample${s}\.pdb$/\1/p")

        if [ ! -f "$file1" ] || [ ! -f "$file2" ]; then
            echo "  Warning: Missing file sequence for sample $s step $i, skipping alignment."
            continue
        fi
        
        "$tmscore_path" "$final_state" "$file2" > "$tran_files_dir/tmscore_tran"
        "$tmscore_path" "$file1" "$file2" > "$tran_files_dir/tmscore_qianhou"

        python "$base_dir/conformation_align.py" "$target_path" "$index2" "$s" || { echo "conformation_align.py failed for sample $s step $index2"; continue; }
    done

    processed_pdbs=($(ls "$tran_out_pdb_dir"/model_*_sample${s}.pdb 2>/dev/null | sort -V))
    
    movie_file="$target_path/model_movie_sample${s}.pdb"
    > "$movie_file"

    echo "  Generating movie file: $(basename "$movie_file")"

    for pdb_file in "${processed_pdbs[@]}"; do
        idx=$(basename "$pdb_file" | sed -n "s/^model_\([0-9]*\)_sample${s}\.pdb$/\1/p")
        
        python "$base_dir/conformation_ensemble.py" "$tran_out_pdb_dir" "$idx" "$target_path" "$s" || { echo "conformation_ensemble.py failed"; continue; }
    done
done

echo "All pathways processed successfully."