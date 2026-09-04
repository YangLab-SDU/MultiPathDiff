import sys
import os

if len(sys.argv) < 5:
    print("Usage: python conformation_ensemble.py <input_dir> <step_index> <output_dir> <sample_index>")
    sys.exit(1)

path_pdb = sys.argv[1]
index = sys.argv[2]
path_out = sys.argv[3]
sample_index = sys.argv[4]

input_filename = f'model_{int(index)}_sample{sample_index}.pdb'
input_path = os.path.join(path_pdb, input_filename)

output_filename = f'model_movie_sample{sample_index}.pdb'
output_path = os.path.join(path_out, output_filename)

if not os.path.exists(input_path):
    print(f"Warning: Input file {input_path} not found, skipping frame.")
    sys.exit(0)

with open(input_path, 'r') as f, open(output_path, "a+") as outfile:
    outfile.write(f'MODEL        {int(index)}\n')
    for line in f:
        if line.startswith("ATOM"):
            outfile.write(line)
        elif line.startswith("END") or line.startswith("TER"):
            outfile.write('ENDMDL\n')
            break