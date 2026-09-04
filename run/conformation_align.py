import numpy as np
import os
import sys

# 参数检查
if len(sys.argv) < 4:
    print("Usage: python conformation_align.py <target_path> <step_index> <sample_index>")
    sys.exit(1)

path_tran = sys.argv[1]
pdb_index = sys.argv[2]
sample_index = sys.argv[3]

TMscore = 0

input_file_path = os.path.join(path_tran, 'tran_files/tmscore_tran')
input_file_path2 = os.path.join(path_tran, 'tran_files/tmscore_qianhou')
output_file_path = os.path.join(path_tran, 'tran_files/tmscore_rot_metrc')

try:
    if not os.path.exists(input_file_path):
        raise Exception(f"File {input_file_path} does not exist")

    with open(input_file_path, "r") as f:
        lines = f.readlines()
        # 读取 TMscore 输出的旋转矩阵 (通常在最后几行)
        matrix_lines = lines[-11:-8]

    with open(input_file_path2, "r") as f:
        lines = f.readlines()
        if len(lines) >= 18:
            TMscore = lines[-18:-17]

    numbers = []
    for line in matrix_lines:
        line = line.strip()
        if line:
            numbers += line.split()

    # 按照 TMscore 格式重组旋转矩阵和平移向量
    with open(output_file_path, "w") as f:
        f.write(f"{numbers[2]}  {numbers[7]}  {numbers[12]}  ")
        f.write(f"{numbers[3]}  {numbers[8]}  {numbers[13]}  ")
        f.write(f"{numbers[4]}  {numbers[9]}  {numbers[14]}  ")
        f.write(f"{numbers[1]}  {numbers[6]}  {numbers[11]}  ")

except Exception as e:
    print(f"An error occurred while processing alignment matrix: {e}")
    sys.exit(1)

rot_to_pdb = np.zeros((3, 3))
trans = np.zeros(3)

# 读取处理后的矩阵
if os.path.exists(output_file_path):
    with open(output_file_path, 'r') as f:
        for line in f.readlines():
            line_data = np.array(line.strip().split('  '))

    # 解析旋转和平移
    if len(line_data) >= 12:
        for i in range(3):
            for j in range(3):
                rot_to_pdb[i][j] = float(line_data[3 * i + j])
            trans[i] = float(line_data[9 + i])
    else:
        print("Error: Parsed matrix data is incomplete.")
        sys.exit(1)

# 定义输入和输出 PDB 路径 (包含 sample_index)
input_pdb = os.path.join(path_tran, f'stage2/Step_{pdb_index}_sample{sample_index}.pdb')
output_pdb = os.path.join(path_tran, f'tran_out_pdb/model_{pdb_index}_sample{sample_index}.pdb')

if not os.path.exists(input_pdb):
    print(f"Error: Input PDB file not found: {input_pdb}")
    sys.exit(1)

with open(input_pdb, 'r') as f, open(output_pdb, 'w') as outfile:
    for line in f.readlines():
        if not line.startswith("ATOM  "):
            continue

        str1 = line[0:30]
        str2 = line[54:80]

        try:
            atom_x = float(line[30:38])
            atom_y = float(line[38:46])
            atom_z = float(line[46:54])
        except ValueError:
            continue

        coordinate = np.array([atom_x, atom_y, atom_z])
        # 应用变换: R * (coord - T)  或者 (coord - T) * R^T 视 TMscore 定义而定
        # 原代码逻辑: coordinate_new = rot_to_pdb.dot(coordinate- trans)
        coordinate_new = rot_to_pdb.dot(coordinate - trans)

        # 格式化坐标，保持 PDB 格式对齐 (8.3f 是标准，这里沿用原代码逻辑保留两位小数但控制宽度)
        str_x = "{:8.3f}".format(coordinate_new[0])
        str_y = "{:8.3f}".format(coordinate_new[1])
        str_z = "{:8.3f}".format(coordinate_new[2])

        new_line = str1 + str_x + str_y + str_z + str2 + '\n'
        outfile.write(new_line)
    outfile.write('TER\n')