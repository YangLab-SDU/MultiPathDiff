#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Run MMseqs2 easy-search for sequence searching with optional CPU or GPU mode")
    parser.add_argument('--query_dir', required=True, help='Directory containing query FASTA files')
    parser.add_argument('--target_db', required=True, help='Path to the target MMseqs2 database')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--tmp_dir_base', required=True, help='Base directory for temporary files')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of concurrent tasks; usually set to 1 for a single query')
    parser.add_argument('--threads_per_task', type=int, default=10, help='Number of CPU threads used per task')
    parser.add_argument('--evalue', type=float, default=1.0, help='E-value threshold')
    parser.add_argument('--sensitivity', type=float, default=5.7, help='MMseqs2 sensitivity parameter')
    parser.add_argument('--max_seqs', type=int, default=10000, help='Maximum number of returned sequences')

    parser.add_argument(
        '--gpu', type=int, choices=[0, 1], default=0,
        help='Whether to enable MMseqs2 GPU search: 1=GPU, 0=CPU. GPU mode adds --gpu 1 to easy-search.'
    )
    parser.add_argument(
        '--gpu_auto_fallback', action='store_true',
        help='If --gpu 1 fails, automatically retry once using CPU. Disabled by default.'
    )

    return parser.parse_args()


def build_mmseqs_cmd(args, query_path, output_path, tmp_dir, use_gpu):
    cmd = [
        'mmseqs', 'easy-search',
        query_path,
        args.target_db,
        output_path,
        tmp_dir,
        '-e', str(args.evalue),
        '-s', str(args.sensitivity),
        '--threads', str(args.threads_per_task),
        '--max-seqs', str(args.max_seqs),
    ]

    if use_gpu:
        cmd.extend(['--gpu', '1'])

    return cmd


def run_one_cmd(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir_base, exist_ok=True)

    fasta_files = sorted(
        f for f in os.listdir(args.query_dir)
        if f.endswith('.fasta') or f.endswith('.fa') or f.endswith('.faa')
    )

    print(args)

    if len(fasta_files) == 0:
        raise RuntimeError(f'No .fasta/.fa/.faa files found in query_dir: {args.query_dir}')

    def run_mmseqs(fasta_file):
        query_path = os.path.join(args.query_dir, fasta_file)

        output_path = os.path.join(args.output_dir, 'msa.m8')
        tmp_dir = os.path.join(args.tmp_dir_base, fasta_file.rsplit('.', 1)[0])
        os.makedirs(tmp_dir, exist_ok=True)

        use_gpu = (args.gpu == 1)
        cmd = build_mmseqs_cmd(args, query_path, output_path, tmp_dir, use_gpu=use_gpu)

        try:
            run_one_cmd(cmd)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            mode = 'GPU' if use_gpu else 'CPU'
            return True, f'{fasta_file} succeeded ({mode}) -> {output_path}'
        except subprocess.CalledProcessError as e:
            if use_gpu and args.gpu_auto_fallback:
                fallback_tmp_dir = tmp_dir + '_cpu_fallback'
                os.makedirs(fallback_tmp_dir, exist_ok=True)
                fallback_cmd = build_mmseqs_cmd(args, query_path, output_path, fallback_tmp_dir, use_gpu=False)
                try:
                    run_one_cmd(fallback_cmd)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    shutil.rmtree(fallback_tmp_dir, ignore_errors=True)
                    return True, f'{fasta_file} GPU failed, automatically retried with CPU successfully -> {output_path}'
                except subprocess.CalledProcessError as e2:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    shutil.rmtree(fallback_tmp_dir, ignore_errors=True)
                    return False, f'{fasta_file} failed in both GPU mode and CPU fallback:\n[GPU stderr]\n{e.stderr}\n[CPU stderr]\n{e2.stderr}'

            shutil.rmtree(tmp_dir, ignore_errors=True)
            mode = 'GPU' if use_gpu else 'CPU'
            return False, f'{fasta_file} failed ({mode}):\n{e.stderr}'

    print(f'Starting processing of {len(fasta_files)} FASTA file(s)')
    failed = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(run_mmseqs, fasta): fasta for fasta in fasta_files}
        for future in as_completed(futures):
            ok, message = future.result()
            print(message)
            if not ok:
                failed.append(message)

    if failed:
        raise RuntimeError('One or more MMseqs2 search tasks failed. Please check the stderr messages above.')


if __name__ == '__main__':
    main()
