#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build multi-pathway MSA files before MSTA generation.

This script replaces the MSA-building part in MultPD_confdiff_force_new_db*.sh.
It can run MMseqs2, JackHMMER, and HHblits in any combination, then writes a
manifest file for the downstream MSTA step:

    <test_dir>/msa/msa_build_manifest.tsv

Manifest columns:
    index, db_name, db_path, use_mode, use_file, use_count, used, tools_used

Downstream MSTA should use only rows with used == 1.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set


TOOL_ALIASES = {
    "all": "all",
    "mmseqs": "mmseqs",
    "mmseqs2": "mmseqs",
    "jackhmmer": "jackhmmer",
    "hmmer": "jackhmmer",
    "hmm": "jackhmmer",
    "hhblits": "hhblits",
    "hh": "hhblits",
}
ALL_TOOLS = ["mmseqs", "jackhmmer", "hhblits"]


@dataclass
class DbInfo:
    index: int
    path: Path
    name: str

    @property
    def mmseqs_db(self) -> Path:
        return self.path / f"target_{self.name}_padded"

    @property
    def fasta(self) -> Path:
        return self.path / f"{self.name}.fasta"

    @property
    def species_id_file(self) -> Path:
        return self.path / f"target_{self.name}_h"


@dataclass
class ManifestRow:
    index: int
    db_name: str
    db_path: Path
    use_mode: str
    use_file: Optional[Path]
    use_count: int
    used: int
    tools_used: str

    def to_tsv(self) -> str:
        use_file = str(self.use_file) if self.use_file else "NA"
        return "\t".join(
            [
                str(self.index),
                self.db_name,
                str(self.db_path),
                self.use_mode,
                use_file,
                str(self.use_count),
                str(self.used),
                self.tools_used,
            ]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-pathway MSA files with selectable tools: MMseqs2, JackHMMER, HHblits."
    )

    # Core paths
    parser.add_argument("--test_dir", required=True, help="Protein test directory, e.g. .../test_data/1ABC_A")
    parser.add_argument("--base_dir", required=True, help="Run directory containing helper scripts")
    parser.add_argument("--db_root", required=True, help="Directory containing per-taxon/per-class AFDB search DB directories")
    parser.add_argument("--hhblits_db", default="", help="HHblits database path; required if hhblits is selected")
    parser.add_argument("--query_fasta", default="", help="Query FASTA. Default: <test_dir>/seq.fasta")

    # Helper scripts
    parser.add_argument("--make_msa_mmseqs_py", default="", help="Default: <base_dir>/make_msa_s_gpu.py")
    parser.add_argument("--make_msa_hmmer_py", default="", help="Default: <base_dir>/make_msa_hmmer.py")
    parser.add_argument("--split_hh_py", default="", help="Default: <base_dir>/split_hhblits_by_species.py")
    parser.add_argument("--merge_sup_py", default="", help="Default: <base_dir>/merge_sup_from_hh_jackhmmer_fixed.py")
    parser.add_argument("--python_cmd", default=sys.executable, help="Python executable used to call helper Python scripts")

    # Tool selection
    parser.add_argument(
        "--tools",
        nargs="+",
        default=["mmseqs,jackhmmer,hhblits"],
        help=(
            "Tools to use. Supports comma or space separated values, e.g. "
            "--tools mmseqs,jackhmmer or --tools mmseqs hhblits. "
            "Aliases: mmseqs2, hmmer, hh. Use 'all' for all three."
        ),
    )
    parser.add_argument("--min_hits", type=int, default=10, help="If MMseqs hits >= this value, use .m8 directly")
    parser.add_argument("--threads", type=int, default=10, help="CPU threads")
    parser.add_argument("--manifest", default="", help="Default: <test_dir>/msa/msa_build_manifest.tsv")

    # Resume / overwrite
    parser.add_argument("--overwrite_search", action="store_true", help="Rerun existing MMseqs2/JackHMMER/HHblits searches")
    parser.add_argument(
        "--reuse_existing_sup",
        action="store_true",
        help="Reuse existing msa_i.sup instead of regenerating it. Default is to regenerate .sup so selected tools are respected.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing")

    # MMseqs2 parameters
    parser.add_argument("--mmseqs_evalue", type=float, default=1.0)
    parser.add_argument("--mmseqs_sensitivity", type=float, default=5.7)
    parser.add_argument("--mmseqs_max_seqs", type=int, default=50000)
    parser.add_argument("--mmseqs_gpu", type=int, choices=[0, 1], default=1)
    parser.add_argument("--mmseqs_gpu_auto_fallback", action="store_true")

    # JackHMMER parameters
    parser.add_argument("--jackhmmer_iterations", type=int, default=3)
    parser.add_argument("--jackhmmer_evalue", type=float, default=1.0)

    # HHblits parameters
    parser.add_argument("--hhblits_iterations", type=int, default=3)
    parser.add_argument("--hhblits_evalue", default="1e-3")
    parser.add_argument("--hhblits_cmd", default="hhblits", help="HHblits command if no conda env is used")
    parser.add_argument(
        "--hhblits_conda_env",
        default="",
        help="If set, HHblits is called as: conda run -n <env> hhblits. Example: hhsuite",
    )
    parser.add_argument("--split_workers", type=int, default=0, help="Workers for split_hhblits_by_species.py. Default: --threads")

    return parser.parse_args()


def parse_tools(raw_tools: Sequence[str]) -> Set[str]:
    tokens: List[str] = []
    for item in raw_tools:
        tokens.extend([x.strip().lower() for x in item.split(",") if x.strip()])

    if not tokens:
        raise ValueError("No tools selected. Use at least one of: mmseqs, jackhmmer, hhblits.")

    selected: Set[str] = set()
    for token in tokens:
        if token not in TOOL_ALIASES:
            raise ValueError(f"Unknown tool '{token}'. Valid tools: mmseqs, jackhmmer, hhblits, all")
        alias = TOOL_ALIASES[token]
        if alias == "all":
            selected.update(ALL_TOOLS)
        else:
            selected.add(alias)

    return selected


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def mkdirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: Sequence[str], *, dry_run: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(shlex.quote(str(x)) for x in cmd)
    print(f"[CMD] {printable}")
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(cmd, check=True, text=True, cwd=str(cwd) if cwd else None)


def list_db_dirs(db_root: Path) -> List[DbInfo]:
    if not db_root.is_dir():
        raise FileNotFoundError(f"db_root does not exist or is not a directory: {db_root}")
    dirs = sorted([p for p in db_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not dirs:
        raise RuntimeError(f"No database directories found under db_root: {db_root}")
    return [DbInfo(index=i + 1, path=p, name=p.name) for i, p in enumerate(dirs)]


def build_paths(args: argparse.Namespace) -> None:
    base_dir = Path(args.base_dir)
    test_dir = Path(args.test_dir)

    if not args.query_fasta:
        args.query_fasta = str(test_dir / "seq.fasta")
    if not args.make_msa_mmseqs_py:
        args.make_msa_mmseqs_py = str(base_dir / "make_msa_s_gpu.py")
    if not args.make_msa_hmmer_py:
        args.make_msa_hmmer_py = str(base_dir / "make_msa_hmmer.py")
    if not args.split_hh_py:
        args.split_hh_py = str(base_dir / "split_hhblits_by_species.py")
    if not args.merge_sup_py:
        args.merge_sup_py = str(base_dir / "merge_sup_from_hh_jackhmmer_fixed.py")
    if not args.manifest:
        args.manifest = str(test_dir / "msa" / "msa_build_manifest.tsv")
    if args.split_workers <= 0:
        args.split_workers = args.threads


def check_required_inputs(args: argparse.Namespace, tools: Set[str]) -> None:
    query_fasta = Path(args.query_fasta)
    if not query_fasta.is_file():
        raise FileNotFoundError(f"Query FASTA not found: {query_fasta}")

    scripts = []
    if "mmseqs" in tools:
        scripts.append(Path(args.make_msa_mmseqs_py))
    if "jackhmmer" in tools:
        scripts.append(Path(args.make_msa_hmmer_py))
    if "hhblits" in tools:
        scripts.append(Path(args.split_hh_py))
    if "jackhmmer" in tools or "hhblits" in tools or "mmseqs" in tools:
        scripts.append(Path(args.merge_sup_py))

    for script in scripts:
        if not script.is_file():
            raise FileNotFoundError(f"Required helper script not found: {script}")

    if "hhblits" in tools and not args.hhblits_db:
        raise ValueError("--hhblits_db is required when hhblits is selected")


def run_mmseqs(args: argparse.Namespace, db: DbInfo, temp_dir: Path, final_m8: Path) -> int:
    if final_m8.is_file() and not args.overwrite_search:
        n = count_lines(final_m8)
        print(f"   -> msa_{db.index}.m8 already exists, skip MMseqs2, hits: {n}")
        return n

    if final_m8.exists() and args.overwrite_search:
        final_m8.unlink()

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_cmd,
        args.make_msa_mmseqs_py,
        "--query_dir",
        args.test_dir,
        "--target_db",
        str(db.mmseqs_db),
        "--output_dir",
        str(temp_dir),
        "--tmp_dir_base",
        str(temp_dir),
        "--max_workers",
        "1",
        "--threads_per_task",
        str(args.threads),
        "--evalue",
        str(args.mmseqs_evalue),
        "--sensitivity",
        str(args.mmseqs_sensitivity),
        "--max_seqs",
        str(args.mmseqs_max_seqs),
        "--gpu",
        str(args.mmseqs_gpu),
    ]
    if args.mmseqs_gpu_auto_fallback:
        cmd.append("--gpu_auto_fallback")

    print(f"   -> Run MMseqs2 for {db.name} ...")
    try:
        run_cmd(cmd, dry_run=args.dry_run)
    except subprocess.CalledProcessError as e:
        print(f"   -> Warning: MMseqs2 failed for {db.name}: return code {e.returncode}")

    tmp_m8 = temp_dir / "msa.m8"
    if args.dry_run:
        return count_lines(final_m8)

    if tmp_m8.is_file():
        final_m8.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_m8), str(final_m8))
        n = count_lines(final_m8)
        print(f"   -> MMseqs2 done, hits: {n}, output: {final_m8}")
    else:
        final_m8.touch()
        n = 0
        print("   -> Warning: MMseqs2 did not generate msa.m8, created empty .m8")

    shutil.rmtree(temp_dir, ignore_errors=True)
    return n


def run_jackhmmer(args: argparse.Namespace, db: DbInfo, temp_dir: Path, final_hmmer: Path) -> int:
    if final_hmmer.is_file() and not args.overwrite_search:
        n = count_lines(final_hmmer)
        print(f"   -> msa_{db.index}.hmmer already exists, skip JackHMMER, lines: {n}")
        return n

    if final_hmmer.exists() and args.overwrite_search:
        final_hmmer.unlink()

    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_cmd,
        args.make_msa_hmmer_py,
        "--query_file",
        args.query_fasta,
        "--target_db",
        str(db.fasta),
        "--output_dir",
        str(temp_dir),
        "--tmp_dir_base",
        str(temp_dir),
        "--iterations",
        str(args.jackhmmer_iterations),
        "--evalue",
        str(args.jackhmmer_evalue),
        "--cpu",
        str(args.threads),
    ]

    print(f"   -> Run JackHMMER for {db.name} ...")
    try:
        run_cmd(cmd, dry_run=args.dry_run)
    except subprocess.CalledProcessError as e:
        print(f"   -> Warning: JackHMMER failed for {db.name}: return code {e.returncode}")

    if args.dry_run:
        return count_lines(final_hmmer)

    for candidate in [temp_dir / "msa.hmmer", temp_dir / "msa.sto"]:
        if candidate.is_file():
            final_hmmer.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(candidate), str(final_hmmer))
            n = count_lines(final_hmmer)
            print(f"   -> JackHMMER done, lines: {n}, output: {final_hmmer}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return n

    print("   -> Warning: JackHMMER did not generate msa.hmmer/msa.sto")
    shutil.rmtree(temp_dir, ignore_errors=True)
    return 0


def build_hhblits_cmd(args: argparse.Namespace, hhr: Path, a3m: Path) -> List[str]:
    if args.hhblits_conda_env:
        cmd = ["conda", "run", "-n", args.hhblits_conda_env, "hhblits"]
    else:
        cmd = shlex.split(args.hhblits_cmd)

    cmd.extend(
        [
            "-i",
            args.query_fasta,
            "-d",
            args.hhblits_db,
            "-o",
            str(hhr),
            "-oa3m",
            str(a3m),
            "-n",
            str(args.hhblits_iterations),
            "-e",
            str(args.hhblits_evalue),
            "-cpu",
            str(args.threads),
        ]
    )
    return cmd


def run_hhblits_and_split(args: argparse.Namespace, dbs: List[DbInfo], hhblit_dir: Path, hh_species_dir: Path) -> None:
    hhr = hhblit_dir / "query.hhr"
    a3m = hhblit_dir / "query.a3m"

    if (not hhr.is_file() or not a3m.is_file()) or args.overwrite_search:
        if args.overwrite_search:
            for p in [hhr, a3m]:
                if p.exists():
                    p.unlink()
        print("--------------------------------------------------------")
        print(">>> Run HHblits once for the query sequence ...")
        run_cmd(build_hhblits_cmd(args, hhr, a3m), dry_run=args.dry_run)
    else:
        print(">>> HHblits result already exists, skip search")

    if args.dry_run:
        return

    if not hhr.is_file() or not a3m.is_file():
        raise RuntimeError(f"HHblits did not generate expected output: {hhr}, {a3m}")

    all_split_exist = all((hh_species_dir / f"msa_{db.index}.hhr").is_file() for db in dbs)
    if all_split_exist and not args.overwrite_search:
        print(f">>> HHblits species split files already exist: msa_1.hhr ~ msa_{len(dbs)}.hhr")
        return

    print("--------------------------------------------------------")
    print(">>> Split HHblits hits by species/taxon group ...")
    cmd = [
        args.python_cmd,
        args.split_hh_py,
        "--hhblits_hhr",
        str(hhr),
        "--db_root",
        args.db_root,
        "--output_dir",
        str(hh_species_dir),
        "--workers",
        str(args.split_workers),
    ]
    run_cmd(cmd, dry_run=args.dry_run)

    missing = [str(hh_species_dir / f"msa_{db.index}.hhr") for db in dbs if not (hh_species_dir / f"msa_{db.index}.hhr").is_file()]
    if missing:
        raise RuntimeError("HHblits split did not generate all expected files. Missing examples: " + ", ".join(missing[:5]))


def run_merge_sup(
    args: argparse.Namespace,
    db: DbInfo,
    final_m8: Path,
    final_hmmer: Path,
    hh_species_file: Path,
    final_sup: Path,
    tools: Set[str],
) -> int:
    if final_sup.is_file() and args.reuse_existing_sup:
        n = count_lines(final_sup)
        print(f"   -> msa_{db.index}.sup already exists, reuse it, hits: {n}")
        return n

    if final_sup.exists() and not args.reuse_existing_sup:
        final_sup.unlink()

    cmd = [
        args.python_cmd,
        args.merge_sup_py,
        "--hh_species_file",
        str(hh_species_file),
        "--species_id_file",
        str(db.species_id_file),
        "--output_sup",
        str(final_sup),
    ]

    if "jackhmmer" in tools and final_hmmer.is_file():
        cmd.extend(["--jackhmmer_file", str(final_hmmer)])
    if "mmseqs" in tools and final_m8.is_file():
        cmd.extend(["--m8_file", str(final_m8)])

    print(f"   -> Merge selected MSA sources into msa_{db.index}.sup ...")
    try:
        run_cmd(cmd, dry_run=args.dry_run)
    except subprocess.CalledProcessError as e:
        print(f"   -> Warning: merge .sup failed for {db.name}: return code {e.returncode}")

    if args.dry_run:
        return count_lines(final_sup)

    n = count_lines(final_sup)
    print(f"   -> .sup hits: {n}, output: {final_sup}")
    return n


def write_manifest(manifest: Path, rows: Iterable[ManifestRow]) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as f:
        f.write("index\tdb_name\tdb_path\tuse_mode\tuse_file\tuse_count\tused\ttools_used\n")
        for row in rows:
            f.write(row.to_tsv() + "\n")
    print("--------------------------------------------------------")
    print(f">>> MSA manifest written to: {manifest}")


def main() -> None:
    args = parse_args()
    build_paths(args)
    tools = parse_tools(args.tools)
    check_required_inputs(args, tools)

    test_dir = Path(args.test_dir)
    db_root = Path(args.db_root)
    msa_dir = test_dir / "msa"
    hhblit_dir = msa_dir / "hhblit"
    hh_species_dir = msa_dir / "hh_species"
    hmmer_dir = msa_dir / "hmmer"
    mkdirs(msa_dir, hhblit_dir, hh_species_dir, hmmer_dir)

    dbs = list_db_dirs(db_root)
    tool_text = ",".join([t for t in ALL_TOOLS if t in tools])

    print("--------------------------------------------------------")
    print(f">>> Selected MSA tools: {tool_text}")
    print(f">>> Number of database groups: {len(dbs)}")
    print(f">>> Query FASTA: {args.query_fasta}")

    # First pass: MMseqs2 and/or JackHMMER. Decide which DBs need .sup.
    m8_counts: Dict[int, int] = {}
    sup_needed: List[DbInfo] = []

    for db in dbs:
        print("--------------------------------------------------------")
        print(f">>> [Database {db.index}/{len(dbs)}] {db.name}")

        final_m8 = msa_dir / f"msa_{db.index}.m8"
        final_hmmer = hmmer_dir / f"msa_{db.index}.hmmer"
        temp_dir = msa_dir / f"temp_{db.index}"

        if "mmseqs" in tools:
            m8_count = run_mmseqs(args, db, temp_dir, final_m8)
        else:
            m8_count = 0
            print("   -> MMseqs2 not selected; ignore existing .m8 for this run")
        m8_counts[db.index] = m8_count

        use_m8_directly = ("mmseqs" in tools and final_m8.is_file() and m8_count >= args.min_hits)
        needs_sup = not use_m8_directly and ("jackhmmer" in tools or "hhblits" in tools)

        if use_m8_directly:
            print(f"   -> MMseqs2 hits >= {args.min_hits}; downstream will use .m8")
        elif needs_sup:
            sup_needed.append(db)
            print("   -> Need .sup for downstream MSTA")
            if "jackhmmer" in tools:
                run_jackhmmer(args, db, temp_dir, final_hmmer)
        else:
            print("   -> No supplement tool selected and .m8 is insufficient/unavailable; this DB will be skipped")

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    # HHblits is run once, only if at least one DB needs .sup.
    if "hhblits" in tools and sup_needed:
        run_hhblits_and_split(args, dbs, hhblit_dir, hh_species_dir)
    elif "hhblits" in tools:
        print(">>> HHblits selected but no DB needs .sup; skip HHblits")

    # Merge .sup for all DBs that need supplement.
    sup_counts: Dict[int, int] = {}
    if sup_needed:
        print("--------------------------------------------------------")
        print(">>> Generate .sup files for DBs needing supplement ...")
    for db in sup_needed:
        final_m8 = msa_dir / f"msa_{db.index}.m8"
        final_hmmer = hmmer_dir / f"msa_{db.index}.hmmer"
        hh_species_file = hh_species_dir / f"msa_{db.index}.hhr"
        final_sup = msa_dir / f"msa_{db.index}.sup"
        sup_counts[db.index] = run_merge_sup(args, db, final_m8, final_hmmer, hh_species_file, final_sup, tools)

    # Write manifest for the MSTA step.
    rows: List[ManifestRow] = []
    for db in dbs:
        final_m8 = msa_dir / f"msa_{db.index}.m8"
        final_sup = msa_dir / f"msa_{db.index}.sup"
        m8_count = m8_counts.get(db.index, count_lines(final_m8) if "mmseqs" in tools else 0)
        sup_count = sup_counts.get(db.index, count_lines(final_sup))

        if "mmseqs" in tools and final_m8.is_file() and m8_count >= args.min_hits:
            rows.append(
                ManifestRow(db.index, db.name, db.path, "m8", final_m8, m8_count, 1, tool_text)
            )
        elif final_sup.is_file() and sup_count > 0 and ("jackhmmer" in tools or "hhblits" in tools):
            rows.append(
                ManifestRow(db.index, db.name, db.path, "sup", final_sup, sup_count, 1, tool_text)
            )
        else:
            rows.append(
                ManifestRow(db.index, db.name, db.path, "skip", None, 0, 0, tool_text)
            )

    manifest = Path(args.manifest)
    write_manifest(manifest, rows)

    used_count = sum(row.used for row in rows)
    print(f">>> Valid MSA groups for MSTA: {used_count}/{len(rows)}")
    if used_count == 0:
        raise RuntimeError("No valid MSA group was produced. Check selected tools and search outputs.")


if __name__ == "__main__":
    main()
