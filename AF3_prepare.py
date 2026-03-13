#!/usr/bin/env python3
"""
AF3_prepare.py

AlphaFold3 only FASTA to JSON job generator.

What this script does
- Reads FASTA input from a file or directory
- Writes AlphaFold3 input.json files
- Optionally writes job.sbatch files
- Optionally writes submit_all.sh
- Supports:
  - single protein jobs
  - bait vs prey PPI jobs
  - homomer copies via --copies
  - ligands and ions via CCD codes
- Defaults to exactly 1 model seed

Important design choice
This script emits ONLY AlphaFold3 dialect JSON and follows the style of:
{
  "name": "...",
  "dialect": "alphafold3",
  "version": 1,
  "modelSeeds": [1],
  "sequences": [
    {
      "protein": {
        "id": ["A","B"],
        "sequence": "...."
      }
    },
    {
      "ligand": {
        "id": ["C"],
        "ccdCodes": ["ZN"]
      }
    }
  ]
}

Examples

Single monomer
  ./AF3_prepare.py my.fasta

Single homomer tetramer
  ./AF3_prepare.py my.fasta --copies 4

Single homomer tetramer with 8 zinc and 4 NAD
  ./AF3_prepare.py my.fasta --copies 4 --ion ZN:8 --ligand NAD:4

PPI using one bait against many prey sequences
  ./AF3_prepare.py preys/ --bait bait.fasta --mode ppi

PPI with 2 bait copies and 1 prey copy
  ./AF3_prepare.py preys/ --bait bait.fasta --mode ppi --bait-copies 2 --prey-copies 1

Write sbatch and a submit script
  ./AF3_prepare.py my.fasta --submit-script
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from Bio import SeqIO


FA_EXTS = {".fasta", ".fa", ".faa", ".fna"}


@dataclass
class ProteinRecord:
    id: str
    seq: str


@dataclass
class CCDItem:
    ccd: str
    count: int


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
{constraint_line}#SBATCH --mem={mem}
#SBATCH --ntasks={ntasks}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/%x_%j.out
#SBATCH --error={logs_dir}/%x_%j.err
#SBATCH --mail-user={mail_user}
#SBATCH --mail-type={mail_type}

echo "#### Running on $(hostname)"
echo "#### Loading module"
module load {module_load}

echo "#### Checking GPU"
nvidia-smi

echo "#### Running AlphaFold3"
export AF3_RESOURCES_DIR={resources_dir}
export AF3_INPUT_DIR={input_dir}
export AF3_OUTPUT_DIR={output_dir}
export AF3_MODEL_PARAMETERS_DIR=${{AF3_RESOURCES_DIR}}/weights
export AF3_DATABASES_DIR=${{AF3_RESOURCES_DIR}}/databases

alphafold \\
    --db_dir=${{AF3_DATABASES_DIR}} \\
    --model_dir=${{AF3_MODEL_PARAMETERS_DIR}} \\
    --input_dir=${{AF3_INPUT_DIR}} \\
    --output_dir=${{AF3_OUTPUT_DIR}}

echo "#### Finished"
"""


def sanitize_name(s: str, max_len: int = 180) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return (s or "job")[:max_len]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


def iter_fasta_paths(p: Path) -> Iterable[Path]:
    if p.is_file():
        if p.suffix.lower() not in FA_EXTS:
            raise ValueError(f"Input file does not look like FASTA: {p}")
        yield p
        return

    if p.is_dir():
        found = False
        for fp in sorted(p.iterdir()):
            if fp.is_file() and fp.suffix.lower() in FA_EXTS:
                found = True
                yield fp
        if not found:
            raise ValueError(f"No FASTA files found in directory: {p}")
        return

    raise FileNotFoundError(str(p))


def parse_fasta_file(path: Path) -> List[ProteinRecord]:
    records: List[ProteinRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            rec_id = sanitize_name(rec.id if rec.id else path.stem)
            seq = str(rec.seq).replace(" ", "").replace("\n", "").upper()
            if not seq:
                continue
            records.append(ProteinRecord(id=rec_id, seq=seq))
    return records


def load_proteins(path: Path) -> List[ProteinRecord]:
    out: List[ProteinRecord] = []
    for fp in iter_fasta_paths(path):
        out.extend(parse_fasta_file(fp))
    return out


def chain_id_from_index(i: int) -> str:
    """
    0 -> A
    1 -> B
    ...
    25 -> Z
    26 -> AA
    27 -> AB
    """
    if i < 0:
        raise ValueError("chain index must be >= 0")

    letters = []
    while True:
        i, rem = divmod(i, 26)
        letters.append(chr(ord("A") + rem))
        if i == 0:
            break
        i -= 1
    return "".join(reversed(letters))


def chain_ids_block(start_idx: int, n: int) -> List[str]:
    return [chain_id_from_index(i) for i in range(start_idx, start_idx + n)]


def parse_ccd_list(items: Optional[List[str]]) -> List[CCDItem]:
    out: List[CCDItem] = []
    if not items:
        return out

    for raw in items:
        raw = raw.strip()
        if not raw:
            continue

        if ":" in raw:
            ccd, count = raw.split(":", 1)
            ccd = ccd.strip().upper()
            count = int(count.strip())
        else:
            ccd = raw.upper()
            count = 1

        if not ccd:
            raise ValueError(f"Empty CCD code in: {raw}")
        if count <= 0:
            raise ValueError(f"CCD count must be positive in: {raw}")

        out.append(CCDItem(ccd=ccd, count=count))

    return out


def expand_ccd_items(items: List[CCDItem]) -> List[str]:
    flat: List[str] = []
    for item in items:
        flat.extend([item.ccd] * item.count)
    return flat


def make_ligand_entries(ccd_codes: List[str], start_chain_index: int) -> List[dict]:
    seqs: List[dict] = []
    for i, ccd in enumerate(ccd_codes):
        cid = chain_id_from_index(start_chain_index + i)
        seqs.append(
            {
                "ligand": {
                    "id": [cid],
                    "ccdCodes": [ccd],
                }
            }
        )
    return seqs


def make_af3_job(
    name: str,
    protein_blocks: List[tuple[str, int]],
    ligand_ccd_codes: Optional[List[str]] = None,
    model_seeds: Optional[List[int]] = None,
    version: int = 1,
) -> dict:
    """
    protein_blocks is a list of:
      [
        (sequence, copies),
        (sequence, copies),
      ]

    Each protein block becomes one AlphaFold3 protein entry with id as a list
    of chain ids, exactly like your working template.
    """
    if model_seeds is None:
        model_seeds = [1]
    if not model_seeds:
        raise ValueError("model_seeds cannot be empty")

    sequences: List[dict] = []
    next_chain_idx = 0

    for seq, copies in protein_blocks:
        if copies <= 0:
            raise ValueError("Protein copies must be >= 1")

        ids = chain_ids_block(next_chain_idx, copies)
        next_chain_idx += copies

        sequences.append(
            {
                "protein": {
                    "id": ids,
                    "sequence": seq,
                }
            }
        )

    if ligand_ccd_codes:
        sequences.extend(make_ligand_entries(ligand_ccd_codes, next_chain_idx))

    return {
        "name": name,
        "dialect": "alphafold3",
        "version": int(version),
        "modelSeeds": [int(x) for x in model_seeds],
        "sequences": sequences,
    }


def module_string(profile: str) -> str:
    return "alphafold/cc7_3-20250304" if profile == "cc7" else "alphafold/cc8_3-20250304"


def default_partition(profile: str) -> str:
    if profile == "cc7":
        return "gpu-2080ti-11g"
    return "gpu-mig-40g,gpu-a100-80g"


def render_sbatch(**kw) -> str:
    constraint_line = ""
    if kw.get("constraint"):
        constraint_line = f"#SBATCH --constraint={kw['constraint']}\n"

    return SBATCH_TEMPLATE.format(
        constraint_line=constraint_line,
        **{k: v for k, v in kw.items() if k != "constraint"},
    )


def choose_default_outdir_name(
    mode: str,
    input_path: Path,
    prey_records: List[ProteinRecord],
    bait: Optional[ProteinRecord],
) -> str:
    if mode == "single" and len(prey_records) == 1:
        return sanitize_name(prey_records[0].id)
    if mode == "ppi" and bait is not None:
        return sanitize_name(bait.id)
    if input_path.is_file():
        return sanitize_name(input_path.stem)
    return sanitize_name(input_path.name)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate AlphaFold3 only input.json files from FASTA."
    )

    p.add_argument("input", help="FASTA file or directory for prey or single jobs")
    p.add_argument("--bait", default=None, help="Single FASTA file for bait in PPI mode")
    p.add_argument("--mode", choices=["auto", "single", "ppi"], default="auto")

    p.add_argument(
        "--outdir",
        default=None,
        help="Output root directory. If omitted, derived from input names.",
    )
    p.add_argument(
        "--project",
        default=None,
        help="Optional grouping folder under outdir.",
    )

    p.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Protein copy number for single mode. Default: 1",
    )
    p.add_argument(
        "--bait-copies",
        type=int,
        default=1,
        help="Bait copy number for PPI mode. Default: 1",
    )
    p.add_argument(
        "--prey-copies",
        type=int,
        default=1,
        help="Prey copy number for PPI mode. Default: 1",
    )

    p.add_argument(
        "--ion",
        action="append",
        default=[],
        help="Ion CCD code, optionally with count. Example: --ion ZN or --ion ZN:8",
    )
    p.add_argument(
        "--ligand",
        action="append",
        default=[],
        help="Ligand CCD code, optionally with count. Example: --ligand NAD or --ligand NAD:4",
    )

    p.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="Explicit model seed. Repeat if you want multiple seeds. Default is a single seed [1].",
    )
    p.add_argument(
        "--version",
        type=int,
        default=1,
        help="AlphaFold3 JSON version field. Default: 1",
    )

    p.add_argument(
        "--name-template",
        default="{bait}_with_{prey}",
        help="Name template for PPI jobs",
    )
    p.add_argument("--skip-self", action="store_true")

    p.add_argument("--no-sbatch", action="store_true", help="Do not write job.sbatch")
    p.add_argument("--submit-script", action="store_true", help="Write submit_all.sh")
    p.add_argument("--dry-run", action="store_true")

    p.add_argument("--af3-module-profile", choices=["cc7", "cc8"], default="cc8")
    p.add_argument("--partition", default=None)
    p.add_argument("--constraint", default=None)
    p.add_argument("--resources-dir", default="/data1/databases/AlphaFold3_resources")
    p.add_argument("--mem", default="128GB")
    p.add_argument("--ntasks", type=int, default=16)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--cpus-per-task", type=int, default=4)
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--mail-user", default="")
    p.add_argument("--mail-type", default="BEGIN,END,FAIL")

    args = p.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "ppi" if args.bait else "single"

    model_seeds = args.seed if args.seed else [1]

    input_path = Path(args.input)
    prey_records = load_proteins(input_path)
    if not prey_records:
        raise SystemExit(f"No sequences found in: {args.input}")

    bait: Optional[ProteinRecord] = None
    if mode == "ppi":
        if not args.bait:
            raise SystemExit("PPI mode requires --bait")

        bait_records = load_proteins(Path(args.bait))
        if len(bait_records) != 1:
            raise SystemExit("Bait must resolve to exactly one sequence")
        bait = bait_records[0]

    ion_items = parse_ccd_list(args.ion)
    ligand_items = parse_ccd_list(args.ligand)
    ligand_ccd_codes = expand_ccd_items(ion_items + ligand_items)

    module_load = module_string(args.af3_module_profile)
    partition = args.partition if args.partition else default_partition(args.af3_module_profile)
    write_sbatch = not args.no_sbatch

    if args.outdir:
        out_root = Path(args.outdir)
    else:
        out_root = Path(choose_default_outdir_name(mode, input_path, prey_records, bait))

    if args.project:
        out_root = out_root / sanitize_name(args.project)

    logs_dir = out_root / "logs"
    job_dirs: List[Path] = []

    def emit(job_name: str, payload: dict) -> None:
        job_name_safe = sanitize_name(job_name)
        job_dir = out_root / job_name_safe
        input_json = job_dir / "input.json"

        job_dirs.append(job_dir)

        if args.dry_run:
            print(f"[dry-run] mkdir -p {job_dir}")
            print(f"[dry-run] write {input_json}")
        else:
            job_dir.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            write_json(input_json, payload)

        if write_sbatch:
            sbatch_txt = render_sbatch(
                job_name=job_name_safe,
                partition=partition,
                constraint=args.constraint,
                mem=args.mem,
                ntasks=args.ntasks,
                gpus=args.gpus,
                cpus_per_task=args.cpus_per_task,
                time=args.time,
                mail_user=args.mail_user,
                mail_type=args.mail_type,
                module_load=module_load,
                resources_dir=args.resources_dir,
                input_dir=str(job_dir),
                output_dir=str(job_dir),
                logs_dir=str(logs_dir),
            )
            sbatch_path = job_dir / "job.sbatch"
            if args.dry_run:
                print(f"[dry-run] write {sbatch_path}")
            else:
                write_text(sbatch_path, sbatch_txt)

    if mode == "single":
        for prey in prey_records:
            payload = make_af3_job(
                name=prey.id,
                protein_blocks=[(prey.seq, args.copies)],
                ligand_ccd_codes=ligand_ccd_codes,
                model_seeds=model_seeds,
                version=args.version,
            )
            emit(prey.id, payload)

    else:
        assert bait is not None
        for prey in prey_records:
            if args.skip_self and prey.id == bait.id:
                continue

            job_name = args.name_template.format(bait=bait.id, prey=prey.id)
            payload = make_af3_job(
                name=job_name,
                protein_blocks=[
                    (bait.seq, args.bait_copies),
                    (prey.seq, args.prey_copies),
                ],
                ligand_ccd_codes=ligand_ccd_codes,
                model_seeds=model_seeds,
                version=args.version,
            )
            emit(job_name, payload)

    if args.submit_script:
        submit_path = out_root / "submit_all.sh"
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            "",
        ]
        for jd in job_dirs:
            rel = jd.relative_to(out_root)
            if write_sbatch:
                lines.append(f'echo "Submitting ./{rel}/job.sbatch"')
                lines.append(f"sbatch ./{rel}/job.sbatch")
        lines.append("")

        if args.dry_run:
            print(f"[dry-run] write {submit_path}")
        else:
            write_text(submit_path, "\n".join(lines))
            submit_path.chmod(0o755)

    if not args.dry_run:
        print(f"### Done")
        print(f"### Output dir: {out_root}")
        print(f"### Jobs: {len(job_dirs)}")
        print(f"### modelSeeds: {model_seeds}")
        print(f"### AlphaFold3 version: {args.version}")
        if ligand_ccd_codes:
            print(f"### Ligands: {', '.join(ligand_ccd_codes)}")


if __name__ == "__main__":
    main()
