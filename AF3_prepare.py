#!/usr/bin/env python3
"""
AF3_prepare.py

Creates per job directories containing:
  - input.json (AlphaFold3 style by default)
  - job.sbatch (optional)
  - submit_all.sh (optional)

This version adds first class support for:
  - ions (via ligand CCD codes such as ZN, MG, CA, NA, CL)
  - small molecules (any CCD code, optionally multiple copies)
  - alphafold3 dialect as the default schema

It still supports alphafoldserver output if you need it.

Examples

Single protein, include zinc:
  ./AF3_prepare.py my.fasta --ion ZN

Single protein, include 2x zinc and 1x magnesium:
  ./AF3_prepare.py my.fasta --ion ZN:2 --ion MG

PPI, include a small molecule ligand ATP and zinc:
  ./AF3_prepare.py prey_dir --bait bait.fasta --mode ppi --ligand ATP --ion ZN

PPI, include multiple ligands:
  ./AF3_prepare.py prey_dir --bait bait.fasta --mode ppi --ligand HEM --ligand NAG:2

Emit alphafoldserver format (no ligands in that schema in this script):
  ./AF3_prepare.py my.fasta --schema alphafoldserver
"""

import json
import random
import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Iterable, Union, Tuple, Dict

from Bio import SeqIO

FA_EXTS = {".fasta", ".fa", ".faa", ".fna"}
JSON_EXTS = {".json"}

DEFAULT_AF3_DIALECT = "alphafold3"
DEFAULT_AF3_VERSION = 2

# Common ion CCD codes (not exhaustive). Users can pass any CCD code.
COMMON_IONS = {
    "ZN", "MG", "CA", "NA", "K", "CL", "MN", "FE", "CU", "CO", "NI", "CD",
}


@dataclass
class SimpleProtein:
    id: str
    seq: str


@dataclass
class CCDItem:
    ccd: str
    count: int


def sanitize_name(s: str, max_len: int = 180) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return (s or "job")[:max_len]


def make_model_seeds(n: int, seed_base: Optional[int] = None) -> List[int]:
    n = int(n)
    if n <= 0:
        return []
    if seed_base is not None:
        b = int(seed_base)
        return list(range(b, b + n))
    return [random.randint(1, 30000) for _ in range(n)]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Union[dict, List[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=4)


def iter_input_paths(p: Path) -> Iterable[Path]:
    if p.is_file():
        yield p
    elif p.is_dir():
        for fp in sorted(p.iterdir()):
            if fp.is_file() and fp.suffix.lower() in (FA_EXTS | JSON_EXTS):
                yield fp
    else:
        raise FileNotFoundError(str(p))


def parse_fasta_file(fasta_file: Path) -> List[SimpleProtein]:
    out: List[SimpleProtein] = []
    with fasta_file.open("r", encoding="utf-8") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            pid = rec.id if rec.id else fasta_file.stem
            out.append(SimpleProtein(id=str(pid), seq=str(rec.seq)))
    return out


def parse_json_file(json_file: Path) -> List[SimpleProtein]:
    with json_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    entries = data if isinstance(data, list) else [data]
    out: List[SimpleProtein] = []
    for entry in entries:
        name = entry.get("name") or json_file.stem
        seq = None
        try:
            if "sequences" in entry and entry["sequences"]:
                s0 = entry["sequences"][0]
                if "proteinChain" in s0:
                    seq = s0["proteinChain"]["sequence"]
                elif "protein" in s0:
                    seq = s0["protein"]["sequence"]
        except Exception:
            pass
        if not seq:
            raise ValueError(f"Cannot find a protein sequence in {json_file}")
        out.append(SimpleProtein(id=str(name), seq=str(seq)))
    return out


def parse_any_file(path: Path) -> List[SimpleProtein]:
    ext = path.suffix.lower()
    if ext in FA_EXTS:
        return parse_fasta_file(path)
    if ext in JSON_EXTS:
        return parse_json_file(path)
    return []


def load_proteins(input_path: Path) -> List[SimpleProtein]:
    prots: List[SimpleProtein] = []
    for fp in iter_input_paths(input_path):
        prots.extend(parse_any_file(fp))
    return prots


def module_string(profile: str) -> str:
    return "alphafold/cc7_3-20250304" if profile == "cc7" else "alphafold/cc8_3-20250304"


def default_partition(profile: str) -> str:
    if profile == "cc7":
        return "gpu-2080ti-11g"
    return "gpu-mig-40g,gpu-a100-80g"


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

echo "#### Running alphafold"

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


def render_sbatch(**kw) -> str:
    constraint_line = ""
    if kw.get("constraint"):
        constraint_line = f"#SBATCH --constraint={kw['constraint']}\n"
    return SBATCH_TEMPLATE.format(
        constraint_line=constraint_line,
        **{k: v for k, v in kw.items() if k != "constraint"},
    )


def parse_ccd_list(items: Optional[List[str]]) -> List[CCDItem]:
    """
    Parse repeated flags like:
      --ion ZN
      --ion ZN:2
      --ligand ATP
      --ligand HEM:3

    Returns list of CCDItem(ccd, count).
    """
    if not items:
        return []
    out: List[CCDItem] = []
    for raw in items:
        raw = raw.strip()
        if not raw:
            continue
        if ":" in raw:
            ccd, cnt = raw.split(":", 1)
            ccd = ccd.strip().upper()
            cnt = int(cnt.strip())
        else:
            ccd = raw.strip().upper()
            cnt = 1
        if not ccd:
            raise ValueError(f"Empty CCD code in '{raw}'")
        if cnt <= 0:
            raise ValueError(f"Count must be positive in '{raw}'")
        out.append(CCDItem(ccd=ccd, count=cnt))
    return out


def expand_ccd_items(items: List[CCDItem]) -> List[str]:
    """
    Expand CCDItem list to a flat list of CCD codes repeated by count.
    """
    flat: List[str] = []
    for it in items:
        flat.extend([it.ccd] * int(it.count))
    return flat


def merge_ccd_items(*groups: List[CCDItem]) -> List[str]:
    """
    Merge multiple CCDItem lists and expand counts.
    """
    merged: List[str] = []
    for g in groups:
        merged.extend(expand_ccd_items(g))
    return merged


def make_ligand_entries(ccd_codes: List[str], start_chain_index: int = 0) -> List[dict]:
    """
    Build AlphaFold3 style ligand entries:
      {"ligand": {"id": ["B"], "ccdCodes": ["ZN"]}}

    We create one ligand entry per ligand instance, each with its own chain id.
    Chain IDs are assigned sequentially from a letter set.
    """
    sequences: List[dict] = []
    for i, ccd in enumerate(ccd_codes):
        chain_id = chain_id_from_index(start_chain_index + i)
        sequences.append({"ligand": {"id": [chain_id], "ccdCodes": [ccd]}})
    return sequences


def chain_id_from_index(i: int) -> str:
    """
    Map 0->A, 1->B, ... 25->Z, 26->AA, 27->AB ...
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


def job_obj_alphafoldserver(name: str, seqs: List[str], counts: List[int], seeds: List[int]) -> dict:
    sequences = []
    for s, c in zip(seqs, counts):
        sequences.append({"proteinChain": {"sequence": s, "count": int(c)}})
    return {
        "name": name,
        "dialect": "alphafoldserver",
        "version": 1,
        "modelSeeds": seeds,
        "sequences": sequences,
    }


def job_obj_alphafold3(
    name: str,
    seqs: List[str],
    chain_ids: List[str],
    seeds: List[int],
    ligand_ccd_codes: Optional[List[str]] = None,
    dialect: str = DEFAULT_AF3_DIALECT,
    version: int = DEFAULT_AF3_VERSION,
    unpaired_msa_path: Optional[str] = None,
) -> dict:
    sequences: List[dict] = []
    for s, cid in zip(seqs, chain_ids):
        prot_obj = {"id": cid, "sequence": s}
        if unpaired_msa_path:
            # Optional and tooling dependent
            prot_obj["unpairedMsaPath"] = unpaired_msa_path
            prot_obj["pairedMsa"] = ""
            prot_obj["templates"] = []
        sequences.append({"protein": prot_obj})

    # Append ligand entries (each as its own "chain" id) after protein chains
    if ligand_ccd_codes:
        start_idx = len(chain_ids)
        sequences.extend(make_ligand_entries(ligand_ccd_codes, start_chain_index=start_idx))

    return {
        "name": name,
        "dialect": dialect,
        "version": int(version),
        "modelSeeds": seeds,
        "sequences": sequences,
    }


def choose_default_outdir_name(
    mode: str,
    prey_input: Path,
    prey_prots: List[SimpleProtein],
    bait: Optional[SimpleProtein],
) -> str:
    if mode == "single" and len(prey_prots) == 1:
        return sanitize_name(prey_prots[0].id)
    if mode == "ppi" and bait is not None:
        return sanitize_name(bait.id)
    if prey_input.is_file():
        return sanitize_name(prey_input.stem)
    return sanitize_name(prey_input.name)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Make AF3 job dirs + input.json + sbatch. Supports AF3 ligands (ions and small molecules)."
    )

    p.add_argument("input", help="Prey input: FASTA/JSON file OR directory")
    p.add_argument("--bait", default=None, help="Optional bait input: FASTA/JSON file OR directory")
    p.add_argument("--mode", choices=["auto", "single", "ppi"], default="auto")

    p.add_argument(
        "--schema",
        choices=["alphafold3", "alphafoldserver"],
        default="alphafold3",
        help="Which JSON schema to emit. Default: alphafold3.",
    )

    p.add_argument("--seeds", type=int, default=5, help="How many modelSeeds to write (default 5)")
    p.add_argument("--seed-base", type=int, default=None, help="Deterministic seeds starting at this int (optional)")

    p.add_argument("--outdir", default=None, help="Output directory root (optional). If omitted, derived from headers.")
    p.add_argument("--project", default=None, help="Optional grouping folder under outdir (for huge batches).")

    p.add_argument("--no-sbatch", action="store_true", help="Do not write job.sbatch files")
    p.add_argument("--submit-script", action="store_true", help="Write submit_all.sh in the output root")

    p.add_argument("--prey-count", type=int, default=1, help="Count for prey chains (alphafoldserver only; default 1)")
    p.add_argument("--bait-count", type=int, default=1, help="Count for bait chain (alphafoldserver only; default 1)")
    p.add_argument("--name-template", default="{bait}_with_{prey}", help="PPI name template")
    p.add_argument("--skip-self", action="store_true")

    # New: ligands / ions for alphafold3
    p.add_argument(
        "--ion",
        action="append",
        default=[],
        help="Add ion(s) by CCD code, optionally with count. Example: --ion ZN or --ion ZN:2",
    )
    p.add_argument(
        "--ligand",
        action="append",
        default=[],
        help="Add small molecule ligand(s) by CCD code, optionally with count. Example: --ligand ATP or --ligand HEM:2",
    )

    # New: control alphafold3 dialect/version fields (default dialect alphafold3)
    p.add_argument("--af3-dialect", default=DEFAULT_AF3_DIALECT, help="AF3 dialect field (default alphafold3)")
    p.add_argument("--af3-version", type=int, default=DEFAULT_AF3_VERSION, help="AF3 version field (default 2)")

    # Optional: include unpairedMsaPath in AF3 protein objects (if you want to point to an a3m)
    p.add_argument(
        "--unpaired-msa-path",
        default=None,
        help="Optional unpairedMsaPath to include in AF3 protein objects (same path written for all chains).",
    )

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

    p.add_argument("--dry-run", action="store_true")

    args = p.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "ppi" if args.bait else "single"

    prey_input = Path(args.input)
    prey_prots = [SimpleProtein(sanitize_name(x.id), x.seq) for x in load_proteins(prey_input)]
    if not prey_prots:
        raise SystemExit(f"No prey sequences found in: {args.input}")

    bait: Optional[SimpleProtein] = None
    if mode == "ppi":
        if not args.bait:
            raise SystemExit("PPI mode requires --bait")
        bait_prots = [SimpleProtein(sanitize_name(x.id), x.seq) for x in load_proteins(Path(args.bait))]
        if len(bait_prots) != 1:
            raise SystemExit("ERROR: --bait must resolve to exactly one sequence for PPI in this script.")
        bait = bait_prots[0]

    # Parse and expand ligand lists
    ion_items = parse_ccd_list(args.ion)
    ligand_items = parse_ccd_list(args.ligand)
    ligand_ccd_codes = merge_ccd_items(ion_items, ligand_items)

    # Friendly hint (does not block) for likely typos
    for it in ion_items:
        if it.ccd.upper() not in COMMON_IONS:
            # not fatal, CCD is broader than this list
            pass

    seeds = make_model_seeds(args.seeds, args.seed_base)
    module_load = module_string(args.af3_module_profile)
    partition = args.partition if args.partition else default_partition(args.af3_module_profile)
    make_sbatch = not args.no_sbatch

    if args.outdir:
        out_root = Path(args.outdir)
    else:
        out_root = Path(choose_default_outdir_name(mode, prey_input, prey_prots, bait))

    if args.project:
        out_root = out_root / sanitize_name(args.project)

    logs_dir = out_root / "logs"
    job_dirs: List[Path] = []

    def emit(job_name: str, obj: dict) -> None:
        job_name_safe = sanitize_name(job_name)
        jd = out_root / job_name_safe
        job_dirs.append(jd)

        input_path = jd / "input.json"
        payload = [obj]

        if args.dry_run:
            print(f"[dry-run] mkdir -p {jd}")
            print(f"[dry-run] mkdir -p {logs_dir}")
            print(f"[dry-run] write {input_path}")
        else:
            jd.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            write_json(input_path, payload)

        if make_sbatch:
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
                input_dir=str(jd),
                output_dir=str(jd),
                logs_dir=str(logs_dir),
            )
            sbatch_path = jd / "job.sbatch"
            if args.dry_run:
                print(f"[dry-run] write {sbatch_path}")
            else:
                write_text(sbatch_path, sbatch_txt)

    if mode == "single":
        for prey in prey_prots:
            if args.schema == "alphafoldserver":
                if ligand_ccd_codes:
                    raise SystemExit(
                        "ERROR: --ion/--ligand currently supported only for --schema alphafold3 in this script."
                    )
                obj = job_obj_alphafoldserver(
                    name=prey.id,
                    seqs=[prey.seq],
                    counts=[args.prey_count],
                    seeds=seeds,
                )
            else:
                obj = job_obj_alphafold3(
                    name=prey.id,
                    seqs=[prey.seq],
                    chain_ids=["A"],
                    seeds=seeds,
                    ligand_ccd_codes=ligand_ccd_codes,
                    dialect=args.af3_dialect,
                    version=args.af3_version,
                    unpaired_msa_path=args.unpaired_msa_path,
                )
            emit(prey.id, obj)
    else:
        assert bait is not None
        bait_id = bait.id
        for prey in prey_prots:
            if args.skip_self and prey.id == bait_id:
                continue
            name = args.name_template.format(bait=bait_id, prey=prey.id)

            if args.schema == "alphafoldserver":
                if ligand_ccd_codes:
                    raise SystemExit(
                        "ERROR: --ion/--ligand currently supported only for --schema alphafold3 in this script."
                    )
                obj = job_obj_alphafoldserver(
                    name=name,
                    seqs=[bait.seq, prey.seq],
                    counts=[args.bait_count, args.prey_count],
                    seeds=seeds,
                )
            else:
                obj = job_obj_alphafold3(
                    name=name,
                    seqs=[bait.seq, prey.seq],
                    chain_ids=["A", "B"],
                    seeds=seeds,
                    ligand_ccd_codes=ligand_ccd_codes,
                    dialect=args.af3_dialect,
                    version=args.af3_version,
                    unpaired_msa_path=args.unpaired_msa_path,
                )
            emit(name, obj)

    if args.submit_script:
        submit_path = out_root / "submit_all.sh"
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            f'echo "Submitting all job.sbatch files under: {out_root}"',
            "",
        ]
        for jd in job_dirs:
            rel = jd.relative_to(out_root)
            if make_sbatch:
                lines.append(f'echo "sbatch ./{rel}/job.sbatch"')
                lines.append(f"sbatch ./{rel}/job.sbatch")
        lines.append("")
        if args.dry_run:
            print(f"[dry-run] write {submit_path}")
        else:
            write_text(submit_path, "\n".join(lines) + "\n")
            submit_path.chmod(0o755)

    if not args.dry_run:
        print(f"### Done. Output dir: {out_root}")
        print(f"### Jobs: {len(job_dirs)}")
        print(f"### Logs dir: {logs_dir}")
        print(f"### Sbatch: {'on' if make_sbatch else 'off'}")
        if args.schema == "alphafold3" and ligand_ccd_codes:
            print(f"### Ligands appended (CCD): {', '.join(ligand_ccd_codes)}")


if __name__ == "__main__":
    main()
