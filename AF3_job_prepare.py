#!/usr/bin/env python3
import json
import random
import argparse
import re
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Iterable, Union

from Bio import SeqIO

FA_EXTS = {".fasta", ".fa", ".faa", ".fna"}
JSON_EXTS = {".json"}


@dataclass
class SimpleProtein:
    id: str
    seq: str


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


def default_partition(profile: str, use_partitions: bool) -> str:
    if not use_partitions:
        return "gpu-short"
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
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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


def get_user_data1_root() -> Path:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown_user"
    return Path("/data1") / user


def abspath_no_symlinks(p: Path) -> Path:
    """
    Absolute path without resolving symlinks.
    This is important on clusters where /data1 is a symlink into /zfsstore.
    """
    return Path(os.path.abspath(str(p.expanduser())))


def ensure_under_data1_user(path: Path, allow_outside: bool) -> None:
    if allow_outside:
        return
    allowed_root = abspath_no_symlinks(get_user_data1_root())
    target = abspath_no_symlinks(path)

    # string based prefix check using normalized absolute paths
    allowed_str = allowed_root.as_posix().rstrip("/") + "/"
    target_str = target.as_posix().rstrip("/") + "/"
    if not target_str.startswith(allowed_str):
        raise SystemExit(
            "ERROR: Output path must be under "
            f"{allowed_root}\n"
            f"Got: {target}\n"
            "If you really want to write elsewhere, pass --allow-outside-data1"
        )


def resolve_project_dir(output_root: Path, project: str) -> Path:
    """
    Build an absolute path without resolving symlinks.
    """
    output_root = abspath_no_symlinks(output_root)
    return abspath_no_symlinks(output_root / project)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Make AF3 job dirs and input.json plus optional sbatch scripts. "
            "Output location defaults to /data1/$USER unless overridden."
        )
    )

    out_group = p.add_mutually_exclusive_group(required=False)
    out_group.add_argument(
        "--project-dir",
        default=None,
        help="Absolute or relative path to the project directory to create (overrides --output-root/--project).",
    )

    p.add_argument(
        "--output-root",
        default=str(get_user_data1_root()),
        help="Base output directory (absolute or relative). Default is /data1/$USER.",
    )

    p.add_argument(
        "--project",
        required=True,
        help="Project subdirectory under --output-root (for example diviva_nterm_mut/nterm_test).",
    )

    p.add_argument("--prey", required=True, help="Prey input: FASTA/JSON file or directory")
    p.add_argument("--bait", default=None, help="Optional bait input: FASTA/JSON file or directory")

    p.add_argument("--mode", choices=["auto", "single", "ppi"], default="auto")

    p.add_argument(
        "--schema",
        choices=["alphafoldserver", "alphafold3"],
        default="alphafoldserver",
        help="Which JSON schema to emit. alphafoldserver matches the working template: proteinChain with sequence+count and no id.",
    )

    p.add_argument("--seeds", type=int, default=10, help="How many modelSeeds to write")
    p.add_argument("--seed-base", type=int, default=None, help="Deterministic seeds starting at this int")

    p.add_argument("--job-dirs", action="store_true", help="Create per-job dirs with input.json")
    p.add_argument("--make-sbatch", action="store_true", help="Write job.sbatch in each job dir (requires --job-dirs)")
    p.add_argument(
        "--submit-script",
        action="store_true",
        help="Write submit_all.sh in project dir (requires --job-dirs and --make-sbatch)",
    )

    p.add_argument("--prey-count", type=int, default=1, help="Count for prey chains (alphafoldserver only)")
    p.add_argument("--bait-count", type=int, default=1, help="Count for bait chain (alphafoldserver only)")
    p.add_argument("--name-template", default="{bait}_with_{prey}", help="PPI name template")
    p.add_argument("--skip-self", action="store_true")

    p.add_argument("--af3-module-profile", choices=["cc7", "cc8"], default="cc8")
    p.add_argument("--use-partitions", action="store_true")
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

    p.add_argument(
        "--allow-outside-data1",
        action="store_true",
        help="Allow writing outside /data1/$USER (default is to block this).",
    )

    args = p.parse_args()

    if args.make_sbatch and not args.job_dirs:
        raise SystemExit("ERROR: --make-sbatch requires --job-dirs")
    if args.submit_script and not (args.job_dirs and args.make_sbatch):
        raise SystemExit("ERROR: --submit-script requires --job-dirs and --make-sbatch")

    if args.project_dir:
        project_dir = abspath_no_symlinks(Path(args.project_dir))
    else:
        project_dir = resolve_project_dir(Path(args.output_root), args.project)

    ensure_under_data1_user(project_dir, args.allow_outside_data1)

    if not args.dry_run:
        project_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    if mode == "auto":
        mode = "ppi" if args.bait else "single"

    prey_prots = [SimpleProtein(sanitize_name(x.id), x.seq) for x in load_proteins(Path(args.prey))]
    if not prey_prots:
        raise SystemExit(f"No prey sequences found in: {args.prey}")

    bait_prots: List[SimpleProtein] = []
    if mode == "ppi":
        if not args.bait:
            raise SystemExit("PPI mode requires --bait")
        bait_prots = [SimpleProtein(sanitize_name(x.id), x.seq) for x in load_proteins(Path(args.bait))]
        if len(bait_prots) != 1:
            raise SystemExit("ERROR: --bait must resolve to exactly one sequence for PPI in this script.")
        bait = bait_prots[0]
    else:
        bait = None

    seeds = make_model_seeds(args.seeds, args.seed_base)

    module_load = module_string(args.af3_module_profile)
    partition = args.partition if args.partition else default_partition(args.af3_module_profile, args.use_partitions)

    job_dirs: List[Path] = []

    def emit(job_name: str, obj: dict):
        job_name_safe = sanitize_name(job_name)

        if args.job_dirs:
            jd = abspath_no_symlinks(project_dir / job_name_safe)
            job_dirs.append(jd)

            input_path = jd / "input.json"
            payload = [obj]

            if args.dry_run:
                print(f"[dry-run] mkdir -p {jd}")
                print(f"[dry-run] write {input_path}")
            else:
                jd.mkdir(parents=True, exist_ok=True)
                write_json(input_path, payload)

            if args.make_sbatch:
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
                )
                sbatch_path = jd / "job.sbatch"
                if args.dry_run:
                    print(f"[dry-run] write {sbatch_path}")
                else:
                    write_text(sbatch_path, sbatch_txt)

        else:
            outpath = project_dir / f"{job_name_safe}.json"
            if args.dry_run:
                print(f"[dry-run] write {outpath}")
            else:
                write_json(outpath, [obj])

    def job_obj_alphafoldserver(name: str, seqs: List[str], counts: List[int], seeds_in: List[int]) -> dict:
        sequences = []
        for s, c in zip(seqs, counts):
            sequences.append({"proteinChain": {"sequence": s, "count": int(c)}})
        return {
            "name": name,
            "dialect": "alphafoldserver",
            "version": 1,
            "modelSeeds": seeds_in,
            "sequences": sequences,
        }

    def job_obj_alphafold3(name: str, seqs: List[str], chain_ids: List[str], seeds_in: List[int]) -> dict:
        sequences = []
        for s, cid in zip(seqs, chain_ids):
            sequences.append({"protein": {"id": cid, "sequence": s}})
        return {
            "name": name,
            "dialect": "alphafold3",
            "version": 1,
            "modelSeeds": seeds_in,
            "sequences": sequences,
        }

    if mode == "single":
        for prey in prey_prots:
            if args.schema == "alphafoldserver":
                obj = job_obj_alphafoldserver(
                    name=prey.id,
                    seqs=[prey.seq],
                    counts=[args.prey_count],
                    seeds_in=seeds,
                )
            else:
                obj = job_obj_alphafold3(
                    name=prey.id,
                    seqs=[prey.seq],
                    chain_ids=["A"],
                    seeds_in=seeds,
                )
            emit(prey.id, obj)

    else:
        bait_id = bait.id
        for prey in prey_prots:
            if args.skip_self and prey.id == bait_id:
                continue
            name = args.name_template.format(bait=bait_id, prey=prey.id)
            if args.schema == "alphafoldserver":
                obj = job_obj_alphafoldserver(
                    name=name,
                    seqs=[bait.seq, prey.seq],
                    counts=[args.bait_count, args.prey_count],
                    seeds_in=seeds,
                )
            else:
                obj = job_obj_alphafold3(
                    name=name,
                    seqs=[bait.seq, prey.seq],
                    chain_ids=["A", "B"],
                    seeds_in=seeds,
                )
            emit(name, obj)

    if args.submit_script:
        submit_path = project_dir / "submit_all.sh"
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            f'echo "Submitting all job.sbatch files under: {project_dir}"',
            "",
        ]
        for jd in job_dirs:
            rel = os.path.relpath(str(jd), str(project_dir))
            lines.append(f'echo "sbatch ./{rel}/job.sbatch"')
            lines.append(f"sbatch ./{rel}/job.sbatch")
        lines.append("")

        if args.dry_run:
            print(f"[dry-run] write {submit_path}")
        else:
            write_text(submit_path, "\n".join(lines) + "\n")
            submit_path.chmod(0o755)

    if not args.dry_run:
        print(f"### Done. Project dir: {project_dir}")
        if args.job_dirs:
            print(f"### Jobs: {len(job_dirs)}")


if __name__ == "__main__":
    main()

