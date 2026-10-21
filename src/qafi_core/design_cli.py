"""Command line for the design loop, so a Nextflow task runs the same code a notebook does.

Two subcommands:

  plan   read a site map, screen the intent, pick the fixed set, write ProteinMPNN's input
  audit  check generated sequences against the fixed set and say whether any of it was touched

The audit is not a formality. A fixed-position file that is read but silently ignored -- a wrong chain
id, a mismatched pdb name -- produces designs that look fine and have changed exactly the residues the
campaign meant to protect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qafi_core import design


def _plan(args: argparse.Namespace) -> None:
    sites = pd.read_parquet(args.sites)
    sites = sites[sites.uniprot_id == args.protein]
    if sites.empty:
        raise SystemExit(f"no site map rows for {args.protein}")
    # the jsonl key has to be the pdb file's name, not the accession: ProteinMPNN matches on it and
    # silently designs everything if it does not find the entry.
    pdb_name = args.pdb_name or f"AF-{args.protein}-F1-model_v6"
    record = design.campaign(
        sites[["position", "fold_cost", "func_cost"]],
        protein=pdb_name,
        chain=args.chain,
        criterion=args.criterion,
        budget=args.budget,
        intent=args.intent,
        note=args.note,
        generator={"proteinmpnn": args.generator},
        model={"site_map": args.site_map_version},
        out_dir=args.out,
    )
    print(
        f"{pdb_name}: fixed {record['n_fixed']} of {record['n_positions']} "
        f"by {record['criterion']}, sha {record['fixed_sha256']}"
    )


def read_fasta(path: str | Path) -> tuple[str, list[str]]:
    """ProteinMPNN's output: the first record is the input sequence, the rest are the designs."""
    seqs = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.startswith(">")
    ]
    if len(seqs) < 2:
        raise SystemExit(f"{path} has no designs")
    return seqs[0], seqs[1:]


def audit(native: str, designs: list[str], fixed: list[int]) -> dict:
    """How much each design changed, and whether it touched a position the campaign fixed."""
    protected = set(fixed)
    rows = []
    for i, d in enumerate(designs, start=1):
        if len(d) != len(native):
            raise SystemExit(f"design {i} is {len(d)} long, the input is {len(native)}")
        changed = {p for p in range(1, len(native) + 1) if native[p - 1] != d[p - 1]}
        rows.append({"design": i, "mutations": len(changed), "violations": sorted(changed & protected)})
    return {
        "designs": len(designs),
        "length": len(native),
        "fixed": len(protected),
        "per_design": rows,
        "clean": all(not r["violations"] for r in rows),
    }


def _audit(args: argparse.Namespace) -> None:
    record = json.loads(Path(args.fixed).read_text())
    # the provenance records where the file was written; a workflow stages it somewhere else, so
    # the path is only a fallback for an interactive run.
    jsonl = args.jsonl or record["fixed_positions_jsonl"]
    fixed_file = json.loads(Path(jsonl).read_text())
    positions = next(iter(next(iter(fixed_file.values())).values()))
    native, designs = read_fasta(args.fasta)
    got = audit(native, designs, positions)
    got["protein"] = record["protein"]
    got["fixed_sha256"] = record["fixed_sha256"]
    Path(args.out).write_text(json.dumps(got, indent=2) + "\n", encoding="utf-8")
    print(
        f"{got['protein']}: {got['designs']} designs, "
        f"{'no position violated' if got['clean'] else 'VIOLATIONS'}"
    )
    if not got["clean"]:
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="pick the positions a redesign must not change")
    p.add_argument("--sites", required=True)
    p.add_argument("--protein", required=True)
    p.add_argument("--chain", default="A")
    p.add_argument(
        "--pdb-name",
        dest="pdb_name",
        default=None,
        help="the key ProteinMPNN matches on; defaults to the AlphaFold file name",
    )
    p.add_argument("--criterion", default="func", choices=sorted(design.CRITERIA))
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--intent", default="stability_engineering")
    p.add_argument("--note", default="")
    p.add_argument("--generator", default="unknown")
    p.add_argument("--site-map-version", dest="site_map_version", default="14d")
    p.add_argument("--out", default=".")
    p.set_defaults(func=_plan)

    a = sub.add_parser("audit", help="check the designs against the fixed set")
    a.add_argument("--fasta", required=True)
    a.add_argument("--fixed", required=True, help="the provenance json written by plan")
    a.add_argument(
        "--jsonl", default=None, help="the fixed-positions file; needed when it is not where plan left it"
    )
    a.add_argument("--out", required=True)
    a.set_defaults(func=_audit)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
