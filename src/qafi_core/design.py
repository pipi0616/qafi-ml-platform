"""The design loop (chapter 15d): the generator proposes, the two energies decide what it may touch.

Chapter 15c measured which ranking finds the positions a redesign must leave alone. This packages that
answer into something runnable: given a protein's site map and a budget, it produces the fixed-position
file ProteinMPNN reads, and records what was decided and why.

The order of events matters and is fixed here so a campaign cannot be assembled a different way by
accident:

  1. ``screen`` -- refuse targets outside the benign scope before anything is computed.
  2. ``rank_positions`` -- order positions by a stated criterion, highest cost first.
  3. ``fixed_positions`` -- take the budget off the top; those are the ones the generator may not change.
  4. ``to_proteinmpnn`` -- write that set in the format ``--fixed_positions_jsonl`` expects.
  5. ``provenance`` -- the record that travels with every candidate.

Nothing here calls ProteinMPNN or the energy model; both are passed in by the caller, so the loop can be
tested without a GPU and reused from a notebook or a Nextflow task alike.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# How a position may be ordered. "func" is 15c's answer; the others are what the literature uses and
# what 15c compared against, kept so a campaign can be re-run with the baseline ranking on purpose.
CRITERIA = {
    "func": "func_cost",  # our functional energy (15c: beats conservation, p = 6.3e-11)
    "fold": "fold_cost",  # our folding energy
    "both": None,  # fold_cost + func_cost, computed
}

# Design is refused outside this scope. The gate is deliberately a allow-list of intents rather than a
# block-list of targets: a campaign states what it is for, and anything unstated does not run.
ALLOWED_INTENTS = frozenset(
    {
        "stability_engineering",  # make a therapeutic or industrial enzyme more robust
        "expression_improvement",  # make it easier to produce
        "benchmark",  # reproduce a published comparison
    }
)


class RefusedError(RuntimeError):
    """Raised when a campaign is outside the scope this module will design for."""


def screen(intent: str, note: str = "") -> dict:
    """Check a campaign's stated intent before any computation; raise unless it is in scope.

    Returns the record to attach to the campaign. ``note`` is free text for the audit trail.
    """
    if intent not in ALLOWED_INTENTS:
        raise RefusedError(
            f"intent {intent!r} is not in the allowed scope {sorted(ALLOWED_INTENTS)}; "
            "this module designs for benign protein engineering only"
        )
    return {"intent": intent, "note": note, "screened_at": datetime.now(UTC).isoformat()}


def rank_positions(sites: pd.DataFrame, by: str = "func") -> pd.DataFrame:
    """Order a protein's positions, the ones a redesign should protect first.

    ``sites`` is the site map of one protein: position, fold_cost, func_cost. The energies are positive
    when favourable, so a costly position has a low value and is ranked first by negating it.
    """
    if by not in CRITERIA:
        raise ValueError(f"unknown criterion {by!r}; use one of {sorted(CRITERIA)}")
    if sites.position.duplicated().any():
        raise ValueError("sites has more than one row for a position; pass one protein's map")

    out = sites.copy()
    out["cost"] = -(out.fold_cost + out.func_cost) if by == "both" else -out[CRITERIA[by]]
    return out.sort_values("cost", ascending=False).reset_index(drop=True)


def fixed_positions(ranked: pd.DataFrame, budget: float) -> list[int]:
    """The positions the generator may not change: the top ``budget`` share of ``ranked``.

    ``budget`` is a share between 0 and 1. Sumida et al. 2024 needed 127 of 221 positions, a share of
    0.57, and had to find it per protein; it is a parameter here for the same reason.
    """
    if not 0 < budget <= 1:
        raise ValueError(f"budget must be a share in (0, 1], got {budget}")
    n = max(1, round(len(ranked) * budget))
    return sorted(int(p) for p in ranked.position.head(n))


def to_proteinmpnn(fixed: dict[str, list[int]], path: str | Path) -> Path:
    """Write ``--fixed_positions_jsonl``: one line, {pdb name: {chain: positions}}, positions 1-based.

    ProteinMPNN reads this file and leaves those residues at their wild-type identity.
    """
    path = Path(path)
    path.write_text(json.dumps(fixed) + "\n", encoding="utf-8")
    return path


def provenance(
    *,
    protein: str,
    criterion: str,
    budget: float,
    fixed: list[int],
    n_positions: int,
    screening: dict,
    generator: dict,
    model: dict,
) -> dict:
    """Everything needed to say where a candidate came from, including a hash of the fixed set.

    ``generator`` and ``model`` name the versions used -- a commit for ProteinMPNN, a run or release id
    for ours -- so two campaigns with the same inputs can be told apart from two that only look alike.
    """
    digest = hashlib.sha256(",".join(map(str, fixed)).encode()).hexdigest()[:16]
    return {
        "protein": protein,
        "criterion": criterion,
        "budget": budget,
        "n_fixed": len(fixed),
        "n_positions": n_positions,
        "fixed_sha256": digest,
        "screening": screening,
        "generator": generator,
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
    }


def campaign(
    sites: pd.DataFrame,
    *,
    protein: str,
    chain: str = "A",
    criterion: str = "func",
    budget: float = 0.5,
    intent: str = "stability_engineering",
    note: str = "",
    generator: dict | None = None,
    model: dict | None = None,
    out_dir: str | Path = ".",
) -> dict:
    """Screen, rank, pick the fixed set, write ProteinMPNN's input, and return the provenance record."""
    screening = screen(intent, note)
    ranked = rank_positions(sites, criterion)
    fixed = fixed_positions(ranked, budget)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = to_proteinmpnn({protein: {chain: fixed}}, out_dir / f"{protein}_fixed.jsonl")
    record = provenance(
        protein=protein,
        criterion=criterion,
        budget=budget,
        fixed=fixed,
        n_positions=len(ranked),
        screening=screening,
        generator=generator or {},
        model=model or {},
    )
    record["fixed_positions_jsonl"] = str(path)
    (out_dir / f"{protein}_provenance.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return record
