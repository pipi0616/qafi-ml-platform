import json
from pathlib import Path

import pandas as pd
import pytest

from qafi_core import design


def sites(n=10):
    """A small site map: position 1 is the costliest on function, position 10 on folding."""
    return pd.DataFrame(
        {
            "position": range(1, n + 1),
            "func_cost": [-(n - i) * 0.5 for i in range(n)],  # position 1 most costly
            "fold_cost": [-i * 0.5 for i in range(n)],  # position n most costly
        }
    )


def test_a_campaign_outside_the_allowed_scope_is_refused_before_anything_runs():
    with pytest.raises(design.RefusedError, match="benign protein engineering"):
        design.screen("make_a_toxin")


def test_an_allowed_intent_returns_a_record_for_the_audit_trail():
    got = design.screen("stability_engineering", note="industrial xylanase")

    assert got["intent"] == "stability_engineering"
    assert got["note"] == "industrial xylanase"
    assert got["screened_at"].endswith("+00:00")


def test_the_criterion_decides_which_end_of_the_protein_is_protected():
    """func and fold disagree by construction here, so a swap of criterion has to change the answer."""
    by_func = design.fixed_positions(design.rank_positions(sites(), "func"), 0.3)
    by_fold = design.fixed_positions(design.rank_positions(sites(), "fold"), 0.3)

    assert by_func == [1, 2, 3]
    assert by_fold == [8, 9, 10]


def test_both_adds_the_two_energies_rather_than_picking_one():
    ranked = design.rank_positions(sites(), "both")

    # the two costs are mirror images, so every position has the same total and order is by position
    assert len(ranked) == 10
    assert ranked.cost.round(6).nunique() == 1


def test_an_unknown_criterion_is_refused():
    with pytest.raises(ValueError, match="unknown criterion"):
        design.rank_positions(sites(), "whatever")


def test_a_site_map_with_two_rows_for_one_position_is_refused():
    doubled = pd.concat([sites(3), sites(3)])
    with pytest.raises(ValueError, match="more than one row"):
        design.rank_positions(doubled, "func")


@pytest.mark.parametrize("budget", [0, -0.1, 1.5])
def test_a_budget_that_is_not_a_share_is_refused(budget):
    with pytest.raises(ValueError, match="share"):
        design.fixed_positions(design.rank_positions(sites(), "func"), budget)


def test_a_tiny_budget_still_protects_one_position():
    assert design.fixed_positions(design.rank_positions(sites(100), "func"), 0.001) == [1]


def test_the_written_file_is_what_proteinmpnn_reads(tmp_path):
    path = design.to_proteinmpnn({"4PN2": {"A": [3, 1, 2]}}, tmp_path / "fixed.jsonl")

    assert json.loads(path.read_text()) == {"4PN2": {"A": [3, 1, 2]}}


def test_a_campaign_records_what_it_decided_and_writes_both_files(tmp_path):
    record = design.campaign(
        sites(20),
        protein="TEST",
        criterion="func",
        budget=0.25,
        intent="stability_engineering",
        note="unit test",
        generator={"proteinmpnn": "8907e667"},
        model={"release": "2026-10-21"},
        out_dir=tmp_path,
    )

    assert record["n_fixed"] == 5 and record["n_positions"] == 20
    assert record["criterion"] == "func" and record["budget"] == 0.25
    assert record["generator"] == {"proteinmpnn": "8907e667"}
    assert len(record["fixed_sha256"]) == 16
    assert json.loads(Path(record["fixed_positions_jsonl"]).read_text()) == {"TEST": {"A": [1, 2, 3, 4, 5]}}
    assert json.loads((tmp_path / "TEST_provenance.json").read_text())["protein"] == "TEST"


def test_two_campaigns_that_protect_different_sets_get_different_hashes(tmp_path):
    a = design.campaign(sites(20), protein="A", criterion="func", budget=0.25, out_dir=tmp_path)
    b = design.campaign(sites(20), protein="B", criterion="fold", budget=0.25, out_dir=tmp_path)

    assert a["fixed_sha256"] != b["fixed_sha256"]


def test_a_refused_campaign_writes_nothing(tmp_path):
    with pytest.raises(design.RefusedError):
        design.campaign(sites(), protein="X", intent="weaponise", out_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []
