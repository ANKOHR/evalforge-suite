from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalforge.core import (
    diff_eval_sets,
    dumps_eval_set,
    forge,
    lint_eval_set,
    load_spec,
)


ROOT = Path(__file__).resolve().parents[1]
TICKET_SPEC = ROOT / "examples" / "support-ticket-triage.yaml"


def test_generation_is_byte_deterministic() -> None:
    spec = load_spec(TICKET_SPEC)
    first = dumps_eval_set(forge(spec, n=80, seed=712)).encode("utf-8")
    second = dumps_eval_set(forge(spec, n=80, seed=712)).encode("utf-8")
    assert first == second


def test_different_seed_changes_eval_set() -> None:
    spec = load_spec(TICKET_SPEC)
    assert dumps_eval_set(forge(spec, n=40, seed=1)) != dumps_eval_set(forge(spec, n=40, seed=2))


def test_forged_ticket_set_covers_all_modes_and_passes_lint() -> None:
    spec = load_spec(TICKET_SPEC)
    result = forge(spec, n=200, seed=42)
    assert result["synthetic"] is True
    assert lint_eval_set(result, spec) == []


def test_lint_flags_missing_failure_mode_coverage() -> None:
    spec = load_spec(TICKET_SPEC)
    result = forge(spec, n=40, seed=42)
    missing_id = spec["failure_modes"][0]["id"]
    for case in result["cases"]:
        case["failure_mode_tags"] = [tag for tag in case["failure_mode_tags"] if tag != missing_id]
        if case.get("target_failure_mode") == missing_id:
            case["target_failure_mode"] = spec["failure_modes"][1]["id"]
    issues = lint_eval_set(result, spec)
    assert any(f"missing failure-mode coverage: {missing_id}" in issue for issue in issues)


def test_regression_traps_are_tagged_and_target_a_declared_mode() -> None:
    spec = load_spec(TICKET_SPEC)
    result = forge(spec, n=30, seed=8)
    declared = {mode["id"] for mode in spec["failure_modes"]}
    traps = [case for case in result["cases"] if case["case_type"] == "regression_trap"]
    assert traps
    for case in traps:
        assert "regression_trap" in case["failure_mode_tags"]
        assert case["target_failure_mode"] in declared
        assert case["target_failure_mode"] in case["failure_mode_tags"]


def test_small_complete_set_keeps_all_case_types_and_mode_coverage() -> None:
    spec = load_spec(TICKET_SPEC)
    result = forge(spec, n=4, seed=8)
    assert {case["case_type"] for case in result["cases"]} == {
        "normal",
        "edge",
        "adversarial",
        "regression_trap",
    }
    assert lint_eval_set(result, spec) == []


def test_adding_failure_mode_adds_tagged_cases_without_generator_changes() -> None:
    spec = load_spec(TICKET_SPEC)
    extended = copy.deepcopy(spec)
    new_mode = copy.deepcopy(extended["failure_modes"][0])
    new_mode["id"] = "new_spec_mode"
    new_mode["description"] = "A newly declared synthetic failure mode."
    extended["failure_modes"].append(new_mode)
    before = forge(spec, n=40, seed=42)
    after = forge(extended, n=40, seed=42)
    assert before["spec_digest"] != after["spec_digest"]
    assert any("new_spec_mode" in case["failure_mode_tags"] for case in after["cases"])
    assert "new_spec_mode" not in before["declared_failure_modes"]


def test_diff_reports_added_removed_modified_and_coverage_delta() -> None:
    old = forge(load_spec(TICKET_SPEC), n=8, seed=6)
    new = copy.deepcopy(old)
    new["cases"] = new["cases"][:1] + new["cases"][2:]
    changed = copy.deepcopy(old["cases"][2])
    changed["ground_truth"] = {"edited": True}
    new["cases"][1] = changed
    new["cases"].append(
        {
            "id": "EF-NEW",
            "case_type": "normal",
            "inputs": {"message": "Synthetic added input"},
            "ground_truth": {"result": "synthetic"},
            "failure_mode_tags": [old["declared_failure_modes"][0]],
            "target_failure_mode": old["declared_failure_modes"][0],
            "difficulty": "easy",
            "synthetic": True,
        }
    )
    result = diff_eval_sets(old, new)
    assert result["added"] == ["EF-NEW"]
    assert result["removed"] == [old["cases"][1]["id"]]
    assert result["modified"] == [old["cases"][2]["id"]]
    assert result["coverage"]["delta"][old["declared_failure_modes"][0]] == 1


def test_lint_detects_duplicate_inputs_and_inconsistent_truth() -> None:
    spec = load_spec(TICKET_SPEC)
    result = forge(spec, n=12, seed=3)
    duplicate = copy.deepcopy(result["cases"][0])
    duplicate["id"] = "EF-DUPLICATE"
    duplicate["ground_truth"] = {"different": True}
    result["cases"].append(duplicate)
    issues = lint_eval_set(result, spec)
    assert any(issue.startswith("duplicate inputs:") for issue in issues)
    assert any(issue.startswith("ground-truth inconsistency:") for issue in issues)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "evalforge", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_forge_smoke(tmp_path: Path) -> None:
    output = tmp_path / "forged.json"
    result = _run_cli("forge", "--spec", str(TICKET_SPEC), "--n", "20", "--seed", "12", "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["synthetic"] is True


def test_cli_lint_smoke(tmp_path: Path) -> None:
    output = tmp_path / "forged.json"
    forged = _run_cli("forge", "--spec", str(TICKET_SPEC), "--n", "20", "--seed", "12", "--output", str(output))
    assert forged.returncode == 0, forged.stderr
    result = _run_cli("lint", "--eval-set", str(output), "--spec", str(TICKET_SPEC))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS:" in result.stdout


def test_cli_diff_smoke(tmp_path: Path) -> None:
    old_path = tmp_path / "old.json"
    new_path = tmp_path / "new.json"
    old_path.write_text(dumps_eval_set(forge(load_spec(TICKET_SPEC), n=8, seed=1)), encoding="utf-8")
    new = forge(load_spec(TICKET_SPEC), n=9, seed=2)
    new_path.write_text(dumps_eval_set(new), encoding="utf-8")
    result = _run_cli("diff", "--old", str(old_path), "--new", str(new_path))
    assert result.returncode == 0, result.stderr
    assert "Added cases:" in result.stdout
    assert "Failure-mode coverage delta:" in result.stdout


def test_forge_rejects_non_positive_case_count() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        forge(load_spec(TICKET_SPEC), n=0)
