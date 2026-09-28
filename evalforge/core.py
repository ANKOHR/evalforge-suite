"""Core task spec loading, deterministic generation, linting, and diffing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

CASE_TYPES = ("normal", "edge", "adversarial", "regression_trap")
_PLACEHOLDER = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")


class EvalForgeError(ValueError):
    """Raised when a task spec or eval set is malformed."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvalForgeError(message)


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a task spec from YAML."""
    try:
        spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvalForgeError(f"cannot read task spec: {exc}") from exc
    validate_spec(spec)
    return spec


def validate_spec(spec: Any) -> None:
    _require(isinstance(spec, dict), "task spec must be a YAML mapping")
    _require(isinstance(spec.get("name"), str) and spec["name"].strip(), "spec.name must be a non-empty string")
    _require(isinstance(spec.get("description"), str) and spec["description"].strip(), "spec.description must be a non-empty string")
    _require(isinstance(spec.get("tools"), list), "spec.tools must be a list")
    for index, tool in enumerate(spec["tools"]):
        _require(isinstance(tool, dict) and isinstance(tool.get("name"), str), f"spec.tools[{index}] needs a name")
        _require(isinstance(tool.get("input_schema"), dict), f"spec.tools[{index}].input_schema must be a mapping")
    criteria = spec.get("success_criteria")
    _require(isinstance(criteria, list) and len(criteria) > 0, "spec.success_criteria must be a non-empty list")
    modes = spec.get("failure_modes")
    _require(isinstance(modes, list) and len(modes) > 0, "spec.failure_modes must be a non-empty list")
    mode_ids: set[str] = set()
    for index, mode in enumerate(modes):
        prefix = f"spec.failure_modes[{index}]"
        _require(isinstance(mode, dict), f"{prefix} must be a mapping")
        mode_id = mode.get("id")
        _require(isinstance(mode_id, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", mode_id) is not None, f"{prefix}.id must be a non-empty identifier")
        _require(mode_id not in mode_ids, f"duplicate failure mode id: {mode_id}")
        mode_ids.add(mode_id)
        _require(isinstance(mode.get("description"), str) and mode["description"].strip(), f"{prefix}.description must be a non-empty string")
        _validate_example(mode.get("example"), f"{prefix}.example")
        for optional in ("edge_example", "adversarial_example", "trap_example"):
            if optional in mode:
                _validate_example(mode[optional], f"{prefix}.{optional}")
    knobs = spec.get("difficulty_knobs")
    _require(isinstance(knobs, dict), "spec.difficulty_knobs must be a mapping")
    levels = knobs.get("levels", ["easy", "medium", "hard"])
    _require(isinstance(levels, list) and levels and all(isinstance(item, str) and item for item in levels), "difficulty_knobs.levels must be a non-empty list of strings")
    _require(len(set(levels)) == len(levels), "difficulty_knobs.levels must not contain duplicates")
    _validate_weights(knobs.get("difficulty_distribution", {}), levels, "difficulty_distribution")
    _validate_weights(knobs.get("case_type_distribution", {}), CASE_TYPES, "case_type_distribution")


def _validate_example(example: Any, label: str) -> None:
    _require(isinstance(example, dict), f"{label} must be a mapping")
    _require("inputs" in example, f"{label}.inputs is required")
    _require("ground_truth" in example, f"{label}.ground_truth is required")
    _require(example["ground_truth"] is not None, f"{label}.ground_truth cannot be null")
    try:
        _canonical(example["inputs"])
        _canonical(example["ground_truth"])
    except (TypeError, ValueError) as exc:
        raise EvalForgeError(f"{label} must contain JSON-compatible values") from exc


def _validate_weights(value: Any, keys: tuple[str, ...] | list[str], label: str) -> None:
    if value is None:
        return
    _require(isinstance(value, dict), f"difficulty_knobs.{label} must be a mapping")
    unknown = sorted(set(value) - set(keys))
    _require(not unknown, f"difficulty_knobs.{label} has unknown keys: {', '.join(unknown)}")
    weights = []
    for key, weight in value.items():
        _require(isinstance(weight, (int, float)) and math.isfinite(weight) and weight >= 0, f"difficulty_knobs.{label}.{key} must be a non-negative number")
        weights.append(float(weight))
    if value:
        _require(sum(weights) > 0, f"difficulty_knobs.{label} weights must sum to more than zero")


def _render(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in context:
                raise EvalForgeError(f"unknown example template field: {key}")
            return str(context[key])

        rendered = _PLACEHOLDER.sub(replace, value)
        if "{{" in rendered or "}}" in rendered:
            raise EvalForgeError("malformed example template placeholder")
        return rendered
    if isinstance(value, list):
        return [_render(item, context) for item in value]
    if isinstance(value, dict):
        return {str(key): _render(item, context) for key, item in value.items()}
    return copy.deepcopy(value)


def _weighted_order(items: list[str], weights: dict[str, float], count: int, seed: int) -> list[str]:
    """Create a seeded, approximately weighted schedule with guaranteed coverage."""
    if count <= 0:
        return []
    effective = {item: float(weights.get(item, 1.0 if not weights else 0.0)) for item in items}
    total = sum(effective.values())
    if total <= 0:
        effective = {item: 1.0 for item in items}
        total = float(len(items))
    quotas = {item: effective[item] * count / total for item in items}
    allocated = {item: int(math.floor(quotas[item])) for item in items}
    remainder = count - sum(allocated.values())
    ranked = sorted(items, key=lambda item: (-(quotas[item] - allocated[item]), items.index(item)))
    for item in ranked[:remainder]:
        allocated[item] += 1
    schedule = [item for item in items for _ in range(allocated[item])]
    # A stable rotation gives variety without depending on global random state.
    if schedule:
        offset = seed % len(schedule)
        schedule = schedule[offset:] + schedule[:offset]
    return schedule


def forge(spec: dict[str, Any], n: int = 200, seed: int = 42) -> dict[str, Any]:
    """Generate a deterministic eval set from a validated task spec."""
    validate_spec(spec)
    _require(isinstance(n, int) and not isinstance(n, bool) and n > 0, "n must be a positive integer")
    _require(isinstance(seed, int) and not isinstance(seed, bool), "seed must be an integer")
    import random

    rng = random.Random(seed)
    modes = spec["failure_modes"]
    mode_ids = [mode["id"] for mode in modes]
    knobs = spec["difficulty_knobs"]
    levels = knobs.get("levels", ["easy", "medium", "hard"])

    # Guarantee one trap per declared failure mode whenever the requested set
    # is large enough, followed by at least one of each other case type.
    schedule: list[tuple[str, str]] = []
    trap_count = min(n, len(modes))
    schedule.extend(("regression_trap", mode_ids[index]) for index in range(trap_count))
    for case_type in ("normal", "edge", "adversarial"):
        if len(schedule) < n:
            mode_id = mode_ids[len(schedule) % len(mode_ids)]
            schedule.append((case_type, mode_id))

    remaining = n - len(schedule)
    extra_types = _weighted_order(list(CASE_TYPES), knobs.get("case_type_distribution", {}), remaining, seed)
    for case_type in extra_types:
        schedule.append((case_type, rng.choice(mode_ids)))

    difficulty_schedule = _weighted_order(list(levels), knobs.get("difficulty_distribution", {}), n, seed)
    # Ensure every configured difficulty is represented in ordinary eval sizes.
    for index, level in enumerate(levels[:min(n, len(levels))]):
        difficulty_schedule[index] = level

    mode_counts: Counter[str] = Counter()
    cases = []
    for index, (case_type, mode_id) in enumerate(schedule, start=1):
        mode = next(item for item in modes if item["id"] == mode_id)
        mode_counts[mode_id] += 1
        difficulty = difficulty_schedule[index - 1]
        example_key = {
            "normal": "example",
            "edge": "edge_example",
            "adversarial": "adversarial_example",
            "regression_trap": "trap_example",
        }[case_type]
        example = mode.get(example_key, mode["example"])
        context = {
            "case_id": f"EF-{index:06d}",
            "index": index,
            "seed": seed,
            "variant": mode_counts[mode_id],
            "failure_mode": mode_id,
            "case_type": case_type,
            "difficulty": difficulty,
        }
        tags = [mode_id]
        if case_type == "regression_trap":
            tags.append("regression_trap")
        cases.append(
            {
                "id": context["case_id"],
                "case_type": case_type,
                "inputs": _render(example["inputs"], context),
                "ground_truth": _render(example["ground_truth"], context),
                "failure_mode_tags": tags,
                "target_failure_mode": mode_id,
                "difficulty": difficulty,
                "synthetic": True,
            }
        )

    spec_fingerprint = hashlib.sha256(_canonical(spec).encode("utf-8")).hexdigest()
    set_payload = {"spec": spec_fingerprint, "n": n, "seed": seed}
    set_id = hashlib.sha256(_canonical(set_payload).encode("utf-8")).hexdigest()[:16]
    return {
        "format_version": 1,
        "eval_set_id": f"{spec['name']}:{set_id}",
        "version": "1.0.0",
        "task_name": spec["name"],
        "task_description": spec["description"],
        "seed": seed,
        "synthetic": True,
        "declared_failure_modes": mode_ids,
        "difficulty_levels": list(levels),
        "case_types": list(CASE_TYPES),
        "spec_digest": spec_fingerprint,
        "cases": cases,
    }


def dumps_eval_set(eval_set: dict[str, Any]) -> str:
    """Serialize an eval set as stable UTF-8 JSON with a final newline."""
    return json.dumps(eval_set, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_eval_set(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalForgeError(f"cannot read eval set: {exc}") from exc
    if not isinstance(value, dict):
        raise EvalForgeError("eval set must be a JSON object")
    return value


def lint_eval_set(eval_set: dict[str, Any], spec: dict[str, Any] | None = None) -> list[str]:
    """Return human-readable quality gaps. An empty list means the set passes."""
    issues: list[str] = []
    if eval_set.get("synthetic") is not True:
        issues.append("eval set must declare synthetic: true")
    cases = eval_set.get("cases")
    if not isinstance(cases, list) or not cases:
        return issues + ["eval set must contain at least one case"]

    expected_modes = [mode["id"] for mode in spec["failure_modes"]] if spec else eval_set.get("declared_failure_modes", [])
    if not isinstance(expected_modes, list) or any(not isinstance(mode, str) for mode in expected_modes):
        issues.append("declared_failure_modes must be a list of strings")
        expected_modes = []
    expected_difficulties = eval_set.get("difficulty_levels", [])
    if spec:
        expected_difficulties = spec["difficulty_knobs"].get("levels", ["easy", "medium", "hard"])
    if not isinstance(expected_difficulties, list):
        issues.append("difficulty_levels must be a list")
        expected_difficulties = []
    ids: set[str] = set()
    coverage: Counter[str] = Counter()
    inputs_seen: dict[str, tuple[str, Any]] = {}
    type_coverage: Counter[str] = Counter()
    difficulty_coverage: Counter[str] = Counter()

    for index, case in enumerate(cases):
        label = f"case at index {index}"
        if not isinstance(case, dict):
            issues.append(f"{label} must be an object")
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id:
            issues.append(f"{label} has no id")
        elif case_id in ids:
            issues.append(f"duplicate case id: {case_id}")
        else:
            ids.add(case_id)
        if case.get("synthetic") is not True:
            issues.append(f"{case_id or label} must declare synthetic: true")
        if "inputs" not in case or case.get("inputs") is None:
            issues.append(f"{case_id or label} has missing inputs")
        if "ground_truth" not in case or case.get("ground_truth") is None:
            issues.append(f"{case_id or label} has missing ground_truth")
        try:
            input_key = _canonical(case.get("inputs"))
            truth = case.get("ground_truth")
            if input_key in inputs_seen:
                previous_id, previous_truth = inputs_seen[input_key]
                if previous_truth != truth:
                    issues.append(f"ground-truth inconsistency: {previous_id} and {case_id} have the same inputs but different ground_truth")
                issues.append(f"duplicate inputs: {previous_id} and {case_id}")
            else:
                inputs_seen[input_key] = (str(case_id), truth)
        except (TypeError, ValueError):
            issues.append(f"{case_id or label} contains non-JSON values")
        tags = case.get("failure_mode_tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            issues.append(f"{case_id or label} failure_mode_tags must be a list of strings")
            tags = []
        for tag in tags:
            if tag in expected_modes:
                coverage[tag] += 1
            elif tag != "regression_trap":
                issues.append(f"{case_id or label} has undeclared failure-mode tag: {tag}")
        target = case.get("target_failure_mode")
        if target is not None and target not in expected_modes:
            issues.append(f"{case_id or label} targets undeclared failure mode: {target}")
        if case.get("case_type") == "regression_trap":
            if "regression_trap" not in tags:
                issues.append(f"{case_id or label} is a regression trap without the regression_trap tag")
            if target not in expected_modes:
                issues.append(f"{case_id or label} regression trap must target a declared failure mode")
            elif target not in tags:
                issues.append(f"{case_id or label} regression trap target {target} is missing from failure_mode_tags")
        case_type = case.get("case_type")
        if case_type not in CASE_TYPES:
            issues.append(f"{case_id or label} has unknown case_type: {case_type}")
        type_coverage[str(case_type)] += 1
        difficulty = case.get("difficulty")
        if isinstance(difficulty, str):
            difficulty_coverage[difficulty] += 1
            if expected_difficulties and difficulty not in expected_difficulties:
                issues.append(f"{case_id or label} has undeclared difficulty level: {difficulty}")

    for mode_id in expected_modes:
        if coverage[mode_id] == 0:
            issues.append(f"missing failure-mode coverage: {mode_id}")

    if len(cases) >= len(CASE_TYPES):
        for case_type in CASE_TYPES:
            if type_coverage[case_type] == 0:
                issues.append(f"missing case type: {case_type}")
    if len(cases) >= len(expected_difficulties):
        for level in expected_difficulties:
            if difficulty_coverage[str(level)] == 0:
                issues.append(f"missing difficulty level: {level}")
    return issues


def diff_eval_sets(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Compare versions by case ID and report declared failure-mode coverage."""
    old_cases = {case.get("id"): case for case in old.get("cases", []) if isinstance(case, dict) and case.get("id") is not None}
    new_cases = {case.get("id"): case for case in new.get("cases", []) if isinstance(case, dict) and case.get("id") is not None}
    old_ids, new_ids = set(old_cases), set(new_cases)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    modified = sorted(case_id for case_id in old_ids & new_ids if _canonical(old_cases[case_id]) != _canonical(new_cases[case_id]))

    def coverage(value: dict[str, Any]) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for case in value.get("cases", []):
            if isinstance(case, dict) and isinstance(case.get("failure_mode_tags"), list):
                counts.update(tag for tag in case["failure_mode_tags"] if isinstance(tag, str) and tag != "regression_trap")
        declared = value.get("declared_failure_modes", [])
        return {mode: counts.get(mode, 0) for mode in declared} if isinstance(declared, list) else dict(sorted(counts.items()))

    old_coverage = coverage(old)
    new_coverage = coverage(new)
    modes = sorted(set(old_coverage) | set(new_coverage))
    coverage_delta = {mode: new_coverage.get(mode, 0) - old_coverage.get(mode, 0) for mode in modes}
    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "coverage": {"old": old_coverage, "new": new_coverage, "delta": coverage_delta},
    }
