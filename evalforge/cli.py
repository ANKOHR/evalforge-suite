"""Command line interface for EvalForge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .core import EvalForgeError, diff_eval_sets, dumps_eval_set, forge, lint_eval_set, load_eval_set, load_spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evalforge", description="Forge and inspect synthetic agent-task eval sets.")
    commands = parser.add_subparsers(dest="command", required=True)

    forge_parser = commands.add_parser("forge", help="generate a deterministic eval set from a YAML task spec")
    forge_parser.add_argument("--spec", required=True, help="task spec YAML file")
    forge_parser.add_argument("--n", type=int, default=200, help="number of cases to generate (default: 200)")
    forge_parser.add_argument("--seed", type=int, default=42, help="generation seed (default: 42)")
    forge_parser.add_argument("--output", "-o", default="eval-set.json", help="output JSON path (default: eval-set.json)")

    lint_parser = commands.add_parser("lint", help="check duplicates, coverage, tags, and case quality")
    lint_parser.add_argument("--eval-set", required=True, help="eval set JSON file")
    lint_parser.add_argument("--spec", help="optional YAML spec to verify expected failure modes and difficulty levels")

    diff_parser = commands.add_parser("diff", help="compare two eval set versions")
    diff_parser.add_argument("--old", required=True, help="older eval set JSON file")
    diff_parser.add_argument("--new", required=True, help="newer eval set JSON file")
    return parser


def _show_diff(result: dict) -> None:
    print(f"Added cases: {len(result['added'])}")
    for case_id in result["added"]:
        print(f"  + {case_id}")
    print(f"Removed cases: {len(result['removed'])}")
    for case_id in result["removed"]:
        print(f"  - {case_id}")
    print(f"Modified cases: {len(result['modified'])}")
    for case_id in result["modified"]:
        print(f"  * {case_id}")
    print("Failure-mode coverage delta:")
    delta = result["coverage"]["delta"]
    if not delta:
        print("  No declared failure modes")
    for mode_id, amount in delta.items():
        print(f"  {mode_id}: {amount:+d}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "forge":
            eval_set = forge(load_spec(args.spec), n=args.n, seed=args.seed)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(dumps_eval_set(eval_set), encoding="utf-8", newline="\n")
            print(f"Forged {len(eval_set['cases'])} synthetic cases for {eval_set['task_name']}.")
            print(f"Eval set: {eval_set['eval_set_id']}")
            print(f"Wrote: {output}")
            return 0
        if args.command == "lint":
            eval_set = load_eval_set(args.eval_set)
            spec = load_spec(args.spec) if args.spec else None
            issues = lint_eval_set(eval_set, spec=spec)
            if issues:
                print(f"FAIL: {len(issues)} issue(s) found in {len(eval_set.get('cases', []))} cases.")
                for issue in issues:
                    print(f"- {issue}")
                return 1
            counts = {mode: 0 for mode in eval_set.get("declared_failure_modes", [])}
            for case in eval_set["cases"]:
                for tag in case.get("failure_mode_tags", []):
                    if tag in counts:
                        counts[tag] += 1
            print(f"PASS: {len(eval_set['cases'])} synthetic cases; all declared failure modes are covered.")
            for mode_id in sorted(counts):
                print(f"- {mode_id}: {counts[mode_id]} case(s)")
            return 0
        if args.command == "diff":
            _show_diff(diff_eval_sets(load_eval_set(args.old), load_eval_set(args.new)))
            return 0
    except (EvalForgeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
