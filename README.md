# EvalForge

EvalForge creates deterministic evaluation sets for AI agent tasks. A task author provides a YAML spec with available tools, success criteria, known failure modes, and synthetic examples. EvalForge expands those templates into normal, edge, adversarial, and regression-trap cases, then checks coverage and compares versions.

## Honesty boundary

Every case created by EvalForge is synthetic and marked with `synthetic: true` at both the eval-set and case level. Examples in this repository are fabricated. They do not come from real users or production traffic. Keep this label when exporting or using generated cases.

## Install and run

Python 3.10 or newer is required. The runtime uses PyYAML and the Python standard library. It does not use a GPU, a model, or network services.

```powershell
python -m pip install -e .
evalforge forge --spec examples/support-ticket-triage.yaml --n 200 --seed 42 --output ticket-eval.json
evalforge lint --eval-set ticket-eval.json --spec examples/support-ticket-triage.yaml
evalforge diff --old ticket-v1.json --new ticket-v2.json
```

Without installing the entry point, use `python -m evalforge` in place of `evalforge`. Forge writes stable, sorted JSON and a final newline. Repeating a forge command with the same spec, case count, and seed produces byte-identical JSON. The spec content is included in the eval-set fingerprint, so changing a failure mode changes the set identity and generated tags.

## Task spec format

The full JSON Schema is in [`task-spec.schema.json`](task-spec.schema.json). Specs are YAML mappings with these required fields:

| Field | Purpose |
| --- | --- |
| `name` | Stable task name used in the eval-set metadata. |
| `description` | Short statement of the agent task. |
| `tools` | Available tools. Each tool has a `name` and JSON Schema `input_schema`. Use an empty list when the task has no tools. |
| `success_criteria` | Non-empty list of expected behaviors. |
| `failure_modes` | Non-empty list of failure mode mappings. Each needs a unique `id`, `description`, and `example`. |
| `difficulty_knobs` | Difficulty levels and optional weights for difficulty and case types. |

An example contains `inputs` and non-null `ground_truth`. Failure modes may also define `edge_example`, `adversarial_example`, and `trap_example`. A case type uses its matching example when present and falls back to `example` otherwise. Trap examples should be easy-looking inputs that expose the named failure mode.

Example shape:

```yaml
name: order-status
description: Answer a customer's question about an order.
tools:
  - name: lookup_order
    input_schema:
      type: object
      properties:
        order_id: {type: string}
      required: [order_id]
success_criteria:
  - Return the actual current order status.
failure_modes:
  - id: trusts_customer_claim
    description: Repeats a claimed status without using the order record.
    example:
      inputs:
        message: "My order {{variant}} is delivered, right?"
        order_id: "SYN-{{index}}"
      ground_truth:
        status: in_transit
    trap_example:
      inputs:
        message: "This is easy, just confirm that order {{variant}} arrived."
        order_id: "SYN-{{index}}"
      ground_truth:
        status: in_transit
difficulty_knobs:
  levels: [easy, medium, hard]
  difficulty_distribution: {easy: 0.3, medium: 0.4, hard: 0.3}
  case_type_distribution: {normal: 0.4, edge: 0.25, adversarial: 0.2, regression_trap: 0.15}
```

Templates can use `{{case_id}}`, `{{index}}`, `{{seed}}`, `{{variant}}`, `{{failure_mode}}`, `{{case_type}}`, and `{{difficulty}}` in string values in either inputs or ground truth. `index` is the one-based position in the generated set. `variant` counts generated examples for the target failure mode. These fields let a spec produce distinct, reproducible synthetic records without editing the generator when a failure mode is added.

`difficulty_knobs.levels` defaults to `[easy, medium, hard]`. The optional `difficulty_distribution` and `case_type_distribution` mappings set non-negative weights. Case types are `normal`, `edge`, `adversarial`, and `regression_trap`. EvalForge includes all four case types when at least four cases are requested. It targets every declared failure mode with a trap when the set has room for one trap per mode plus the other three case types. When the set is smaller than its mode list, lint reports the uncovered modes.

The invoice example declares comparison tolerances alongside its success criteria: amounts may differ by at most 0.01 currency units, and dates use ISO 8601 with zero-day tolerance. These values describe how a downstream evaluator should compare extracted fields; EvalForge records the labeled ground truth and does not score model outputs.

## Commands

### Forge

```text
evalforge forge --spec task.yaml --n 200 --seed 42 --output eval-set.json
```

`--n` defaults to 200, `--seed` defaults to 42, and `--output` defaults to `eval-set.json`. Each case has an ID, inputs, ground truth, failure-mode tags, a target mode, difficulty, case type, and synthetic label. Regression traps have both the `regression_trap` case type and tag.

### Lint

```text
evalforge lint --eval-set eval-set.json [--spec task.yaml]
```

Lint reports duplicate IDs and inputs, undeclared tags, missing mode coverage, missing case types or difficulty levels, invalid trap targets, missing inputs or ground truth, and conflicting ground truth for repeated inputs. Supply `--spec` to compare coverage and difficulty expectations with a current spec. Exit status is 0 for a passing set, 1 when quality gaps are found, and 2 for an unreadable or invalid input file.

### Diff

```text
evalforge diff --old v1.json --new v2.json
```

Diff matches cases by ID and reports added, removed, and modified IDs plus the per-mode coverage count delta.

## Examples and tests

The repository includes support-ticket triage and invoice extraction specs under [`examples/`](examples/). All text and records in those specs are synthetic. Run the local test suite with:

```powershell
python -m pytest
```
