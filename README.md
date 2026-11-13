# Genie Agent Optimizer

A Genie Code skill that benchmarks an existing Genie Agent (Genie space) and iteratively improves it. It combines the native Genie benchmark (SQL correctness) with an MLflow LLM judge that scores the tone of Genie's written answers. Then it tries one config change at a time on a per-developer working copy and keeps a change only if the score improves.

## Contents

| File | Purpose |
| --- | --- |
| `SKILL.md` | Instructions Genie Code follows: inputs, the optimisation loop, change heuristics, guardrails and report format. |
| `genie_optimizer.py` | Helper module (Genie REST API + MLflow plumbing) that Genie Code loads into a Python session. |
| `README.md` | This file. |

## Usage

1. Put this folder under `/Workspace/Users/<you>/.assistant/skills/` so Genie Code can see the skill.
2. Open Genie Code in a notebook on **serverless** compute. The skill needs Python. The Genie agent editor only allows read-only SQL, so the skill stops there and gives you a prompt to run from a notebook instead.
3. Ask:

   > Use my genie-agent-optimizer skill to optimise Genie space `<space_id>`

   Optionally add your own tone guidelines for the judge.

Requirements:
* CAN EDIT on the original Genie space (to export its config).
* The space already has benchmark questions with SQL answers (Benchmarks tab in the Genie UI). The skill never invents them.
* `databricks-sdk` and `mlflow>=3.4` (both preinstalled on serverless).
* Access to the judge model endpoint (default `databricks-claude-sonnet-4-6`).

## How it works

1. **Working copy.** The skill creates (once) or resets your personal copy of the original space, titled `<original title> [optimizer copy - <your name>]`, in your home folder. The copy is identified by a marker line in its description: `[genie-optimizer] source=<id> developer=<email>`. Each developer gets their own copy, so several people can optimise the same space without conflicting. All experiments run on the copy.
2. **Baseline.** Runs the native benchmark (`/eval-runs`) and asks each benchmark question through the Conversation API to get the written answer. An MLflow `make_judge` judge then scores each answer's tone.
3. **Diagnose and change.** Genie Code reads the failing questions and the judge's rationales, then makes **one** targeted change (text instruction, example SQL, SQL snippet, join spec, column description, or trusted function).
4. **Re-benchmark and decide.** A change is kept only if accuracy does not drop **and** the composite score rises by at least 0.01. Otherwise the copy is reverted. The loop stops at the first accepted change, with at most 3 attempts.
5. **Verify and report.** Checks the copy differs from the original only by the accepted change, that benchmarks are untouched and that the original is unchanged. Then it reports a results table with links to the MLflow experiment, each MLflow run and each Genie benchmark run.
6. **Promote (optional).** Copies the change to the original space only after you explicitly confirm. Promotion refuses to run if the original changed since the copy was synced.

### Scoring

* Accuracy = `num_correct / num_questions` from the native eval run (`NEEDS_REVIEW` counts as not correct).
* Tone = mean judge label, `1 - poor` to `5 - excellent`. A missing written answer scores 1.
* Composite = `0.7 x accuracy + 0.3 x (tone - 1) / 4` (range 0-1).

Genie is not deterministic, so a gain that comes down to one question flipping is re-checked with `repeats=2`.

### What gets logged to MLflow

Experiment: `/Users/<you>/genie_optimizer_<original space id>`. Each benchmark is a parent run with:
* metrics `accuracy`, `tone_mean`, `composite`
* artifacts `serialized_space.json` (the full config that was tested) and `details.json` (per-question results)
* tags `genie_space_id`, `genie_eval_run_ids`, `genie_space_url`, linking the run to its native Genie benchmark run

Each question's trace carries the `tone` assessment and a `genie_benchmark` pass/fail assessment copied from the native eval run.

## Guardrails

* Never edits the original space except through `promote_to_original`, after explicit user confirmation.
* Never edits `benchmarks`. It also never copies a benchmark's exact question and SQL into the example queries (that would game the score). Patterns are generalised instead.
* Doesn't touch tables, warehouse or permissions.
* One change per attempt, so every result can be traced to its change.

## Rollback

Every tested config is saved as `serialized_space.json` on its MLflow run. To restore one on the copy:

```python
cfg = mlflow.artifacts.load_dict(f"runs:/{run_id}/serialized_space.json")
put_space(SPACE, cfg, etag=get_space(SPACE)[0]["etag"])
```

Restoring a config on the original counts as a promotion and needs explicit confirmation.

## Genie API notes (found in testing)

* `instructions.text_instructions` accepts at most one item: append to its `content` list.
* `PATCH` replaces the whole `serialized_space`. Send the full config with the current `etag` (a content hash).
* Id-bearing lists must be sorted by id. `benchmarks` is returned in descending id order and is left as-is.
* Creating a space from an export auto-adds join specs from Unity Catalog PK/FK constraints, so the helper always follows the create with a `PATCH` to get an exact copy.
* Eval-run results contain only SQL, not the written answer, so tone uses the Conversation API.

## Known limitations

* No detection of concurrent writes to the same copy, for example from a second session of the same user. Check `config_diff` after a run if results look odd.
* Helper state (the working-copy sync snapshot) lives in the Python session. If the session restarts, rebuild the snapshot from the baseline run's `serialized_space.json` before promoting.
* Auto Mode may block `exec()` of files under `.assistant/`. Pasting the helper code into the session works as a fallback.
* The `/genie/rooms/<id>/benchmarks` deep link is unverified. If it doesn't open the Benchmarks tab, open Benchmarks > Evaluations in the space and find the run by its `genie_eval_run_id`.
