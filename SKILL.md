---
name: genie-agent-optimizer
description: Benchmark and iteratively optimise an existing Genie Agent (Genie space) on a per-developer working copy. Runs the native Genie benchmark plus an MLflow LLM judge on the tone of the written answers, proposes and applies one config change to the copy, re-benchmarks, and keeps it only if the score improves (up to 3 attempts), then reports. Read when the user asks to optimise, tune, improve, or benchmark-and-fix a Genie Agent / Genie space.
---

# Genie Agent optimiser

A closed loop: **sync working copy -> baseline -> propose one change -> apply -> re-benchmark -> keep or revert**, at most 3 attempts. You (Genie Code) do the diagnosis and choose the changes; the helper module does the API and MLflow plumbing.

**All benchmarking and changes happen on the developer's working copy, never on the original.**
* One copy per developer per original agent, titled `<original title> [optimizer copy - <developer name>]`, in the developer's home folder. Several people can run the process on the same agent without conflicting.
* The copy is found by a marker line in its description (`[genie-optimizer] source=<original id> developer=<email>`), not by title. Don't remove that line.
* On a re-run, the existing copy is reset to the original's current state (full config, benchmarks included) and work starts from there. Earlier experiment changes on the copy are discarded; they are still in MLflow.
* Never create a copy any other way, and never trash copies without the user's permission.

Helper: `genie_optimizer.py` in this folder (same directory as this SKILL.md). It is plain Python using `databricks-sdk` and `mlflow>=3.4` (serverless already has both).

## Inputs to confirm (ask only if missing)

* `space_id` of the **original** Genie Agent. If the user gives a name or URL, resolve with `GET /api/2.0/genie/spaces` (the id is the 32-hex part of `/genie/rooms/<id>`). The user needs CAN EDIT on it (to export its config). If they point at an optimizer copy, `get_or_create_working_copy` refuses: ask for the original.
* Tone guidelines for the judge. Default is `DEFAULT_TONE_GUIDELINES` (professional, concise, answer first, plain business language, units and dates). Use the user's wording if they give one.
* The space must already have benchmark questions (SQL answers). If none, stop and tell the user to add them in the Genie UI (Benchmarks tab); don't invent them.

Don't ask for confirmation before each attempt: the user asked for the loop by invoking this skill, and it only edits their own copy. Say up front which copy is used (title, and whether it was created or reset) and that every config version is saved in MLflow.

## Before Step 0: check you can run Python here

The helper needs Python with write access to the Genie API and MLflow. Some places Genie Code runs, such as the Genie agent editor, only allow read-only SQL. Check this before doing anything else:

* If your instructions limit code execution to read-only SQL, or a Python `executeCode` call is refused, do NOT attempt Step 0 or any later step. Do not clone the space, run benchmarks or edit instructions as a substitute.
* Instead, stop and tell the user, in two or three short sentences:
  * This optimiser has to run where Python is available, because the current editor only allows read-only SQL.
  * Open a notebook on serverless compute, open Genie Code there, and paste the prompt below.
* Fill in the prompt for them. If the user is viewing a Genie space, use its id (the 32-hex part of `/genie/rooms/<id>`). If it is an optimizer copy (its description contains `[genie-optimizer] source=<id>`), use the `source` id, i.e. the original:

  > Use my genie-agent-optimizer skill to optimise Genie space `<original space_id>`

  Add their tone guidelines to the prompt if they gave any.
* Then end the turn. Do not offer a reduced version of the loop unless the user asks for one.

## Step 0: load the helper

Import `genie_optimizer.py` as a normal module in one `executeCode` Python call (`default` session, serverless). Do not `exec()` the file content and do not paste it inline.

```python
import sys, importlib
from databricks.sdk import WorkspaceClient
_p = f"/Workspace/Users/{WorkspaceClient().current_user.me().user_name}/.assistant/skills/genie-agent-optimizer"
if _p not in sys.path:
    sys.path.insert(0, _p)
import genie_optimizer
importlib.reload(genie_optimizer)   # picks up edits to the file; harmless on first load
from genie_optimizer import *
```

The star import exposes every public helper plus `w`, `json`, `copy` and `mlflow`, so the snippets below work unchanged. Private names (`_tone_num`, `_host`, ...) are not imported; use `genie_optimizer._name` if you need one. All later steps run in that same session so variables persist. If the file is edited mid-session, re-run this block (reload, then the star import again).

## Step 1: sync the working copy and run the baseline

```python
SOURCE = "<original space_id>"
wc = get_or_create_working_copy(SOURCE)  # creates once per developer, else resets the copy to the original
SPACE = wc["copy_id"]                    # every later call uses the COPY
print(wc["title"], "created" if wc["created"] else "reset from original")
EXP = default_experiment(SOURCE)         # /Users/<me>/genie_optimizer_<original id>: history per original
judge = make_tone_judge()                # or make_tone_judge(guidelines="...")
baseline = benchmark(SPACE, "baseline", EXP, judge=judge)
ORIGINAL = copy.deepcopy(baseline["config"])   # copy's starting point = original's config; also in MLflow
best = baseline
print(baseline["accuracy"], baseline["tone_mean"], baseline["composite"])
print(run_links(baseline, EXP, SPACE))   # keep these for the report
print(json.dumps(failures_summary(baseline), indent=1, default=str))
```

What `benchmark()` does:
* Native benchmark: `POST /api/2.0/genie/spaces/{id}/eval-runs`, polls to `DONE`, reads each result's `assessment` (GOOD/BAD/NEEDS_REVIEW) and `assessment_reasons`. Accuracy = `num_correct / num_questions` (NEEDS_REVIEW counts as not correct).
* Tone: the eval-run results contain only SQL, not Genie's written answer. So it asks every benchmark question through the Conversation API (`start-conversation`, one fresh conversation each, 3 at a time), takes the `text` attachment, and scores it with an MLflow `make_judge` judge via `mlflow.genai.evaluate`. The judge returns a label (`1 - poor` ... `5 - excellent`), not a bare number, because the MLflow UI shows a numeric 1 as "Pass"; the helper converts labels back to 1-5 for `tone_mean`. No written answer scores 1.
* Each question's trace in MLflow also gets a `genie_benchmark` pass/fail assessment copied from the native eval run (pass only if GOOD; the rationale holds the eval run id, verdict and reasons), so correctness in MLflow matches the Genie benchmark run. The tone run logs `tone_mean` and `genie_benchmark_accuracy` metrics.
* Composite = 0.7 x accuracy + 0.3 x (tone - 1)/4, range 0-1.
* Logs a parent MLflow run with metrics `accuracy`, `tone_mean`, `composite`, the config snapshot and per-question details, and tags `genie_space_id`, `genie_eval_run_ids` and `genie_space_url` that tie the MLflow run to its native Genie benchmark run.

A run of 6 questions takes about 2-3 minutes. Keep the default 10-minute `executeCode` timeout for up to ~20 questions; raise it for bigger sets.

## Step 2: diagnose and choose ONE change

Use `failures_summary(best)`, plus `best["config"]`, to pick the single change most likely to raise the composite. Map symptoms to config parts:

| Symptom | Change |
| --- | --- |
| `LLM_JUDGE_MISSING_OR_INCORRECT_JOIN`, wrong tables | add or fix `instructions.join_specs`, add an `example_question_sqls` entry |
| `..._FILTER`, `EMPTY_RESULT`, entity lookup misses | text instruction on the filter or lookup rule, `sql_snippets.filters`, an example SQL that shows the pattern |
| `..._AGGREGATION`, `INCORRECT_METRIC_CALCULATION` | `sql_snippets.measures`, example SQL |
| `INCORRECT_TABLE_OR_FIELD_USAGE`, `WRONG_COLUMNS` | `data_sources.tables[].column_configs` descriptions/synonyms, text instruction |
| `INSTRUCTION_COMPLIANCE_OR_MISSING_BUSINESS_LOGIC` | rewrite the relevant text instruction to be explicit |
| Low tone scores (read the rationales) | a "Response style" section in the text instruction that addresses the issues the judge named |

Rules:
* Change one thing (or one tightly related group) per attempt so the result is attributable.
* Never edit `benchmarks`, and don't paste a benchmark's exact question+SQL into `example_question_sqls`: that games the score. Generalise the pattern (different entity, same shape).
* Don't touch tables, warehouse or permissions; don't drop existing instructions unless they are clearly the cause.
* Each attempt should be a different idea. If attempt N fails, use what its results showed.

## Step 3: apply, re-benchmark, keep or revert

Edit a deep copy of the best config, then:

```python
cand = copy.deepcopy(best["config"])
# --- example edits ---
# cand["instructions"]["text_instructions"][0]["content"].append("\n## Response style\n* Lead with the answer ...\n")
# cand["instructions"]["example_question_sqls"].append({"id": new_id(), "question": ["..."], "sql": ["SELECT ..."]})
print(config_diff(best["config"], cand))
meta_now, _ = get_space(SPACE)
put_space(SPACE, cand, etag=meta_now["etag"], baseline=ORIGINAL)   # validates first
attempt = benchmark(SPACE, f"attempt-{n}: <short change description>", EXP, judge=judge)
print(run_links(attempt, EXP, SPACE))
if is_improvement(attempt, best):
    best = attempt                       # keep; stop (or continue only if the user asked for more)
else:
    m, _ = get_space(SPACE)
    put_space(SPACE, best["config"], etag=m["etag"])    # revert before the next idea
```

`SPACE` is always the copy here. Never pass `SOURCE` to `put_space`.

Accept rule (`is_improvement`): accuracy must not drop AND composite must rise by at least 0.01. Stop on the first accepted attempt. Otherwise try a new change, up to 3 attempts in total.

API constraints (found in testing):
* `instructions.text_instructions` may hold **at most one item**. Append to `text_instructions[0]["content"]` (a list of strings); never add a second item. `validate_config` checks this.
* New ids: `new_id()` (32 hex). `put_space` sorts id lists, except `benchmarks`, which it leaves untouched.
* `PATCH` replaces the whole `serialized_space`; always send the full config, with the current `etag` (it's a hash of the content).
* Creating a space from an export isn't an exact copy: Genie auto-adds join_specs from UC PK/FK constraints. `get_or_create_working_copy` always follows the create with a PATCH, then checks `config_diff` against the original. If it prints `WARNING: copy differs`, report it, because results on the copy may not transfer.
* If a `put_space` call fails validation or the API rejects it, fix the config; that doesn't count as an attempt.

Noise: Genie isn't deterministic (e.g. 5/6 vs 6/6 on an unchanged space). If an accepted gain comes down to one question flipping, or the user wants more confidence, re-run with `repeats=2` before reporting it as an improvement.

## Step 4: verify the final state

* `get_space(SPACE)` and `config_diff(ORIGINAL, current)`. It should be empty if nothing was kept, or equal the accepted change only.
* Confirm `current["benchmarks"] == ORIGINAL["benchmarks"]`.
* Confirm the original is untouched: `config_diff(wc["source_config"], get_space(SOURCE)[1]) == []`.

## Step 5: report

Keep it short:
* Baseline vs each attempt: a table with accuracy (x/n), tone mean, composite, kept or reverted, and two link columns for every row, taken from `run_links(...)`: the MLflow run (`mlflow_run`) and the Genie benchmark run (`genie_benchmarks` link, with its `genie_eval_run_id`, which the user picks under Benchmarks > Evaluations in the copy).
* ALWAYS include these links, every time, even if the user didn't ask: the MLflow experiment (`mlflow_experiment`) and the working copy (`genie_space`). Write them as full clickable markdown links built from the workspace host (`w.config.host`), never as bare paths or ids only.
* The `/benchmarks` deep link is a best guess at the UI route. If the user reports it doesn't open the Benchmarks tab, link the copy (`genie_space`) instead and tell them to open Benchmarks > Evaluations and find the run by its `genie_eval_run_id` (the run's timestamp also matches the MLflow run's start time).
* For each attempt: what changed (from `config_diff` + your one-line description), why you chose it, what happened to the failing questions and to tone.
* Final state of the copy (with the improvement, or reverted to the original's config), with the links above (each MLflow run has `serialized_space.json` and `details.json`).
* If an attempt was kept, offer to promote it to the original. Only after the user explicitly says yes, run `promote_to_original(SOURCE, SPACE, wc["source_config"])`. It copies everything except benchmarks and refuses if someone changed the original since the sync. Report the diff it returns.
* If all 3 failed: the most likely reasons and what to try next (e.g. benchmark answer is wrong or ambiguous, needs a metric view or data change, needs more benchmark questions).
* Point out remaining suspected benchmark problems (expected SQL that looks wrong) instead of "fixing" the agent to match them.

## Rollback

To restore any version on the copy: `cfg = mlflow.artifacts.load_dict(f"runs:/{run_id}/serialized_space.json")`, then `put_space(SPACE, cfg, etag=get_space(SPACE)[0]["etag"])`. A fresh run of the skill resets the copy to the original anyway. Applying an old version to the original counts as a promotion and needs the user's explicit confirmation.
