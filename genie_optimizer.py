"""Helpers for the genie-agent-optimizer skill.

Load in a Python session with:
    exec(open("/Workspace/Users/james.graham@databricks.com/.assistant/skills/genie-agent-optimizer/genie_optimizer.py").read())

All experiments run on the developer's working copy (get_or_create_working_copy); the original is only
changed by promote_to_original, after explicit user confirmation.
"""
import json, time, uuid, copy
from concurrent.futures import ThreadPoolExecutor
from databricks.sdk import WorkspaceClient
import mlflow
from mlflow.genai.judges import make_judge
from mlflow.entities import AssessmentSource, AssessmentSourceType
from typing import Literal

w = WorkspaceClient()
_API = "/api/2.0/genie/spaces"
TERMINAL = {"DONE", "EVALUATION_FAILED", "EVALUATION_CANCELLED", "EVALUATION_TIMEOUT"}

# ---------------------------------------------------------------- space config
def get_space(space_id):
    """Return (space_metadata, serialized_space_dict). Needs CAN EDIT."""
    r = w.api_client.do("GET", f"{_API}/{space_id}", query={"include_serialized_space": "true"})
    return r, json.loads(r["serialized_space"])

def new_id():
    """32-hex id for new instructions / examples / snippets."""
    return uuid.uuid4().hex

def _sort_ids(obj):
    """Genie expects id-bearing lists sorted by id; sort recursively."""
    if isinstance(obj, dict):
        return {k: _sort_ids(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [_sort_ids(v) for v in obj]
        if items and all(isinstance(i, dict) and "id" in i for i in items):
            items = sorted(items, key=lambda i: i["id"])
        return items
    return obj

def validate_config(cfg, baseline):
    """Cheap pre-flight checks before PATCH. Raises on known API rejections / unsafe edits."""
    ti = cfg.get("instructions", {}).get("text_instructions", [])
    if len(ti) > 1:
        raise ValueError("text_instructions must contain at most one item: append to text_instructions[0]['content']")
    if cfg.get("benchmarks") != baseline.get("benchmarks"):
        raise ValueError("benchmarks changed: never edit benchmarks while optimising")
    ids = []
    def walk(o):
        if isinstance(o, dict):
            if "id" in o and isinstance(o["id"], str): ids.append(o["id"])
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
    walk(cfg)
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate ids: {sorted(dup)[:5]}")

def put_space(space_id, serialized, etag=None, baseline=None):
    """Full replacement of the space config. Returns the new metadata (with etag).
    Pass baseline (the original config) to run validate_config first.
    Never edit `benchmarks` (that would game the score); its order is left as-is."""
    if baseline is not None:
        validate_config(serialized, baseline)
    s = {k: (v if k == "benchmarks" else _sort_ids(v)) for k, v in serialized.items()}
    body = {"serialized_space": json.dumps(s)}
    if etag:
        body["etag"] = etag
    return w.api_client.do("PATCH", f"{_API}/{space_id}", body=body)

def config_diff(before, after):
    """Short human-readable list of what changed between two configs."""
    out = []
    def walk(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{path}.{k}" if path else k)
        elif isinstance(a, list) and isinstance(b, list) and all(isinstance(x, dict) and "id" in x for x in a + b):
            ia, ib = {x["id"]: x for x in a}, {x["id"]: x for x in b}
            for i in ib.keys() - ia.keys(): out.append(f"ADDED   {path}[{i}]")
            for i in ia.keys() - ib.keys(): out.append(f"REMOVED {path}[{i}]")
            for i in ia.keys() & ib.keys(): walk(ia[i], ib[i], f"{path}[{i}]")
        elif a != b:
            out.append(f"CHANGED {path}")
    walk(before, after, "")
    return out

# ---------------------------------------------------------------- working copy (one per developer per source agent)
COPY_MARKER = "[genie-optimizer]"

def _developer():
    me = w.current_user.me()
    return me.user_name, (me.display_name or me.user_name.split("@")[0])

def _marker(source_id, user_name):
    return f"{COPY_MARKER} source={source_id} developer={user_name}"

def _list_spaces():
    token, out = None, []
    while True:
        q = {"page_size": 100, **({"page_token": token} if token else {})}
        page = w.api_client.do("GET", _API, query=q)
        out += page.get("spaces", [])
        token = page.get("next_page_token")
        if not token:
            return out

def find_working_copy(source_id):
    """Return the caller's existing copy of source_id (or None). Matched on the description marker,
    so renaming the original or the copy does not create a second copy."""
    user_name, _ = _developer()
    mk = _marker(source_id, user_name)
    hits = sorted([s for s in _list_spaces() if mk in (s.get("description") or "")],
                  key=lambda s: s.get("create_time", ""))
    if len(hits) > 1:
        print(f"WARNING: {len(hits)} copies found for {mk}; using the oldest {hits[0]['space_id']}. "
              "Ask the user before trashing the others.")
    return hits[0] if hits else None

def get_or_create_working_copy(source_id):
    """Create the developer's copy of source_id once; on later runs reset it to the original's current state.

    Title: '<original title> [optimizer copy - <developer>]'. Lives in the developer's home folder.
    Returns {"copy_id", "title", "created", "source_config"}.
    Never modifies the original.
    """
    user_name, display = _developer()
    src_meta, src_cfg = get_space(source_id)
    if COPY_MARKER in (src_meta.get("description") or ""):
        raise ValueError("source_id is itself an optimizer copy: point at the original agent")
    title = f"{src_meta['title']} [optimizer copy - {display}]"
    desc = ((src_meta.get("description") or "") + "\n\n" + _marker(source_id, user_name)).strip()
    existing = find_working_copy(source_id)
    if existing is None:
        r = w.api_client.do("POST", _API, body={
            "title": title, "description": desc,
            "warehouse_id": src_meta["warehouse_id"],
            "serialized_space": src_meta["serialized_space"],
            "parent_path": f"/Workspace/Users/{user_name}"})
        copy_id, created = r["space_id"], True
    else:
        copy_id, created = existing["space_id"], False
    # Always PATCH: resets an existing copy, and on a new copy removes the join_specs Genie
    # auto-adds from UC PK/FK constraints at creation (create alone does not give an exact copy).
    cur, _ = get_space(copy_id)
    w.api_client.do("PATCH", f"{_API}/{copy_id}", body={
        "title": title, "description": desc,
        "warehouse_id": src_meta["warehouse_id"],
        "serialized_space": src_meta["serialized_space"],   # full reset to the original, benchmarks included
        "etag": cur["etag"]})
    _, copy_cfg = get_space(copy_id)
    drift = config_diff(src_cfg, copy_cfg)
    if drift:
        print("WARNING: copy differs from original after sync:", drift[:10])
    return {"copy_id": copy_id, "title": title, "created": created, "source_config": src_cfg}

def promote_to_original(source_id, copy_id, source_config):
    """Copy the optimised config (everything except benchmarks) from the copy to the ORIGINAL agent.
    Only call after the user explicitly confirms. source_config = wc["source_config"] from the sync.
    Refuses if the original changed since the sync (etag is a content hash, so compare content explicitly)."""
    _, copy_cfg = get_space(copy_id)
    src_meta, src_cfg = get_space(source_id)
    if config_diff(source_config, src_cfg):
        raise RuntimeError("Original agent changed since the copy was synced; re-run the process instead of overwriting.")
    new_cfg = copy.deepcopy(copy_cfg)
    new_cfg["benchmarks"] = src_cfg.get("benchmarks", {})
    put_space(source_id, new_cfg, etag=src_meta.get("etag"))
    _, after = get_space(source_id)
    return config_diff(src_cfg, after)

# ---------------------------------------------------------------- native benchmark
def run_native_benchmark(space_id, question_ids=None, timeout_s=1800, poll_s=10):
    """Start a Genie benchmark eval run, wait, and return per-question results."""
    body = {"benchmark_question_ids": question_ids} if question_ids else {}
    run = w.api_client.do("POST", f"{_API}/{space_id}/eval-runs", body=body)
    rid = run["eval_run_id"]
    start = time.time()
    while run.get("eval_run_status") not in TERMINAL:
        if time.time() - start > timeout_s:
            raise TimeoutError(f"eval run {rid} still {run.get('eval_run_status')}")
        time.sleep(poll_s)
        run = w.api_client.do("GET", f"{_API}/{space_id}/eval-runs/{rid}")
    if run["eval_run_status"] != "DONE":
        raise RuntimeError(f"eval run {rid} ended {run['eval_run_status']}")
    results, token = [], None
    while True:
        q = {"page_size": 100, **({"page_token": token} if token else {})}
        page = w.api_client.do("GET", f"{_API}/{space_id}/eval-runs/{rid}/results", query=q)
        for r in page.get("eval_results", []):
            d = w.api_client.do("GET", f"{_API}/{space_id}/eval-runs/{rid}/results/{r['result_id']}")
            actual = [x.get("response") for x in d.get("actual_response", []) if x.get("response_type") == "SQL"]
            results.append({
                "question_id": r["benchmark_question_id"],
                "question": r.get("question"),
                "assessment": d.get("assessment"),            # GOOD / BAD / NEEDS_REVIEW
                "reasons": d.get("assessment_reasons", []),
                "expected_sql": r.get("benchmark_answer"),
                "actual_sql": actual[0] if actual else None,
            })
        token = page.get("next_page_token")
        if not token:
            break
    n = run.get("num_questions") or len(results)
    return {
        "eval_run_id": rid,
        "num_questions": n,
        "num_correct": run.get("num_correct", 0),
        "num_needs_review": run.get("num_needs_review", 0),
        "accuracy": (run.get("num_correct", 0) / n) if n else 0.0,
        "results": results,
    }

# ---------------------------------------------------------------- written answers (Conversation API)
def ask_genie(space_id, question, timeout_s=300, poll_s=3):
    """Ask one question in a fresh conversation; return the written text answer."""
    r = w.api_client.do("POST", f"{_API}/{space_id}/start-conversation", body={"content": question})
    cid, mid = r["conversation_id"], r["message_id"]
    start = time.time()
    while True:
        m = w.api_client.do("GET", f"{_API}/{space_id}/conversations/{cid}/messages/{mid}")
        if m.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "QUERY_RESULT_EXPIRED"):
            break
        if time.time() - start > timeout_s:
            break
        time.sleep(poll_s)
    texts = [a["text"]["content"] for a in m.get("attachments", []) or [] if "text" in a and a["text"].get("content")]
    return {"question": question, "status": m.get("status"), "response": "\n\n".join(texts),
            "conversation_id": cid, "message_id": mid}

def collect_answers(space_id, questions, max_workers=3):
    """Ask each benchmark question in parallel (keep workers low: Genie rate limits)."""
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda q: ask_genie(space_id, q), questions))

# ---------------------------------------------------------------- tone judge
DEFAULT_TONE_GUIDELINES = (
    "Professional, concise and confident. Leads with the direct answer, then key supporting figures. "
    "Uses plain business language (no SQL or column-name jargon), states units/currency and dates, "
    "notes caveats briefly, and never invents data that is not in the result."
)

# Tone is a labelled 1-5 scale, not a bare int: the MLflow UI renders a numeric 1 as "Pass".
TONE_LABELS = ("1 - poor", "2 - weak", "3 - acceptable", "4 - good", "5 - excellent")

def _tone_num(v):
    """'4 - good' -> 4.0; also accepts plain numbers (older runs). None if unparseable."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    try:
        return float(str(v).strip().split()[0])
    except (ValueError, IndexError):
        return None

def make_tone_judge(guidelines=DEFAULT_TONE_GUIDELINES, model="databricks:/databricks-claude-sonnet-4-6"):
    return make_judge(
        name="tone",
        instructions=(
            "You are grading the TONE and WRITING QUALITY of a data assistant's written answer, "
            "not whether the numbers are correct.\n"
            f"Target tone: {guidelines}\n\n"
            "User question: {{ inputs }}\n"
            "Assistant written answer: {{ outputs }}\n\n"
            "Pick exactly one label: '5 - excellent' = fully matches the target tone; '4 - good' = minor issues; "
            "'3 - acceptable' = clear issues; '2 - weak' = major issues; "
            "'1 - poor' = wrong tone, confusing, or no written answer. Explain the main issue in one or two sentences."
        ),
        feedback_value_type=Literal["1 - poor", "2 - weak", "3 - acceptable", "4 - good", "5 - excellent"],
        model=model,
    )

def _log_benchmark_verdicts(trace_by_q, native):
    """Attach the native Genie verdict to each question's trace as a pass/fail 'genie_benchmark'
    assessment (pass only if GOOD; BAD and NEEDS_REVIEW fail, matching accuracy)."""
    if not native:
        return
    for r in native["results"]:
        tid = trace_by_q.get(r["question"])
        if not tid:
            continue
        reasons = ", ".join(r.get("reasons") or []) or "none"
        mlflow.log_feedback(
            trace_id=tid, name="genie_benchmark", value=(r["assessment"] == "GOOD"),
            rationale=f"Genie eval run {native['eval_run_id']}: {r['assessment']} (reasons: {reasons})",
            source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="genie_eval_run"))

def judge_tone(answers, judge, experiment_path, run_name, native=None):
    """Score written answers with the tone judge via mlflow.genai.evaluate. Returns mean (1-5) and rows.
    Pass native (a run_native_benchmark result) to also log each question's Genie verdict on its trace."""
    mlflow.set_experiment(experiment_path)
    data = [{"inputs": {"question": a["question"]},
             "outputs": {"response": a["response"] or "(no written answer)"}} for a in answers]
    with mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None) as run:
        res = mlflow.genai.evaluate(data=data, scorers=[judge])
        df = res.tables["eval_results"]
        by_q, trace_by_q = {}, {}
        for _, r in df.iterrows():
            q = (r.get("request") or {}).get("question")
            fb = next((x for x in (r.get("assessments") or []) if x.get("assessment_name") == "tone"), {})
            by_q[q] = {"score": _tone_num((fb.get("feedback") or {}).get("value", r.get("tone/value"))),
                       "rationale": fb.get("rationale")}
            if r.get("trace_id"):
                trace_by_q[q] = r.get("trace_id")
        _log_benchmark_verdicts(trace_by_q, native)
        rows = []
        for a in answers:
            hit = by_q.get(a["question"], {})
            rows.append({"question": a["question"], "response": a["response"],
                         "score": hit.get("score"), "rationale": hit.get("rationale")})
        scores = [r["score"] if r["score"] is not None else 1.0 for r in rows]  # missing score counts as 1
        tone_mean = sum(scores) / len(scores) if scores else 0.0
        mlflow.log_metric("tone_mean", tone_mean)   # labels have no built-in mean, so log it explicitly
        if native:
            mlflow.log_metric("genie_benchmark_accuracy", native["accuracy"])
    return {"tone_mean": tone_mean, "rows": rows, "mlflow_run_id": run.info.run_id}

# ---------------------------------------------------------------- full benchmark + scoring
def composite(accuracy, tone_mean, w_acc=0.7, w_tone=0.3):
    """0-1 score: weighted benchmark accuracy + normalised tone (1-5 -> 0-1)."""
    return w_acc * accuracy + w_tone * max(0.0, (tone_mean - 1) / 4)

def default_experiment(space_id):
    """Flat path in the user's home (MLflow will not create parent folders)."""
    return f"/Users/{w.current_user.me().user_name}/genie_optimizer_{space_id}"

def benchmark(space_id, label, experiment_path, judge=None, repeats=1):
    """Native benchmark + tone judge. Logs to MLflow (config snapshot as artifact). repeats>1 averages."""
    judge = judge or make_tone_judge()
    meta, cfg = get_space(space_id)
    questions = [q["question"][0] for q in cfg.get("benchmarks", {}).get("questions", [])]
    if not questions:
        raise ValueError("Space has no benchmark questions: add some in the Genie UI first.")
    mlflow.set_experiment(experiment_path)
    with mlflow.start_run(run_name=label) as parent:
        mlflow.log_dict(cfg, "serialized_space.json")
        mlflow.log_params({"space_id": space_id, "label": label, "repeats": repeats})
        natives, tones = [], []
        for i in range(repeats):
            natives.append(run_native_benchmark(space_id))
            answers = collect_answers(space_id, questions)
            tones.append(judge_tone(answers, judge, experiment_path, f"{label}-tone-{i}", native=natives[-1]))
        acc = sum(n["accuracy"] for n in natives) / repeats
        tone = sum(t["tone_mean"] for t in tones) / repeats
        score = composite(acc, tone)
        mlflow.log_metrics({"accuracy": acc, "tone_mean": tone, "composite": score})
        mlflow.set_tags({"genie_space_id": space_id,
                         "genie_eval_run_ids": ",".join(n["eval_run_id"] for n in natives),
                         "genie_space_url": f"{_host()}/genie/rooms/{space_id}"})
        mlflow.log_dict({"native": natives, "tone": tones}, "details.json")
    return {"label": label, "accuracy": acc, "tone_mean": tone, "composite": score,
            "native": natives[-1], "tone": tones[-1], "etag": meta.get("etag"),
            "config": cfg, "mlflow_run_id": parent.info.run_id}

def _host():
    return w.config.host.rstrip("/")

def run_links(b, experiment_path, space_id):
    """Direct links for one benchmark() result: MLflow experiment, MLflow run, Genie copy and its
    Benchmarks tab, plus the native eval_run_id(s) to pick in the Evaluations tab."""
    exp = mlflow.get_experiment_by_name(experiment_path)
    exp_url = f"{_host()}/ml/experiments/{exp.experiment_id}"
    return {
        "label": b["label"],
        "mlflow_experiment": exp_url,
        "mlflow_run": f"{exp_url}/runs/{b['mlflow_run_id']}",
        "genie_space": f"{_host()}/genie/rooms/{space_id}",
        "genie_benchmarks": f"{_host()}/genie/rooms/{space_id}/benchmarks",
        "genie_eval_run_id": b["native"]["eval_run_id"],
    }

def failures_summary(b):
    """Compact view of what to fix: failing benchmark questions + lowest-tone answers."""
    bad = [{k: r[k] for k in ("question", "assessment", "reasons", "expected_sql", "actual_sql")}
           for r in b["native"]["results"] if r["assessment"] != "GOOD"]
    low = sorted(b["tone"]["rows"], key=lambda r: r["score"] or 0)[:5]
    return {"failing_benchmarks": bad, "lowest_tone": low}

def is_improvement(new, best, min_gain=0.01):
    """Accept only if accuracy did not drop AND composite improved by min_gain."""
    return new["accuracy"] >= best["accuracy"] and new["composite"] >= best["composite"] + min_gain
