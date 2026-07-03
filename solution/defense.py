"""
Pipeline fault detector: baseline thresholds plus stream-aware heuristics
for subtle faults that sit inside published 3-sigma bounds.
"""
import math

from api import Verdict

LINEAGE_WINDOW = 15
LINEAGE_SIGMA_K = 0.8
NULL_RATE_SUBTLE = 0.0075
STALENESS_SUBTLE = 6.0
EMBEDDING_CENTROID_SUBTLE = 0.028
EMBEDDING_AGE_SUBTLE = 35.0


def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def _tool_error(result):
    return isinstance(result, dict) and "error" in result


def _mean(values):
    return sum(values) / len(values)


def _pstdev(values):
    if len(values) < 2:
        return 0.0
    avg = _mean(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / len(values))


def _lineage_duration_spike(duration_ms, history):
    if len(history) < 5:
        return False
    window = history[-LINEAGE_WINDOW:]
    mean = _mean(window)
    stdev = _pstdev(window) or 1e-9
    return duration_ms > mean + LINEAGE_SIGMA_K * stdev


def check_data_batch(payload, ctx):
    profile = ctx.tools.batch_profile(payload["batch_id"])
    if _tool_error(profile):
        return Verdict(alert=False, pillar="checks")

    b = ctx.baseline
    row_count = profile["row_count"]
    null_rate = profile["null_rate"]["customer_id"]
    mean_amount = profile["mean_amount"]
    staleness = profile["staleness_min"]

    alert = (
        row_count < b["row_count_min"]
        or row_count > b["row_count_max"]
        or null_rate > b["null_rate_max"]
        or mean_amount < b["mean_amount_min"]
        or mean_amount > b["mean_amount_max"]
        or staleness > b["staleness_min_max"]
        or null_rate > NULL_RATE_SUBTLE
        or staleness > STALENESS_SUBTLE
    )
    return Verdict(alert=alert, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    diff = ctx.tools.contract_diff(
        payload["contract_id"], payload["checkpoint_batch_id"]
    )
    if _tool_error(diff):
        return Verdict(alert=False, pillar="contracts")

    b = ctx.baseline
    alert = bool(diff["violations"]) or diff["freshness_delay_min"] > b[
        "freshness_delay_max_min"
    ]
    return Verdict(alert=alert, pillar="contracts")


def check_lineage_run(payload, ctx):
    graph = ctx.tools.lineage_graph_slice(payload["run_id"])
    if _tool_error(graph):
        return Verdict(alert=False, pillar="lineage")

    b = ctx.baseline
    duration = graph["duration_ms"]
    upstream = graph["actual_upstream"]
    downstream = graph["actual_downstream_count"]
    history = ctx.state.setdefault("lineage_durations", [])

    alert = (
        duration > b["lineage_duration_ms_max"]
        or downstream == 0
        or "raw.customers" not in upstream
        or _lineage_duration_spike(duration, history)
    )
    history.append(duration)
    return Verdict(alert=alert, pillar="lineage")


def check_feature_materialization(payload, ctx):
    drift = ctx.tools.feature_drift(payload["feature_view"], payload["batch_id"])
    if _tool_error(drift):
        return Verdict(alert=False, pillar="ai_infra")

    alert = drift["mean_shift_sigma"] > ctx.baseline["feature_mean_shift_sigma_max"]
    return Verdict(alert=alert, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    drift = ctx.tools.embedding_drift(payload["corpus"], payload["chunk_batch_id"])
    if _tool_error(drift):
        return Verdict(alert=False, pillar="ai_infra")

    b = ctx.baseline
    centroid = drift["centroid_shift"]
    age = drift["avg_doc_age_days"]
    alert = (
        centroid > b["embedding_centroid_shift_max"]
        or age > b["corpus_avg_doc_age_days_max"]
        or centroid > EMBEDDING_CENTROID_SUBTLE
        or age > EMBEDDING_AGE_SUBTLE
    )
    return Verdict(alert=alert, pillar="ai_infra")
