"""Finite, forward intervals on the delivered media clock."""
import math


def union_intervals(spans, duration=None):
    valid = []
    for span in spans or []:
        try:
            a, b = map(float, span)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(a) or not math.isfinite(b) or b <= a:
            continue
        a, b = max(0.0, a), max(0.0, b)
        if duration is not None:
            a, b = min(a, duration), min(b, duration)
        if b > a:
            valid.append((a, b))
    merged = []
    for a, b in sorted(valid):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def union_seconds(spans, duration=None):
    return sum(b - a for a, b in union_intervals(spans, duration))


def repeat_evidence(report, duration=None):
    """Shot similarity is a candidate; only a successful temporal scan proves seconds."""
    deg = (report or {}).get("degeneracy") or {}
    scan = deg.get("repeats") or {}
    raw = scan.get("duplicate_union")
    reason = scan.get("abstained") or ""
    if not isinstance(raw, list):
        reason = reason or "时间轴区间缺失"
    elif any(not isinstance(s, (list, tuple)) or len(s) != 2 or
             not union_intervals([s], duration) for s in raw):
        reason = reason or "时间轴区间无效"
    elif not raw and scan.get("duplicate_seconds", 0):
        reason = reason or "重复秒数缺少时间轴区间"
    spans = union_intervals(raw, duration) if not reason else []
    return {"spans": spans, "seconds": union_seconds(spans), "abstained": reason,
            "n_shot_pairs": len(deg.get("near_duplicate_shots") or []),
            "n_runs": scan.get("n_spans", len(spans))}
