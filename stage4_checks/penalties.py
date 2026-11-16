"""Explicit prerequisite ownership, applied before merging different episodes."""
from shared.types import Measurement
from shared.types import Verdict


def dependent(key, owner, reason, **detail):
    return Measurement(key, Verdict.NOT_APPLICABLE, reason=reason,
                       detail={**detail, "suppressed_by": owner})


def normalize_episode(ms, ledger=None):
    """Missing audio belongs to the track check, not every downstream metric."""
    track = ms.get("spec.audio_track")
    if track and track.verdict in (Verdict.FAIL, Verdict.VOID):
        for key in ("av.clipping", "av.dynamic_range", "av.long_silence",
                    "av.voice_face_match", "av.speaker_consistency"):
            old = ms.get(key)
            if old is not None:
                ms[key] = dependent(key, track.key, "音轨缺失/全静音已计入音轨合规；此项不重复计分",
                                    **{k: v for k, v in old.detail.items() if k != "suppressed_by"})
                if ledger is not None:
                    ledger.add("penalty_prerequisite", [track.key], key, "annotate",
                               "缺失音轨的从属测量不再扣分")
    resolution, aspect = ms.get("spec.resolution"), ms.get("spec.aspect")
    if resolution and aspect and resolution.verdict is Verdict.FAIL and aspect.verdict is Verdict.FAIL:
        ms[aspect.key] = dependent(aspect.key, resolution.key,
                                  "同一画面尺寸错误已计入分辨率，不重复扣画幅", **aspect.detail)
        if ledger is not None:
            ledger.add("penalty_prerequisite", [resolution.key], aspect.key, "annotate",
                       "同一输出几何错误只扣一次")
