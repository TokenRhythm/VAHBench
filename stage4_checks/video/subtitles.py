"""Compare cue text in its own time window; coarse ASR stays uncertain.

Word timestamps are preferred. A segment crossing a cue boundary cannot
locate a substring inside that segment, so it cannot certify alignment.
"""
from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher


def _tokens(text):
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    return re.findall(r"[\u3400-\u9fff]|[^\W_]+(?:'[^\W_]+)*", text)


def _text_spans(text, units, min_match=0.8):
    want = _tokens(text)
    words, owners = [], []
    for j, unit in enumerate(units):
        tokens = _tokens(unit.get("text", ""))
        words.extend(tokens)
        owners.extend((j, i, len(tokens)) for i in range(len(tokens)))
    if not want or not words:
        return
    # Preserve order and multiplicity: repeating one word must not satisfy
    # every occurrence of that word in the cue. Short phrases require exactness.
    n = len(want)
    exact = [i for i in range(len(words) - n + 1) if words[i:i + n] == want]
    if exact:
        for i in exact:
            first, last = owners[i], owners[i + n - 1]
            yield units[first[0]], units[last[0]], first[1] == 0 and last[1] == last[2] - 1
        return
    need = n if n < 5 else math.ceil(n * min_match)
    for i in range(len(words)):
        for size in range(need, n + (n - need) + 1):
            candidate = words[i:i + size]
            blocks = [b for b in SequenceMatcher(None, want, candidate, autojunk=False).get_matching_blocks() if b.size]
            if sum(b.size for b in blocks) >= need:
                first = owners[i + blocks[0].b]
                last = owners[i + blocks[-1].b + blocks[-1].size - 1]
                # A phrase inside a coarse segment has no recoverable onset
                # or offset. Do not invent uniform word times for it.
                bounded = first[1] == 0 and last[1] == last[2] - 1
                yield units[first[0]], units[last[0]], bounded


def text_matches(text, units, min_match=0.8):
    return next(_text_spans(text, units, min_match), None) is not None


def _span(item, allow_point=False):
    try:
        a, b = float(item["start"]), float(item["end"])
    except (KeyError, TypeError, ValueError):
        return None
    valid = 0 <= a <= b if allow_point else 0 <= a < b
    return (a, b) if math.isfinite(a) and math.isfinite(b) and valid else None


def align_cues(cues, transcript, *, tolerance_s=0.5, min_text_match=0.8):
    units = []
    for seg in transcript.get("segments", []):
        words = seg.get("words") or []
        if words and all(_span(w, allow_point=True) and w.get("word", "").strip() for w in words):
            units.extend({**w, "text": w["word"], "word_timed": True} for w in words)
        else:
            units.append(seg)
    valid = []
    for unit in units:
        span = _span(unit, allow_point=unit.get("word_timed", False))
        if span:
            valid.append({**unit, "start": span[0], "end": span[1]})
    rows = []
    for cue in cues:
        span = _span(cue)
        state = "unknown"
        if span and valid:
            a, b = span
            contained = [u for u in valid if u["start"] >= a - tolerance_s
                         and u["end"] <= b + tolerance_s]
            overlap = [u for u in valid if u["end"] > a - tolerance_s
                       and u["start"] < b + tolerance_s]
            matches = list(_text_spans(cue["text"], contained, min_text_match))
            overlap_matches = list(_text_spans(cue["text"], overlap, min_text_match))
            if any(bounded and abs(first["start"] - a) <= tolerance_s
                   and abs(last["end"] - b) <= tolerance_s
                   for first, last, bounded in matches):
                state = "aligned"
            elif matches and all(bounded for _, _, bounded in matches):
                state = "mistimed"  # also catches a subtitle held far too long
            elif any(bounded and first.get("word_timed") and last.get("word_timed")
                     for first, last, bounded in overlap_matches):
                state = "mistimed"
            elif any(bounded and last["end"] - first["start"] <= b - a + 2 * tolerance_s
                     for first, last, bounded in overlap_matches):
                state = "mistimed"  # a whole short ASR segment has usable bounds
            elif overlap_matches:
                state = "unknown"  # a broad segment cannot place its words
            elif len(valid) != len(units):
                state = "unknown"  # text may belong to an untimed segment
            elif text_matches(cue["text"], units, min_text_match):
                state = "mistimed"
            else:
                state = "unheard"
        rows.append({"text": cue["text"], "start": cue.get("start"),
                     "end": cue.get("end"), "status": state})
    return rows


def _ordered_token_pairs(caption, spoken, weights):
    """One-to-one word matches in order, preferring nearby speech times."""
    nc = len(caption)
    # Weighted LCS: order and repeated-word multiplicity both matter. Store
    # only directions and two numeric rows, not a matrix of Python floats.
    directions = [bytearray(len(spoken) + 1) for _ in range(nc + 1)]
    prev = [0.0] * (len(spoken) + 1)
    for i, (word, ci) in enumerate(caption, 1):
        cur = [0.0] * (len(spoken) + 1)
        for j, (heard, ui) in enumerate(spoken, 1):
            best, direction = (prev[j], 1) if prev[j] >= cur[j - 1] else (cur[j - 1], 2)
            if word == heard:
                value = prev[j - 1] + weights[ci, ui]
                if value > best:
                    best, direction = value, 3
            cur[j], directions[i][j] = best, direction
        prev = cur
    pairs = []
    i, j = nc, len(spoken)
    while i and j:
        direction = directions[i][j]
        if direction == 3:
            pairs.append((caption[i - 1][1], spoken[j - 1][1]))
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1

    return pairs


def score_alignment(cues, transcript, *, tolerance_s=0.5, min_text_match=0.8,
                    max_cue_s=7.0):
    """Bidirectional, one-to-one timed word credit.

    An ordered alignment consumes each subtitle/ASR token at most once.
    Missing speech, invented subtitles and repeated cues all remain in the
    denominator. Boundary error has a continuous cost beyond the tolerance;
    a subtitle held throughout a video cannot collect full credit.
    Coarse segments can certify only a whole segment assigned to one cue.
    """
    # 1. Normalise the subtitle and ASR populations, preserving every token.
    cues = sorted(cues, key=lambda c: (_span(c) or (float("inf"),))[0])
    rows = align_cues(cues, transcript, tolerance_s=tolerance_s,
                      min_text_match=min_text_match)
    caption = [(word, ci) for ci, cue in enumerate(cues) for word in _tokens(cue["text"])]
    spoken, units = [], []
    for seg in transcript.get("segments", []):
        words = seg.get("words") or []
        fine = words and all(_span(w, allow_point=True) and _tokens(w.get("word", "")) for w in words)
        for unit in (words if fine else [seg]):
            toks = _tokens(unit.get("word" if fine else "text", ""))
            ui = len(units)
            units.append({"span": _span(unit, allow_point=bool(fine)),
                          "fine": bool(fine), "n": len(toks)})
            spoken.extend((word, ui) for word in toks)
    # Untimed transcript text cannot certify alignment, but must not vanish
    # from the recall denominator when a provider omits its segment records.
    ns = max(len(spoken), len(_tokens(transcript.get("text", ""))))
    nc = len(caption)
    if not nc or not ns or not rows or all(r["status"] == "unknown" for r in rows):
        return {"score": None, "rows": rows, "subtitle_words": nc, "speech_words": ns}

    # 2. Match words once, in order, weighting by time proximity.
    cue_spans = [_span(c) for c in cues]
    weights = {}
    for ci, cs in enumerate(cue_spans):
        for ui, unit in enumerate(units):
            us = unit["span"]
            if cs is None or us is None:
                weights[ci, ui] = 0.0
            else:
                distance = max(cs[0] - us[1], us[0] - cs[1], 0.0)
                weights[ci, ui] = 1.0 / (1.0 + distance / max(tolerance_s, 0.001))

    pairs = _ordered_token_pairs(caption, spoken, weights)

    # 3. Certify coarse segments, then discount boundary/display-time errors.
    allocations = {}
    for ci, ui in pairs:
        allocations.setdefault(ui, []).append(ci)
    credited = 0.0
    for ci, cs in enumerate(cue_spans):
        matched = [ui for owner, ui in pairs if owner == ci and (
            units[ui]["fine"] or (len(allocations[ui]) == units[ui]["n"]
                                   and len(set(allocations[ui])) == 1))]
        quality = 0.0
        if matched and cs is not None:
            a = min(units[ui]["span"][0] for ui in matched)
            b = max(units[ui]["span"][1] for ui in matched)
            error = max(0.0, abs(cs[0] - a) - tolerance_s) + max(0.0, abs(cs[1] - b) - tolerance_s)
            union = max(cs[1], b) - min(cs[0], a)
            quality = max(0.0, 1.0 - error / max(union, 1e-9))
            # Bounding the whole speech span is insufficient: concatenating
            # an episode into one caption must not erase all internal timing
            # errors. Limit credit in proportion to excessive display time.
            quality *= min(1.0, max_cue_s / (cs[1] - cs[0]))
            if cs[1] - cs[0] > max_cue_s and rows[ci]["status"] == "aligned":
                rows[ci]["status"] = "mistimed"
        credited += len(matched) * quality
        rows[ci]["matched_words"] = len(matched)
        rows[ci]["timing_credit"] = round(quality, 6)
    return {"score": 2.0 * credited / (nc + ns), "rows": rows,
            "subtitle_words": nc, "speech_words": ns, "credited_words": credited,
            "precision": credited / nc, "recall": credited / ns}
