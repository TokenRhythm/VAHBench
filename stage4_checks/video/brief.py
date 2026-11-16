"""Expectations: the brief's demands, each verified three ways.

Every expectation compares three things the rest of the pipeline kept apart:

    the brief      what was asked            (BriefFacts)
    the trajectory what the agent planned    (artifacts it produced on the way)
    the delivery   what shipped              (transcript, faces, files)

and records which of the four outcomes it found -- satisfied, planned but not
delivered, delivered without a trace of planning, or ignored -- in the
measurement's detail, because "ignored" and "tried and failed" are different
findings even when both score zero.

Evidence is pulled from the bus on demand; nothing here assumes a module has
already run. Every cross-channel move -- a silence that voids a word count,
a sheet that seeds a face match, a subtitle held against a transcript -- is
written to the bus's flow ledger so the report can show it.
"""
from __future__ import annotations

import re
from pathlib import Path

from stage1_inputs.brief import BriefFacts
from stage1_inputs.brief import normalise
from stage4_checks.reports import _abstain
from stage4_checks.reports import _m
from stage4_checks.reports import dig
from shared.types import Scale
from shared.types import Verdict
from stage4_checks.penalties import dependent


MIN_SPEECH_WORDS = 5    # fewer ASR words than this is "no dialogue heard"


def _tri(planned: bool | None, delivered: bool | None) -> str:
    if planned is None or delivered is None:
        return "unknown"
    return {(True, True): "satisfied", (True, False): "planned_not_delivered",
            (False, True): "delivered_unplanned", (False, False): "ignored"}[(planned, delivered)]


def _worst(ms):
    """Per-episode measurements -> the worst one, tagged with which episode."""
    from stage5_scoring.node import DEFAULT_CUT_POINTS
    from stage5_scoring.node import measurement_rank
    from stage5_scoring.registry import SOURCE_MAPPINGS
    ms = [m for m in ms if m is not None]
    if not ms:
        return None
    result = min(ms, key=lambda m: measurement_rank(m, DEFAULT_CUT_POINTS, SOURCE_MAPPINGS.get(m.key, "band")))
    gaps = [m.detail.get("episode", m.reason) for m in ms if m.verdict is Verdict.ABSTAIN]
    if gaps:
        result.detail.update(missing_episodes=gaps,
                             episode_results=[{"verdict": m.verdict.value, "value": m.value,
                                               "reason": m.reason} for m in ms])
    return result


def _audio_dependency(bus, key, video):
    _flow(bus, "penalty_prerequisite", ["spec.audio_track"], key, "annotate",
          f"{video}：音轨缺失/全静音已在音轨合规扣分；从属检查不重复计分")
    return dependent(key, "spec.audio_track", "缺失音轨已扣分，此项无可评价对象", episode=str(video))


def _ledger(bus):
    return getattr(bus, "ledger", None)


def _flow(bus, rule, src, dst, effect, note=""):
    led = _ledger(bus)
    if led is not None:
        led.add(rule, src, dst, effect, note)


# ------------------------------------------------------------ dialogue ---
def word_cap(brief: BriefFacts, bus, videos: list[Path]):
    key = "brief.word_cap"
    if brief.word_cap is None:
        return _m(key, Verdict.NOT_APPLICABLE, reason="任务未指定词数上限")
    per = []
    for v in videos:
        t = bus.transcript(v)
        if t is None:
            per.append(_abstain(key, "无转录"))
        elif t.get("silent"):
            per.append(_audio_dependency(bus, key, v.name))
        elif not _dialogue_heard(t)[0]:
            # A live track with no intelligible speech (music, ambience) is
            # the silent case wearing a waveform: "0 < 125" is not a pass.
            why = _dialogue_heard(t)[1]
            per.append(_m(key, Verdict.ABSTAIN,
                          reason=f"{v.name}：音轨有声但转不出台词（{why}），零词不是达标",
                          episode=v.name, n_words=t.get("n_words"), asr_model=t.get("model")))
            _flow(bus, "dialogue_unmeasured", ["asr"], key, "annotate",
                  f"{v.name}: ASR 不足以确定台词量（{why}），不能认定超限")
        else:
            n, cap = t["n_words"], brief.word_cap
            over = max(0.0, (n - cap) / cap)
            extra = {}
            got = bus.subtitles_for(v) if hasattr(bus, "subtitles_for") else None
            if got:
                extra["subtitle_words"] = sum(len(c["text"].split()) for c in got[1]
                                              if _spoken(c["text"]))
            per.append(_m(key, Verdict.PASS if n <= cap else Verdict.FAIL, Scale.RATE, over,
                          reason="" if n <= cap else f"{v.name}：{n} 词，上限 {cap}",
                          episode=v.name, n_words=n, cap=cap, asr_model=t.get("model"),
                          **extra))
    return _worst(per)


def _find_line(line: str, segments) -> tuple[bool, float | None]:
    want = normalise(line)
    toks = want.split()
    text = normalise(" ".join(s["text"] for s in segments))
    if want in text:
        for s in segments:
            if toks[0] in normalise(s["text"]).split():
                return True, float(s["start"])
        return True, None
    # tolerate ASR slips: >= 80% of the line's tokens in order within a window
    words = text.split()
    n = len(toks)
    for i in range(0, max(1, len(words) - n + 1)):
        win = words[i:i + n + 2]
        hit = sum(1 for t in toks if t in win)
        if hit >= max(2, int(0.8 * n)):
            return True, None
    return False, None


def mandated_lines(brief: BriefFacts, bus, videos: list[Path], texts, facts=None):
    """Two measurements: is the line spoken; is it spoken in the reversal band."""
    if not brief.mandated_lines:
        return [_m(k, Verdict.NOT_APPLICABLE, reason="任务未指定必须出现的台词")
                for k in ("brief.mandated_line", "brief.mandated_line_placement")]
    out = []
    band = brief.band(brief.line_placement) if brief.line_placement else None
    if band is None:
        out.append(_m("brief.mandated_line_placement", Verdict.NOT_APPLICABLE,
                      reason="任务未指定台词出现时间"))
    plan_name = getattr(getattr(facts, "claims", None), "planning_artifact", None)
    for line in brief.mandated_lines:
        planned_in = [p.name for p, body in texts if normalise(line) in normalise(body)]
        planned = bool(planned_in)          # the texts were searched: absence is "not planned"
        found_any, at, ep = False, None, None
        silent_all = True
        unknown = 0
        for v in videos:
            t = bus.transcript(v)
            if t is None:
                unknown += 1
                continue
            if not t.get("silent"):
                silent_all = False
            ok, ts = _find_line(line, t.get("segments", []))
            if ok:
                found_any, at, ep = True, ts, v.name
                break
        if not found_any and unknown:
            out.append(_abstain("brief.mandated_line", "部分转录缺失，不能确认指定台词未出现"))
            continue
        if silent_all and videos:
            out.append(_audio_dependency(bus, "brief.mandated_line", "all"))
            out.append(_audio_dependency(bus, "brief.mandated_line_placement", "all"))
            continue
        tri = _tri(planned, found_any)
        out.append(_m("brief.mandated_line", Verdict.PASS if found_any else Verdict.FAIL,
                      reason="" if found_any else f"指定台词未出现：「{line[:60]}」"
                      + ("（剧本文件里有，成片里没有：执行失败）" if planned else "（轨迹里也没有：被忽略）"),
                      line=line, episode=ep, at_seconds=at, triangulation=tri,
                      planned_in_trajectory=planned, planned_in=planned_in[:5],
                      planning_artifact=plan_name))
        _flow(bus, "planned_vs_delivered", ["claims.planning_artifact", "texts", "asr"],
              "brief.mandated_line", "annotate",
              f"指定台词：轨迹文本{'含' if planned else '不含'}该句"
              f"{('（' + '、'.join(planned_in[:3]) + '）') if planned_in else ''}，"
              f"成片{'有' if found_any else '无'} → {tri}")
        if found_any and band is not None:
            if at is None:
                out.append(_abstain("brief.mandated_line_placement", "台词命中但时间戳不可定位"))
            else:
                inside = band.start - 3 <= at <= band.end + 3
                out.append(_m("brief.mandated_line_placement",
                              Verdict.PASS if inside else Verdict.FAIL,
                              reason="" if inside else
                              f"台词在 {at:.0f}s，任务要求在反转带 {band.start:.0f}–{band.end:.0f}s",
                              at_seconds=at, band=[band.start, band.end], episode=ep))
        elif not found_any and band is not None:
            out.append(dependent("brief.mandated_line_placement", "brief.mandated_line",
                                 "台词缺失已扣分，不重复扣出现时间"))
    return out


def no_voiceover(brief: BriefFacts, bus, videos: list[Path]):
    """An offscreen speaker is not evidence of narration (e.g. reverse shots)."""
    key = "brief.no_voiceover"
    if brief.voiceover_allowed is not False:
        return _m(key, Verdict.NOT_APPLICABLE, reason="任务未禁止旁白")
    return _abstain(key, "画外对白不能判为旁白；当前缺少区分对白与叙述的成片证据")


# ---------------------------------------------------------- subtitles ---
_BRACKETED = re.compile(r"^\s*[\[(【（].*[\])】）]\s*$", re.S)
_CUE_LINE = re.compile(r"\d{1,2}:\d{2}(?:[.,:]\d{1,3})?\s*[–\-—>]+\s*\d{1,2}:\d{2}")


def _spoken(text: str) -> bool:
    """A cue that is dialogue, not on-screen text like `[APPENDIX — M. CHEN]`."""
    t = (text or "").strip()
    if not t or _BRACKETED.match(t):
        return False
    # Screen titles with an author credit, also when exported without brackets.
    if t.isupper() and re.search(r"[—–]\s*[A-Z]\.\s*[A-Z]{2,}\s*$", t):
        return False
    t = re.sub(r"[\[(【][^\])】]*[\])】]", " ", t)
    return bool(re.search(r"[^\W_]", t))


def _dialogue_cues(cues) -> tuple[list[dict], str]:
    """The cues that are lines of dialogue, and why the rest were dropped.

    Bracketed screen text and explicit section tags are excluded. Ordinary
    repeated dialogue stays; speaker tags are stripped before ASR matching.
    """
    section = re.compile(r"^\s*\[(?:HOOK|CONTEXT|SETUP|CONFLICT|TWIST|CLIFFHANGER|CTA|TITLE|OUTRO)\]", re.I)
    speech = [{**c, "text": re.sub(r"[\[(【][^\])】]*[\])】]", " ", c["text"]).strip()}
              for c in cues if _spoken(c["text"]) and not section.match(c["text"])]
    dropped = sum(bool(section.match(c["text"])) for c in cues)
    return speech, f"{dropped} 条是章节标签" if dropped else ""


def _text_has_cues(texts) -> bool:
    """Subtitle text with a time axis inside a *document*. Subtitle files
    themselves are excluded -- they are judged by their cues, and a
    placeholder .srt must not re-enter through its own timestamps."""
    return any(len(_CUE_LINE.findall(body)) >= 3 for p, body in texts
               if Path(p).suffix.lower() not in (".srt", ".vtt", ".ass"))


def _dialogue_heard(t: dict) -> tuple[bool, str]:
    """Did the transcript carry intelligible dialogue, or is it Whisper
    filling silence? On codex's music-only episodes the base model returned
    「請問」 x 24 and "Of course." x 7 -- a short phrase looped -- which a
    word count alone reads as dialogue. Three deterministic signatures:
    too few words, one segment text repeated across most segments, or a
    vocabulary a quarter the size of the word count."""
    n = int(t.get("n_words") or 0)
    if n < MIN_SPEECH_WORDS:
        return False, f"转出 {n} 词"
    segs = [normalise(s.get("text", "")) for s in (t.get("segments") or [])]
    segs = [s for s in segs if s]
    if len(segs) >= 3:
        top = max(set(segs), key=segs.count)
        if segs.count(top) / len(segs) >= 0.6:
            return False, f"转录是同一短语重复 {segs.count(top)} 次（「{top[:20]}」，ASR 幻觉特征）"
    toks = normalise(t.get("text") or " ".join(segs)).split()
    if toks and len(set(toks)) / len(toks) < 0.25:
        return False, f"{len(toks)} 词只有 {len(set(toks))} 个不同词（ASR 幻觉特征）"
    return True, ""


def _subtitle_files_with_dialogue(paths) -> bool:
    """At least one subtitle file carries a spoken cue. A file that cannot
    be read is given the benefit of the doubt; one that parses to nothing
    but `[LABEL]` lines is not."""
    from stage3_evidence.collector import parse_subtitles
    for p in paths:
        try:
            cues = parse_subtitles(Path(p).read_text(encoding="utf-8", errors="replace"),
                                   Path(p).suffix.lower())
        except OSError:
            return True
        if _dialogue_cues(cues)[0]:
            return True
    return False


def subtitles_aligned(brief: BriefFacts, bus, videos: list[Path], texts, cuts=None):
    """The agent's subtitle file against what the track actually says.

    brief (subtitles demanded) -> trajectory (an .srt it wrote) -> delivery
    (Whisper on the audio). A cue nobody speaks is a line that was planned and
    never voiced -- the text channel of audio-visual alignment.
    """
    key = "brief.subtitles_aligned"
    from stage4_checks.video.subtitles import score_alignment
    from stage5_scoring.node import DEFAULT_CUT_POINTS
    alignment = {**DEFAULT_CUT_POINTS["subtitle_alignment"],
                 **((cuts or {}).get("subtitle_alignment") or {})}
    demanded = any(d.key == "subtitles" for d in brief.deliverables)
    per = []
    for v in videos:
        got = bus.subtitles_for(v) if hasattr(bus, "subtitles_for") else None
        if got is None:
            if demanded:
                per.append(_abstain(key, f"{v.name}：未定位到本集时间轴字幕") if _text_has_cues(texts)
                           else _m(key, Verdict.VOID, reason=f"{v.name}：未交付本集字幕", episode=v.name))
            continue
        path, cues = got
        speech, dropped = _dialogue_cues(cues)
        t = bus.transcript(v)
        if t is None:
            per.append(_abstain(key, "无转录"))
            continue
        if t.get("silent"):
            per.append(_audio_dependency(bus, key, v.name))
            continue
        n_words = int(t.get("n_words") or 0)
        heard, why_not = _dialogue_heard(t)
        alignment_result = score_alignment(speech, t, **alignment) if speech else None
        if (not heard and alignment_result and
                any(r["status"] == "aligned" for r in alignment_result["rows"])):
            # A timed cue can corroborate short OR repeated dialogue. A word
            # overlap elsewhere in the track cannot override the ASR filter.
            heard = True
        if not speech:
            # codex's ep1.srt: "[HOOK] The Woman Behind the Numbers" x 6 --
            # section labels in a subtitle container. Against a track that
            # carries dialogue that is a subtitle file with no subtitles.
            what = dropped or f"{len(cues)} 条全是方括号里的屏幕文字"
            if heard:
                per.append(_m(key, Verdict.FAIL, Scale.RATE, 1.0,
                              reason=f"{v.name}：字幕里没有一条台词（{what}），而音轨转出 {n_words} 词",
                              episode=v.name, subtitle_file=path.name, n_cues=0,
                              placeholder_cues=len(cues), asr_words=n_words,
                              asr_model=t.get("model")))
            else:
                # A placeholder file over a track with no dialogue. Not an
                # abstention: the agent's own delivery made this unmeasurable,
                # and abstaining would rank a fake subtitle file above no file
                # at all (which VOIDs below). ABSTAIN leaves the denominator;
                # VOID does not (EVAL_DESIGN 6.2).
                per.append(_m(key, Verdict.VOID,
                              reason=f"{v.name}：字幕无台词（{what}），音轨也无可辨语音（{why_not}）：无可对齐，且是交付本身造成的",
                              episode=v.name, subtitle_file=path.name, placeholder_cues=len(cues),
                              asr_words=n_words, asr_model=t.get("model")))
            continue
        if not heard:
            # Real-looking cues over a track with no dialogue: lines that were
            # written and never voiced.
            per.append(_m(key, Verdict.FAIL, Scale.RATE, 1.0,
                          reason=f"{v.name}：字幕有 {len(speech)} 条台词，音轨却无可辨语音（{why_not}）：写了没配音",
                          episode=v.name, subtitle_file=path.name, n_cues=len(speech),
                          n_unheard=len(speech), asr_words=n_words, asr_model=t.get("model")))
            continue
        rows = alignment_result["rows"]
        missed = [r["text"] for r in rows if r["status"] == "unheard"]
        mistimed = [r["text"] for r in rows if r["status"] == "mistimed"]
        unknown = sum(r["status"] == "unknown" for r in rows)
        rate = None if alignment_result["score"] is None else 1.0 - alignment_result["score"]
        verdict = Verdict.ABSTAIN if rate is None else (Verdict.FAIL if rate > 1e-9 else Verdict.PASS)
        reason = (f"{v.name}：{unknown}/{len(speech)} 条字幕缺少足够精细的语音时间戳"
                  if rate is None else (f"{v.name}：{len(missed)} 条未听到，{len(mistimed)} 条时间错位，双向词对齐缺失率 {rate:.1%}"
                                   if rate else ""))
        per.append(_m(key, verdict, Scale.RATE, rate,
                      reason=reason,
                      episode=v.name, subtitle_file=path.name, n_cues=len(speech),
                      n_unheard=len(missed), unheard=missed[:5],
                      n_mistimed=len(mistimed), n_unknown=unknown,
                      cue_alignment=rows, alignment_config=alignment,
                      subtitle_words=alignment_result["subtitle_words"],
                      speech_words=alignment_result["speech_words"],
                      alignment_precision=alignment_result.get("precision"),
                      alignment_recall=alignment_result.get("recall"),
                      asr_words=t.get("n_words"), asr_model=t.get("model")))
    if not per or all(m.verdict is Verdict.ABSTAIN and "未定位到" in m.reason for m in per):
        if not demanded:
            return _m(key, Verdict.NOT_APPLICABLE, reason="任务未要求字幕，交付里也没有字幕文件")
        if _text_has_cues(texts):
            return _abstain(key, "字幕仅以文本形式交付（无 .srt/.vtt/.ass），无时间轴文件可与音轨比对")
        return _m(key, Verdict.VOID, reason="任务要求字幕，交付里没有任何字幕文件")
    _flow(bus, "subtitles_vs_transcript", ["brief.deliverables", "files:.srt", "asr"], key,
          "reinterpret", f"{len(per)} 集的字幕文件逐条对照转录")
    return _worst(per)


# ----------------------------------------------------------- repetition --
# ---------------------------------------------------------- deliverables --
# Wording as agents actually write it: claudecode's FINAL_DELIVERY.md uses
# markdown headings ("## Series title options", "## Platform tags",
# "## Next-episode teasers") with the items on the lines below.
_TEXT_HINTS = {
    "episode_titles": re.compile(r"\bepisode\s*\d+\s*[:\-–|]|\bepisode titles?\b", re.I),
    "series_title": re.compile(r"\b(series title|title options?|series name)\b", re.I),
    "description": re.compile(r"\b(description|synopsis|logline)\b", re.I),
    "tags": re.compile(r"\b(tags?|hashtags?)\b\s*[:\-–]?", re.I),
    "teaser": re.compile(r"\bteasers?\b", re.I),
}


def _count_tags(body: str, at: int) -> int:
    """Hashtags or list items in the lines after a tags label."""
    tail = body[at: at + 600].splitlines()
    n = 0
    for line in tail[:8]:
        line = line.strip()
        if not line:
            continue
        if re.match(r"#{1,6}\s", line):              # a markdown heading, not a #hashtag
            break
        hashes = re.findall(r"#\w+", line)
        if hashes:
            n += len(hashes)
        elif line.startswith(("-", "*", "•")) or re.match(r"\d+[.)]", line):
            n += 1
        elif "," in line:
            n += len([x for x in line.split(",") if x.strip()])
    return n


def deliverables(brief: BriefFacts, bus, videos: list[Path], texts):
    key = "brief.deliverables"
    if not brief.deliverables:
        return None
    imgs = bus.produced_images()
    subs = bus.files((".srt", ".ass", ".vtt"))
    missing, found, how = [], [], {}
    owned_elsewhere = {}
    for d in brief.deliverables:
        if d.key in ("episodes", "subtitles"):
            owned_elsewhere[d.key] = "C" if d.key == "episodes" else "brief.subtitles_aligned"
            continue
        ok = False
        if d.key == "episodes":
            ok = bool(videos) and (d.count is None or len(videos) >= d.count)
            how[d.key] = "videos"
        elif d.key in ("series_cover", "episode_covers"):
            covers = [p for p in imgs if "cover" in p.stem.lower()]
            need = d.count if (d.key == "episode_covers" and d.count) else 1
            ok = len(covers) >= need
            how[d.key] = f"{len(covers)} cover image(s)"
        elif d.key == "subtitles":
            # A subtitle file with at least one spoken cue, or subtitle text
            # with a time axis in a document ("complete subtitle text" can
            # legitimately be a table). A file of bracketed section labels is
            # a container, not subtitles.
            with_lines = _subtitle_files_with_dialogue(subs)
            ok = with_lines or _text_has_cues(texts)
            how[d.key] = ("files" if with_lines else
                          "files_without_dialogue" if subs and not ok else
                          "text" if ok else "none")
        else:
            pat = _TEXT_HINTS.get(d.key)
            ok = bool(pat) and any(pat.search(body) for _, body in texts)
            how[d.key] = "text" if ok else "none"
            if ok and d.key == "tags" and d.count:
                n = 0
                for _, body in texts:
                    for m in pat.finditer(body):
                        n = max(n, _count_tags(body, m.end()))
                ok = n >= d.count
                how[d.key] = f"text:{n} tags"
        (found if ok else missing).append(d.text)
    n = len(missing) + len(found)
    if not n:
        return _m(key, Verdict.NOT_APPLICABLE, reason="交付项均已在完成度或字幕检查计分",
                  owned_elsewhere=owned_elsewhere)
    rate = len(missing) / n
    _flow(bus, "planned_vs_delivered", ["brief.deliverables", "texts", "produced", "files"],
          key, "annotate", f"{len(found)}/{len(brief.deliverables)} 项交付物在轨迹产物里找到")
    return _m(key, Verdict.PASS if not missing else Verdict.FAIL, Scale.RATE, rate,
              reason="" if not missing else "缺少交付物：" + "；".join(missing)[:160],
              found=found, missing=missing, evidence=how, owned_elsewhere=owned_elsewhere)


# ------------------------------------------------- the continuity contract --
# Words that carry no wardrobe information. Kept small on purpose: the brief's
# visual lines are already terse ("Victor wears immaculate charcoal suits and a
# red tie"), so anything not here is a term worth carrying forward.
_VISUAL_STOP = {
    "the", "and", "with", "wears", "wear", "wearing", "carries", "carry", "has",
    "her", "his", "their", "она", "one", "a", "an", "is", "are", "by", "in",
    "of", "distinguished", "preserve", "across", "all", "episodes", "relative",
}


def _visual_terms(brief: BriefFacts) -> dict[str, set[str]]:
    """Per character, the content words of the brief's wardrobe line.

    The character's own name tokens are dropped: "Maya wears restrained navy
    officewear" would otherwise score a contract that says "Maya" and nothing
    about what she looks like.
    """
    out: dict[str, set[str]] = {}
    for c in brief.characters:
        if not c.visual:
            continue
        drop = _VISUAL_STOP | set(c.tokens)
        terms = {w for w in re.findall(r"[a-z]{3,}", c.visual.lower()) if w not in drop}
        if terms:
            out[c.name] = terms
    return out


def _term_coverage(terms: dict[str, set[str]], haystack: str):
    """Which of the brief's wardrobe terms survive into the agent's own text."""
    hay = set(re.findall(r"[a-z]{3,}", (haystack or "").lower()))
    per, found, total = {}, 0, 0
    for name, ts in terms.items():
        hit = ts & hay
        per[name] = {"n_terms": len(ts), "found": len(hit),
                     "missing": sorted(ts - hit)[:8]}
        found += len(hit)
        total += len(ts)
    return per, found, total


def continuity_contract(brief: BriefFacts, facts=None):
    """Did the agent write the brief's wardrobe down before generating?

    The first half of the only information flow this benchmark can attribute to
    the agent alone, and it closes without a single pixel:

        brief "navy officewear, silver mechanical pencil behind one ear"
          -> the agent's own contract          <- here
          -> the agent's own prompts           <- specs_in_prompts
          -> the delivered shot                <- O3/O4, and the model's job

    Measured on `data/harness` prompt_01: opensquilla, openclaw and hermes each
    wrote a named bible; claudecode embedded one in the generation script it
    wrote; codex's CONTINUITY.md is 263 bytes of episode structure with no
    character in it. Nothing here decodes a frame.
    """
    key = "brief.continuity_contract"
    terms = _visual_terms(brief)
    if not terms:
        return None                      # the brief specifies no wardrobe
    text = getattr(getattr(facts, "continuity", None), "text", None)
    if not text:
        return _abstain(key, "轨迹里找不到 agent 写下的连续性文本")
    per, found, total = _term_coverage(terms, text)
    # The unit is the character, not the word. An earlier version passed only
    # on word-for-word coverage and nothing could reach it: these are
    # paraphrases ("restrained navy officewear" against "Restrained navy
    # officewear (navy blazer, white blouse)"), so every harness failed and the
    # check separated nobody. Per character -- did the agent describe this
    # person at all -- is the same unit `characters_anchored` uses, and on the
    # real case it splits 4/4 for four harnesses against 1/4 for codex.
    # Word coverage stays as the magnitude.
    unspecified = sorted(n for n, v in per.items() if not v["found"])
    return _m(key, Verdict.PASS if not unspecified else Verdict.FAIL,
              Scale.RATE, (total - found) / float(total),
              reason="" if not unspecified else
              f"{len(unspecified)}/{len(per)} 个角色在 agent 自己的连续性文本里"
              f"没有任何外观描述：{'、'.join(unspecified)}",
              source=getattr(facts.continuity, "source", None),
              characters_specified=len(per) - len(unspecified),
              characters_total=len(per), per_character=per)


def specs_in_prompts(brief: BriefFacts, facts=None):
    """A contract nobody quotes is an asset the agent built and did not use.

    The same failure `brief.characters_anchored` finds on the image side --
    the agent drew four character sheets and the delivered faces matched none
    of them -- asked on the text side, where it needs no model at all.
    """
    key = "brief.specs_in_prompts"
    terms = _visual_terms(brief)
    if not terms:
        return None
    prompts = getattr(getattr(facts, "requested", None), "prompts", None)
    if not prompts:
        # Three of five harnesses log prompts only inside their trajectory
        # exports, where no field separates a generation prompt from the
        # scaffolding's system prompt. ABSTAIN until the LM backend fills it.
        return _abstain(key, "轨迹里没有可确定性恢复的逐次 prompt 文本")
    per, found, total = _term_coverage(terms, "\n".join(prompts))
    # Same unit as continuity_contract, for the same reason: did each character
    # reach the prompts at all. Word coverage is the magnitude.
    unquoted = sorted(n for n, v in per.items() if not v["found"])
    return _m(key, Verdict.PASS if not unquoted else Verdict.FAIL,
              Scale.RATE, (total - found) / float(total),
              reason="" if not unquoted else
              f"{len(unquoted)}/{len(per)} 个角色的外观从未进入 agent 自己的 prompt："
              f"{'、'.join(unquoted)}",
              n_prompts=len(prompts),
              characters_quoted=len(per) - len(unquoted),
              characters_total=len(per), per_character=per)


def _distinctive_location_tokens(brief: BriefFacts) -> dict[str, set[str]]:
    """Per brief location, the words that belong to it and no other.

    "room" sits in two of prompt_01's four locations and resolves nothing;
    "conference", "elevator", "workspace", "suite" each pick out one. Deriving
    that from the brief beats hand-listing it: another brief has other
    locations, and nothing here needs editing.
    """
    per = {loc: {w for w in re.findall(r"[a-z]{3,}", loc.lower())
                 if w not in _VISUAL_STOP}
           for loc in brief.locations}
    counts: dict[str, int] = {}
    for terms in per.values():
        for w in terms:
            counts[w] = counts.get(w, 0) + 1
    return {loc: {w for w in terms if counts[w] == 1} for loc, terms in per.items()}


def declared_shot_locations(brief: BriefFacts, facts=None):
    """The agent's shot plan, with each span resolved to a brief location.

    This is the raw material for replacing a guess. `visual_consistency`
    groups shots into locations by clustering CLIP features at a fixed
    distance, because nothing ever told it which shots share a room -- and a
    cut from the conference room to the open plan is correct storytelling that
    the clustering has to infer. The agent wrote the answer down before it
    generated anything.

    Returns `[(start, end, location_or_None), ...]`. A span that names no
    location stays None: coverage is partial by nature -- "Maya at her desk,
    silver pencil behind ear" places nobody -- and a guess dressed as a
    declaration would be worse than the clustering it replaces.
    """
    plan = getattr(getattr(facts, "continuity", None), "shot_plan", None)
    if not plan or not brief.locations:
        return []
    distinctive = _distinctive_location_tokens(brief)
    out = []
    for shot in plan:
        words = set(re.findall(r"[a-z]{3,}", shot.text.lower()))
        scored = [(len(distinctive[loc] & words), loc) for loc in brief.locations]
        best_n, best_loc = max(scored)
        out.append((shot.start, shot.end, best_loc if best_n else None))
    return out


def scene_plan(brief: BriefFacts, facts=None):
    """Of the brief's locations, how many did the agent plan a shot in?

    Same shape as `characters_anchored`, and for the same reason: a rate over
    the things the brief named, not over the agent's own beats. Demanding that
    every beat state a room would punish good writing -- "Maya at her desk,
    silver pencil behind ear" continues the previous location and is supposed
    to -- while the question that matters is whether the agent's plan visits
    the world the brief specified.
    """
    key = "brief.scene_plan"
    if not brief.locations:
        return None
    plan = getattr(getattr(facts, "continuity", None), "shot_plan", None)
    if not plan:
        return _abstain(key, "轨迹里没有可确定性恢复的分镜计划")
    resolved = declared_shot_locations(brief, facts)
    used = {loc for _, _, loc in resolved if loc}
    n = len(brief.locations)
    miss = n - len(used)
    return _m(key, Verdict.PASS if miss == 0 else Verdict.FAIL,
              Scale.RATE, miss / float(n),
              reason="" if miss == 0 else
              f"任务书的 {n} 个地点里，分镜计划只到过 {len(used)} 个",
              n_planned_shots=len(resolved),
              n_shots_placed=sum(1 for _, _, l in resolved if l),
              locations_used=sorted(used),
              locations_missing=sorted(set(brief.locations) - used))


# ------------------------------------------------------------ characters --
def characters_anchored(brief: BriefFacts, bus, videos: list[Path], facts=None):
    """brief name -> trajectory sheet -> video face. O3 without task-supplied
    references: the agent's own character sheets are the references."""
    key = "brief.characters_anchored"
    if not brief.characters:
        if brief.declared_character_count:
            return _abstain(key, "任务指定了角色数量，但具名角色未完整解析")
        return _m(key, Verdict.NOT_APPLICABLE, reason="任务未指定具名角色")
    n_img = getattr(getattr(facts, "calls", None), "image_gen", None)
    sheets = bus.character_sheets(brief.characters)
    planned = {c.name: (c.name in sheets) for c in brief.characters}
    if not sheets:
        if n_img == 0:
            why, tri = "轨迹没有任何图像生成调用：角色设计被跳过", "ignored"
        elif n_img:
            why, tri = f"轨迹生成了 {n_img} 张图，但没有以角色名命名的角色图，无参考可锚定", "unnamed"
        else:
            why, tri = "轨迹里没有以角色名命名的角色图，无参考可锚定", "no_sheet"
        _flow(bus, "image_triangulation", ["calls.image_gen", "produced"], key, "reinterpret", why)
        return _m(key, Verdict.ABSTAIN, reason=why, image_gen_calls=n_img,
                  triangulation={c.name: tri for c in brief.characters})
    _flow(bus, "image_triangulation", ["calls.image_gen", "produced"], key, "annotate",
          f"{len(sheets)}/{len(brief.characters)} 个角色有轨迹角色图（image_gen={n_img}）")

    # Series-level, not worst-episode. A principal character need not appear
    # in every episode -- prompt 1's CFO is written to enter in episode 3 --
    # so "unmatched in ep2" is storytelling, not a defect. A character is
    # anchored if some delivered episode shows a face that matches its sheet;
    # which episodes did is kept per character for the reader.
    seen: dict[str, list[str]] = {n: [] for n in sheets}
    best: dict[str, float] = {n: 0.0 for n in sheets}
    compared: dict[str, set[str]] = {n: set() for n in sheets}
    ran = 0
    for v in videos:
        rep = bus.seeded_visual(v, brief.characters, brief, facts)
        names = list((bus.seeds.get(str(v)) or {}).get("sheets") or sheets)
        ri = (rep or {}).get("character_identity", {}).get("reference_identity") if rep else None
        if not ri or not ri.get("n_references"):
            continue
        ran += 1
        per_ref = ri.get("per_reference") or []
        for name, entry in zip(names, per_ref):
            if name not in sheets or "matches" not in entry:
                continue
            compared[name].add(str(v))
            sim = float(entry.get("best_similarity") or 0.0)
            best[name] = max(best.get(name, 0.0), sim)
            if entry.get("matches"):
                seen.setdefault(name, []).append(v.name)
    if not ran:
        return _abstain(key, "带参考图的视觉比对未产出")

    names = [c.name for c in brief.characters]
    unmatched = [n for n in names if not seen.get(n)]
    missing_sheets = [n for n in names if n not in sheets]
    # A negative claim needs every delivered episode checked. Positive
    # series-level existence needs only one match. Summary counts cannot be
    # assigned to arbitrary names, and missing sheets are not failed faces.
    incomplete = [n for n in unmatched if len(compared.get(n, set())) < len(videos)]
    if missing_sheets or incomplete or (brief.declared_character_count is not None
                                      and brief.declared_character_count != len(names)):
        return _m(key, Verdict.ABSTAIN,
                  reason="角色参考或逐角色比对证据不完整，不能只按可测角色给分",
                  missing_sheets=missing_sheets, incomplete_characters=incomplete,
                  n_characters_in_brief=len(names), n_references=len(sheets),
                  n_matched=len(names) - len(unmatched),
                  episodes_seen=seen, best_similarity=best)
    # This leaf uses fraction mapping: all unmatched must mean zero for any
    # cast size. The old band-era floor inflated small casts' scores.
    rate = len(unmatched) / float(len(names))
    # The reason carries the margin, because the verdict rests on one number:
    # visual_consistency's 0.4 ArcFace threshold, set for photo references and
    # never fitted for a drawn character sheet against a generated face.
    # "Evelyn 0.32 vs 0.4" lets a reader judge absence versus a near miss.
    return _m(key, Verdict.PASS if not unmatched else Verdict.FAIL, Scale.RATE, rate,
              reason="" if not unmatched else
              f"{len(unmatched)}/{len(names)} 个有角色图的角色在任何一集里都找不到匹配人脸："
              + "、".join(f"{n}（最高相似度 {best[n]:.2f}，阈值 0.4）" for n in unmatched),
              n_references=len(names), n_matched=len(names) - len(unmatched),
              rate_denominator=len(names),
              episodes_seen={n: seen.get(n, []) for n in names},
              best_similarity={n: round(best[n], 3) for n in names},
              sheets={k: str(p.name) for k, p in sheets.items()},
              sheet_sources=dict(getattr(bus, "sheet_sources", {})),
              planned=planned, image_gen_calls=n_img,
              triangulation={c.name: ("sheet" if planned[c.name] else "no_sheet")
                             for c in brief.characters},
              n_characters_in_brief=len(brief.characters))


def scene_fidelity(brief: BriefFacts, bus, videos: list[Path], facts=None):
    """visual.scene_reference on the agent's own conditioning keyframes: does
    delivered shot i geometrically match the image clip i was generated from.
    Shot boundaries come from the artifact module's splices (evidence bus), so
    the pairing is by structure the pixels recovered, not by guess."""
    key = "visual.scene_reference"
    per, n_seeded = [], 0
    for v in videos:
        rep = bus.seeded_visual(v, brief.characters, brief, facts)
        seeds = bus.seeds.get(str(v)) or {}
        if not seeds.get("scene_references"):
            per.append(_abstain(key, f"{v.name}：没有可对应的场景参考图"))
            continue
        n_seeded += 1
        src = rep.get("scene_reference_check") if rep else None
        if not src or not src.get("n_shots"):
            per.append(_abstain(key, f"{v.name}：场景参考比对未产出（人脸模型缺失或模块失败）"))
            continue
        n = int(src["n_shots"])
        ok = int(src.get("n_shots_matching_own_ref") or 0)
        per.append(_m(key, Verdict.PASS if ok >= n else Verdict.FAIL, Scale.RATE,
                      (n - ok) / float(n),
                      reason="" if ok >= n else
                      f"{v.name}：{n - ok}/{n} 个片段与 agent 自己的条件关键帧几何上对不上",
                      episode=v.name, n_shots=n, n_matching=ok,
                      shots_json=seeds.get("shots_json"),
                      unmatched_shots=[r.get("shot") for r in (src.get("per_shot") or [])
                                       if not r.get("matches_own_ref")][:12]))
    if not n_seeded:
        return _abstain(key, "轨迹里没有逐片段的条件关键帧（或与像素段数对不齐），场景忠实度无参考可比")
    if any(m.verdict is Verdict.ABSTAIN for m in per):
        return _m(key, Verdict.ABSTAIN, reason="部分集场景参考或比对结果缺失",
                  episodes=[{"verdict": m.verdict.value, "reason": m.reason, **m.detail} for m in per])
    return _worst(per)


def asset_utilisation(brief: BriefFacts, bus, videos: list[Path], facts=None):
    """provenance.conditioning_edges as process: of the things that should
    have been anchored, how many did the agent build a reference for.

    units        the brief's named characters + the delivered clips
    strong edge  a sheet named after the character / a keyframe named after
                 the clip (content-checkable, and checked by O3)
    Declared edge counts without a target binding cannot estimate coverage.
    Pixel cuts do not establish source clips or missing reference assets.
    """
    key = "provenance.conditioning_edges"
    units = strong = 0
    detail: dict = {}
    bound_targets: set[Path] = set()
    if brief.characters:
        sheets = bus.character_sheets(brief.characters)
        units += len(brief.characters)
        strong += len(sheets)
        bound_targets.update(Path(p).resolve() for p in sheets.values())
        detail["characters"] = {"n": len(brief.characters), "with_sheet": len(sheets),
                                "sources": dict(getattr(bus, "sheet_sources", {}))}
    n_clips = n_kf = 0
    unknown_inventory = []
    for v in videos:
        clips = (bus.reference_clips(v) if hasattr(bus, "reference_clips") else
                 bus.episode_clips(v) if hasattr(bus, "episode_clips") else [])
        n = len(clips)
        if not clips:
            unknown_inventory.append(str(v))
        bound_targets.update(Path(p).resolve() for p in clips)
        kf = bus.keyframes_for_clips(v) if hasattr(bus, "keyframes_for_clips") else {}
        n_clips += n
        n_kf += min(n, len(kf))
    units += n_clips
    strong += n_kf
    detail["clips"] = {"n": n_clips, "with_keyframe": n_kf}
    cond = getattr(facts, "conditioning", None) if facts is not None else None
    declared = len(cond) if cond is not None else None
    detail["declared_edges"] = declared
    if unknown_inventory:
        if strong == 0 and declared == 0:
            return _m(key, Verdict.FAIL, Scale.RATE, 1.0,
                      reason="未找到参考资产，且轨迹明确未使用参考；不依赖未确定的源片段数量",
                      unbound_videos=unknown_inventory, **detail)
        return _m(key, Verdict.ABSTAIN,
                  reason="交付片段与源素材未能对应，不能用画面切换次数推断缺少参考资产",
                  unbound_videos=unknown_inventory, **detail)

    # Finding some character sheets does not resolve other declared inputs.
    # Otherwise adding one real sheet could turn an unknown binding into a
    # large false failure for every clip. Complete physical coverage remains
    # measurable even when an unrelated declaration cannot be located.
    resolve = getattr(bus, "_resolve_asset", None)
    delivered_targets = {Path(v).resolve() for v in videos}
    unbound = []
    for edge in cond or []:
        if not edge.references:
            continue
        target = resolve(edge.target) if callable(resolve) else None
        # A located extra asset (e.g. a cover) is outside these units. A
        # declaration for a whole episode still cannot assign its clip inputs.
        if target is None or (target in delivered_targets and target not in bound_targets):
            unbound.append(edge.target)
    if strong < units and unbound:
        return _m(key, Verdict.ABSTAIN,
                  reason="部分参考引用未能绑定到具体角色或交付片段，不能将未绑定视为未使用",
                  unbound_conditioning_targets=unbound, **detail)

    if units == 0:
        if declared == 0:
            return _m(key, Verdict.FAIL, reason="轨迹显示生成时未使用任何一致性参考资产", **detail)
        return _abstain(key, "任务无具名角色、像素也未恢复出片段结构，无从统计资产利用")
    weak = 0
    if strong == 0 and declared:
        return _abstain(key, "引用记录未能绑定到具体角色或交付片段，不能按引用条数推算覆盖率")
    if strong == 0 and not declared:
        if declared == 0:
            return _m(key, Verdict.FAIL, Scale.RATE, 1.0,
                      reason=f"未建任何参考资产：角色图 0/{detail.get('characters', {}).get('n', 0)}，"
                             f"片段关键帧 0/{n_clips}；轨迹也明确未用参考", **detail)
        return _abstain(key, "产物里没有以角色/片段命名的参考图，轨迹也未记录条件引用（§2.4 全无）")
    covered = strong + weak
    rate = 1.0 - covered / float(units)
    _flow(bus, "asset_utilisation", ["produced", "conditioning", "brief.characters", "artifact.splices"],
          key, "annotate", f"{covered}/{units} 个应锚定单元有参考资产（强 {strong}，弱 {weak}）")
    return _m(key, Verdict.PASS if rate == 0.0 else Verdict.FAIL, Scale.RATE, rate,
              reason="" if rate == 0.0 else
              f"{units - covered}/{units} 个应锚定单元没有参考资产"
              f"（角色图 {detail.get('characters', {}).get('with_sheet', 0)}/"
              f"{detail.get('characters', {}).get('n', 0)}，片段关键帧 {n_kf}/{n_clips}）"
              + ("；仅有轨迹声明、无可核对文件（弱证据）" if weak else ""),
              strong_edges=strong, weak_edges=weak, units=units, **detail)


# ----------------------------------------------------------------- entry --
def evaluate_text(brief: BriefFacts | None, facts=None):
    """The brief x trajectory checks: 2.3.b / c / d, the planning leaves that
    close without a pixel.

    They take neither the bus nor the videos because they need neither -- the
    brief's wardrobe against the agent's own contract, its prompts and its
    shot plan. Kept apart from `evaluate` so a run whose modules did not run,
    or which delivered nothing at all, is still scored on what it planned.
    That is exactly the run one most wants to know it about.
    """
    if brief is None:
        return []
    return [m for m in (continuity_contract(brief, facts),
                        specs_in_prompts(brief, facts),
                        scene_plan(brief, facts)) if m is not None]


def evaluate(brief: BriefFacts | None, bus, facts=None, cuts=None):
    """The rest: everything that reads the delivery through the bus."""
    if brief is None:
        return []
    videos = bus.delivered_videos()
    if not videos:
        return []
    texts = bus.text_artifacts()
    out = []
    for m in (deliverables(brief, bus, videos, texts),
              word_cap(brief, bus, videos),
              no_voiceover(brief, bus, videos),
              subtitles_aligned(brief, bus, videos, texts, cuts),
              asset_utilisation(brief, bus, videos, facts)):
        if m is not None:
            out.append(m)
    out.extend(mandated_lines(brief, bus, videos, texts, facts))
    return out
