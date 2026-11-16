"""The brief, compiled: prompt.md -> BriefFacts, one per prompt.

Until this existed the upstream brief entered the evaluation as four numbers
(episodes, seconds, aspect, resolution). Everything else it demands -- four
named characters with described wardrobe, a mandated line placed in the
reversal, a spoken-word cap, a ban on voiceover, a deliverables checklist --
never reached any check. A video with every character wrong but consistent
from shot to shot scored full marks on visual consistency.

The prompts are templated: 24 bold field labels recur across all ten, so this
is regex-first and deterministic. Anything it cannot parse stays None, and an
expectation built on a None simply does not exist for that prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "prof", "sir", "dame", "lady", "lord"}


@dataclass
class Character:
    name: str                 # "Maya Chen" / "Mrs. Alvarez"
    first: str                # "maya" / "alvarez" -- first non-honorific token
    descriptor: str = ""      # "29, a quiet financial analyst ..."
    visual: str = ""          # "wears restrained navy officewear and ..."

    @property
    def tokens(self) -> list[str]:
        """Name tokens a trajectory might file a sheet under, honorifics
        dropped: claudecode named Mrs. Alvarez's sheet `alvarez.png`."""
        return [t for t in re.split(r"[\s.'-]+", self.name.lower())
                if t and t not in _HONORIFICS and len(t) > 1]


@dataclass
class TimingBand:
    start: float
    end: float
    label: str                # hook | conflict | escalation | reversal | payoff


@dataclass
class Deliverable:
    key: str                  # cover_frame | episode_titles | description | tags | subtitles | teaser | ...
    text: str                 # the phrase as written
    count: int | None = None  # "five platform tags" -> 5 ; "60-word description" -> 60


@dataclass
class BriefFacts:
    prompt_no: int
    title: str = ""
    episodes: int | None = None
    seconds_per_episode: float | None = None
    aspect_ratio: str | None = None
    characters: list[Character] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    word_cap: int | None = None
    voiceover_allowed: bool | None = None      # None = the brief does not say
    mandated_lines: list[str] = field(default_factory=list)
    timing_bands: list[TimingBand] = field(default_factory=list)
    deliverables: list[Deliverable] = field(default_factory=list)
    line_placement: str | None = None          # band label the mandated line must land in
    declared_character_count: int | None = None
    parse_warnings: list[str] = field(default_factory=list)
    raw: dict[str, str] = field(default_factory=dict)

    def band(self, label: str) -> TimingBand | None:
        return next((b for b in self.timing_bands if b.label == label), None)


_LABEL = re.compile(r"^- \*\*([^*]+)\*\*:\s*(.*)$", re.M)
_PROMPT_SPLIT = re.compile(r"^### Prompt (\d+):\s*(.*)$", re.M)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _num(tok: str) -> int | None:
    tok = tok.lower()
    if tok.isdigit():
        return int(tok)
    return _WORDS.get(tok)


def _fields(body: str) -> dict[str, str]:
    return {m.group(1).strip(): m.group(2).strip() for m in _LABEL.finditer(body)}


def _characters(chars_txt: str, visual_txt: str) -> list[Character]:
    """`**Characters**: Four principal characters. Maya Chen, 29, ...; Victor Hale, 41, ...`
    `**Character Visual Design**: Maya wears ...; Victor wears ...`"""
    out: list[Character] = []
    # "Four principal characters." / "Five central characters." / "Six
    # recurring characters." -- the count sentence varies per prompt, and a
    # strip that only knew "principal" silently dropped the first character
    # of prompts 3, 5 and 7 (Iris, Zoe, Mara).
    body = re.sub(r"^\w+\s+(?:\w+\s+)?characters?\.\s*", "", chars_txt.strip(), flags=re.I)
    for part in re.split(r";\s*(?:and\s+)?", body):
        # The final item in the brief is commonly introduced as "and the ..."
        # and quoted aliases such as “M. Bell”. Normalize only the structural
        # punctuation; apostrophes inside names remain intact.
        part = re.sub(r"^\s*and\s+", "", part.strip(), flags=re.I)
        part = re.sub(r"^\s*the\s+", "", part, flags=re.I)
        part = part.translate(str.maketrans({"“": "", "”": "", '"': ""}))
        m = re.match(r"\s*((?:[A-Z][\w'.-]*\.?\s+)*[A-Z][\w'-]+),\s*(.*)$", part)
        if not m:
            continue
        name = m.group(1).strip()
        c = Character(name=name, first="", descriptor=m.group(2).strip().rstrip(".,"))
        c.first = c.tokens[0] if c.tokens else name.split()[0].lower()
        out.append(c)
    if visual_txt:
        for c in out:
            m = re.search(rf"\b{re.escape(c.name.split()[0])}\b[^;.]*?(?:wears|carries|is distinguished|has|sports|keeps)[^;]*",
                          visual_txt)
            if m:
                c.visual = m.group(0).strip()
    return out


def _locations(setting: str) -> list[str]:
    m = re.search(r"including\s+(.*)$", setting, flags=re.I)
    src = m.group(1) if m else setting
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", src.rstrip(". "))
    return [p.strip(" .") for p in parts if 2 < len(p.strip()) < 80]


def _dialogue(txt: str) -> tuple[int | None, bool | None, list[str]]:
    cap = re.search(r"(?:below|under|no more than|at most|within)\s+(\d+)\s+(?:spoken\s+)?words", txt, re.I)
    vo = None
    if re.search(r"\bno (?:explanatory |narrated |spoken )?voice-?over\b", txt, re.I):
        vo = False
    elif re.search(r"voice-?over", txt, re.I):
        vo = True
    lines = [s.strip() for s in re.findall(r"[“\"]([^”\"]{6,200})[”\"]", txt)]
    return (int(cap.group(1)) if cap else None), vo, lines


_BAND = re.compile(r"(\d+)\s*[–-]\s*(\d+)\s*seconds?\s+for\s+(?:the\s+)?([a-z][a-z -]*?)(?:,|;|\.|\s+and\b|$)", re.I)
_BAND_LABEL = {"hook": "hook", "conflict": "conflict", "escalation": "escalation",
               "reversal": "reversal", "cliff": "payoff", "payoff": "payoff",
               "resolution": "payoff", "climax": "reversal", "setup": "conflict"}


def _bands(txt: str) -> list[TimingBand]:
    out: list[TimingBand] = []
    for a, b, label in _BAND.findall(txt):
        key = next((v for k, v in _BAND_LABEL.items() if k in label.lower()), label.strip().lower())
        out.append(TimingBand(float(a), float(b), key))
    return out


def _deliv_key(phrase: str) -> str | None:
    p = phrase.lower()
    if "subtitle" in p:                 # before "title": "subtitle" contains it
        return "subtitles"
    if "cover" in p:
        return "episode_covers" if "episode" in p else "series_cover"
    if "title" in p:
        return "episode_titles" if "episode" in p else "series_title"
    if "description" in p or "synopsis" in p or "logline" in p:
        return "description"
    if "tag" in p:
        return "tags"
    if "subtitle" in p:
        return "subtitles"
    if "teaser" in p:
        return "teaser"
    if "episode" in p:
        return "episodes"
    return None


def _deliverables(txt: str) -> list[Deliverable]:
    """One entry per distinct key: "series and episode covers" is two, "episode
    titles" and "titles" are one (the larger count wins on collision)."""
    body = re.sub(r"^Deliver\s+", "", txt.strip(), flags=re.I)
    by_key: dict[str, Deliverable] = {}
    for phrase in re.split(r",\s*(?:and\s+)?|\s+and\s+(?=(?:a|an|the|one|two|three|four|five|six|\d)\b)", body):
        phrase = phrase.strip(" .")
        if not phrase:
            continue
        keys = [_deliv_key(phrase)]
        if re.search(r"series and episode covers?", phrase, re.I):
            keys = ["series_cover", "episode_covers"]
        cnt = None
        m = re.search(r"(\d+)-word", phrase)
        if m:
            cnt = int(m.group(1))
        else:
            # "all three finished episodes", "five platform tags", "six episode
            # covers": the count is the first number-word anywhere in the
            # phrase, not necessarily its first word.
            for tok in re.findall(r"\b(\w+)\b", phrase):
                if _num(tok) is not None:
                    cnt = _num(tok)
                    break
        for key in keys:
            if key is None:
                continue
            old = by_key.get(key)
            if old is None or (cnt or 0) > (old.count or 0):
                by_key[key] = Deliverable(key=key, text=phrase, count=cnt)
    return list(by_key.values())


def parse_body(body: str, prompt_no: int = 0, title: str = "") -> BriefFacts:
    """One prompt's body -> BriefFacts. The body is the text under a
    `### Prompt N:` heading in prompt.md, or the same template wherever a
    trajectory embedded it (the task the agent was given is usually its first
    message, so most trajectories carry the brief verbatim)."""
    f = _fields(body)
    bf = BriefFacts(prompt_no=prompt_no, title=title, raw=f)
    fm = re.search(r"\((\w+) episodes?\)|(\d+)\s+episodes?", f.get("Format", ""), re.I)
    if fm:
        bf.episodes = _num(fm.group(1) or fm.group(2))
    cm = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
                   r"(?:principal|central|recurring)\s+characters?\b",
                   f.get("Characters", ""), re.I)
    if cm:
        bf.declared_character_count = _num(cm.group(1))
    dur = f.get("Single-episode duration", "")
    sm = re.search(r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m)\b", dur, re.I)
    if sm:
        n, unit = float(sm.group(1)), sm.group(2).lower()
        bf.seconds_per_episode = n * 60 if unit.startswith("m") else n
    # Placement only where the brief asks for it: prompt 1 wants its line
    # "near the final reversal"; prompt 5's line is merely "recurring".
    dn = f.get("Dialogue Norms", "")
    if re.search(r"(near|at|during|in|before|after) the (final |last )?(reversal|climax|turn)", dn, re.I):
        bf.line_placement = "reversal"
    am = re.search(r"(\d+:\d+)", f.get("Aspect ratio", ""))
    if am:
        bf.aspect_ratio = am.group(1)
    bf.characters = _characters(f.get("Characters", ""), f.get("Character Visual Design", ""))
    if (bf.declared_character_count is not None
            and len(bf.characters) != bf.declared_character_count):
        bf.parse_warnings.append(
            f"角色解析数 {len(bf.characters)} != 任务书声明数 "
            f"{bf.declared_character_count}")
    bf.locations = _locations(f.get("Setting", ""))
    bf.word_cap, bf.voiceover_allowed, bf.mandated_lines = _dialogue(f.get("Dialogue Norms", ""))
    bf.timing_bands = _bands(f.get("Timing Allocation", ""))
    bf.deliverables = _deliverables(f.get("Final Output Checklist", ""))
    return bf


def parse_prompts(text: str) -> dict[int, BriefFacts]:
    """All prompts in prompt.md, keyed by number."""
    out: dict[int, BriefFacts] = {}
    marks = list(_PROMPT_SPLIT.finditer(text))
    for i, m in enumerate(marks):
        body = text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)]
        bf = parse_body(body, int(m.group(1)), m.group(2).strip())
        out[bf.prompt_no] = bf
    return out


# ------------------------------------------------ the brief, from the run --
# Labels that identify the template. Three of them in one text is a brief,
# not a mention of one.
_BRIEF_MARKERS = ("Format", "Single-episode duration", "Characters", "Setting",
                  "Dialogue Norms", "Timing Allocation", "Final Output Checklist")
_BRIEF_END = re.compile(r"\[Delivery Requirements\]|### Prompt \d+:", re.M)
_SCAN_EXT = (".md", ".txt", ".json", ".jsonl", ".log", ".js", ".yaml", ".yml", ".ts", ".py")
_SCAN_SKIP = ("node_modules", "__pycache__", ".git", "site-packages")


def _unescape(text: str) -> str:
    """A brief inside a JSON string has `\\n` for newlines and `\\"` for quotes;
    the label regex anchors on line starts, so put the lines back."""
    if "\\n" in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace('\\"', '"')
    return text


def brief_from_text(text: str) -> BriefFacts | None:
    """The first brief the text contains, or None. A `### Prompt N:` heading
    keeps its number; a bare template parses as prompt 0 with no title."""
    text = _unescape(text)
    got = parse_prompts(text)
    if got:
        return next(iter(got.values()))
    starts = [m.start() for m in re.finditer(r"^- \*\*Format\*\*:", text, re.M)]
    for st in starts:
        end = _BRIEF_END.search(text, st + 1)
        body = text[st: end.start() if end else st + 12000]
        f = _fields(body)
        if sum(1 for k in _BRIEF_MARKERS if k in f) >= 3:
            return parse_body(body, 0, "")
    return None


def iter_run_texts(run_root: Path, max_files: int = 400, max_bytes: int = 6_000_000):
    """(path, text) for the run's own text files, hidden directories included
    on purpose -- hermes keeps its trajectory under `.hermes/`."""
    root = Path(run_root)
    if root.is_file():
        root = root.parent
    n = 0
    for p in sorted(root.rglob("*")):
        if n >= max_files:
            break
        if not p.is_file() or p.suffix.lower() not in _SCAN_EXT:
            continue
        if any(part in _SCAN_SKIP for part in p.parts):
            continue
        n += 1
        try:
            if p.stat().st_size > max_bytes:
                continue
            yield p, p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue


def find_brief_in_run(run_root: Path, max_files: int = 400,
                      max_bytes: int = 6_000_000) -> BriefFacts | None:
    """Scan the run's own text for the brief it was given, verbatim."""
    for _p, text in iter_run_texts(run_root, max_files, max_bytes):
        if "Dialogue Norms" not in text and "Single-episode duration" not in text:
            continue
        bf = brief_from_text(text)
        if bf is not None:
            return bf
    return None


# ------------------------------------------ the brief, by its footprints --
_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
_SHEET_DIR = re.compile(r"character|cast|sheet|ref", re.I)


def _sheet_stems(run_root: Path, max_files: int = 3000) -> set[str]:
    """Stems of images filed under a characters/cast/sheets folder: the way
    every harness so far names its character sheets (`assets/characters/
    maya.png`), and a name list nothing in prose has to state."""
    out: set[str] = set()
    n = 0
    for p in Path(run_root).rglob("*"):
        if n >= max_files:
            break
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXT:
            continue
        n += 1
        if _SHEET_DIR.search(p.parent.name):
            stem = re.sub(r"[_\-]?(sheet|ref|reference|v?\d+)$", "", p.stem.lower())
            if 2 < len(stem) < 30 and stem.isalpha():
                out.add(stem)
    return out


def score_brief_evidence(bf: BriefFacts, text: str, stems: set[str]) -> float:
    """How much of `bf` the run's own text and sheet names echo. Title and a
    `Prompt N` mention are strong; each character name, the mandated line
    and each sheet stem matching a character token count once."""
    t = normalise(text)
    score = 0.0
    if bf.title and normalise(bf.title) in t:
        score += 3
    if bf.prompt_no and re.search(rf"\bprompt[ _\-]?0*{bf.prompt_no}\b(?!\d)", t):
        score += 2
    for c in bf.characters:
        if normalise(c.name) in t:
            score += 1
        if any(tok in stems for tok in c.tokens):
            score += 1
    for line in bf.mandated_lines:
        if normalise(line) in t:
            score += 2
    return score


def match_brief_by_content(briefs: dict[int, BriefFacts], text: str,
                           stems: set[str]) -> BriefFacts | None:
    """The brief the run's footprints point at, when they point clearly:
    best score at least 3, and ahead of the runner-up by at least 2."""
    if not briefs:
        return None
    ranked = sorted(((score_brief_evidence(bf, text, stems), n) for n, bf in briefs.items()),
                    reverse=True)
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else (0.0, None))
    if best[0] >= 3 and best[0] - second[0] >= 2:
        return briefs[best[1]]
    return None


_WORD_CAP = re.compile(r"(?:<|below|under|no more than|at most|within|max(?:imum)?)\s*"
                       r"(\d{2,3})\s+(?:spoken\s+)?words", re.I)
_NO_VO = re.compile(r"\bno (?:explanatory |narrated |spoken )?voice-?\s?over\b", re.I)
_MANDATED = re.compile(r"mandated line[^\"“]{0,40}[\"“]([^\"”]{6,200})[\"”]", re.I)
_TITLE = re.compile(r"(?:^|\n)\s*(?:#+\s*)?(?:trajectory\s*[—\-:]+\s*)?prompt\s+\d+\s*[,:—\-]+\s*[\"“]([^\"”\n]{4,80})[\"”]", re.I)
_PROMPT_NO = re.compile(r"\bprompt[ _\-]?(\d{1,2})\b", re.I)
_FULL_NAME_TAIL = r")\s+([A-Z][a-z]{1,20})\b"


def reconstruct_brief(run_root: Path, text: str | None = None,
                      stems: set[str] | None = None) -> BriefFacts | None:
    """A partial brief from the run's footprints, for a run that never quotes
    the template. Only what the run states is filled; the rest stays None
    and the expectations that need it never exist. None when not even a cast
    or a constraint could be found."""
    root = Path(run_root)
    if text is None:
        text = "\n".join(t for _p, t in iter_run_texts(root))
    if stems is None:
        stems = _sheet_stems(root)
    chars: list[Character] = []
    for stem in sorted(stems):
        m = re.search(r"\b(" + re.escape(stem.capitalize()) + _FULL_NAME_TAIL, text)
        name = f"{m.group(1)} {m.group(2)}" if m else stem.capitalize()
        chars.append(Character(name=name, first=stem))
    bf = BriefFacts(prompt_no=0)
    bf.characters = chars
    m = _WORD_CAP.search(text)
    bf.word_cap = int(m.group(1)) if m else None
    bf.voiceover_allowed = False if _NO_VO.search(text) else None
    bf.mandated_lines = list(dict.fromkeys(x.strip() for x in _MANDATED.findall(text)))
    m = _TITLE.search(text)
    bf.title = m.group(1).strip() if m else ""
    m = _PROMPT_NO.search(text)
    bf.prompt_no = int(m.group(1)) if m else 0
    if not chars and bf.word_cap is None and bf.voiceover_allowed is None \
            and not bf.mandated_lines:
        return None
    return bf


def _match_verbatim(briefs: dict[int, BriefFacts], run) -> tuple[BriefFacts | None, str]:
    """case name, then the brief the trajectory quotes verbatim."""
    n = prompt_number(getattr(run, "case", "") or "")
    if n is not None and n in (briefs or {}):
        return briefs[n], "case_name"
    root = getattr(run, "root", None)
    if root is None:
        return None, "none"
    emb = find_brief_in_run(Path(root))
    if emb is None:
        return None, "none"
    if emb.prompt_no and emb.prompt_no in (briefs or {}):
        return briefs[emb.prompt_no], "embedded_number"
    if emb.title:
        for bf in (briefs or {}).values():
            if bf.title.strip().lower() == emb.title.strip().lower():
                return bf, "embedded_title"
    # A bare template has no number and no title; its cast line does not
    # change between prompt.md and the copy the agent was handed.
    for field_name in ("Characters", "One-sentence synopsis"):
        mine = normalise(emb.raw.get(field_name, ""))
        if len(mine) < 40:
            continue
        for bf in (briefs or {}).values():
            if normalise(bf.raw.get(field_name, "")) == mine:
                return bf, "embedded_content"
    return emb, "embedded_text"


def match_brief(briefs: dict[int, BriefFacts], run) -> tuple[BriefFacts | None, str]:
    """Which brief this run answers, and how that was decided.

        case_name         the case directory is `prompt_N...` and prompt.md has N
        embedded_number   the trajectory embeds `### Prompt N:` and prompt.md has N
        embedded_title    the embedded brief's title matches one in prompt.md
        embedded_content  the embedded brief's cast line matches one in prompt.md
        embedded_text     no prompt.md match; the embedded brief is used as-is
        content_match     nothing verbatim, but the run's own text and sheet
                          names point clearly at one brief in prompt.md
                          (claudecode compiles the brief into series.js and
                          names it in TRAJECTORY.md; it never quotes it)
        reconstructed     no prompt.md to point at: a partial brief rebuilt
                          from the run's footprints (cast from sheet stems,
                          word cap, voiceover rule, mandated line)
        none              nothing to compile expectations from

    Nothing here is asked of the user.
    """
    bf, how = _match_verbatim(briefs, run)
    if bf is not None:
        return bf, how
    root = getattr(run, "root", None)
    if root is None:
        return None, "none"
    text = "\n".join(t for _p, t in iter_run_texts(Path(root)))
    stems = _sheet_stems(Path(root))
    got = match_brief_by_content(briefs or {}, text, stems)
    if got is not None:
        return got, "content_match"
    rec = reconstruct_brief(Path(root), text, stems)
    if rec is not None:
        return rec, "reconstructed"
    return None, "none"


def apply_brief_targets(runs, matched: dict) -> int:
    """A target no record supplied is filled from the matched brief --
    episodes, seconds and aspect are properties of the prompt. Returns how
    many runs were filled."""
    from stage2_delivery.discover import Target
    n = 0
    for r in runs:
        bf, _how = matched.get(r.run_id, (None, "none"))
        if bf is not None and not r.target.known and (bf.episodes or bf.seconds_per_episode):
            r.target = Target(bf.episodes, bf.seconds_per_episode, bf.aspect_ratio, None,
                              source="brief")
            n += 1
    return n


def load_briefs(data_root: Path) -> dict[int, BriefFacts]:
    for cand in (data_root / "prompt.md", data_root.parent / "prompt.md"):
        if cand.is_file():
            return parse_prompts(cand.read_text(encoding="utf-8"))
    return {}


_CASE_NO = re.compile(r"prompt[_\-]?(\d+)", re.I)


def prompt_number(case_name: str) -> int | None:
    m = _CASE_NO.search(case_name)
    return int(m.group(1)) if m else None


def normalise(s: str) -> str:
    """For matching a mandated line against a transcript: case, punctuation,
    curly quotes and whitespace all fold."""
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\w\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()
