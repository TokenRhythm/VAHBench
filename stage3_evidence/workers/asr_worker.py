"""Transcribe one video with Whisper. Runs inside the av venv (it has whisper
and torch-cpu); the orchestrator never imports torch.

    python asr_worker.py --video V --out O [--model base]

Output: {"model", "language", "text", "n_words", "segments": [{start, end, text}]}
Greedy decoding, fixed model name, temperature 0 -- the same reproducibility
stance as every other model-backed step: fixed weights extract, rules judge.
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--language", default=None)
    a = ap.parse_args()

    import whisper  # noqa: WPS433  (venv-local)
    model = whisper.load_model(a.model, device="cpu")
    res = model.transcribe(a.video, language=a.language, temperature=0.0,
                           beam_size=None, best_of=None, fp16=False,
                           condition_on_previous_text=True, verbose=False, word_timestamps=True)
    segs = [{"start": round(float(s["start"]), 2), "end": round(float(s["end"]), 2),
             "text": s["text"].strip(),
             "words": [{"start": round(float(w["start"]), 3),
                        "end": round(float(w["end"]), 3), "word": w["word"]}
                       for w in s.get("words", [])]} for s in res.get("segments", [])]
    text = " ".join(s["text"] for s in segs).strip()
    payload = {"model": f"whisper-{a.model}", "language": res.get("language"),
               "text": text, "n_words": len(text.split()), "segments": segs,
               "timing_schema": "word-timestamps-v1"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
