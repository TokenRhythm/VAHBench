"""The tool-event reader, and the fixes the sixth audit of the checklist found.

Three harnesses export structured tool calls in three record shapes. Two of
them were declared "prose only" by a comment in the extractor after one
failed lookup, and every count, error, parameter and reference they carry was
left for the language model. These pin that all three shapes are read, that
the reading is exact, and that the traps the HTTP reader already fell into
(polls counted as generations) are not re-dug here.

Fixtures are cut down from the real events.jsonl files under data/harness.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


import stage1_inputs.extractors as extractors                                       # noqa: E402
from stage1_inputs.extractors import deterministic_extract
from stage1_inputs.extractors import lm_extract     # noqa: E402
from stage1_inputs.schema import CallInventory
from stage1_inputs.schema import RequestedParams
from stage1_inputs.schema import TrajectoryFacts
from stage1_inputs.schema import compare

PROMPT = ("Vertical 9:16 cinematic clip. Maya in a navy blazer with a silver "
          "pencil behind her ear watches Victor through glass.")
MUSIC = "Subtle corporate thriller score, cold and tense, sparse electronic pulse."


def _run(files: dict[str, object]) -> Path:
    d = Path(tempfile.mkdtemp())
    for name, body in files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, list):
            body = "\n".join(json.dumps(r, ensure_ascii=False) for r in body)
        p.write_text(body, encoding="utf-8")
    return d


# ---------------------------------------------------------------- shapes --
def openclaw_events():
    def call(i, name, args):
        return {"type": "tool.call", "ts": f"t{i}",
                "data": {"toolCallId": f"c{i}", "name": name, "arguments": args}}

    def result(i, name, is_error, text="ok", details=None):
        return {"type": "tool.result",
                "data": {"message": {"toolCallId": f"c{i}", "toolName": name,
                                     "isError": is_error,
                                     "content": [{"type": "text", "text": text}],
                                     "details": details or {}}}}
    return [
        {"type": "session.started"},
        {"type": "assistant.message", "ts": "a1", "content": "planning"},
        call(1, "write", {"path": "story/series_bible.md", "content": "..."}),
        result(1, "write", False, "Successfully wrote 2963 bytes"),
        call(2, "music_generate", {"prompt": MUSIC, "durationSeconds": 60,
                                   "filename": "ep1_score.mp3"}),
        result(2, "music_generate", False),
        call(3, "image_generate", {"prompt": PROMPT + " Cover.", "aspectRatio": "9:16",
                                   "filename": "cover.png"}),
        result(3, "image_generate", False, details={"status": "started"}),
        call(4, "image_generate", {"action": "status"}),           # a poll
        result(4, "image_generate", False),
        call(5, "video_generate", {"prompt": PROMPT + " Shot one.", "aspectRatio": "9:16",
                                   "resolution": "720P", "filename": "ep1_hook.mp4"}),
        result(5, "video_generate", False),
        call(6, "video_generate", {"action": "status"}),           # a poll
        result(6, "video_generate", False),
        call(7, "exec", {"command": "ffmpeg ..."}),
        result(7, "exec", True, "ffmpeg: command not found"),
        {"type": "assistant.message", "ts": "a2", "content": "done"},
    ]


def hermes_events(duplicate=True):
    """OpenAI-style tool_calls on the assistant message; plugin tools wrapped
    in a generic `tool_call`; every assistant turn dumped twice."""
    def assistant(i, tcs):
        row = {"type": "message", "ts": f"t{i}", "role": "assistant",
               "content": "", "tool_calls": tcs, "source": "response_dump"}
        rows = [row]
        if duplicate:
            rows.append({**row, "source": "request_dump"})
        return rows

    def tc(i, name, args):
        return {"id": f"call_{i}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}

    def wrapped(i, name, args):
        return tc(i, "tool_call", {"name": name, "arguments": args})

    def tool(i, content):
        return {"type": "message", "ts": f"r{i}", "role": "tool",
                "content": json.dumps(content), "tool_call_id": f"call_{i}",
                "source": "request_dump"}
    rows = [{"type": "message", "role": "user", "content": "You are a director..."}]
    rows += assistant(1, [tc(1, "image_generate", {"prompt": PROMPT, "aspect_ratio": "portrait"})])
    rows.append(tool(1, {"success": True, "image": "$HERMES_HOME/cache/gen_1.png",
                         "model": "google/gemini-3-pro-image-preview"}))
    rows += assistant(2, [wrapped(2, "video_generate", {
        "prompt": PROMPT + " Push in.", "aspect_ratio": "9:16", "duration": 5,
        "resolution": "720p", "image_url": "$WORKSPACE_DIR/assets/frame_ep1_hook.png"})])
    rows.append(tool(2, {"success": True, "video": "$SCRATCH/v2.mp4",
                         "model": "bytedance/seedance-2.0-mini"}))
    rows += assistant(3, [wrapped(3, "video_generate", {
        "prompt": PROMPT + " Rejected.", "aspect_ratio": "9:16", "duration": 11,
        "image_url": "bad"})])
    rows.append(tool(3, {"success": False, "video": None,
                         "error": "OpenRouter rejected the video request (HTTP 400): Invalid URL"}))
    rows += assistant(4, [tc(4, "clarify", {"choices": ["a", "b"]})])
    rows.append(tool(4, {"error": "Clarify tool is not available in this execution context."}))
    rows += assistant(5, [tc(5, "terminal", {"command": "ls"})])
    rows.append(tool(5, {"output": "ok", "exit_code": 0, "error": None}))
    return rows


def opensquilla_events():
    def assistant(i, blocks):
        return {"type": "message", "ts": f"t{i}", "turnIdx": i, "role": "assistant",
                "content": [{"type": "text", "text": "..."}] + blocks}

    def block(i, name, args):
        return {"type": "toolCall", "name": name, "id": f"call_{i}", "arguments": args}

    def result(i, name, payload, is_error=False):
        return {"type": "message", "ts": f"r{i}", "role": "toolResult",
                "toolCallId": f"call_{i}", "toolName": name, "isError": is_error,
                "content": [{"type": "text", "text": json.dumps(payload)}]}
    return [
        {"type": "llm_call", "model": "m", "input": {}, "output": {}},
        assistant(1, [block(1, "image_generate", {"prompt": PROMPT, "size": "1024x1536"})]),
        result(1, "image_generate", {"status": "ok", "path": "$WORKSPACE_DIR/gen-1.png",
                                     "model": "google/gemini-3-pro-image-preview"}),
        assistant(2, [block(2, "music_generate", {"prompt": MUSIC, "duration_seconds": 90,
                                                  "output_path": "assets/audio/bgm.wav"})]),
        result(2, "music_generate", {"status": "not_available", "tool": "music_generate",
                                     "note": "ElevenLabs music provider is disabled or not configured."}),
        assistant(3, [block(3, "exec_command", {"command": "ffmpeg -i x.png -vf zoompan ..."})]),
        result(3, "exec_command", {"stdout": "", "exit_code": 0}),
    ]


class TestOpenclawShape(unittest.TestCase):
    def setUp(self):
        self.root = _run({".openclaw/trajectory-exports/trajectory/events.jsonl":
                          openclaw_events()})
        self.f = deterministic_extract(self.root, self.root)

    def test_counts_by_tool_kind(self):
        self.assertEqual(self.f.calls.total, 7)
        self.assertEqual(self.f.calls.image_gen, 1)
        self.assertEqual(self.f.calls.video_gen, 1)
        self.assertEqual(self.f.calls.audio_gen, 1)
        self.assertIn("events.jsonl", self.f.formats_seen)

    def test_status_polls_are_not_generations(self):
        """The 7.5x trap the HTTP reader fell into, not re-dug: a call with
        {"action": "status"} against a generation tool is a poll, so it lands
        in `total` and in none of the generation counts."""
        self.assertEqual(self.f.calls.total, 7)
        self.assertEqual(self.f.calls.video_gen, 1)

    def test_errors_from_the_explicit_flag(self):
        """One explicit failure the log never shows succeeding: one unrecovered
        attempt out of five (the poll is part of the attempt it waits on)."""
        e = self.f.errors
        self.assertEqual((e.n_attempts, e.n_recovered, e.n_unrecovered), (None, None, None))
        self.assertTrue(self.f.reliability_audit["unknown"])  # queued image has no linked terminal result

    def test_prompts_are_image_and_video_only_in_call_order(self):
        """Regression: prompts were inserted at index 0 one by one, which
        reversed them, and music prompts rode along into a list that is
        scored for wardrobe words."""
        self.assertEqual(len(self.f.requested.prompts), 2)
        self.assertTrue(self.f.requested.prompts[0].endswith("Cover."))
        self.assertTrue(self.f.requested.prompts[1].endswith("Shot one."))
        self.assertFalse(any("thriller score" in p for p in self.f.requested.prompts))

    def test_requested_parameters(self):
        self.assertEqual(self.f.requested.aspect_ratios, ["9:16"])
        self.assertEqual(self.f.requested.resolutions, ["720P"])

    def test_produced_from_filename_arguments_and_write_calls(self):
        paths = [Path(a.path).name for a in self.f.produced]
        self.assertIn("series_bible.md", paths)
        self.assertIn("cover.png", paths)
        self.assertIn("ep1_hook.mp4", paths)
        self.assertIn("ep1_score.mp3", paths)
        kinds = {Path(a.path).name: a.kind for a in self.f.produced}
        self.assertEqual(kinds["ep1_score.mp3"], "audio")
        self.assertEqual(kinds["series_bible.md"], "text")


class TestHermesShape(unittest.TestCase):
    def setUp(self):
        self.root = _run({".hermes/trajectory-exports/trajectory/events.jsonl":
                          hermes_events()})
        self.f = deterministic_extract(self.root, self.root)

    def test_wrapped_plugin_calls_are_unwrapped(self):
        self.assertEqual(self.f.calls.video_gen, 2)
        self.assertEqual(self.f.calls.image_gen, 1)

    def test_duplicated_dumps_count_once(self):
        """hermes writes every assistant turn as a request dump and a
        response dump. Read naively that doubled every inventory figure."""
        self.assertEqual(self.f.calls.total, 5)

    def test_errors_from_the_json_content(self):
        # HTTP 400 on a bad reference URL is the agent's and never recovers:
        # one unrecovered attempt. A tool the harness does not provide is the
        # environment's and leaves the sequence -- it is nobody's attempt.
        e = self.f.errors
        self.assertEqual((e.n_attempts, e.n_recovered, e.n_unrecovered), (None, None, None))
        self.assertTrue(self.f.reliability_audit["unknown"])  # failed request has no task identity

    def test_rejected_video_requests_generate_nothing(self):
        self.assertEqual(self.f.requested.clip_durations, [5.0])
        self.assertEqual(self.f.requested.generated_seconds, 5.0)
        self.assertEqual(self.f.requested.generated_seconds_source, "per_request")

    def test_conditioning_from_image_url(self):
        """The docstring used to say no format records this."""
        self.assertEqual(len(self.f.conditioning), 1)
        self.assertIn("frame_ep1_hook.png", self.f.conditioning[0].references[0])
        self.assertEqual(self.f.conditioning[0].kind, "scene")

    def test_produced_from_result_paths(self):
        names = sorted(Path(a.path).name for a in self.f.produced)
        self.assertEqual(names, ["gen_1.png", "v2.mp4"])

    def test_only_unread_fields_are_left_for_the_model(self):
        gaps = extractors.needs_lm(self.f)
        self.assertIn("errors.n_unrecovered", gaps)
        self.assertIn("errors.n_attempts", gaps)
        for g in ("produced", "requested.prompts",
                  "conditioning", "requested.generated_seconds"):
            self.assertNotIn(g, gaps)


class TestOpensquillaShape(unittest.TestCase):
    def setUp(self):
        self.root = _run({".opensquilla/trajectory-exports/trajectory/events.jsonl":
                          opensquilla_events()})
        self.f = deterministic_extract(self.root, self.root)

    def test_tool_call_blocks_in_content(self):
        self.assertEqual(self.f.calls.image_gen, 1)
        self.assertEqual(self.f.calls.audio_gen, 1)
        self.assertEqual(self.f.calls.total, 3)
        self.assertEqual(len(self.f.requested.prompts), 1)

    def test_provider_not_configured_is_infrastructure(self):
        """The provider failure is the environment's: it leaves the sequence
        instead of becoming an unrecovered attempt, so the run reads clean.

        The shell operation cannot be tied to a production target, but it
        succeeded, so it cannot be hiding a failure: the determined task is
        reported and the unattributed record is counted beside it rather than
        blanking the reading."""
        e = self.f.errors
        self.assertEqual((e.n_attempts, e.n_recovered, e.n_unrecovered), (1, 0, 0))
        self.assertEqual(self.f.reliability_audit["excluded_infrastructure"], 1)
        self.assertEqual(self.f.reliability_audit["n_attempts"], 1)  # image only; shell production target is unknown
        self.assertEqual(self.f.reliability_audit["n_unattributed_records"], 1)
        self.assertEqual(self.f.reliability_audit["n_open_tasks"], 0)

    def test_no_video_generation_is_zero_seconds_exactly(self):
        """opensquilla built its episodes with ffmpeg zoompan over stills.
        A complete inventory with no video call in it is 0 s generated, and
        that is a finding, not a gap."""
        self.assertEqual(self.f.calls.video_gen, 0)
        self.assertEqual(self.f.requested.generated_seconds, 0.0)
        self.assertEqual(self.f.requested.generated_seconds_source, "per_request")

    def test_zero_video_requests_does_not_prove_zero_footage(self):
        from stage4_checks.reports import from_delivery
        from shared.types import Verdict
        ms = {m.key: m for m in from_delivery(3, None, 180.0, self.f)}
        self.assertIs(ms["provenance.duration_provenance"].verdict, Verdict.ABSTAIN)


class TestTraceReaderConsistency(unittest.TestCase):
    def test_rejected_posts_and_models_by_row(self):
        trace = [
            {"ts": 2.0, "seq": 1, "url": "https://x/api/v1/videos", "method": "POST",
             "status": 202, "duration": 8, "elapsed_ms": 1.0, "model": "vid-a"},
            {"ts": 1.0, "seq": 2, "url": "https://x/api/v1/images", "method": "POST",
             "status": 200, "elapsed_ms": 1.0, "model": "img-a"},
            {"ts": 3.0, "seq": 3, "url": "https://x/api/v1/videos", "method": "POST",
             "status": 400, "duration": 8, "elapsed_ms": 1.0, "model": "vid-a"},
        ]
        root = _run({"trace.jsonl": trace})
        f = deterministic_extract(root, root)
        # The 400 generated nothing.
        self.assertEqual(f.requested.generated_seconds, 8.0)
        # Rows are classified individually: the timestamp sort that load_calls
        # applies (image before video here) must not shuffle the kinds. Both
        # video POSTs count as requests; only the accepted one made seconds.
        self.assertEqual(f.calls.video_gen, 2)
        self.assertEqual(f.calls.image_gen, 1)

    def test_trace_with_no_video_posts_is_zero_generated(self):
        trace = [{"ts": 1.0, "seq": 1, "url": "https://x/api/v1/chat/completions",
                  "status": 200, "elapsed_ms": 1.0}]
        root = _run({"trace.jsonl": trace})
        f = deterministic_extract(root, root)
        self.assertEqual(f.requested.generated_seconds, 0.0)


class TestCostLedger(unittest.TestCase):
    def test_unbilled_kinds_are_unobservable_not_zero(self):
        """A ledger records what the skill charged for. A run that spoke its
        lines through a free local engine has audio calls the ledger cannot
        see, and audio_gen=0 was feeding the triangulation as 'the audio
        requirement was ignored'."""
        root = _run({"costs.jsonl": [{"kind": "image", "id": "g", "cost": 0.1},
                                     {"kind": "video", "id": "v", "cost": 0.5}]})
        f = deterministic_extract(root, root)
        self.assertEqual(f.calls.image_gen, 1)
        self.assertEqual(f.calls.video_gen, 1)
        self.assertIsNone(f.calls.audio_gen)

    def test_job_handles_are_not_model_ids(self):
        """The ledger's `id` is the model on image rows and a job handle on
        video rows; 18 handles were being reported as 18 video models."""
        root = _run({"costs.jsonl": [{"kind": "image", "id": "google/gemini-3.1-flash-image", "cost": 0.1},
                                     {"kind": "video", "id": "4TrG4XXG3og03Cc7A9yG", "cost": 0.5},
                                     {"kind": "video", "id": "UzM28cdOxXgva5iLonBh", "cost": 0.5}]})
        f = deterministic_extract(root, root)
        self.assertIsNone(f.produced)     # no `out` rows, so nothing to carry a model

    def test_stale_absolute_paths_are_relocated_under_the_run(self):
        """claudecode's ledger points into data/claudecode/..., a layout the
        data has left. 30 produced edges resolved to nothing."""
        root = _run({"assets/characters/maya.png": "png",
                     "costs.jsonl": [{"kind": "image", "id": "g", "cost": 0.1,
                                      "out": "C:\\old\\data\\claudecode\\p1\\assets\\characters\\maya.png"},
                                     {"kind": "image", "id": "g", "cost": 0.1,
                                      "out": "C:\\old\\gone\\nowhere.png"}]})
        f = deterministic_extract(root, root)
        self.assertTrue(Path(f.produced[0].path).is_file())
        self.assertTrue(f.produced[0].path.endswith("assets/characters/maya.png"))
        self.assertEqual(f.produced[1].path, "C:/old/gone/nowhere.png")   # as written


class TestWorkspaceScan(unittest.TestCase):
    def test_refs_in_a_generation_script_are_conditioning_edges(self):
        js = ("shots = [{\n"
              "  refs: ['maya', 'victor'],\n"
              "  image: 'Maya in a navy blazer with a silver pencil behind her ear, "
              "Victor beside her in a charcoal suit, glass room.',\n"
              "}];\n")
        root = _run({"series.js": js, "assets/characters/maya.png": "x"})
        f = deterministic_extract(root, root)
        self.assertEqual(len(f.conditioning), 1)
        self.assertEqual(f.conditioning[0].kind, "character")
        self.assertEqual(f.conditioning[0].references,
                         ["assets/characters/maya.png", "victor"])

    def test_more_scaffolding_is_skipped(self):
        root = _run({"SOUL.md": "prompt: 'You are becoming someone. Be genuinely helpful, not performatively.'",
                     "TOOLS.md": "video: 'Preferred voice for narration is Nova and the default speaker is the kitchen.'",
                     "USER.md": "x", "IDENTITY.md": "x"})
        f = deterministic_extract(root, root)
        self.assertIsNone(f.requested.prompts)

    def test_asset_manifest_is_not_a_bible(self):
        """`characters.json` matched the bible pattern and, being long, won."""
        root = _run({"characters.json": json.dumps({"maya": "assets/maya.png"} | {
                         f"x{i}": f"assets/x{i}.png" for i in range(200)}),
                     "series_bible.md": "- **Maya** navy blazer, silver pencil.\n"})
        f = deterministic_extract(root, root)
        self.assertEqual(f.continuity.source, "series_bible.md")

    def test_transcript_is_not_a_plan(self):
        self.assertIsNone(extractors._PLAN_FILE.search("transcript.md"))
        self.assertIsNotNone(extractors._PLAN_FILE.search("script.md"))

    def test_episode_heading_must_head_a_line(self):
        """A sentence mentioning 'episode 2' inside episode 1's plan used to
        relabel every shot after it; so did `ep1.mp4` in an output listing."""
        body = ("# EPISODE 1\n"
                "- **S1 (0-4s, Hook)**: Victor applauded inside the glass room today\n"
                "- **S2 (4-18s, Setup)**: this beat pays off in episode 2 when Maya returns\n"
                "outputs: ep3.mp4 written\n"
                "- **S3 (18-30s, Turn)**: Maya at her desk, pencil behind her ear\n")
        plan = extractors._scan_shot_plan(body)
        self.assertEqual([s.episode for s in plan], [1, 1, 1])
        body2 = "# EPISODE 1\n- (0-4s): a shot in the room\n## Episode 2\n- (0-4s): another shot in a hall\n"
        self.assertEqual([s.episode for s in extractors._scan_shot_plan(body2)], [1, 2])

    def test_lm_parts_skip_dependency_trees(self):
        root = _run({"node_modules/x/debug.log": "noise", "build/run.txt": "noise",
                     "run_video.log": "stage keys ok",
                     ".openclaw/trajectory-exports/trajectory/events.jsonl":
                         openclaw_events()})
        f = TrajectoryFacts()
        parts = extractors._lm_parts(root, f)
        joined = "\n".join(parts)
        self.assertIn("stage keys ok", joined)
        self.assertNotIn("noise", joined)


class TestCompareNormalisation(unittest.TestCase):
    def test_relative_path_and_bare_filename_agree(self):
        a, b = TrajectoryFacts(), TrajectoryFacts()
        a.continuity.source = "receipts_series/scripts/00_series_bible.md"
        b.continuity.source = "00_series_bible.md"
        self.assertIn("continuity.source", compare(a, b)["agree"])

    def test_measured_floats_agree_within_rounding_but_counts_do_not(self):
        a = TrajectoryFacts(requested=RequestedParams(generated_seconds=180.004),
                            calls=CallInventory(video_gen=18))
        b = TrajectoryFacts(requested=RequestedParams(generated_seconds=180.0),
                            calls=CallInventory(video_gen=17))
        r = compare(a, b)
        self.assertIn("requested.generated_seconds", r["agree"])
        self.assertEqual([d["field"] for d in r["differ"]], ["calls.video_gen"])

    def test_round_trip_through_dict(self):
        root = _run({".hermes/trajectory-exports/trajectory/events.jsonl": hermes_events()})
        f = deterministic_extract(root, root)
        g = TrajectoryFacts.from_dict(json.loads(json.dumps(f.to_dict())))
        self.assertEqual(g.to_dict(), f.to_dict())
        self.assertEqual(compare(f, g)["differ"], [])

    def test_real_differences_still_differ(self):
        a, b = TrajectoryFacts(), TrajectoryFacts()
        a.continuity.source = "story/series_bible.md"
        b.continuity.source = "creative_package.md"
        self.assertEqual([d["field"] for d in compare(a, b)["differ"]],
                         ["continuity.source"])


class TestLMChunkMerge(unittest.TestCase):
    def _run(self, replies: list[str], files: dict | None = None):
        root = _run(files or {"a.md": "x" * 50, "b.md": "y" * 50})
        it = iter(replies)
        seen = []

        def call_fn(prompt):
            seen.append(prompt)
            return next(it)
        # Include the repeated source-path heading in the per-file budget.
        f = lm_extract(root, model="m", revision="r", call_fn=call_fn, max_chars=80)
        return f, seen

    def test_counts_are_summed_across_slices(self):
        """First-wins made a three-slice log report the first third's counts."""
        f, seen = self._run([json.dumps({"calls": {"video_gen": 4, "total": 10},
                                         "errors": {"n_unrecovered": 1}}),
                             json.dumps({"calls": {"video_gen": 6, "total": 12},
                                         "errors": {"n_unrecovered": 0}})])
        self.assertEqual(len(seen), 2)
        self.assertEqual(f.calls.video_gen, 10)
        self.assertEqual(f.calls.total, 22)
        self.assertEqual(f.errors.n_unrecovered, 1)
        self.assertIn("slice 1 of 2", seen[0])
        self.assertIn("IN THIS SLICE", seen[0])

    def test_single_slice_is_told_it_is_whole(self):
        root = _run({"a.md": "short"})
        seen = []
        lm_extract(root, model="m", revision="r",
                   call_fn=lambda p: seen.append(p) or "{}")
        self.assertIn("whole execution log", seen[0])

    def test_totals_take_the_larger_reading(self):
        f, _ = self._run([json.dumps({"requested": {"generated_seconds": 180}}),
                          json.dumps({"requested": {"generated_seconds": 60}})])
        self.assertEqual(f.requested.generated_seconds, 180.0)

    def test_shot_plan_dedups_on_span_not_wording(self):
        """hermes came back with 57 shots for a 40-shot plan: the same shot
        from two slices with slightly different text."""
        f, _ = self._run([
            json.dumps({"continuity": {"shot_plan": [
                {"start": 0, "end": 3, "text": "Victor applauded inside the room", "episode": 1}]}}),
            json.dumps({"continuity": {"shot_plan": [
                {"start": 0, "end": 3, "text": "Victor applauded inside the glass room, wide", "episode": 1},
                {"start": 3, "end": 8, "text": "Maya watches through the glass", "episode": 1}]}})])
        self.assertEqual(len(f.continuity.shot_plan), 2)
        self.assertIn("wide", f.continuity.shot_plan[0].text)   # richer text kept


class TestLMTypeCoercion(unittest.TestCase):
    def _run(self, payload):
        root = _run({"TRAJECTORY.md": "x"})
        return lm_extract(root, model="m", revision="r", call_fn=lambda p: json.dumps(payload))

    def test_numeric_strings_are_accepted(self):
        f = self._run({"errors": {"n_unrecovered": "2", "n_attempts": "10"},
                       "requested": {"generated_seconds": "180", "clip_durations": ["10", 10]}})
        self.assertEqual(f.errors.n_unrecovered, 2)
        self.assertEqual(f.errors.n_attempts, 10)
        self.assertEqual(f.requested.generated_seconds, 180.0)
        self.assertEqual(f.requested.clip_durations, [10.0, 10.0])

    def test_wrong_shapes_are_dropped_and_noted_not_crashed(self):
        f = self._run({"calls": {"total": "many", "video_gen": True},
                       "claims": {"declared_deliverables": {"a": 1}},
                       "requested": {"generated_seconds": [1, 2]}})
        self.assertIsNone(f.calls.total)
        self.assertIsNone(f.calls.video_gen)
        self.assertIsNone(f.claims.declared_deliverables)
        self.assertIsNone(f.requested.generated_seconds)
        self.assertTrue(any("类型不符" in n for n in f.notes))

    def test_a_string_where_a_list_is_wanted_is_wrapped(self):
        f = self._run({"claims": {"deviations": "backend redirected"}})
        self.assertEqual(f.claims.deviations, ["backend redirected"])

    def test_prompt_asks_for_clips_per_episode_and_forbids_derived_spans(self):
        seen = []
        root = _run({"TRAJECTORY.md": "x"})
        lm_extract(root, model="m", revision="r", call_fn=lambda p: seen.append(p) or "{}")
        self.assertIn("declared_clips_per_episode", seen[0])
        self.assertIn("deriving times from durations is estimating", seen[0])
        self.assertIn("Music, speech and sound prompts do", seen[0])


class TestMergeRederives(unittest.TestCase):
    def test_derived_total_is_recomputed_over_merged_counts(self):
        """The LM's derived figure used the LM's own video_gen; the merged
        checklist keeps the deterministic count beside it."""
        import run as run_eval
        base = TrajectoryFacts(calls=CallInventory(video_gen=18))
        lm = TrajectoryFacts(calls=CallInventory(video_gen=6),
                             requested=RequestedParams(clip_durations=[10.0]))
        extractors.derive_generated_seconds(lm)
        self.assertEqual(lm.requested.generated_seconds, 60.0)
        merged = run_eval._merge_facts(base, lm)
        self.assertEqual(merged.calls.video_gen, 18)
        self.assertEqual(merged.requested.generated_seconds, 180.0)
        self.assertEqual(merged.requested.generated_seconds_source, "derived")


class TestObservedAbsenceIsNotUnobservable(unittest.TestCase):
    """An empty reading and an unreadable channel are different facts.

    `None` means the log has no such channel, and the check abstains -- out of
    the denominator. `[]` means this reader saw every generation call and none
    of them carried a reference, and the check fails -- in the denominator.
    Recording the second as the first said "the trajectory does not record
    references" about openclaw, whose 23 generation calls record their
    arguments in full and simply contain no image, and it spent an LM call per
    run re-deriving an answer already in hand: `conditioning` was the largest
    single gap across `data/`, 69 runs of 80.
    """

    def setUp(self):
        self.root = _run({".openclaw/trajectory-exports/trajectory/events.jsonl":
                          openclaw_events()})
        self.f = deterministic_extract(self.root, self.root)

    def test_generation_calls_without_references_read_as_empty_not_none(self):
        self.assertIsNotNone(self.f.conditioning)
        self.assertEqual(self.f.conditioning, [])
        # the precondition: there were generation calls to look at
        self.assertTrue(self.f.calls.image_gen or self.f.calls.video_gen)

    def test_an_observed_absence_fails_rather_than_abstains(self):
        import stage4_checks.reports as extract
        from shared.types import Verdict
        m = {x.key: x for x in extract.from_facts(self.f)}["provenance.conditioning_edges"]
        self.assertIs(m.verdict, Verdict.FAIL)
        self.assertEqual(m.detail["n_edges"], 0)

    def test_it_is_no_longer_a_gap_for_the_model(self):
        self.assertNotIn("conditioning", extractors.needs_lm(self.f))

    def test_a_log_with_no_generation_calls_stays_unobservable(self):
        """The claim is "we saw the calls and none had a reference". With no
        generation call in the log there is nothing to have seen, so None."""
        events = "\n".join(json.dumps(e) for e in [
            {"type": "tool.call", "ts": "t1",
             "data": {"toolCallId": "c1", "name": "write",
                      "arguments": {"path": "notes.md", "content": "x"}}},
            {"type": "tool.result", "ts": "t1", "data": {"toolCallId": "c1"}},
        ])
        root = _run({".openclaw/trajectory-exports/trajectory/events.jsonl": events})
        f = deterministic_extract(root, root)
        self.assertIsNone(f.conditioning)
        self.assertIn("conditioning", extractors.needs_lm(f))

    def test_a_generation_script_still_overturns_an_observed_absence(self):
        """The zero is only as good as "this reader saw every generation
        call". An agent that generates from a script it wrote calls the API
        outside the tool channel, and `_from_agent_text` runs after this one."""
        # A generation script the agent wrote: a prompt with the character
        # sheets it was conditioned on beside it, claudecode's series.js shape.
        script = ("const series = { cover: { refs: ['maya', 'victor'], image: "
                  "'Two-shot separated by a vertical steel-mullioned glass "
                  "partition that splits the frame, the woman in the navy "
                  "blazer in cold sharp focus on the right, the man in "
                  "charcoal softer on the left.' } };")
        root = _run({".openclaw/trajectory-exports/trajectory/events.jsonl":
                     openclaw_events(), "series.js": script})
        f = deterministic_extract(root, root)
        self.assertTrue(f.conditioning)
        self.assertEqual(f.conditioning[0].kind, "character")
        self.assertEqual(sorted(f.conditioning[0].references), ["maya", "victor"])


class TestHttpTraceCannotObserveConditioning(unittest.TestCase):
    def test_a_request_record_carries_body_bytes_not_the_body(self):
        """The one backend that must keep saying None: an HTTP trace records
        that a call was made, never what was in it."""
        trace = [{"ts": 1.0, "seq": 1, "url": "https://x/api/v1/videos",
                  "method": "POST", "status": 202, "duration": 8},
                 {"ts": 2.0, "seq": 2, "url": "https://x/api/v1/images",
                  "method": "POST", "status": 200}]
        root = _run({"trace.jsonl": "\n".join(json.dumps(r) for r in trace)})
        f = deterministic_extract(root, root)
        self.assertIsNone(f.conditioning)
        self.assertIn("conditioning", extractors.needs_lm(f))


if __name__ == "__main__":
    unittest.main(verbosity=2)
