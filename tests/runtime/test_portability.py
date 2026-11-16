"""Relocate the source tree and launch it outside its checkout.

These checks exercise path resolution and discovery, not detector/model quality.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from shared.paths import REPO_ROOT


class RelocatedCheckoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name).resolve()
        cls.checkout = cls.base / "另一台电脑 with spaces" / "VAHBench"
        cls.checkout.mkdir(parents=True)
        for name in ("shared", "config", "stage1_inputs", "stage2_delivery",
                     "stage3_evidence", "stage4_checks", "stage5_scoring"):
            shutil.copytree(REPO_ROOT / name, cls.checkout / name,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copy2(REPO_ROOT / "run.py", cls.checkout / "run.py")
        shutil.copy2(REPO_ROOT / "pyproject.toml", cls.checkout / "pyproject.toml")
        cls.foreign_cwd = cls.base / "启动目录 with spaces"
        cls.foreign_cwd.mkdir()
        cls.env = {key: value for key, value in os.environ.items()
                   if not key.startswith("VAB_EVAL_") and key not in {
                       "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"}}
        cls.env["PYTHONDONTWRITEBYTECODE"] = "1"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def execute(self, *args):
        result = subprocess.run([sys.executable, *args], cwd=self.foreign_cwd,
                                env=self.env, capture_output=True, text=True,
                                encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return result.stdout

    def code(self, body):
        return self.execute("-c", "import sys; sys.path.insert(0, " +
                            repr(str(self.checkout)) + ")\n" + body)

    def test_absolute_launcher_works_outside_checkout(self):
        self.assertIn("--data", self.execute(str(self.checkout / "run.py"), "--help"))

    def test_config_thresholds_workers_and_detectors_follow_checkout(self):
        output = self.code("""
import hashlib, json
from shared.paths import REPO_ROOT
from shared.settings import Settings, DEFAULT_CONFIG
from run import load_cut_points, cut_points_hash
from stage4_checks.dispatch import SCRIPTS
from stage3_evidence import collector
cfg = Settings.load()
assert DEFAULT_CONFIG == REPO_ROOT / 'config/default.yaml'
assert cfg.lm.api_key_file is None
assert all(path.is_relative_to(REPO_ROOT) for path in SCRIPTS.values())
worker = collector.WORKER
assert worker == REPO_ROOT / 'stage3_evidence/workers/asr_worker.py'
assert worker.is_file()
print(json.dumps({'root': str(REPO_ROOT), 'config': cfg.source,
                  'threshold_hash': cut_points_hash(load_cut_points(None)),
                  'configured_threshold_hash': cut_points_hash(load_cut_points(__import__('pathlib').Path(cfg.cut_points))),
                  'config_sha256': hashlib.sha256(DEFAULT_CONFIG.read_bytes()).hexdigest()}))
""")
        record = json.loads(output)
        self.assertEqual(Path(record["root"]), self.checkout)
        self.assertEqual(Path(record["config"]), self.checkout / "config/default.yaml")
        self.assertEqual(record["threshold_hash"], record["configured_threshold_hash"])
        self.assertEqual(record["config_sha256"], hashlib.sha256(
            (REPO_ROOT / "config/default.yaml").read_bytes()).hexdigest())

    def test_discovery_writes_to_callers_relative_output_directory(self):
        data = self.foreign_cwd / "输入 data" / "harness" / "model" / "prompt_01"
        data.mkdir(parents=True)
        (data / "run.json").write_text('{"target":{"episodes":1}}', encoding="utf-8")
        output = self.code("""
from unittest.mock import patch
import run
# No media are present. Only bypass the ffprobe capability check so discovery
# can be exercised on machines without FFmpeg; no probe result is fabricated.
with patch('shared.preflight.shutil.which', return_value=sys.executable):
    raise SystemExit(run.main(['--data', '输入 data', '--out-dir', '输出 reports', '--discover-only']))
""")
        self.assertIn("1", output)
        summary = self.foreign_cwd / "输出 reports" / "summary.json"
        self.assertTrue(summary.is_file())
        records = json.loads(summary.read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["evaluation_status"], "discovery_only")
        self.assertIsNone(records[0]["S"])
        self.assertFalse((self.checkout / "输出 reports").exists())


if __name__ == "__main__":
    unittest.main()
