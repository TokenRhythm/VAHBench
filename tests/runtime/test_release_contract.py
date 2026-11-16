"""Public configuration and CLI behavior must be explicit and fail closed."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared.settings import Settings
from shared.preflight import Preflight
from run import main, load_cut_points


class ConfigurationContract(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp) / "settings.yaml"
            path.write_text(text)
            return Settings.load(path)

    def test_empty_module_list_disables_modules(self):
        self.assertEqual(self.load("modules: []").modules, ())

    def test_legacy_none_selection_remains_supported(self):
        self.assertEqual(self.load("modules: [none]").modules, ())

    def test_unknown_module_and_setting_are_errors(self):
        for text in ("modules: [visua]", "modules: visual", "job: 1",
                     "lm: {modle: typo}", "lm: [bad]", "[not, a, mapping]",
                     "[]", "false", "lm: false"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_invalid_budgets_are_errors(self):
        for text in ("jobs: 0", "lm: {max_chars: 0}", "lm: {max_tokens: -1}"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_missing_config_and_threshold_files_are_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            for loader in (Settings.load, load_cut_points):
                with self.subTest(loader=loader), self.assertRaises(ValueError):
                    loader(Path(tmp) / "missing.yaml")

    def test_bad_thresholds_do_not_silently_change_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "thresholds.yaml"
            for text in ("bad: [", "[1, 2]", "epslion: 0.9"):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_cut_points(path)


class PublicCLIContract(unittest.TestCase):
    def test_full_evaluation_stops_before_extraction_when_detectors_missing(self):
        with patch("run.preflight.check", return_value=Preflight()), \
             patch("run.install_key"), patch("run.find_runs") as discover, \
             patch("run.extractors.deterministic_extract") as extract, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data", "unused"]), 2)
        discover.assert_not_called()
        extract.assert_not_called()

    def test_discovery_does_not_load_credentials(self):
        with patch("run.preflight.check", return_value=Preflight()), \
             patch("run.install_key") as key, patch("run.find_runs", return_value=[]), \
             tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data", "unused", "--out-dir", tmp,
                                   "--discover-only"]), 2)
        key.assert_not_called()

    def test_partial_opt_in_can_reach_discovery(self):
        with patch("run.preflight.check", return_value=Preflight()), \
             patch("run.install_key"), patch("run.find_runs", return_value=[]) as discover, \
             tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--data", "unused", "--out-dir", tmp,
                                   "--allow-partial"]), 2)
        discover.assert_called_once()
