import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import pandas as pd

from l1obs.cli import _build_parser, _load_constellation_manifest, main
from test_constellation import synthetic_constellation


class ConstellationCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.datasets = synthetic_constellation()
        self.manifest = self.root / "inputs" / "constellation.json"
        self.manifest.parent.mkdir()
        entries = {}
        for spacecraft, (name, data) in enumerate(self.datasets.items()):
            entries[name] = {}
            for kind, frame in data.items():
                frame = frame.rename_axis("time")
                extension = ".csv" if spacecraft % 2 or spacecraft == 0 else ".parquet"
                filename = f"{spacecraft}_{kind}{extension}"
                path = self.manifest.parent / filename
                if extension == ".csv":
                    frame.to_csv(path)
                else:
                    frame.to_parquet(path)
                entries[name][kind] = filename
        self.manifest.write_text(json.dumps(entries))
        patcher = patch("socket.create_connection", side_effect=AssertionError("network forbidden"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        plt.close("all")

    def invoke(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["l1obs", "constellation", "--manifest", str(self.manifest), *arguments]), \
                redirect_stdout(output), redirect_stderr(errors):
            with self.assertRaises(SystemExit) as exit_result:
                main()
        return exit_result.exception.code, output.getvalue(), errors.getvalue()

    def test_loads_mixed_local_formats_relative_to_manifest_without_alignment(self):
        loaded = _load_constellation_manifest(self.manifest)
        self.assertEqual(list(loaded), list(self.datasets))
        for name, data in self.datasets.items():
            for kind, expected in data.items():
                expected = expected.rename_axis("time")
                pd.testing.assert_frame_equal(loaded[name][kind], expected, check_freq=False)
        self.assertNotEqual(len(loaded["Wind"]["magnetic"]), len(loaded["Wind"]["positions"]))

    def test_command_generates_png_pdf_and_closes_figure(self):
        output = self.root / "plots" / "comparison.png"
        code, stdout, stderr = self.invoke("--coordinate-system", "GSE", "--output", str(output), "--pdf")
        self.assertEqual(code, 0, stderr)
        self.assertGreater(output.stat().st_size, 1000)
        self.assertGreater(output.with_suffix(".pdf").stat().st_size, 1000)
        self.assertIn(str(output), stdout)
        self.assertIn(str(output.with_suffix(".pdf")), stdout)
        self.assertFalse(plt.get_fignums())

    def test_options_are_forwarded_and_default_output_is_used(self):
        with patch("l1obs.cli.default_paths") as paths, \
                patch("l1obs.cli.plot_six_spacecraft_B_and_geometry") as plot, \
                patch("l1obs.cli.plt.close") as close:
            paths.return_value.output = self.root / "output"
            names = list(self.datasets)[::-1]
            code, _, stderr = self.invoke(
                "--coordinate-system", "GSE", "--start", "2026-09-01T00:00:01Z",
                "--end", "2026-09-01T00:00:50Z", "--independent-b-ylim",
                "--magnitude-column", "b_mag", "--magnetic-max-gap", "5s",
                "--position-max-gap", "2min", "--show", "--spacecraft-order", *names,
            )
            self.assertEqual(code, 0, stderr)
            options = plot.call_args.kwargs
            self.assertEqual(options["output_path"], self.root / "output" / "constellation.png")
            self.assertEqual(options["time_range"], (pd.Timestamp("2026-09-01T00:00:01Z"),
                                                     pd.Timestamp("2026-09-01T00:00:50Z")))
            self.assertFalse(options["common_B_ylim"])
            self.assertEqual(options["magnitude_column"], "b_mag")
            self.assertEqual(options["magnetic_max_gap"], pd.Timedelta(seconds=5))
            self.assertEqual(options["position_max_gap"], pd.Timedelta(minutes=2))
            self.assertTrue(options["show"])
            self.assertEqual(options["spacecraft_names"], names)
            close.assert_called_once_with(plot.return_value)

    def test_custom_columns_and_units_are_forwarded(self):
        with patch("l1obs.cli.plot_six_spacecraft_B_and_geometry") as plot, patch("l1obs.cli.plt.close"):
            code, _, stderr = self.invoke("--b-components", "Bx", "By", "Bz",
                                          "--position-components", "x", "y", "z",
                                          "--b-units", "T", "--position-units", "AU")
            self.assertEqual(code, 0, stderr)
            options = plot.call_args.kwargs
            self.assertEqual(options["B_components"], ("Bx", "By", "Bz"))
            self.assertEqual(options["position_components"], ("x", "y", "z"))
            self.assertEqual(options["B_units"], "T")
            self.assertEqual(options["position_units"], "AU")

    def test_bad_count_duplicate_names_and_malformed_manifest(self):
        for text, message in [("{}", "exactly six"), ("[]", "exactly six"),
                              ('{"Wind": {}, "Wind": {}}', "Duplicate manifest key"),
                              ("not JSON", "Cannot read constellation manifest")]:
            with self.subTest(text=text):
                self.manifest.write_text(text)
                code, _, stderr = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn(message, stderr)
                self.assertNotIn("Traceback", stderr)

    def test_missing_file_bad_timestamp_and_missing_time_column(self):
        entries = json.loads(self.manifest.read_text())
        path = self.manifest.parent / entries["Wind"]["magnetic"]
        path.unlink()
        code, _, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("Cannot load Wind magnetic", stderr)
        path = self.manifest.parent / "bad.csv"
        entries["Wind"]["magnetic"] = "bad.csv"
        self.manifest.write_text(json.dumps(entries))
        for contents in ("bx_gse,by_gse,bz_gse\n1,2,3\n", "time,bx_gse\nnot-a-date,1\n"):
            path.write_text(contents)
            code, _, stderr = self.invoke()
            self.assertEqual(code, 1)
            self.assertIn("Cannot load Wind magnetic", stderr)
            self.assertNotIn("Traceback", stderr)

    def test_unsupported_extension_and_invalid_manifest_entry(self):
        entries = json.loads(self.manifest.read_text())
        entries["Wind"]["magnetic"] = "field.pkl"
        self.manifest.write_text(json.dumps(entries))
        code, _, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn(".parquet and .csv", stderr)
        entries["Wind"] = {"magnetic": 10}
        self.manifest.write_text(json.dumps(entries))
        code, _, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("requires magnetic and positions", stderr)

    def test_paired_range_gap_validation_and_plot_errors(self):
        code, _, stderr = self.invoke("--start", "2026-09-01")
        self.assertEqual(code, 1)
        self.assertIn("--start and --end together", stderr)
        code, _, stderr = self.invoke("--magnetic-max-gap", "0s")
        self.assertEqual(code, 2)
        self.assertIn("Invalid gap duration", stderr)
        code, _, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn("coordinate metadata", stderr)
        code, _, stderr = self.invoke("--coordinate-system", "GSM")
        self.assertEqual(code, 1)
        self.assertIn("coordinate-system mismatch", stderr)
        code, _, stderr = self.invoke("--coordinate-system", "GSE", "--output", "comparison.pdf")
        self.assertEqual(code, 1)
        self.assertIn("PNG path", stderr)

    def test_existing_commands_and_constellation_help(self):
        parser = _build_parser()
        self.assertEqual(parser.parse_args(["positions", "--time", "2026-09-01"]).command, "positions")
        self.assertEqual(parser.parse_args(["timeseries", "--start", "2026-09-01"]).command, "timeseries")
        code, stdout, _ = self.invoke("--help")
        self.assertEqual(code, 0)
        self.assertIn("--manifest", stdout)
        self.assertIn("--position-max-gap", stdout)


if __name__ == "__main__":
    unittest.main()
