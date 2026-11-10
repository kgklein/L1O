"""Offline end-to-end CLI tests with synthetic CSV/Parquet manifest pairs."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from l1obs.cli import main, _build_parser
from l1obs.analysis.shocks.pipeline import run_event
from test_shock_data import LocalEventFixture
from test_shock_timing import NORMAL, SPEED


class ShockCliTests(LocalEventFixture):
    def invoke(self, operation, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["l1obs", "shock", operation, "--event", str(self.path), *extra]), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as result:
                main()
        return result.exception.code, stdout.getvalue(), stderr.getvalue()

    def synthetic_accepted_event(self):
        epoch = self.start + pd.Timedelta(seconds=120)
        # Use a faster synthetic shock to keep every crossing inside fixture data.
        speed = SPEED * 4
        times = epoch + pd.to_timedelta((self.positions - self.positions.mean(axis=0)) @ NORMAL / speed, unit="s")
        self.config["arrivals"] = {name: {"time": time.isoformat(), "uncertainty_s": 1., "accepted": True,
                                        "method": "manual_selected", "notes": "synthetic test only"}
                                   for name, time in zip(self.names, times)}
        self.config["monte_carlo"] = {"samples": 40, "seed": 15, "recompute_positions": True}
        self.save_event()
        return speed

    def test_inspect_outputs_and_fit_gate(self):
        code, stdout, stderr = self.invoke("inspect", "--no-relative", "--zoom-seconds", "25")
        self.assertEqual(code, 0, stderr)
        output = self.root / "results"
        for filename in ("arrivals.csv", "inspection.json", "shock_inspection.png", "shock_inspection.pdf",
                         "shock_components.png", "shock_zoom.png", "shock_geometry.png"):
            self.assertGreater((output / filename).stat().st_size, 100)
        inspection = json.loads((output / "inspection.json").read_text())
        self.assertEqual(inspection["status"], "inspection_only")
        self.assertFalse(any(row["accepted"] for row in inspection["arrivals"]))
        code, _, stderr = self.invoke("fit", "--no-relative")
        self.assertEqual(code, 1)
        self.assertIn("accepted manual", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(json.loads((output / "shock_timing.json").read_text())["status"], "failed")
        self.assertFalse(plt.get_fignums())

    def test_full_fit_outputs_units_uncertainties_and_no_input_mutation(self):
        speed = self.synthetic_accepted_event()
        content = self.path.read_text()
        code, stdout, stderr = self.invoke("fit", "-v", "--zoom-seconds", "20")
        self.assertEqual(code, 0, stderr)
        self.assertIn("Speed:", stdout)
        output = self.root / "results"
        report = json.loads((output / "shock_timing.json").read_text())
        self.assertAlmostEqual(report["fit"]["speed_km_s"], speed, places=6)
        self.assertEqual(report["fit"]["method"], "weighted")
        self.assertEqual(report["fit"]["dof"], 2)
        self.assertEqual(report["monte_carlo"]["failed"], 0)
        self.assertEqual(len(report["leave_one_out"]), 6)
        self.assertEqual(len(report["geometry"]["interpolation"]), 6)
        self.assertEqual(self.path.read_text(), content)
        self.assertTrue(report["software_versions"]["numpy"])
        self.assertEqual(report["units"]["speed"], "km/s")
        residuals = pd.read_csv(output / "timing_residuals.csv")
        self.assertEqual(len(residuals), 6)
        self.assertLess(residuals.residual_s.abs().max(), 1e-7)
        for stem in ("shock_inspection", "shock_components", "shock_zoom", "shock_relative", "shock_geometry", "timing_diagnostics"):
            for suffix in (".png", ".pdf"):
                self.assertGreater((output / (stem + suffix)).stat().st_size, 1000)
        self.assertFalse(plt.get_fignums())

    def test_unweighted_four_member_subset_and_partial_uncertainties(self):
        self.synthetic_accepted_event()
        for name in self.names:
            self.config["arrivals"][name].pop("uncertainty_s")
        self.save_event()
        with patch("l1obs.analysis.shocks.pipeline._inspection_figures"), \
             patch("l1obs.analysis.shocks.pipeline.save_figure", side_effect=lambda fig, path: plt.close(fig)):
            report = run_event(self.path, operation="fit", spacecraft=self.names[:4])
        self.assertIsNone(report["monte_carlo"])
        self.assertEqual(report["fit"]["dof"], 0)
        self.assertEqual(report["fit"]["method"], "unweighted")
        self.assertFalse(report["leave_one_out"])
        self.config["arrivals"]["ACE"]["uncertainty_s"] = 1
        self.save_event()
        code, _, stderr = self.invoke("fit")
        self.assertEqual(code, 1)
        self.assertIn("Supply uncertainties for all", stderr)

    def test_missing_event_and_cli_compatibility(self):
        args = _build_parser().parse_args(["shock", "-vv", "inspect", "--event", str(self.path)])
        self.assertTrue(args.vv)
        args = _build_parser().parse_args(["shock", "inspect", "-v", "--event", str(self.path)])
        self.assertTrue(args.verbose)
        self.path.unlink()
        code, _, stderr = self.invoke("inspect")
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", stderr)
