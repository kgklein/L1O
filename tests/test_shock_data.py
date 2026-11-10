"""Stage 1A local inputs and bounded geometry, using unittest conventions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT
from l1obs.analysis.shocks import load_event, load_event_data, position_at
from l1obs.analysis.shocks.visualization import plot_inspection, plot_geometry


class LocalEventFixture(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.names = list(SPACECRAFT)
        self.start = pd.Timestamp("2026-03-31T23:58:00Z")
        self.times = pd.date_range(self.start, periods=241, freq="s")
        self.positions = np.array([[0, 0, 0], [80000, 0, 0], [0, 70000, 0],
                                   [0, 0, 60000], [-70000, 10000, 10000], [20000, -50000, -30000]], float)
        self.positions += [1500000, 0, 0]
        self.manifest = self.root / "inputs.json"
        entries = {}
        for i, name in enumerate(self.names):
            b = np.where(np.arange(241) < 120, 5., 10.)
            mag = pd.DataFrame({"bx_gse": b, "by_gse": 0., "bz_gse": 0., "b_mag": b}, index=self.times)
            pos = pd.DataFrame(np.tile(self.positions[i], (2, 1)), columns=["x_km", "y_km", "z_km"],
                               index=self.times[[0, -1]])
            entries[name] = {}
            for kind, frame in (("magnetic", mag), ("positions", pos)):
                suffix = ".csv" if i % 2 else ".parquet"
                path = self.root / f"{name}_{kind}{suffix}"
                frame.rename_axis("time").to_csv(path) if suffix == ".csv" else frame.to_parquet(path)
                entries[name][kind] = path.name
        self.manifest.write_text(json.dumps(entries))
        self.config = {"event_id": "synthetic", "start": self.start.isoformat(), "end": self.times[-1].isoformat(),
                       "spacecraft": self.names, "inputs": {"manifest": "inputs.json", "native_cadence": True,
                       "quality_masks_applied": True, "coordinate_system": "GSE", "magnetic_units": "nT",
                       "position_units": "km"}, "output_dir": "results"}
        self.path = self.root / "event.json"
        self.save_event()
        network = patch("socket.create_connection", side_effect=AssertionError("network forbidden"))
        network.start()
        self.addCleanup(network.stop)
        self.addCleanup(plt.close, "all")

    def save_event(self):
        self.path.write_text(json.dumps(self.config))
        return load_event(self.path)


class EventDataTests(LocalEventFixture):
    def test_incomplete_event_and_native_local_mixed_formats(self):
        event = load_event(self.path)
        self.assertFalse(event.arrivals)
        loaded = load_event_data(event)
        self.assertEqual(list(loaded.magnetic), self.names)
        self.assertEqual(loaded.magnetic["IMAP"].index[-1].day, 1)
        self.assertEqual(loaded.magnetic["WIND"].index[0].day, 31)
        self.assertEqual(len(loaded.magnetic["ADITYA-L1"]), 241)
        self.assertEqual(event.output_dir, self.root / "results")
        self.assertEqual(len(loaded.provenance["spacecraft"]["ACE"]["files"]["magnetic"]["sha256"]), 64)

    def test_missing_inputs_and_resampling_assertion(self):
        entries = json.loads(self.manifest.read_text())
        entries.pop("ACE")
        self.manifest.write_text(json.dumps(entries))
        with self.assertRaisesRegex(ValueError, "missing spacecraft"):
            load_event_data(load_event(self.path))
        self.config["inputs"]["native_cadence"] = False
        with self.assertRaisesRegex(ValueError, "native_cadence"):
            load_event_data(self.save_event())

    def test_magnitude_not_filled_and_components_derived_when_absent(self):
        path = self.root / "WIND_magnetic.parquet"
        frame = pd.read_parquet(path)
        frame.loc[frame.index[3], "b_mag"] = np.nan
        frame.to_parquet(path)
        loaded = load_event_data(load_event(self.path))
        self.assertTrue(np.isnan(loaded.magnetic["WIND"].b_mag.iloc[3]))
        frame = frame.drop(columns="b_mag")
        frame.loc[frame.index[3], "bx_gse"] = np.nan
        frame.to_parquet(path)
        loaded = load_event_data(load_event(self.path))
        self.assertTrue(np.isnan(loaded.magnetic["WIND"].b_mag.iloc[3]))
        self.assertIn("calculated", loaded.magnetic["WIND"].attrs["magnitude_source"])

    def test_bad_event_manual_state_and_duplicate_alias(self):
        for changes in ({"spacecraft": ["WIND", "Wind"]}, {"end": self.start.isoformat()},
                        {"arrivals": {"ACE": {"accepted": True, "method": "auto", "time": self.start.isoformat()}}},
                        {"arrivals": {"ACE": {"accepted": True, "method": "manual_selected"}}},
                        {"arrivals": {"ACE": {"uncertainty_s": -1}}}):
            original = self.config.copy()
            self.config.update(changes)
            with self.assertRaises(ValueError):
                self.save_event()
            self.config = original

    def test_figures_and_actual_epoch_positions(self):
        event = load_event(self.path)
        data = load_event_data(event)
        for figure in (plot_inspection(data.magnetic, window=(event.start, event.end)),
                       plot_geometry(self.names, self.positions)):
            self.assertGreater(len(figure.axes), 0)
            plt.close(figure)
        pos = data.ephemerides["ACE"].copy()
        pos.iloc[1, 0] += 240
        vector, metadata = position_at(pos, self.start + pd.Timedelta(seconds=30))
        self.assertAlmostEqual(vector[0], self.positions[1, 0] + 30)
        self.assertEqual(metadata["bracket_gap_s"], 240)
        for time in (self.start - pd.Timedelta(seconds=1), self.times[-1] + pd.Timedelta(seconds=1)):
            with self.assertRaisesRegex(ValueError, "not bracketed"):
                position_at(pos, time)
        with self.assertRaisesRegex(ValueError, "gap"):
            position_at(pos, self.start + pd.Timedelta(seconds=30), max_gap_s=100)
        pos.iloc[1, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid bracketing"):
            position_at(pos, self.start + pd.Timedelta(seconds=30))

    def test_metadata_mismatch_resampled_data_and_nan_rows(self):
        path = self.root / "WIND_magnetic.parquet"
        frame = pd.read_parquet(path)
        for attrs, message in (({"resampled": True}, "resampled"),
                               ({"coordinate_system": "GSM"}, "GSE"), ({"units": "T"}, "units")):
            frame.attrs = attrs
            frame.to_parquet(path)
            with self.assertRaisesRegex(ValueError, message):
                load_event_data(load_event(self.path))
        frame.attrs = {}
        frame.iloc[4, :] = np.nan
        frame.to_parquet(path)
        loaded = load_event_data(load_event(self.path))
        self.assertTrue(loaded.magnetic["WIND"].iloc[4].isna().all())
        self.assertEqual(len(loaded.magnetic["WIND"]), 241)

    def test_timezone_conversion_duplicate_json_and_bad_options(self):
        self.config["start"] = "2026-03-31T16:58:00-07:00"
        self.assertEqual(self.save_event().start, self.start)
        self.config["monte_carlo"] = {"recompute_positions": "false"}
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.save_event()
        self.config.pop("monte_carlo")
        self.config["detection"] = {"windows_s": 30}
        with self.assertRaisesRegex(ValueError, "Unknown detection"):
            self.save_event()
        self.path.write_text('{"event_id":"one","event_id":"two"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_event(self.path)
