"""Stage 1B ramp selection and explicit validation."""
import unittest

import numpy as np
import pandas as pd

from l1obs.analysis.shocks.arrivals import (
    DetectionOptions, detect_arrivals, build_arrival_table, accepted_arrivals, validate_magnetic_support,
)
from test_shock_data import LocalEventFixture


def ramp_frame(times=None):
    times = np.arange(301.) if times is None else np.asarray(times)
    values = 5 + 5 * np.clip((times - 145) / 10, 0, 1)
    return pd.DataFrame({"b_mag": values}, index=pd.Timestamp("2026-03-31T23:57:30Z") + pd.to_timedelta(times, unit="s"))


class DetectionTests(unittest.TestCase):
    def test_ramp_midpoint_duration_and_cadence_floor(self):
        frame = ramp_frame()
        result = detect_arrivals(frame, options=DetectionOptions(window_s=30))
        self.assertEqual(result.status, "proposed")
        best = result.candidates[0]
        self.assertEqual(pd.Timestamp(best["crossing_time"]), pd.Timestamp("2026-04-01T00:00:00Z"))
        self.assertAlmostEqual(best["ramp_duration_s"], 8)
        self.assertGreaterEqual(best["uncertainty_s"], .5)
        self.assertFalse(best["accepted"])

    def test_nan_and_timestamp_gaps_are_not_bridged(self):
        for mode in ("nan", "missing"):
            frame = ramp_frame()
            if mode == "nan":
                frame.iloc[140:160] = np.nan
            else:
                frame = frame.drop(frame.index[140:160])
            result = detect_arrivals(frame, options=DetectionOptions(window_s=30))
            self.assertEqual(result.status, "no_candidate")
            with self.assertRaisesRegex(ValueError, "invalid|gap"):
                validate_magnetic_support(frame, pd.Timestamp("2026-04-01T00:00:00Z"))

    def test_nonuniform_cadence_and_multiple_discontinuities(self):
        frame = ramp_frame(np.arange(0, 300, 1.13))
        self.assertEqual(detect_arrivals(frame, options=DetectionOptions(window_s=30)).status, "proposed")
        frame = ramp_frame(np.arange(600.))
        frame["b_mag"] = np.where(np.arange(600) < 150, 5., np.where(np.arange(600) < 400, 10., 15.))
        result = detect_arrivals(frame, options=DetectionOptions(window_s=30))
        self.assertEqual(result.status, "ambiguous")
        self.assertGreaterEqual(len(result.candidates), 2)
        self.assertTrue(all(not c["accepted"] for c in result.candidates))

    def test_naive_native_times_are_interpreted_as_utc(self):
        frame = ramp_frame()
        frame.index = frame.index.tz_localize(None)
        original = frame.index.copy()
        result = detect_arrivals(frame, ("2026-03-31T23:58:00Z", "2026-04-01T00:02:00Z"))
        self.assertEqual(result.status, "proposed")
        pd.testing.assert_index_equal(frame.index, original)
        validate_magnetic_support(frame, "2026-04-01T00:00:00Z")

    def test_timestamp_offset_and_empty_window(self):
        frame = ramp_frame()
        result = detect_arrivals(frame, timestamp_offset_s=.5, timestamp_uncertainty_s=5)
        self.assertEqual(pd.Timestamp(result.candidates[0]["crossing_time"]), pd.Timestamp("2026-04-01T00:00:00.5Z"))
        self.assertGreaterEqual(result.candidates[0]["uncertainty_s"], 5)
        self.assertEqual(detect_arrivals(frame, ("2027-01-01", "2027-01-02")).status, "missing")


class ManualSelectionTests(LocalEventFixture):
    def test_automatic_proposals_cannot_be_fitted_and_manual_is_preserved(self):
        from l1obs.analysis.shocks import load_event_data
        event = self.save_event()
        data = load_event_data(event)
        table, _ = build_arrival_table(event, data.magnetic)
        with self.assertRaisesRegex(ValueError, "accepted manual"):
            accepted_arrivals(table)
        time = self.times[121].isoformat()
        self.config["arrivals"] = {n: {"time": time, "uncertainty_s": 2, "method": "manual_validated",
                                         "accepted": True, "notes": "synthetic validation"} for n in self.names}
        event = self.save_event()
        content = self.path.read_text()
        for _ in range(2):
            table, _ = build_arrival_table(event, data.magnetic)
            self.assertEqual(list(table.crossing_time), [time] * 6)
            self.assertEqual(len(accepted_arrivals(table)), 6)
        self.assertEqual(self.path.read_text(), content)

    def test_subcadence_precision_needs_explicit_justification(self):
        from l1obs.analysis.shocks import load_event_data
        self.config["arrivals"] = {"ACE": {"time": self.times[120].isoformat(), "uncertainty_s": .01,
                                          "method": "manual_selected", "accepted": True}}
        event = self.save_event()
        data = load_event_data(event)
        with self.assertRaisesRegex(ValueError, "precision_justification"):
            build_arrival_table(event, data.magnetic)
        self.config["arrivals"]["ACE"]["precision_justification"] = "Synthetic precision experiment"
        table, _ = build_arrival_table(self.save_event(), data.magnetic)
        self.assertEqual(table.loc[table.spacecraft == "ACE", "uncertainty_s"].iloc[0], .01)
