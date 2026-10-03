from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT
from l1obs.fetch.sscweb import EphemerisFetchError
from l1obs.proc.ephemeris import get_positions_at_time, interpolate_position


class InterpolatePositionTests(unittest.TestCase):
    def setUp(self):
        self.times = pd.to_datetime(
            ["2026-09-01T00:00:00Z", "2026-09-01T00:10:00Z"]
        )
        self.positions = np.array([[0.0, 10.0, -10.0], [10.0, 30.0, 10.0]])

    def test_midpoint(self):
        result = interpolate_position(
            self.times, self.positions, "2026-09-01T00:05:00Z"
        )
        np.testing.assert_allclose(result, [5.0, 20.0, 0.0])

    def test_exact_timestamp(self):
        result = interpolate_position(
            self.times, self.positions, "2026-09-01T00:10:00Z"
        )
        np.testing.assert_allclose(result, self.positions[1])

    def test_target_must_be_bracketed(self):
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            interpolate_position(
                self.times, self.positions, "2026-08-31T23:59:00Z"
            )
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            interpolate_position(
                self.times, self.positions, "2026-09-01T00:11:00Z"
            )

    def test_rejects_malformed_position_shape(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            interpolate_position(self.times, [[1.0], [2.0]], self.times[0])

    def test_single_non_exact_sample_is_insufficient(self):
        with self.assertRaisesRegex(ValueError, "not bracketed"):
            interpolate_position(
                [self.times[0]], [[1.0, 2.0, 3.0]], "2026-09-01T00:01:00Z"
            )


class ConfigurationTests(unittest.TestCase):
    def test_registry_order_and_solar_id(self):
        self.assertEqual(
            list(SPACECRAFT),
            ["WIND", "ACE", "DSCOVR", "ADITYA-L1", "IMAP", "SOLAR-1"],
        )
        self.assertEqual(SPACECRAFT["SOLAR-1"].ssc_id, "solar1")

    @staticmethod
    def _ephemeris(offset):
        index = pd.to_datetime(
            ["2026-08-31T23:55:00Z", "2026-09-01T00:05:00Z"]
        )
        return pd.DataFrame(
            {
                "x_km": [offset, offset + 10.0],
                "y_km": [offset + 10.0, offset + 30.0],
                "z_km": [offset - 10.0, offset + 10.0],
            },
            index=index,
        )

    @patch("l1obs.proc.ephemeris.fetch_ephemeris")
    def test_partial_failure_preserves_success_and_provenance(self, fetch):
        def response(spacecraft_id, *args, **kwargs):
            if spacecraft_id == "ace":
                raise EphemerisFetchError("missing")
            return self._ephemeris(0.0)

        fetch.side_effect = response
        configuration = get_positions_at_time(
            "2026-09-01T00:00:00Z", spacecraft=["wind", "ACE"]
        )

        self.assertEqual(configuration.time.tzinfo, timezone.utc)
        self.assertEqual(configuration.positions["spacecraft"].tolist(), ["Wind"])
        np.testing.assert_allclose(
            configuration.positions.loc[0, ["x_km", "y_km", "z_km"]].astype(float),
            [5.0, 20.0, 0.0],
        )
        self.assertEqual(
            configuration.positions.loc[0, "sample_before"],
            pd.Timestamp("2026-08-31T23:55:00Z"),
        )
        self.assertEqual(
            configuration.positions.loc[0, "sample_after"],
            pd.Timestamp("2026-09-01T00:05:00Z"),
        )
        self.assertIn("ACE", configuration.errors)

    @patch("l1obs.proc.ephemeris.fetch_ephemeris")
    def test_all_failures_raise_aggregate_error(self, fetch):
        fetch.side_effect = EphemerisFetchError("offline")
        with self.assertRaisesRegex(RuntimeError, "No spacecraft positions"):
            get_positions_at_time(
                datetime(2026, 9, 1, tzinfo=timezone.utc),
                spacecraft=["Wind", "ACE"],
            )


if __name__ == "__main__":
    unittest.main()
