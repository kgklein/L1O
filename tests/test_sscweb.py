import unittest
from unittest.mock import Mock, patch

import pandas as pd

from l1obs.fetch.sscweb import EphemerisFetchError, fetch_ephemeris


class FetchEphemerisTests(unittest.TestCase):
    @staticmethod
    def _response():
        from sscws.coordinates import CoordinateSystem

        return {
            "HttpStatus": 200,
            "Data": [
                {
                    "Id": "wind",
                    "Time": [
                        pd.Timestamp("2026-09-01T00:00:00Z").to_pydatetime(),
                        pd.Timestamp("2026-09-01T00:10:00Z").to_pydatetime(),
                    ],
                    "Coordinates": [
                        {
                            "CoordinateSystem": CoordinateSystem.GSE,
                            "X": [1.0, 2.0],
                            "Y": [3.0, 4.0],
                            "Z": [5.0, 6.0],
                        }
                    ],
                }
            ],
        }

    @patch("l1obs.fetch.sscweb.SscWs")
    def test_successful_response_is_normalized(self, sscws):
        client = Mock()
        client.get_locations.return_value = self._response()
        sscws.return_value = client

        result = fetch_ephemeris(
            "wind", "2026-09-01T00:00:00Z", "2026-09-01T00:20:00Z"
        )

        self.assertEqual(result.index.name, "time")
        self.assertEqual(str(result.index.tz), "UTC")
        self.assertEqual(result.columns.tolist(), ["x_km", "y_km", "z_km"])
        self.assertEqual(result.iloc[1].tolist(), [2.0, 4.0, 6.0])

    @patch("l1obs.fetch.sscweb.SscWs")
    def test_http_failure_raises_useful_error(self, sscws):
        sscws.return_value.get_locations.return_value = {
            "HttpStatus": 503,
            "ErrorDescription": "temporarily unavailable",
        }
        with self.assertRaisesRegex(EphemerisFetchError, "HTTP 503"):
            fetch_ephemeris(
                "wind", "2026-09-01T00:00:00Z", "2026-09-01T00:20:00Z"
            )

    @patch("l1obs.fetch.sscweb.SscWs")
    def test_missing_data_raises(self, sscws):
        sscws.return_value.get_locations.return_value = {
            "HttpStatus": 200,
            "Data": [],
        }
        with self.assertRaisesRegex(EphemerisFetchError, "no ephemeris"):
            fetch_ephemeris(
                "wind", "2026-09-01T00:00:00Z", "2026-09-01T00:20:00Z"
            )

    @patch("l1obs.fetch.sscweb.SscWs")
    def test_inconsistent_coordinate_lengths_raise(self, sscws):
        response = self._response()
        response["Data"][0]["Coordinates"][0]["Z"] = [5.0]
        sscws.return_value.get_locations.return_value = response
        with self.assertRaisesRegex(EphemerisFetchError, "inconsistent"):
            fetch_ephemeris(
                "wind", "2026-09-01T00:00:00Z", "2026-09-01T00:20:00Z"
            )


if __name__ == "__main__":
    unittest.main()
