import tempfile
from datetime import datetime, timezone
from pathlib import Path
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from l1obs.proc.ephemeris import SpacecraftConfiguration
from l1obs.viz.configuration import calculate_centroid, plot_configuration


class ConfigurationPlotTests(unittest.TestCase):
    def setUp(self):
        self.positions = pd.DataFrame(
            {
                "spacecraft": ["Wind", "ACE", "DSCOVR"],
                "x_km": [0.0, 3000.0, 6000.0],
                "y_km": [-3000.0, 0.0, 3000.0],
                "z_km": [1000.0, 2000.0, 3000.0],
                "sample_before": pd.to_datetime(
                    ["2026-08-31T23:55:00Z"] * 3
                ),
                "sample_after": pd.to_datetime(
                    ["2026-09-01T00:05:00Z"] * 3
                ),
            }
        )
        self.configuration = SpacecraftConfiguration(
            time=datetime(2026, 9, 1, tzinfo=timezone.utc),
            frame="GSE",
            positions=self.positions,
        )

    def tearDown(self):
        plt.close("all")

    def test_centroid_is_arithmetic_mean(self):
        centroid = calculate_centroid(self.positions)
        self.assertEqual(centroid.to_dict(), {
            "x_km": 3000.0,
            "y_km": 0.0,
            "z_km": 2000.0,
        })

    def test_plot_has_three_equal_scale_panels_and_can_save(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "configuration.png"
            figure = plot_configuration(self.configuration, output=output)

            self.assertIsNotNone(figure)
            self.assertEqual(len(figure.axes), 3)
            self.assertEqual(
                [axis.get_title() for axis in figure.axes],
                ["GSE X-Y", "GSE X-Z", "GSE Y-Z"],
            )
            self.assertTrue(all(axis.get_aspect() == 1.0 for axis in figure.axes))
            self.assertTrue(
                all(axis.get_xlim() == figure.axes[0].get_xlim() for axis in figure.axes)
            )
            self.assertTrue(
                all(axis.get_ylim() == figure.axes[0].get_ylim() for axis in figure.axes)
            )
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
