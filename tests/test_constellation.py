"""Synthetic loaded-data examples; no spacecraft services are contacted."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd

from l1obs.config import SPACECRAFT_COLORS
from l1obs.viz.constellation import (
    _align_geometry,
    _gap_limit,
    _prepare,
    plot_six_spacecraft_B_and_geometry,
)


START = pd.Timestamp("2026-09-01T00:00:00Z")
OFFSETS = np.array([[10, -7, 1], [-13, 5, 9], [4, 19, -12],
                    [22, -11, 6], [-8, -17, -3], [1, 2, 18]], dtype=float)
VELOCITIES = np.array([[.1, .2, -.1], [-.2, .05, .1], [.3, -.1, .2],
                       [-.05, .15, -.2], [.2, -.3, .05], [-.15, .1, .3]])


def absolute_positions(spacecraft, seconds):
    seconds = np.asarray(seconds)
    translation = np.array([1_500_000, 200_000, -40_000]) + seconds[:, None] * [12, -4, 2]
    return translation + OFFSETS[spacecraft] + seconds[:, None] * VELOCITIES[spacecraft]


def synthetic_constellation():
    """Six asymmetric moving tracks with independently sampled magnetic data."""
    datasets = {}
    names = ["Wind", "ACE", "DSCOVR", "Aditya-L1", "IMAP", "SOLAR-1"]
    for spacecraft, name in enumerate(names):
        magnetic_seconds = np.arange(0, 60.01, [0.5, 1, 1.5, 2, 2.5, 3][spacecraft])
        phase = magnetic_seconds / 7
        magnetic = pd.DataFrame({
            "bx_gse": (spacecraft + 1) * np.sin(phase),
            "by_gse": 2 * np.cos(phase) + spacecraft,
            "bz_gse": -np.cos(phase / 2) - spacecraft,
            "b_mag": np.full(len(phase), 100.0),
        }, index=START + pd.to_timedelta(magnetic_seconds, unit="s"))
        position_seconds = np.arange(0, 61, [2, 3, 4, 5, 6, 10][spacecraft])
        positions = pd.DataFrame(absolute_positions(spacecraft, position_seconds),
                                 columns=["x_km", "y_km", "z_km"],
                                 index=START + pd.to_timedelta(position_seconds, unit="s"))
        datasets[name] = {"magnetic": magnetic, "positions": positions}
    return datasets


class ConstellationPlotTests(unittest.TestCase):
    def setUp(self):
        self.datasets = synthetic_constellation()
        patcher = patch("socket.create_connection", side_effect=AssertionError("network forbidden"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        plt.close("all")

    def plot(self, **kwargs):
        return plot_six_spacecraft_B_and_geometry(self.datasets, coordinate_system="GSE", **kwargs)

    def align(self, datasets=None, position_max_gap=None):
        source = self.datasets if datasets is None else datasets
        frames = [_prepare(entry["positions"], ("x_km", "y_km", "z_km"), name)
                  for name, entry in source.items()]
        return _align_geometry(frames, START, START + pd.Timedelta(seconds=60),
                               [_gap_limit(frame.index, position_max_gap) for frame in frames])

    def test_instantaneous_center_interpolation_and_relative_sums(self):
        times, relative, center = self.align()
        seconds = (times - START).total_seconds().to_numpy()
        expected = np.stack([absolute_positions(spacecraft, seconds) for spacecraft in range(6)])
        np.testing.assert_allclose(center, expected.mean(axis=0), atol=1e-8, rtol=0)
        np.testing.assert_allclose(relative, expected - expected.mean(axis=0), atol=1e-8, rtol=0)
        np.testing.assert_allclose(relative.sum(axis=0), 0, atol=1e-8, rtol=0)
        self.assertFalse(np.allclose(center[0], center[-1]))
        self.assertGreater(len(times), len(self.datasets["SOLAR-1"]["positions"]))

    def test_nine_axes_distinct_projections_labels_colors_and_markers(self):
        figure = self.plot()
        self.assertEqual(len(figure.axes), 9)
        _, relative, _ = self.align()
        for axis, pair in zip(figure.axes[6:], [(0, 1), (0, 2), (1, 2)]):
            self.assertEqual(axis.get_aspect(), 1.0)
            self.assertEqual(axis.get_xlabel(), f"Δ{'xyz'[pair[0]]} [km]")
            self.assertEqual(axis.get_ylabel(), f"Δ{'xyz'[pair[1]]} [km]")
            for spacecraft, name in enumerate(self.datasets):
                track, start, end = axis.lines[spacecraft * 3:spacecraft * 3 + 3]
                np.testing.assert_allclose(track.get_xdata(), relative[spacecraft, :, pair[0]])
                np.testing.assert_allclose(track.get_ydata(), relative[spacecraft, :, pair[1]])
                self.assertEqual(start.get_marker(), "o")
                self.assertEqual(end.get_marker(), "^")
                self.assertEqual(start.get_markerfacecolor(), "none")
                np.testing.assert_allclose(start.get_xdata(), [relative[spacecraft, 0, pair[0]]])
                np.testing.assert_allclose(end.get_ydata(), [relative[spacecraft, -1, pair[1]]])
                label = figure.axes[spacecraft].texts[0]
                self.assertEqual(label.get_text(), name)
                self.assertEqual(to_rgba(label.get_color()), to_rgba(track.get_color()))
            self.assertEqual(axis.lines[-1].get_marker(), "+")
        legend = figure.axes[6].get_legend()
        self.assertEqual([text.get_text() for text in legend.get_texts()], list(self.datasets))
        self.assertEqual(to_rgba(figure.axes[0].texts[0].get_color()), to_rgba(SPACECRAFT_COLORS["WIND"]))
        self.assertTrue(all(axis.get_xlim() == figure.axes[0].get_xlim() for axis in figure.axes[:6]))
        self.assertTrue(all(axis.get_ylim() == figure.axes[0].get_ylim() for axis in figure.axes[:6]))
        self.assertTrue(all(not axis.get_xticklabels() for axis in figure.axes[:5]))
        for axis in figure.axes[:6]:
            self.assertEqual([line.get_color() for line in axis.lines], ["black", "#1f77b4", "#d62728", "#2ca02c"])
            self.assertEqual([line.get_linestyle() for line in axis.lines], ["-", "-", "--", ":"])

    def test_native_magnetic_samples_and_default_vector_norm(self):
        figure = self.plot()
        for axis, entry in zip(figure.axes[:6], self.datasets.values()):
            magnetic = entry["magnetic"]
            self.assertEqual(len(axis.lines[0].get_xdata()), len(magnetic))
            np.testing.assert_allclose(axis.lines[0].get_ydata(),
                                       np.linalg.norm(magnetic.iloc[:, :3].to_numpy(), axis=1))
            self.assertFalse(np.allclose(axis.lines[0].get_ydata(), magnetic.b_mag))

    def test_explicit_archive_magnitude_preserves_nan_and_independent_y_limits(self):
        self.datasets["Wind"]["magnetic"].iloc[4, 3] = np.nan
        figure = self.plot(magnitude_column="b_mag", common_B_ylim=False)
        self.assertTrue(np.isnan(figure.axes[0].lines[0].get_ydata()[4]))
        self.assertEqual(figure.axes[0].lines[0].get_ydata()[0], 100)
        self.assertIn("b_mag", figure.axes[0].get_title(loc="left"))
        other = self.plot(common_B_ylim=False)
        self.assertNotEqual(other.axes[0].get_ylim(), other.axes[5].get_ylim())

    def test_magnetic_nan_and_missing_timestamp_gaps_are_not_connected(self):
        field = self.datasets["Wind"]["magnetic"]
        field.loc[START + pd.Timedelta(seconds=2), "bx_gse"] = np.nan
        self.datasets["Wind"]["magnetic"] = field.drop(field.loc[START + pd.Timedelta(seconds=10):START + pd.Timedelta(seconds=20)].index)
        figure = self.plot()
        line = figure.axes[0].lines[0]
        times = pd.DatetimeIndex(line.get_xdata())
        self.assertTrue(np.isnan(line.get_ydata()[times.get_loc(START + pd.Timedelta(seconds=2))]))
        self.assertTrue(np.isnan(line.get_ydata()[times.get_loc(START + pd.Timedelta(seconds=15))]))
        override = self.plot(magnetic_max_gap="20s")
        self.assertNotIn(START + pd.Timedelta(seconds=15), pd.DatetimeIndex(override.axes[0].lines[0].get_xdata()))

    def test_invalid_position_masks_all_six_without_redefining_center(self):
        self.datasets["Wind"]["positions"].loc[START + pd.Timedelta(seconds=20), "x_km"] = np.nan
        times, relative, center = self.align()
        invalid = (times > START + pd.Timedelta(seconds=18)) & (times < START + pd.Timedelta(seconds=22))
        self.assertTrue(np.isnan(relative[:, invalid]).all())
        self.assertTrue(np.isnan(center[invalid]).all())
        complete = np.isfinite(center).all(axis=1)
        np.testing.assert_allclose(relative[:, complete].sum(axis=0), 0, atol=1e-8)

    def test_missing_position_intervals_and_override(self):
        positions = self.datasets["Wind"]["positions"]
        self.datasets["Wind"]["positions"] = positions.drop(positions.loc[START + pd.Timedelta(seconds=10):START + pd.Timedelta(seconds=20)].index)
        times, relative, _ = self.align()
        inside = (times > START + pd.Timedelta(seconds=8)) & (times < START + pd.Timedelta(seconds=22))
        self.assertTrue(np.isnan(relative[:, inside]).all())
        _, unbroken, _ = self.align(position_max_gap="20s")
        self.assertTrue(np.isfinite(unbroken).all())

    def test_collective_missing_timestamp_gap_inserts_geometry_break(self):
        for entry in self.datasets.values():
            positions = entry["positions"]
            entry["positions"] = positions.loc[(positions.index <= START + pd.Timedelta(seconds=10)) |
                                                (positions.index >= START + pd.Timedelta(seconds=50))]
        times, relative, _ = self.align(position_max_gap="15s")
        gap = (times > START + pd.Timedelta(seconds=10)) & (times < START + pd.Timedelta(seconds=50))
        self.assertTrue(gap.any())
        self.assertTrue(np.isnan(relative[:, gap]).all())

    def test_geometry_no_extrapolation_single_point_and_zero_extent(self):
        for entry in self.datasets.values():
            entry["positions"] = entry["positions"].iloc[:1].copy()
            entry["positions"].iloc[0, :] = [1, 2, 3]
        times, relative, _ = self.align()
        self.assertEqual(times.tolist(), [START])
        np.testing.assert_array_equal(relative, 0)
        figure = self.plot()
        self.assertTrue(np.isfinite(figure.axes[6].get_xlim()).all())
        self.assertEqual(len(figure.axes[6].lines[0].get_xdata()), 1)

    def test_geometry_clip_and_magnetic_requested_range(self):
        self.datasets["Wind"]["positions"] = self.datasets["Wind"]["positions"].iloc[2:-2]
        figure = self.plot(time_range=(START, START + pd.Timedelta(seconds=60)))
        self.assertIn("00:00:04", figure.axes[-1].texts[0].get_text())
        self.assertIn("00:00:56", figure.axes[-1].texts[0].get_text())

    def test_frame_and_unit_metadata_and_custom_component_names(self):
        for entry in self.datasets.values():
            entry["magnetic"].attrs.update(frame="gse", units={"bx_gse": "nT", "by_gse": "nT", "bz_gse": "nT"})
            entry["positions"].attrs.update(coordinate_system="GSE", units="km")
        figure = plot_six_spacecraft_B_and_geometry(self.datasets)
        self.assertIn("GSE", figure.axes[0].get_title(loc="left"))
        for entry in self.datasets.values():
            entry["magnetic"] = entry["magnetic"].rename(columns={"bx_gse": "Bx", "by_gse": "By", "bz_gse": "Bz"})
            entry["magnetic"].attrs.update(frame="GSM", units="nT")
            entry["positions"].attrs["coordinate_system"] = "GSM"
        custom = plot_six_spacecraft_B_and_geometry(self.datasets, B_components=("Bx", "By", "Bz"))
        self.assertIn("GSM", custom.axes[6].get_title())

    def test_invalid_frame_and_unit_declarations_raise(self):
        cases = [
            ("magnetic", {"frame": "GSM"}, {}),
            ("positions", {"frame": "GSM"}, {}),
            ("positions", {"frame": "GSE", "coordinate_system": "GSM"}, {}),
            ("positions", {"units": "AU"}, {}),
            ("positions", {"units": {"x_km": "km", "y_km": "m"}}, {}),
            ("magnetic", {"units": "T"}, {}),
            ("positions", {}, {"position_units": "AU"}),
        ]
        for kind, attrs, options in cases:
            with self.subTest(kind=kind, attrs=attrs, options=options):
                self.datasets = synthetic_constellation()
                self.datasets["Wind"][kind].attrs.update(attrs)
                with self.assertRaisesRegex(ValueError, "mismatch|units"):
                    self.plot(**options)
        with self.assertRaisesRegex(ValueError, "coordinate metadata"):
            plot_six_spacecraft_B_and_geometry(synthetic_constellation())
        with self.assertRaisesRegex(ValueError, "mismatch"):
            plot_six_spacecraft_B_and_geometry(synthetic_constellation(), coordinate_system="GSM")

    def test_count_order_missing_columns_and_invalid_times(self):
        with self.assertRaisesRegex(ValueError, "Exactly six"):
            plot_six_spacecraft_B_and_geometry(dict(list(self.datasets.items())[:5]))
        with self.assertRaisesRegex(ValueError, "permutation"):
            self.plot(spacecraft_names=["Wind"] * 6)
        for kind, column in [("magnetic", "bx_gse"), ("positions", "z_km")]:
            data = synthetic_constellation()
            data["Wind"][kind] = data["Wind"][kind].drop(columns=column)
            with self.assertRaisesRegex(ValueError, f"missing columns: {column}"):
                plot_six_spacecraft_B_and_geometry(data, coordinate_system="GSE")
        field = self.datasets["Wind"]["magnetic"]
        self.datasets["Wind"]["magnetic"] = pd.concat([field, field.iloc[:1]])
        with self.assertRaisesRegex(ValueError, "unique timestamps"):
            self.plot()
        self.datasets = synthetic_constellation()
        self.datasets["Wind"]["magnetic"].index = pd.date_range(START, periods=121, freq="500ms").where(np.arange(121) != 2)
        with self.assertRaisesRegex(ValueError, "NaT"):
            self.plot()

    def test_invalid_ranges_no_overlap_and_no_simultaneously_valid_positions(self):
        with self.assertRaisesRegex(ValueError, "start before end"):
            self.plot(time_range=(START, START))
        with self.assertRaisesRegex(ValueError, "no magnetic samples"):
            self.plot(time_range=(START + pd.Timedelta(days=1), START + pd.Timedelta(days=2)))
        for override in ("0s", "-1s", pd.NaT):
            with self.assertRaisesRegex(ValueError, "positive"):
                self.plot(position_max_gap=override)
        self.datasets["Wind"]["positions"].index += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "Non-overlapping"):
            self.plot()
        self.datasets = synthetic_constellation()
        self.datasets["Wind"]["positions"].iloc[:, :] = np.nan
        with self.assertRaisesRegex(ValueError, "no valid positions"):
            self.plot()
        self.datasets = synthetic_constellation()
        # Each spacecraft has some valid positions, but disjoint valid runs.
        self.datasets["Wind"]["positions"].iloc[1:-1, :] = np.nan
        self.datasets["ACE"]["positions"].iloc[[0, -1], :] = np.nan
        with self.assertRaisesRegex(ValueError, "No simultaneously valid"):
            self.plot()

    def test_inputs_unchanged_naive_unsorted_times_and_color_order(self):
        originals = synthetic_constellation()
        names = list(self.datasets)
        figure = self.plot(spacecraft_names=names[::-1], colors={"SOLAR-1": "magenta"})
        self.assertEqual(figure.axes[0].texts[0].get_text(), "SOLAR-1")
        self.assertEqual(figure.axes[0].texts[0].get_color(), "magenta")
        for name in names:
            for kind in ("magnetic", "positions"):
                pd.testing.assert_frame_equal(self.datasets[name][kind], originals[name][kind])
                frame = self.datasets[name][kind].iloc[::-1].copy()
                frame.index = frame.index.tz_localize(None)
                self.datasets[name][kind] = frame
        naive = self.plot()
        self.assertEqual(len(naive.axes), 9)
        custom = {f"Craft {index}": data for index, data in enumerate(self.datasets.values())}
        first = plot_six_spacecraft_B_and_geometry(custom, coordinate_system="GSE")
        second = plot_six_spacecraft_B_and_geometry(custom, spacecraft_names=list(custom)[::-1], coordinate_system="GSE")
        self.assertEqual(first.axes[0].texts[0].get_color(), second.axes[5].texts[0].get_color())
        self.assertEqual(len({axis.texts[0].get_color() for axis in first.axes[:6]}), 6)

    def test_all_nan_magnetic_panel_and_infinities(self):
        self.datasets["Wind"]["magnetic"].iloc[:, :3] = np.nan
        self.datasets["ACE"]["magnetic"].iloc[0, 0] = np.inf
        figure = self.plot()
        self.assertIn("No valid B samples", [text.get_text() for text in figure.axes[0].texts])
        self.assertTrue(np.isnan(figure.axes[1].lines[0].get_ydata()[0]))

    def test_png_pdf_output_show_and_local_style(self):
        before = plt.rcParams["font.size"]
        with tempfile.TemporaryDirectory() as directory, patch("matplotlib.pyplot.show") as show:
            path = Path(directory) / "nested" / "constellation.png"
            figure = self.plot(output_path=path, save_pdf=True, show=True)
            show.assert_called_once()
            self.assertGreater(path.stat().st_size, 1000)
            self.assertGreater(path.with_suffix(".pdf").stat().st_size, 1000)
            self.assertTrue(plt.fignum_exists(figure.number))
        self.assertEqual(plt.rcParams["font.size"], before)
        with self.assertRaisesRegex(ValueError, "PNG"):
            self.plot(output_path="plot.pdf")
        with self.assertRaisesRegex(ValueError, "requires output_path"):
            self.plot(save_pdf=True)


if __name__ == "__main__":
    unittest.main()
