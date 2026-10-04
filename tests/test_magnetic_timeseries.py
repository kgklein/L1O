import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from l1obs.cli import _magnetic_columns, _run_timeseries
from l1obs.fetch.cdaweb import FetchResult
from l1obs.proc.resample import resample_to_1s


class MagneticTimeseriesTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-09-01T00:00:00Z")
        self.end = self.start + pd.Timedelta(seconds=13)
        self.times = pd.date_range(self.start, periods=5, freq="3s", name="time")
        self.frame = pd.DataFrame({
            "bx_gse": [1, 4, np.nan, 10, 13],
            "by_gse": [2, 5, np.nan, 11, 14],
            "bz_gse": [3, 6, np.nan, 12, 15],
            "b_mag": [20, 30, np.nan, 50, 60],
        }, index=self.times)

    def test_legacy_components_keep_archive_magnitude_and_prefixed_interface(self):
        result = _magnetic_columns(self.frame, "WIND")
        np.testing.assert_equal(result.iloc[0].to_numpy(), [1, 2, 3, 20, -1, -2, 3, 20])
        self.assertEqual(result.columns.tolist(), [
            "WIND_bx_gse", "WIND_by_gse", "WIND_bz_gse", "WIND_b_mag",
            "WIND_Br", "WIND_Bt", "WIND_Bn", "WIND_Bmag",
        ])
        self.assertTrue(result.iloc[2].isna().all())

    def test_valid_runs_interpolate_but_invalid_gaps_do_not(self):
        result = resample_to_1s(self.frame, self.start, self.end, preserve_nan_gaps=True)
        np.testing.assert_array_equal(result.bx_gse.iloc[:4], [1, 2, 3, 4])
        self.assertTrue(result.iloc[4:9].isna().all().all())
        np.testing.assert_array_equal(result.bx_gse.iloc[9:], [10, 11, 12, 13])

    def test_columns_have_independent_gaps_and_no_endpoint_extrapolation(self):
        frame = self.frame.copy()
        frame.loc[self.times[2], "b_mag"] = 40
        result = resample_to_1s(frame, self.start - pd.Timedelta(seconds=1),
                                self.end + pd.Timedelta(seconds=1), preserve_nan_gaps=True)
        self.assertTrue(result.iloc[0].isna().all())
        self.assertTrue(result.iloc[-1].isna().all())
        self.assertEqual(result.loc[self.times[2], "b_mag"], 40)
        self.assertTrue(np.isnan(result.loc[self.times[2], "bx_gse"]))

    def test_single_valid_point_empty_and_all_invalid_frames(self):
        for frame in (self.frame.iloc[0:0], self.frame * np.nan):
            result = resample_to_1s(frame, self.start, self.end, preserve_nan_gaps=True)
            self.assertEqual(len(result), 13)
            self.assertTrue(result.isna().all().all())
        result = resample_to_1s(self.frame.iloc[:1], self.start, self.end, preserve_nan_gaps=True)
        self.assertEqual(result.b_mag.iloc[0], 20)
        self.assertTrue(result.iloc[1:].isna().all().all())

    def test_default_resampling_still_interpolates_and_extrapolates(self):
        result = resample_to_1s(self.frame, self.start - pd.Timedelta(seconds=1),
                                self.end + pd.Timedelta(seconds=1))
        self.assertFalse(result.isna().any().any())
        self.assertEqual(result.loc[self.times[2], "bx_gse"], 7)
        self.assertEqual(result.b_mag.iloc[0], 20)
        self.assertEqual(result.b_mag.iloc[-1], 60)

    def test_cli_resamples_magnetic_and_plasma_separately_and_plots_archive_values(self):
        datasets = {"WIND": {"mag": "WI_H0_MFI", "plasma": "WI_PM_3DP"},
                    "ACE": {"mag": "AC_H3_MFI"},
                    "DSCOVR": {"mag": "DSCOVR_H0_MAG"},
                    "IMAP": {"mag": "IMAP_MAG_L2_NORM-GSE"}}
        plasma = pd.DataFrame({"Np": [1, 4]}, index=[self.start, self.times[-1]])
        imap = self.frame.copy()
        imap.index = pd.date_range(self.start, periods=5, freq="500ms", name="time")

        def fetch(dataset, *args, **kwargs):
            if dataset == "IMAP_MAG_L2_NORM-GSE":
                frame = imap
            else:
                frame = plasma if dataset == "WI_PM_3DP" else self.frame
            return FetchResult(frame, dataset, {})

        with tempfile.TemporaryDirectory() as directory, \
                patch("l1obs.cli.DATASETS", datasets), \
                patch("l1obs.cli.fetch_cdaweb_dataset", side_effect=fetch), \
                patch("l1obs.cli.save_hdf5") as save, \
                patch("l1obs.cli.plot_timeseries") as plot:
            args = Namespace(start=self.start, outdir=directory, cachedir=directory, force=False)
            self.assertEqual(_run_timeseries(args), 0)
            merged = save.call_args.args[0]
            pd.testing.assert_frame_equal(plot.call_args.args[0], merged)
            for mission in ("WIND", "ACE", "DSCOVR"):
                self.assertEqual(merged[f"{mission}_Bmag"].iloc[0], 20)
                self.assertTrue(merged[f"{mission}_Bmag"].iloc[4:9].isna().all())
                self.assertTrue(merged[f"{mission}_Bmag"].iloc[13:].isna().all())
            self.assertFalse(merged.WIND_Np.isna().any())
            self.assertEqual(merged.WIND_Np.iloc[6], 2.5)
            self.assertEqual(merged.IMAP_Bmag.iloc[0], 20)
            self.assertTrue(np.isnan(merged.IMAP_Bmag.iloc[1]))
            self.assertEqual(merged.IMAP_Bmag.iloc[2], 60)
            self.assertTrue(merged.IMAP_Bmag.iloc[3:].isna().all())
            self.assertEqual(len(merged), 3600)


if __name__ == "__main__":
    unittest.main()
