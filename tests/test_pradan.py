"""Local-only Aditya-L1 fixtures mirror the inspected Level-2 schema."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from l1obs.config import DATASETS
from l1obs.fetch.magnetic import fetch_magnetic_field
from l1obs.fetch import pradan


class PradanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.start = pd.Timestamp("2026-09-20T00:00:00Z")
        self.end = self.start + pd.Timedelta(minutes=1)
        # Fail on any attempted network access, including the other providers.
        for target in ("socket.socket.connect", "l1obs.fetch.magnetic.fetch_cdaweb_dataset",
                       "l1obs.fetch.magnetic.fetch_ncei_product"):
            mock = patch(target, side_effect=AssertionError("Unexpected network/provider call"))
            mock.start()
            self.addCleanup(mock.stop)

    def dataset(self, times=None):
        times = pd.date_range(self.start, periods=6, freq="10s") if times is None else times
        n = len(times)
        return xr.Dataset({
            "time": ("utc_time", times.asi8 / 1e9, {"units": "Seconds (UNIX time)",
                    "description": "Epoch starts at Jan 1, 1970 00:00:00, bin-width = 10second"}),
            "Bx_gse": ("Bx", np.full(n, 3, dtype="float32")),
            "By_gse": ("By", np.full(n, 4, dtype="float32")),
            "Bz_gse": ("Bz", np.full(n, 12, dtype="float32")),
            "Quality_flag_10s_data": ("utc_time", np.ones(n, dtype="float32")),
            "Bx_gsm": ("Bx", np.full(n, 99.)),
            "x_gse": ("Bx", np.full(n, 1_500_000.)),
            "Bx_gse_error": ("Bx", np.full(n, 0.1)),
        }, attrs={"Fill_value": "-9999.0", "processing_level": "Level 2"})

    def write(self, ds=None, day=None):
        day = self.start if day is None else day
        path = self.root / f"L2_AL1_MAG_{day.strftime('%Y%m%d')}_V00.nc"
        (self.dataset() if ds is None else ds).to_netcdf(path, engine="netcdf4")
        return path

    def fetch(self, start=None, end=None, force=False):
        return fetch_magnetic_field("Aditya-L1", self.start if start is None else start,
                                    self.end if end is None else end, self.root, force=force)

    def test_registry_normalization_time_native_cadence_and_derived_magnitude(self):
        self.write()
        self.assertEqual(DATASETS["ADITYA-L1"], {"mag": pradan.PRODUCT, "mag_provider": "pradan"})
        result = self.fetch()
        self.assertEqual(result.dataset_id, pradan.PRODUCT)
        self.assertEqual(result.used_vars, pradan.USED_VARS)
        self.assertEqual(result.df.columns.tolist(), ["bx_gse", "by_gse", "bz_gse", "b_mag"])
        pd.testing.assert_index_equal(result.df.index, pd.date_range(self.start, periods=6, freq="10s", name="time"))
        np.testing.assert_array_equal(result.df.iloc[0], [3, 4, 12, 13])
        self.assertEqual(result.df.attrs["coordinate_system"], "GSE")
        self.assertIn("derived", result.df.attrs["magnitude_source"])

    def test_fill_and_quality_mask_before_magnitude(self):
        ds = self.dataset()
        ds.Bx_gse.values[1] = -9999
        ds.By_gse.values[2] = np.inf
        ds.Quality_flag_10s_data.values[:] = [1, 1, 1, 0, -9999, np.nan]
        self.write(ds)
        frame = self.fetch().df
        self.assertEqual(frame.b_mag.iloc[0], 13)
        self.assertTrue(frame.iloc[3:].isna().all().all())
        self.assertTrue(np.isnan(frame.bx_gse.iloc[1]))
        self.assertEqual(frame.by_gse.iloc[1], 4)
        self.assertTrue(frame.b_mag.iloc[1:].isna().all())

    def test_metadata_bounds_and_fills_are_inclusive(self):
        ds = self.dataset()
        ds.Bx_gse.attrs.update({"valid_min": -3., "valid_max": 3., "missing_value": -88.})
        ds.Bx_gse.values[:] = [-3, 3, -4, 4, -88, -9999]
        ds.By_gse.attrs["valid_range"] = [4., 4.]
        self.write(ds)
        frame = self.fetch().df
        np.testing.assert_array_equal(frame.bx_gse.iloc[:2], [-3, 3])
        np.testing.assert_array_equal(frame.b_mag.iloc[:2], [13, 13])
        self.assertTrue(frame.bx_gse.iloc[2:].isna().all())
        self.assertTrue(frame.b_mag.iloc[2:].isna().all())

    def test_multi_day_trim_and_no_extra_midnight_file(self):
        midnight = self.start + pd.Timedelta(days=1)
        self.write(self.dataset(pd.date_range(midnight - pd.Timedelta(seconds=30), periods=3, freq="10s")))
        self.write(self.dataset(pd.date_range(midnight, periods=3, freq="10s")), midnight)
        frame = self.fetch(midnight - pd.Timedelta(seconds=15), midnight + pd.Timedelta(seconds=15)).df
        pd.testing.assert_index_equal(frame.index, pd.DatetimeIndex([
            midnight - pd.Timedelta(seconds=10), midnight, midnight + pd.Timedelta(seconds=10)], name="time"))
        result = self.fetch(midnight - pd.Timedelta(seconds=30), midnight)
        self.assertEqual(len(result.df), 3)
        self.assertLess(result.df.index[-1], midnight)

    def test_discovery_uses_requested_dates_and_reports_all_missing_files(self):
        self.write()
        (self.root / "L1_AL1_MAG_20260921_V00.nc").touch()
        (self.root / "L2_AL1_MAG_20260921_V01.nc").touch()
        with self.assertRaisesRegex(RuntimeError, "20260921_V00.nc.*20260922_V00.nc"):
            self.fetch(self.start, self.start + pd.Timedelta(days=3))
        later = self.start + pd.Timedelta(days=5)
        self.write(self.dataset(pd.date_range(later, periods=2, freq="10s")), later)
        result = self.fetch(later, later + pd.Timedelta(seconds=20))
        self.assertEqual(result.df.index[0], later)

    def test_files_reread_force_and_naive_times(self):
        ds = self.dataset()
        self.write(ds)
        self.assertEqual(self.fetch(self.start.tz_localize(None), self.end.tz_localize(None)).df.b_mag.iloc[0], 13)
        ds.Bx_gse.values[:] = 0
        self.write(ds)
        for force in (False, True):
            self.assertEqual(self.fetch(force=force).df.b_mag.iloc[0], np.sqrt(160))
        self.assertFalse(list(self.root.glob("*.parquet")))

    def test_malformed_shapes_missing_variables_times_and_units(self):
        cases = [(self.dataset().drop_vars("By_gse"), "Missing variable By_gse"),
                 (self.dataset().assign(Bx_gse=("short", np.zeros(5))), "must have shape")]
        for ds, message in cases:
            with self.subTest(message=message):
                self.write(ds)
                with self.assertRaisesRegex(RuntimeError, message):
                    self.fetch()
        ds = self.dataset()
        ds.time.values[1] = ds.time.values[0]
        self.write(ds)
        with self.assertRaisesRegex(RuntimeError, "duplicate timestamps"):
            self.fetch()
        ds.time.values[0] = -9999
        self.write(ds)
        with self.assertRaisesRegex(RuntimeError, "Invalid Unix timestamps"):
            self.fetch()
        ds.time.attrs["units"] = "seconds since 1958"
        self.write(ds)
        with self.assertRaisesRegex(RuntimeError, "Unsupported time units"):
            self.fetch()

    def test_cli_consumes_local_provider_and_keeps_derived_magnitude(self):
        from argparse import Namespace
        from l1obs.cli import _run_timeseries

        self.write()
        with patch("l1obs.cli.DATASETS", {"ADITYA-L1": DATASETS["ADITYA-L1"]}), \
             patch("l1obs.cli.save_hdf5") as save, \
             patch("l1obs.cli.plot_timeseries"):
            args = Namespace(start=self.start, cachedir=self.root, outdir=self.root, force=False)
            self.assertEqual(_run_timeseries(args), 0)
        frame = save.call_args.args[0]
        self.assertEqual(frame["ADITYA-L1_Bmag"].iloc[0], 13)
        self.assertEqual(frame["ADITYA-L1_b_mag"].iloc[0], 13)
        self.assertEqual(frame["ADITYA-L1_Br"].iloc[0], -3)
        self.assertTrue(frame["ADITYA-L1_Bmag"].iloc[51:].isna().all())

    def test_bad_intervals_and_no_samples(self):
        self.write()
        for start, end in ((self.start, self.start), (self.end, self.start), (pd.NaT, self.end)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.fetch(start, end)
        with self.assertRaisesRegex(RuntimeError, "No samples"):
            self.fetch(self.end, self.end + pd.Timedelta(seconds=10))
        with self.assertRaisesRegex(ValueError, "Unsupported local"):
            pradan.fetch_pradan_product("other", self.start, self.end, self.root)
