"""Offline NOAA provider tests using real local NetCDF serialization."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import xarray as xr

from l1obs.config import DATASETS
from l1obs.fetch import ncei
from l1obs.fetch.cdaweb import FetchResult
from l1obs.fetch.models import FetchResult as SharedResult
from l1obs.fetch.magnetic import fetch_magnetic_field


class NceiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.start = pd.Timestamp("2026-04-22T00:00:00Z")
        # Unexpected requests must fail rather than accessing the network.
        self.transport = patch("l1obs.fetch.ncei.urlopen", side_effect=AssertionError("Unexpected network call"))
        self.open = self.transport.start()
        self.addCleanup(self.transport.stop)

    def dataset(self, times=None):
        if times is None:
            times = pd.date_range(self.start, periods=6, freq="s")
        counts = (times - pd.Timestamp("1958-01-01T00:00:00Z")) / pd.Timedelta(microseconds=1)
        n = len(times)
        return xr.Dataset({
            "time_sec": ("record", np.asarray(counts), {"units": ncei.TIME_UNITS}),
            "b_gse_sec": (("record", "xyz"), np.tile([3., 4., 0.], (n, 1)).astype("float32")),
            "b_gse_sphr_sec": (("record", "xyz"), np.tile([20., 0.1, 0.2], (n, 1))),
            "flags_summary": ("record", np.zeros(n)),
            "flags": ("record", np.ones(n)),  # Summary, not this bitmask, controls filtering.
        })

    def write(self, ds, name="fixture.nc"):
        path = self.root / name
        ds.to_netcdf(path, engine="netcdf4")
        return path

    def entry(self, day=None, revision="20260511T175724Z"):
        day = self.start if day is None else day
        identifier = f"discovered_{day.strftime('%Y%m%d')}_p{revision}_pub.nc"
        return {"id": identifier, "product": ncei.PRODUCT, "satellite": "SOLAR-1",
                "time_coverage_start": day.isoformat(),
                "time_coverage_end": (day + pd.Timedelta(days=1)).isoformat(),
                "file_link": f"https://example.test/{identifier}"}

    def serve(self, entries, files):
        def response(url, **kwargs):
            if url.startswith(ncei.API_URL):
                return io.BytesIO(json.dumps({"status": {"code": 200}, "data": entries}).encode())
            return io.BytesIO(files[url])
        self.open.side_effect = response

    def test_registry_wrapper_and_result_compatibility(self):
        self.assertIs(FetchResult, SharedResult)
        self.assertEqual(DATASETS["SOLAR-1"], {"mag": ncei.PRODUCT, "mag_provider": "ncei"})
        for mission in ["Wind", "ACE", "DSCOVR", "IMAP"]:
            with patch("l1obs.fetch.magnetic.fetch_cdaweb_dataset") as fetch:
                fetch_magnetic_field(mission, self.start, self.start + pd.Timedelta(seconds=1), self.root)
                self.assertEqual(fetch.call_args.args[0], DATASETS[mission.upper()]["mag"])
        with patch("l1obs.fetch.magnetic.fetch_ncei_product") as fetch:
            fetch_magnetic_field("solar-1", self.start, self.start + pd.Timedelta(seconds=1), self.root)
            self.assertEqual(fetch.call_args.args[0], ncei.PRODUCT)
        with self.assertRaisesRegex(ValueError, "No magnetic"):
            fetch_magnetic_field("SOHO", self.start, self.start, self.root)

    def test_native_normalization_archive_magnitude_and_summary_quality(self):
        ds = self.dataset()
        ds.flags_summary.values[:] = [0, 1, 2, np.nan, -9999, 0]
        frame = ncei._read_file(self.write(ds))
        self.assertEqual(frame.columns.tolist(), ["bx_gse", "by_gse", "bz_gse", "b_mag"])
        self.assertEqual(frame.index.name, "time")
        self.assertEqual(frame.index[0], self.start)
        self.assertEqual(float(ds.time_sec.values[0]), 2155507200000000.)
        self.assertEqual(str(frame.index.tz), "UTC")
        np.testing.assert_array_equal(np.diff(frame.index.asi8), np.full(5, 1_000_000_000))
        np.testing.assert_array_equal(frame.iloc[0], [3, 4, 0, 20])
        self.assertTrue(frame.iloc[1:5].isna().all().all())
        self.assertEqual(frame.b_mag.iloc[-1], 20)

    def test_fill_bounds_and_component_boundaries(self):
        ds = self.dataset()
        ds.b_gse_sec.attrs.update({"_FillValue": -9999., "valid_min": [-5, -6, -7], "valid_max": [5, 6, 7]})
        ds.b_gse_sec.values[:4] = [[-5, 6, 7], [-9999, 4, 0], [6, -7, 8], [3, 4, np.inf]]
        ds.b_gse_sphr_sec.attrs["valid_range"] = [10., 30.]
        ds.b_gse_sphr_sec.values[:, 0] = [10, 30, 31, -9999, 9, 20]
        frame = ncei._read_file(self.write(ds))
        np.testing.assert_array_equal(frame.iloc[0], [-5, 6, 7, 10])
        self.assertTrue(np.isnan(frame.bx_gse.iloc[1]))
        self.assertEqual(frame.by_gse.iloc[1], 4)
        self.assertEqual(frame.b_mag.iloc[1], 30)
        self.assertTrue(frame.iloc[2].isna().all())
        self.assertTrue(np.isnan(frame.bz_gse.iloc[3]))
        self.assertTrue(frame.b_mag.iloc[2:5].isna().all())

    def test_missing_value_scalar_bounds_and_invalid_quality_metadata(self):
        ds = self.dataset()
        ds.b_gse_sec.attrs.update({"missing_value": -8888., "valid_min": -4., "valid_max": 4.})
        ds.b_gse_sec.values[0] = [-8888, -4, 4]
        ds.flags_summary.attrs["missing_value"] = 0.  # Invalid metadata sentinel wins over good code.
        frame = ncei._read_file(self.write(ds))
        self.assertTrue(frame.isna().all().all())
        del ds.flags_summary.attrs["missing_value"]
        frame = ncei._read_file(self.write(ds, "bounds.nc"))
        self.assertTrue(np.isnan(frame.bx_gse.iloc[0]))
        np.testing.assert_array_equal(frame.iloc[0, 1:3], [-4, 4])

    def test_unsigned_quality_and_fill_fallback_without_metadata(self):
        ds = self.dataset()
        ds["flags_summary"] = ("record", np.array([0, 1, 2, 255, 0, 0], dtype="uint8"))
        ds.b_gse_sec.values[4, 1] = -9999
        ds.b_gse_sphr_sec.values[5, 0] = -9999
        for variable in ds.data_vars.values():
            variable.encoding["_FillValue"] = None
        frame = ncei._read_file(self.write(ds))
        self.assertEqual(frame.b_mag.iloc[0], 20)
        self.assertTrue(frame.iloc[1:4].isna().all().all())
        self.assertTrue(np.isnan(frame.by_gse.iloc[4]))
        self.assertTrue(np.isnan(frame.b_mag.iloc[5]))
        self.assertEqual(frame.bx_gse.iloc[5], 3)

    def test_discovery_overlap_revisions_and_query(self):
        previous = self.entry(self.start - pd.Timedelta(days=1))
        following = self.entry(self.start + pd.Timedelta(days=1))
        old, latest = self.entry(revision="20260501T000000Z"), self.entry()
        wrong = dict(latest, product="ops_mag-l3_solar1")
        self.serve([previous, following, wrong, latest, old], {})
        self.assertEqual(ncei._discover(self.start, self.start + pd.Timedelta(days=1)), [latest])
        query = parse_qs(urlparse(self.open.call_args.args[0]).query)
        self.assertEqual(query["prod"], [ncei.PRODUCT])
        self.assertEqual(query["sat"], ["SOLAR-1"])
        self.assertEqual(query["start_time"], ["2026-04-22T00:00:00Z"])
        self.assertEqual(query["end_time"], ["2026-04-23T00:00:00Z"])
        self.assertEqual(len(self.open.call_args_list), 1)

    def test_missing_days_and_ambiguous_revisions(self):
        self.serve([self.entry()], {})
        with self.assertRaisesRegex(RuntimeError, "2026-04-23"):
            ncei._discover(self.start, self.start + pd.Timedelta(days=2))
        self.serve([self.entry(), dict(self.entry(), id="unknown.nc")], {})
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            ncei._discover(self.start, self.start + pd.Timedelta(seconds=2))

    def test_multi_day_exact_trim_cache_and_force(self):
        boundary = self.start + pd.Timedelta(days=1)
        times = [pd.date_range(boundary - pd.Timedelta(seconds=3), periods=3, freq="s"),
                 pd.date_range(boundary, periods=4, freq="s")]
        entries = [self.entry(), self.entry(boundary)]
        files = {e["file_link"]: self.write(self.dataset(t), f"day{i}.nc").read_bytes()
                 for i, (e, t) in enumerate(zip(entries, times))}
        self.serve(entries, files)
        start, end = boundary - pd.Timedelta(seconds=2), boundary + pd.Timedelta(seconds=2)
        result = fetch_magnetic_field("SOLAR-1", start, end, self.root)
        pd.testing.assert_index_equal(result.df.index, pd.date_range(start, periods=4, freq="s", name="time"))
        self.assertEqual(result.dataset_id, ncei.PRODUCT)
        self.assertEqual(result.used_vars, ncei.USED_VARS)
        self.assertEqual(len(self.open.call_args_list), 4)
        self.open.side_effect = AssertionError("Cache hit accessed network")
        cached = fetch_magnetic_field("SOLAR-1", start, end, self.root)
        pd.testing.assert_frame_equal(result.df, cached.df)
        self.assertEqual(cached.used_vars, {"_cached": "true"})
        # A different interval reuses raw files, but still discovers the correct revision.
        self.serve(entries, files)
        before = self.open.call_count
        fetch_magnetic_field("SOLAR-1", start, end - pd.Timedelta(seconds=1), self.root)
        self.assertEqual(self.open.call_count - before, 2)
        before = self.open.call_count
        fetch_magnetic_field("SOLAR-1", start, end, self.root, force=True)
        self.assertEqual(self.open.call_count - before, 4)
        self.assertFalse(list(self.root.rglob("*.part")))

    def test_failed_download_cleanup(self):
        self.open.side_effect = OSError("download interrupted")
        with self.assertRaisesRegex(OSError, "interrupted"):
            ncei._download(self.entry(), self.root, False)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_malformed_products_and_times(self):
        cases = [self.dataset().drop_vars("b_gse_sec"),
                 self.dataset().assign(b_gse_sec=(("record", "short"), np.zeros((6, 2)))),
                 self.dataset().assign(flags_summary=("different", np.zeros(6)))]
        for i, ds in enumerate(cases):
            with self.subTest(i=i), self.assertRaisesRegex(RuntimeError, "Missing variable|aligned shape"):
                ncei._read_file(self.write(ds, f"malformed{i}.nc"))
        ds = self.dataset()
        ds.time_sec.values[1] = ds.time_sec.values[0]
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            ncei._read_file(self.write(ds, "duplicate.nc"))
        ds.time_sec.attrs["_FillValue"] = -9999.
        ds.time_sec.values[0] = -9999
        with self.assertRaisesRegex(RuntimeError, "Invalid time"):
            ncei._read_file(self.write(ds, "invalid.nc"))
        ds.time_sec.attrs["units"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "Unsupported time"):
            ncei._read_file(self.write(ds, "units.nc"))

    def test_discovery_failed_response_and_invalid_interval(self):
        self.open.side_effect = lambda *a, **k: io.BytesIO(b'{"status":{"code":500},"data":[]}')
        with self.assertRaisesRegex(RuntimeError, "response"):
            ncei._discover(self.start, self.start + pd.Timedelta(seconds=1))
        with self.assertRaisesRegex(ValueError, "after start"):
            ncei.fetch_ncei_product(ncei.PRODUCT, self.start, self.start, self.root)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            ncei.fetch_ncei_product("wrong", self.start, self.start, self.root)

    def test_duplicate_records_conflict_or_deduplicate(self):
        frames = [ncei._read_file(self.write(self.dataset())) for _ in range(2)]
        with patch.object(ncei, "_discover", return_value=[self.entry(), self.entry()]), \
             patch.object(ncei, "_download", return_value=self.root / "unused.nc"), \
             patch.object(ncei, "_read_file", side_effect=frames):
            result = ncei.fetch_ncei_product(ncei.PRODUCT, self.start, self.start + pd.Timedelta(seconds=6), self.root)
            self.assertEqual(len(result.df), 6)
        frames[1] = frames[1].copy()
        frames[1].iloc[0, 0] = 100
        with patch.object(ncei, "_discover", return_value=[self.entry(), self.entry()]), \
             patch.object(ncei, "_download", return_value=self.root / "unused.nc"), \
             patch.object(ncei, "_read_file", side_effect=frames), \
             self.assertRaisesRegex(RuntimeError, "Conflicting duplicate"):
            ncei.fetch_ncei_product(ncei.PRODUCT, self.start, self.start + pd.Timedelta(seconds=6), self.root, force=True)
