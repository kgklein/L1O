import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
from cdasws.datarepresentation import DataRepresentation

from l1obs.config import DATASETS
from l1obs.fetch.cdaweb import _cache_path, _resolve_vars, fetch_cdaweb_dataset
from l1obs.fetch.variable_map import MAGNETIC_PRODUCTS


class MagneticFetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.start = pd.Timestamp("2026-09-01T00:00:00Z")
        self.end = self.start + pd.Timedelta(seconds=20)
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.client_class = stack.enter_context(patch("l1obs.fetch.cdaweb.CdasWs"))
        self.client = self.client_class.return_value
        stack.enter_context(patch("socket.create_connection", side_effect=AssertionError("network forbidden")))

    def product(self, dataset, vector=None, magnitude=None, flags=None, dtype=np.float64):
        spec = MAGNETIC_PRODUCTS[dataset]
        vector = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                            if vector is None else vector, dtype=dtype)
        magnitude = np.asarray([10, 20, 30] if magnitude is None else magnitude, dtype=dtype)
        time = pd.date_range(self.start, periods=len(vector), freq=pd.Timedelta(seconds=spec.cadence_seconds))
        attrs = {"FILLVAL": dtype(-1e31), "VALIDMIN": -100.0, "VALIDMAX": 100.0}
        variables = {
            spec.vector: xr.DataArray(vector, dims=(spec.time, "component"), attrs=attrs.copy()),
            spec.magnitude: xr.DataArray(magnitude, dims=(spec.time,), attrs=attrs.copy()),
        }
        if spec.quality:
            variables[spec.quality] = xr.DataArray(
                [0] * len(vector) if flags is None else flags,
                dims=(spec.time,), attrs={"FILLVAL": -999, "VALIDMIN": 0, "VALIDMAX": 65535},
            )
        return xr.Dataset(variables, coords={spec.time: time.tz_localize(None)})

    def fetch(self, dataset="WI_H0_MFI", data=None, force=False):
        self.client.get_data.return_value = ({"http": {"status_code": 200}},
                                             self.product(dataset) if data is None else data)
        return fetch_cdaweb_dataset(dataset, self.start, self.end, self.cache, force=force)

    def test_exact_mission_mappings_native_cadence_and_archive_magnitude(self):
        expected = {
            "WIND": ("WI_H0_MFI", "Epoch3", "B3GSE", "B3F1", 3, None),
            "ACE": ("AC_H3_MFI", "Epoch", "BGSEc", "Magnitude", 1, None),
            "DSCOVR": ("DSCOVR_H0_MAG", "Epoch1", "B1GSE", "B1F1", 1, "FLAG1"),
            "IMAP": ("IMAP_MAG_L2_NORM-GSE", "epoch", "b_gse", "magnitude", 0.5, "quality_flags"),
        }
        self.assertEqual(set(MAGNETIC_PRODUCTS), {item[0] for item in expected.values()})
        for mission, (dataset, time, vector, magnitude, cadence, quality) in expected.items():
            with self.subTest(mission=mission):
                self.assertEqual(DATASETS[mission]["mag"], dataset)
                spec = MAGNETIC_PRODUCTS[dataset]
                self.assertEqual((spec.time, spec.vector, spec.magnitude, spec.cadence_seconds, spec.quality),
                                 (time, vector, magnitude, cadence, quality))
                used = {"time": time, "B_GSE": vector, "B_MAG": magnitude}
                if quality:
                    used["quality"] = quality
                self.assertEqual(_resolve_vars(self.client, dataset), used)
                result = self.fetch(dataset)
                self.assertEqual(result.used_vars, used)
                self.assertEqual(result.df.columns.tolist(), ["bx_gse", "by_gse", "bz_gse", "b_mag"])
                self.assertEqual(result.df.index.name, "time")
                self.assertEqual(str(result.df.index.tz), "UTC")
                self.assertEqual(result.df.index.tolist(),
                                 pd.date_range(self.start, periods=3, freq=pd.Timedelta(seconds=cadence)).tolist())
                np.testing.assert_array_equal(result.df.iloc[0], [1, 2, 3, 10])
                self.assertNotEqual(result.df.b_mag.iloc[0], np.linalg.norm([1, 2, 3]))
                requested = [vector, magnitude] + ([quality] if quality else [])
                self.client.get_data.assert_called_with(
                    dataset, requested, self.start.to_pydatetime(), self.end.to_pydatetime(),
                    dataRepresentation=DataRepresentation.XARRAY,
                )
        self.client.get_variables.assert_not_called()

    def test_fill_masking_at_source_precision_preserves_rows(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                data = self.product("WI_H0_MFI", vector=[[1, -1e31, 3], [-1e31] * 3, [7, 8, 9]],
                                    magnitude=[10, -1e31, 30], dtype=dtype)
                # No bounds: these records must be masked by FILLVAL itself.
                for name in ["B3GSE", "B3F1"]:
                    data[name].attrs = {"FILLVAL": -1e31}
                with self.assertLogs("l1obs.fetch.cdaweb", level="WARNING"):
                    df = self.fetch(data=data, force=True).df
                self.assertEqual(len(df), 3)
                self.assertTrue(np.isnan(df.by_gse.iloc[0]))
                self.assertEqual(df.bx_gse.iloc[0], 1)
                self.assertTrue(df.iloc[1].isna().all())
                self.assertEqual(df.b_mag.iloc[2], 30)

    def test_missing_fill_metadata_uses_float32_and_float64_sentinels(self):
        data = self.product("WI_H0_MFI", vector=[[float(np.float32(-1e31)), -1e31, 3], [4, 5, 6], [7, 8, 9]])
        for name in ["B3GSE", "B3F1"]:
            data[name].attrs = {}
        with self.assertLogs("l1obs.fetch.cdaweb", level="WARNING"):
            df = self.fetch(data=data).df
        self.assertTrue(df.iloc[0, :2].isna().all())
        self.assertEqual(df.bz_gse.iloc[0], 3)

    def test_component_bounds_and_scalar_bounds_are_inclusive(self):
        data = self.product("WI_H0_MFI", vector=[[-10, -20, -30], [10, 20, 30], [-11, 21, np.inf]],
                            magnitude=[0, 100, 101])
        data.B3GSE.attrs.update(VALIDMIN=[-10, -20, -30], VALIDMAX=[10, 20, 30])
        data.B3F1.attrs.update(VALIDMIN=0, VALIDMAX=100)
        df = self.fetch(data=data).df
        np.testing.assert_array_equal(df.iloc[0], [-10, -20, -30, 0])
        np.testing.assert_array_equal(df.iloc[1], [10, 20, 30, 100])
        self.assertTrue(df.iloc[2].isna().all())

    def test_dscovr_nonzero_missing_and_fill_flags_mask_all_fields(self):
        data = self.product("DSCOVR_H0_MAG", vector=[[1, 2, 3]] * 5,
                            magnitude=[10] * 5, flags=[0, 1, 2, np.nan, -999])
        df = self.fetch("DSCOVR_H0_MAG", data).df
        np.testing.assert_array_equal(df.iloc[0], [1, 2, 3, 10])
        self.assertTrue(df.iloc[1:].isna().all().all())
        self.assertEqual(len(df), 5)

    def test_imap_nonzero_missing_and_fill_flags_mask_all_fields(self):
        dataset = "IMAP_MAG_L2_NORM-GSE"
        data = self.product(dataset, vector=[[1, 2, 3]] * 6,
                            magnitude=[10] * 6, flags=[0, 1, 2, np.nan, -999, 65536])
        df = self.fetch(dataset, data).df
        np.testing.assert_array_equal(df.iloc[0], [1, 2, 3, 10])
        self.assertTrue(df.iloc[1:].isna().all().all())
        self.assertEqual(len(df), 6)
        self.assertEqual(df.index[1] - df.index[0], pd.Timedelta(milliseconds=500))

    def test_imap_metadata_masks_fills_and_component_and_scalar_bounds(self):
        dataset = "IMAP_MAG_L2_NORM-GSE"
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                data = self.product(dataset,
                                    vector=[[-10, -20, -30], [10, 20, 30],
                                            [-11, 21, -1e31], [1, 2, 3], [4, 5, 6]],
                                    magnitude=[0, 100, -1e31, -1, 101], dtype=dtype)
                data.b_gse.attrs.update(VALIDMIN=[-10, -20, -30], VALIDMAX=[10, 20, 30])
                data.magnitude.attrs.update(VALIDMIN=0, VALIDMAX=100)
                df = self.fetch(dataset, data, force=True).df
                np.testing.assert_array_equal(df.iloc[0], [-10, -20, -30, 0])
                np.testing.assert_array_equal(df.iloc[1], [10, 20, 30, 100])
                self.assertTrue(df.iloc[2].isna().all())
                self.assertTrue(df.b_mag.iloc[3:].isna().all())
                np.testing.assert_array_equal(df.iloc[3, :3], [1, 2, 3])

    def test_imap_fill_metadata_masks_values_without_validity_bounds(self):
        dataset = "IMAP_MAG_L2_NORM-GSE"
        data = self.product(dataset, vector=[[1, -1e31, 3], [4, 5, 6], [7, 8, 9]],
                            magnitude=[10, -1e31, 30], dtype=np.float32)
        for name in ["b_gse", "magnitude"]:
            data[name].attrs = {"FILLVAL": -1e31}
        with self.assertLogs("l1obs.fetch.cdaweb", level="WARNING"):
            df = self.fetch(dataset, data).df
        self.assertTrue(np.isnan(df.by_gse.iloc[0]))
        self.assertTrue(np.isnan(df.b_mag.iloc[1]))
        self.assertEqual(df.bx_gse.iloc[0], 1)

    def test_imap_missing_epoch_or_quality_and_malformed_vector_raise(self):
        dataset = "IMAP_MAG_L2_NORM-GSE"
        for name in ("epoch", "quality_flags"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, f"Missing required variables: {name}"):
                    self.fetch(dataset, self.product(dataset).drop_vars(name))
        data = self.product(dataset)
        data["b_gse"] = xr.DataArray(np.ones((3, 2)), dims=("epoch", "bad_component"))
        with self.assertRaisesRegex(RuntimeError, "shape"):
            self.fetch(dataset, data)

    def test_component_fill_attributes_and_scalar_lower_bound(self):
        data = self.product("WI_H0_MFI", vector=[[-999, 2, 3], [4, -998, 6], [7, 8, -997]],
                            magnitude=[-1, 0, 100])
        data.B3GSE.attrs.update(FILLVAL=[-999, -998, -997], VALIDMIN=-1000)
        data.B3F1.attrs.update(VALIDMIN=0)
        df = self.fetch(data=data).df
        self.assertTrue(np.isnan(df.bx_gse.iloc[0]))
        self.assertTrue(np.isnan(df.by_gse.iloc[1]))
        self.assertTrue(np.isnan(df.bz_gse.iloc[2]))
        np.testing.assert_array_equal(df.b_mag, [np.nan, 0, 100])

    def test_timezone_aware_epoch_is_converted_to_utc(self):
        data = self.product("WI_H0_MFI")
        epoch = pd.date_range(self.start, periods=3, freq="3s").tz_convert("America/Phoenix")
        data = data.assign_coords(Epoch3=epoch.tolist())
        df = self.fetch(data=data).df
        self.assertEqual(df.index[0], self.start)
        self.assertEqual(str(df.index.tz), "UTC")

    def test_nonmagnetic_fetch_and_cache_keep_existing_response_behavior(self):
        raw = pd.DataFrame({"Np": [1.0, 2.0]},
                           index=pd.date_range("2026-09-01", periods=2, freq="64s"))
        self.client.get_variables.return_value = [{"Name": "Np"}]
        self.client.get_data.return_value = {"data": raw}
        result = fetch_cdaweb_dataset("AC_H0_SWE", self.start, self.end, self.cache)
        self.assertEqual(result.used_vars, {"NP": "Np"})
        self.assertEqual(result.df.columns.tolist(), ["Np"])
        self.assertEqual(str(result.df.index.tz), "UTC")
        self.client.get_data.assert_called_once_with(
            "AC_H0_SWE", ["Np"], self.start.to_pydatetime().replace(tzinfo=None),
            self.end.to_pydatetime().replace(tzinfo=None), data_type="pandas",
        )
        self.client_class.reset_mock()
        cached = fetch_cdaweb_dataset("AC_H0_SWE", self.start, self.end, self.cache)
        self.client_class.assert_not_called()
        pd.testing.assert_frame_equal(cached.df, result.df, check_freq=False)

    def test_cache_round_trip_is_offline_and_force_refetches(self):
        for dataset in ("WI_H0_MFI", "IMAP_MAG_L2_NORM-GSE"):
            with self.subTest(dataset=dataset):
                first = self.fetch(dataset)
                self.client_class.reset_mock()
                cached = fetch_cdaweb_dataset(dataset, self.start, self.end, self.cache)
                self.client_class.assert_not_called()
                pd.testing.assert_frame_equal(cached.df, first.df)
                self.fetch(dataset, force=True)
                self.client_class.assert_called_once()

    def test_legacy_magnetic_cache_is_ignored_and_plasma_name_is_unchanged(self):
        stamp = "20260901T000000_20260901T000020"
        legacy = self.cache / f"WI_H0_MFI_{stamp}.parquet"
        pd.DataFrame({"old": [1]}, index=[self.start]).to_parquet(legacy)
        result = self.fetch()
        self.assertNotIn("old", result.df)
        self.client.get_data.assert_called_once()
        self.assertTrue(legacy.exists())
        self.assertEqual(_cache_path(self.cache, "AC_H0_SWE", self.start, self.end).name,
                         f"AC_H0_SWE_{stamp}.parquet")

    def test_missing_required_variables_raise(self):
        for name in ("Epoch3", "B3GSE", "B3F1"):
            with self.subTest(name=name):
                data = self.product("WI_H0_MFI").drop_vars(name)
                with self.assertRaisesRegex(RuntimeError, f"Missing required variables: {name}"):
                    self.fetch(data=data)
        data = self.product("DSCOVR_H0_MAG").drop_vars("FLAG1")
        with self.assertRaisesRegex(RuntimeError, "FLAG1"):
            self.fetch("DSCOVR_H0_MAG", data)

    def test_malformed_vector_shape_and_unaligned_scalar_raise(self):
        data = self.product("WI_H0_MFI")
        data["B3GSE"] = xr.DataArray(np.ones((3, 2)), dims=("Epoch3", "bad_component"))
        with self.assertRaisesRegex(RuntimeError, "shape"):
            self.fetch(data=data)
        data = self.product("WI_H0_MFI")
        data["B3F1"] = xr.DataArray([1, 2, 3], dims=("other_time",))
        with self.assertRaisesRegex(RuntimeError, "align"):
            self.fetch(data=data)

    def test_empty_product_raises(self):
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            self.fetch(data=self.product("WI_H0_MFI").isel(Epoch3=slice(0, 0)))

    def test_http_failure_and_no_data_raise(self):
        for status_code, data in [(503, None), (200, None)]:
            self.client.get_data.return_value = ({"http": {"status_code": status_code}}, data)
            with self.assertRaisesRegex(RuntimeError, f"HTTP {status_code}"):
                fetch_cdaweb_dataset("WI_H0_MFI", self.start, self.end, self.cache)


if __name__ == "__main__":
    unittest.main()
