"""Stage 1C planar recovery, sensitivity, and controlled uncertainty coverage."""
import unittest

import numpy as np
import pandas as pd

from l1obs.analysis.shocks.timing import fit_timing, leave_one_out, angular_separation_deg
from l1obs.analysis.shocks.uncertainty import monte_carlo_timing


POSITIONS = np.array([[0, 0, 0], [80000, 0, 0], [0, 70000, 0], [0, 0, 60000],
                      [-70000, 10000, 10000], [20000, -50000, -30000]], float) + [1500000, 0, 0]
NORMAL = np.array([-.9, .3, .1])
NORMAL /= np.linalg.norm(NORMAL)
SPEED = 500.
NAMES = ["WIND", "ACE", "DSCOVR", "ADITYA-L1", "IMAP", "SOLAR-1"]
EPOCH = pd.Timestamp("2026-04-01T00:00:00Z")


def exact_times(positions=POSITIONS):
    offsets = (positions - positions.mean(axis=0)) @ NORMAL / SPEED
    return EPOCH + pd.to_timedelta(offsets, unit="s")


class TimingTests(unittest.TestCase):
    def test_exact_solution_sign_units_and_date_boundary(self):
        fit = fit_timing(POSITIONS, exact_times(), spacecraft=NAMES)
        np.testing.assert_allclose(fit.normal_gse, NORMAL, atol=1e-10)
        self.assertAlmostEqual(fit.speed_km_s, SPEED, places=7)
        np.testing.assert_allclose(fit.slowness_s_km, NORMAL / SPEED, atol=1e-12)
        self.assertLess(fit.max_abs_residual_s, 1e-8)
        self.assertEqual(fit.dof, 2)
        self.assertEqual(fit.geometry_rank, 3)
        self.assertEqual(fit.observed_times.min().day, 31)
        self.assertEqual(fit.observed_times.max().day, 1)
        delta = (fit.observed_times - fit.predicted_times).total_seconds()
        np.testing.assert_allclose(delta, fit.residuals_s, atol=1e-9)

    def test_equal_uncertainties_and_translation_invariance(self):
        times = exact_times() + pd.to_timedelta([.1, -1, .5, -.5, 1.5, 1], unit="s")
        weighted = fit_timing(POSITIONS, times, np.ones(6) * 2)
        unweighted = fit_timing(POSITIONS, times, np.ones(6) * 2, method="unweighted")
        np.testing.assert_allclose(weighted.normal_gse, unweighted.normal_gse, atol=1e-14)
        self.assertAlmostEqual(weighted.speed_km_s, unweighted.speed_km_s, places=10)
        shifted = fit_timing(POSITIONS + [1e8, -3e8, 4e8], times + pd.Timedelta(days=500), np.ones(6) * 2)
        np.testing.assert_allclose(shifted.slowness_s_km, weighted.slowness_s_km, atol=1e-14)
        self.assertAlmostEqual(weighted.chi_square, np.sum((weighted.residuals_s / 2) ** 2))

    def test_noisy_recovery_and_differential_precision(self):
        rng = np.random.default_rng(123)
        sigma = np.array([1, 1, 1, 5, .5, 1])
        noisy = exact_times() + pd.to_timedelta(rng.normal(size=6) * sigma, unit="s")
        fit = fit_timing(POSITIONS, noisy, sigma)
        self.assertLess(abs(fit.speed_km_s - SPEED), 15)
        self.assertLess(angular_separation_deg(fit.normal_gse, NORMAL), 2)
        corrupted = exact_times() + pd.to_timedelta([40, 0, 0, 0, 0, 0], unit="s")
        weighted = fit_timing(POSITIONS, corrupted, [100, 1, 1, 1, 1, 1])
        unweighted = fit_timing(POSITIONS, corrupted)
        self.assertLess(abs(weighted.speed_km_s - SPEED), abs(unweighted.speed_km_s - SPEED))
        self.assertLess(angular_separation_deg(weighted.normal_gse, NORMAL), .01)

    def test_geometry_rank_condition_and_zero_slowness(self):
        for scale, message in ((0., "Rank-deficient"), (1e-9, "Ill-conditioned")):
            positions = POSITIONS.copy()
            positions[:, 2] *= scale
            with self.assertRaisesRegex(ValueError, message):
                fit_timing(positions, exact_times())
        positions = POSITIONS.copy()
        positions[:, 2] *= 1e-4
        self.assertTrue(any("conditioning" in f for f in fit_timing(positions, exact_times()).flags))
        with self.assertRaisesRegex(ValueError, "slowness"):
            fit_timing(POSITIONS, [EPOCH] * 6)

    def test_missing_and_invalid_errors_and_four_member_solution(self):
        for sigma in ([1] * 5, [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, np.nan], [1, 1, 1, 1, 1, -1]):
            with self.assertRaisesRegex(ValueError, "uncertainties"):
                fit_timing(POSITIONS, exact_times(), sigma)
        with self.assertRaisesRegex(ValueError, "At least four"):
            fit_timing(POSITIONS[:3], exact_times()[:3])
        with self.assertRaisesRegex(ValueError, "equal length"):
            fit_timing(POSITIONS, exact_times()[:5])
        fit = fit_timing(POSITIONS[:4], exact_times()[:4], np.ones(4))
        self.assertEqual(fit.dof, 0)
        self.assertIsNone(fit.reduced_chi_square)

    def test_covariance_gls_and_bad_covariance(self):
        sigma = np.ones(6)
        covariance = np.diag(sigma) + np.ones((6, 6)) * 4
        fit = fit_timing(POSITIONS, exact_times(), covariance_s2=covariance)
        self.assertAlmostEqual(fit.speed_km_s, SPEED, places=7)
        for covariance in (np.zeros((6, 6)), np.eye(5), np.eye(6) + np.triu(np.ones((6, 6)), 1)):
            with self.assertRaises(ValueError):
                fit_timing(POSITIONS, exact_times(), covariance_s2=covariance)

    def test_leave_one_out_incorrect_arrival_and_subset_failures(self):
        times = exact_times() + pd.to_timedelta([0, 0, 0, 0, 40, 0], unit="s")
        full = fit_timing(POSITIONS, times, np.ones(6), spacecraft=NAMES)
        results = leave_one_out(POSITIONS, times, np.ones(6), spacecraft=NAMES, full_fit=full)
        recovered = results[4]
        self.assertEqual(recovered["excluded_spacecraft"], "IMAP")
        self.assertAlmostEqual(recovered["speed_km_s"], SPEED, places=7)
        self.assertLess(recovered["rms_residual_s"], 1e-8)
        self.assertGreater(recovered["angular_difference_deg"], .1)
        self.assertEqual(len(results), 6)
        positions = POSITIONS.copy()
        positions[:, 2] = 0
        positions[0, 2] = 10000
        results = leave_one_out(positions, exact_times(positions))
        self.assertEqual(results[0]["status"], "failed")
        self.assertIn("Rank-deficient", results[0]["error"])


    def test_misfit_and_ols_diagnostic_flags(self):
        times = exact_times() + pd.to_timedelta([40, 0, 0, 0, 0, 0], unit="s")
        fit = fit_timing(POSITIONS, times, np.ones(6))
        self.assertTrue(any("Large residuals" in f for f in fit.flags))
        self.assertGreater(fit.reduced_chi_square, 4)
        fit = fit_timing(POSITIONS, times, np.arange(1, 7), method="unweighted")
        self.assertTrue(any("not a calibrated" in f for f in fit.flags))


class MonteCarloTests(unittest.TestCase):
    def test_fixed_seed_reproducibility_directional_measures(self):
        args = dict(spacecraft=NAMES, samples=250, seed=991, recompute_positions=False)
        first = monte_carlo_timing(POSITIONS, exact_times(), np.ones(6), **args)
        second = monte_carlo_timing(POSITIONS, exact_times(), np.ones(6), **args)
        self.assertEqual(first, second)
        self.assertEqual(first["failed"], 0)
        self.assertLess(first["speed_interval_95_km_s"][0], SPEED)
        self.assertGreater(first["speed_interval_95_km_s"][1], SPEED)
        self.assertGreater(first["angular_radius_95_deg"], first["angular_radius_68_deg"])

    def test_ephemeris_recomputed_and_failures_reported(self):
        times = exact_times()
        ephemerides = {}
        for name, position, time in zip(NAMES, POSITIONS, times):
            ephemerides[name] = pd.DataFrame([position, position + [10, 0, 0]],
                index=[time, time + pd.Timedelta(seconds=10)], columns=["x_km", "y_km", "z_km"])
        result = monte_carlo_timing(POSITIONS, times, np.ones(6), spacecraft=NAMES,
                                    samples=150, seed=1, ephemerides=ephemerides)
        self.assertGreater(result["failed"], 0)
        self.assertEqual(result["failed"] + result["successful"], 150)
        self.assertTrue(result["failure_reasons"])
        self.assertTrue(result["recomputed_positions"])
        self.assertEqual(len(result["draws"]), 150)
        # Supply broad brackets; moving positions must affect the propagated fit.
        for name, position, time in zip(NAMES, POSITIONS, times):
            ephemerides[name] = pd.DataFrame([position - [10000, 0, 0], position + [10000, 0, 0]],
                index=[time - pd.Timedelta(seconds=100), time + pd.Timedelta(seconds=100)],
                columns=["x_km", "y_km", "z_km"])
        moving = monte_carlo_timing(POSITIONS, times, np.ones(6), spacecraft=NAMES, samples=100,
                                    seed=1, ephemerides=ephemerides)
        fixed = monte_carlo_timing(POSITIONS, times, np.ones(6), spacecraft=NAMES, samples=100,
                                   seed=1, recompute_positions=False)
        self.assertEqual(moving["failed"], 0)
        self.assertNotEqual(moving["speed_interval_95_km_s"], fixed["speed_interval_95_km_s"])

    def test_broad_normal_distribution_and_invalid_mc_options(self):
        result = monte_carlo_timing(POSITIONS, exact_times(), np.ones(6) * 500,
                                    samples=100, seed=4, recompute_positions=False)
        self.assertEqual(result["status"], "unstable")
        self.assertTrue(result["flags"])
        for args in ({"samples": 1}, {"seed": None}, {"seed": -1}):
            with self.assertRaises(ValueError):
                monte_carlo_timing(POSITIONS, exact_times(), np.ones(6), recompute_positions=False, **args)

    def test_controlled_interval_coverage(self):
        # 60 independent noisy events; broad binomial tolerances avoid flaky tests.
        # Intervals are estimated by independent parametric timing draws per event.
        rng = np.random.default_rng(922)
        counts = np.zeros(4, dtype=int)
        for i in range(60):
            times = exact_times() + pd.to_timedelta(rng.normal(size=6), unit="s")
            result = monte_carlo_timing(POSITIONS, times, np.ones(6), samples=180,
                                        seed=i, recompute_positions=False)
            counts[0] += result["speed_interval_68_km_s"][0] <= SPEED <= result["speed_interval_68_km_s"][1]
            counts[1] += result["speed_interval_95_km_s"][0] <= SPEED <= result["speed_interval_95_km_s"][1]
            angle = angular_separation_deg(result["central_normal_gse"], NORMAL)
            counts[2] += angle <= result["angular_radius_68_deg"]
            counts[3] += angle <= result["angular_radius_95_deg"]
        self.assertTrue(25 <= counts[0] <= 53, counts)
        self.assertTrue(50 <= counts[1] <= 60, counts)
        self.assertTrue(25 <= counts[2] <= 53, counts)
        self.assertTrue(50 <= counts[3] <= 60, counts)
