"""Reproducible timing-error propagation; every failed draw is accounted for."""
from __future__ import annotations

from collections import Counter
import logging

import numpy as np
import pandas as pd

from .data import R_COLUMNS, prepare_table
from .event import positive
from .geometry import _position_at_prepared
from .timing import fit_timing, angular_separation_deg, _noise_model

log = logging.getLogger(__name__)


def monte_carlo_timing(positions_km, times, uncertainties_s=None, *, spacecraft=None,
                       samples=2000, seed=0, ephemerides=None, recompute_positions=True,
                       ephemeris_max_gap_s=900, covariance_s2=None, method="auto", condition_limit=1e6) -> dict:
    """Independent Gaussian 1-sigma timing errors, or supplied Gaussian covariance.

    Positions can be recomputed at each perturbed epoch using the same bounded
    interpolation as the nominal fit. Ephemeris uncertainty is not modeled.
    Returned intervals are conditional on successful draws, explicitly flagged
    if any draws failed. The raw draw table includes failed rows and errors.
    """
    if isinstance(samples, bool) or int(samples) != samples or samples < 2:
        raise ValueError("Monte Carlo samples must be an integer >= 2")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("A deterministic nonnegative integer Monte Carlo seed is required")
    samples = int(samples)
    limit = positive(ephemeris_max_gap_s, "ephemeris_max_gap_s")
    full = fit_timing(positions_km, times, uncertainties_s, spacecraft=spacecraft,
                      method=method, covariance_s2=covariance_s2, condition_limit=condition_limit)
    sigma, covariance = _noise_model(len(full.spacecraft), uncertainties_s, covariance_s2)
    if covariance is None:
        raise ValueError("Monte Carlo requires supplied timing uncertainties/covariance")
    prepared = None
    if recompute_positions:
        if ephemerides is None or any(name not in ephemerides for name in full.spacecraft):
            raise ValueError("Recomputing positions requires ephemerides for every fit spacecraft")
        prepared = {name: prepare_table(ephemerides[name], R_COLUMNS, "km", name) for name in full.spacecraft}
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((samples, len(full.spacecraft))) @ np.linalg.cholesky(covariance).T
    rows, failures, speeds, normals = [], Counter(), [], []
    condition_flags = 0
    for index, perturbation in enumerate(draws):
        row = {"realization": index, "status": "failed", "error": ""}
        try:
            perturbed = full.observed_times + pd.to_timedelta(perturbation, unit="s")
            positions = positions_km if prepared is None else np.asarray([
                _position_at_prepared(prepared[name], time, limit)[0]
                for name, time in zip(full.spacecraft, perturbed)])
            fit = fit_timing(positions, perturbed, sigma, spacecraft=full.spacecraft, method=method,
                             covariance_s2=covariance, condition_limit=condition_limit)
            speeds.append(fit.speed_km_s)
            normals.append(fit.normal_gse)
            weak = fit.geometry_condition > 1e3 or fit.design_condition > 1e3
            condition_flags += int(weak)
            row.update(status="ok", speed_km_s=fit.speed_km_s, nx_gse=fit.normal_gse[0],
                       ny_gse=fit.normal_gse[1], nz_gse=fit.normal_gse[2],
                       rms_residual_s=fit.rms_residual_s, max_abs_residual_s=fit.max_abs_residual_s,
                       chi_square=fit.chi_square, reduced_chi_square=fit.reduced_chi_square,
                       geometry_condition=fit.geometry_condition, design_condition=fit.design_condition,
                       weak_conditioning=weak, residuals_s=dict(zip(full.spacecraft, fit.residuals_s.tolist())))
        except (ValueError, np.linalg.LinAlgError, OverflowError) as exc:
            failures[str(exc).split(": ")[-1]] += 1
            row["error"] = str(exc)
        rows.append(row)
    success = len(speeds)
    failed = samples - success
    summary = {"samples": samples, "seed": int(seed), "successful": success, "failed": failed,
               "failure_fraction": failed / samples, "failure_reasons": dict(failures),
               "weak_conditioning_realizations": condition_flags,
               "error_model": "multivariate Gaussian" if covariance_s2 is not None else "independent Gaussian",
               "timing_covariance_s2": covariance.tolist(), "recomputed_positions": recompute_positions,
               "ephemeris_max_gap_s": limit, "intervals_conditional_on_success": True,
               "status": "ok" if failed == 0 else "flagged",
               "flags": [], "draws": rows}
    if condition_flags:
        summary["status"] = "flagged"
        summary["flags"].append(f"{condition_flags}/{samples} realizations have weak conditioning (>1000)")
    if failed:
        summary["flags"].append(f"{failed}/{samples} Monte Carlo fits failed; intervals condition on successful draws")
    if success < max(2, .8 * samples):
        summary["status"] = "unstable"
        summary["flags"].append("Fewer than 80% of draws succeeded; uncertainty summary is unreliable")
    if success >= 2:
        normals = np.asarray(normals)
        resultant = normals.mean(axis=0)
        concentration = float(np.linalg.norm(resultant))
        central = full.normal_gse if concentration < 1e-12 else resultant / concentration
        angles = angular_separation_deg(normals, central)
        summary.update(speed_median_km_s=float(np.median(speeds)),
                       speed_interval_68_km_s=np.percentile(speeds, [16, 84]).tolist(),
                       speed_interval_95_km_s=np.percentile(speeds, [2.5, 97.5]).tolist(),
                       central_normal_gse=central.tolist(), central_normal_definition="normalized mean of successful unit normals",
                       normal_mean_resultant_length=concentration,
                       angular_radius_68_deg=float(np.percentile(angles, 68)),
                       angular_radius_95_deg=float(np.percentile(angles, 95)),
                       angle_from_nominal_68_deg=float(np.percentile(angular_separation_deg(normals, full.normal_gse), 68)),
                       angle_from_nominal_95_deg=float(np.percentile(angular_separation_deg(normals, full.normal_gse), 95)))
        if concentration < .9 or summary["angular_radius_95_deg"] > 45:
            summary["status"] = "unstable"
            summary["flags"].append("Broad/multidirectional normal distribution; a single confidence cone is insufficient")
        if any(float(np.dot(normal, full.normal_gse)) < 0 for normal in normals):
            if summary["status"] == "ok":
                summary["status"] = "flagged"
            summary["flags"].append("Some realizations reverse propagation hemisphere relative to nominal fit")
    log.info("Monte Carlo: %d successful, %d failed, %d weakly conditioned", success, failed, condition_flags)
    return summary
